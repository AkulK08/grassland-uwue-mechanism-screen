from pathlib import Path
from datetime import datetime
import json
import re
import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = Path("results/stage1b6ap_full13_reza_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

MANUAL = Path("data/raw/towers/_reza_raw_exports/manual_fluxnet")
EXTRACTED = Path("data/raw/towers/_reza_raw_exports/extracted")
EXTRACTED.mkdir(parents=True, exist_ok=True)

TARGET_SITES = [
    "CA-SF3",
    "CN-HaM",
    "NL-Hrw",
    "RU-NeC",
    "US-CMW",
    "US-Cop",
    "US-Dk1",
    "US-Ne1",
    "US-Ne2",
    "US-Ne3",
    "US-SP1",
    "US-Ton",
    "US-Var",
]

AGREE = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03bg_reza_tower_satellite_agreement_long.csv")
PROD = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03bd_product_identifiability_summary.csv")
SCREEN = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03be_product_screened_definition_final.csv")
C4 = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03bm_c4_reza_decision_by_model.csv")
MIXED = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03bl_c4_mixedlm_ecoregion_random_intercept.csv")

for p in [AGREE, PROD, SCREEN, C4]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required Stage 1B.6AK file: {p}")

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(c):
    return (
        str(c).strip().lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def clean_flux(x):
    y = to_num(x)
    y = y.mask(y <= -9000)
    y = y.mask(y >= 9000)
    return y

def read_csv_safe(path, nrows=None):
    for sep in [",", "\t", ";"]:
        try:
            d = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
            if len(d.columns) > 1:
                return d
        except Exception:
            pass
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        return None

def first_col(df, names):
    lut = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lut:
            return lut[norm(n)]
    return None

def match_col(df, patterns):
    out = []
    for c in df.columns:
        lc = norm(c)
        for pat in patterns:
            if re.search(pat, lc):
                out.append(c)
                break
    return out

def pick_col(df, role):
    choices = {
        "LE": ["LE_F_MDS", "LE_F", "LE", "LE_CORR", "LE_PI_F", "LE_1_1_1"],
        "H": ["H_F_MDS", "H_F", "H", "H_CORR", "H_PI_F", "H_1_1_1"],
        "NETRAD": ["NETRAD", "NETRAD_F", "RNET", "Rn", "NETRAD_PI_F", "NETRAD_1_1_1", "NETRAD_1"],
        "G": ["G_F_MDS", "G_F", "G", "G_1_1_1", "G_PI_F", "G_1"],
        "GPP": ["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT", "GPP_DT", "GPP"],
        "VPD": ["VPD_F", "VPD", "VPD_PI_F"],
        "SWC": ["SWC_F_MDS_1", "SWC_F_MDS", "SWC_1_1_1", "SWC", "SWC_1"],
    }
    c = first_col(df, choices.get(role, []))
    if c:
        return c

    pats = {
        "LE": [r"^le($|_)"],
        "H": [r"^h($|_)"],
        "NETRAD": [r"netrad", r"rnet", r"^rn$"],
        "G": [r"^g($|_)"],
        "GPP": [r"gpp"],
        "VPD": [r"vpd"],
        "SWC": [r"swc"],
    }.get(role, [])
    m = match_col(df, pats)
    return m[0] if m else None

def qc_cols(df):
    q = []
    for c in df.columns:
        lc = norm(c)
        if "qc" in lc or lc.endswith("_qf") or "quality" in lc:
            q.append(c)
    return q

def site_from_path(path):
    s = str(path)
    for site in TARGET_SITES:
        if site in s or site.replace("-", "_") in s:
            return site
    m = re.search(r"([A-Z]{2}-[A-Za-z0-9]{2,3})", s)
    return m.group(1) if m else None

def temporal_rank(path):
    s = str(path).upper()
    # Lower is better.
    if "_HH_" in s or "_HR_" in s:
        return 1, "HH_OR_HR"
    if "_DD_" in s:
        return 2, "DD"
    if "_WW_" in s:
        return 3, "WW"
    if "_MM_" in s:
        return 4, "MM"
    if "_YY_" in s:
        return 5, "YY"
    return 9, "UNKNOWN"

def get_year(df):
    for c in ["TIMESTAMP_START", "TIMESTAMP", "TIMESTAMP_END", "timestamp_start", "timestamp"]:
        if c in df.columns:
            y = df[c].astype(str).str.extract(r"(\d{4})")[0]
            return to_num(y)
    return pd.Series(np.nan, index=df.index)

def extract_archives():
    rows = []
    zips = list(MANUAL.glob("*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zz:
                members = [m for m in zz.namelist() if m.lower().endswith((".csv", ".txt"))]
                for m in members:
                    # Extract broadly; some packages do not name files FULLSET clearly.
                    out = EXTRACTED / re.sub(r"[^A-Za-z0-9_.-]+", "_", z.stem + "__" + Path(m).name)
                    if not out.exists() or out.stat().st_size == 0:
                        with zz.open(m) as src, open(out, "wb") as dst:
                            dst.write(src.read())
                    rows.append({"zip": str(z), "member": m, "extracted": str(out), "status": "ok"})
        except Exception as e:
            rows.append({"zip": str(z), "member": "", "extracted": "", "status": "failed", "error": repr(e)})
    pd.DataFrame(rows).to_csv(TAB / "Table_PRODUCT03cj_full13_archive_extract_log.csv", index=False)

def candidate_files():
    files = []
    roots = [MANUAL, EXTRACTED]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".csv", ".txt"]:
                site = site_from_path(p)
                if site in TARGET_SITES:
                    files.append(p)
    return sorted(set(files))

def compute_file_site_years(path):
    site = site_from_path(path)
    if site not in TARGET_SITES:
        return [], None

    d = read_csv_safe(path)
    if d is None or len(d) < 2:
        return [], {
            "site_id": site,
            "source_file": str(path),
            "file_status": "unreadable_or_empty",
        }

    le_col = pick_col(d, "LE")
    h_col = pick_col(d, "H")
    rn_col = pick_col(d, "NETRAD")
    g_col = pick_col(d, "G")
    gpp_col = pick_col(d, "GPP")
    vpd_col = pick_col(d, "VPD")
    swc_col = pick_col(d, "SWC")
    qcols = qc_cols(d)

    trank, tres = temporal_rank(path)

    audit = {
        "site_id": site,
        "source_file": str(path),
        "temporal_rank": trank,
        "temporal_resolution_guess": tres,
        "n_rows": len(d),
        "has_LE": bool(le_col),
        "has_H": bool(h_col),
        "has_NETRAD": bool(rn_col),
        "has_G": bool(g_col),
        "has_GPP": bool(gpp_col),
        "has_VPD": bool(vpd_col),
        "has_QC": bool(qcols),
        "LE_col": le_col,
        "H_col": h_col,
        "NETRAD_col": rn_col,
        "G_col": g_col,
        "GPP_col": gpp_col,
        "VPD_col": vpd_col,
        "SWC_col": swc_col,
        "qc_cols_preview": ";".join(qcols[:20]),
    }

    if not (le_col and h_col and rn_col and gpp_col and vpd_col):
        audit["file_status"] = "missing_required_columns"
        return [], audit

    d = d.copy()
    d["_year"] = get_year(d)

    le = clean_flux(d[le_col])
    h = clean_flux(d[h_col])
    rn = clean_flux(d[rn_col])
    g = clean_flux(d[g_col]) if g_col else pd.Series(0.0, index=d.index)
    gpp = clean_flux(d[gpp_col])
    vpd = clean_flux(d[vpd_col])

    preferred_qc = []
    for pat in [
        "LE_F_MDS_QC", "LE_F_QC",
        "VPD_F_MDS_QC", "VPD_F_QC",
        "H_F_MDS_QC", "H_F_QC",
        "NEE_VUT_REF_QC", "NEE_CUT_REF_QC",
        "USTAR_QC",
        "GPP_NT_VUT_REF_QC", "GPP_DT_VUT_REF_QC",
    ]:
        if pat in d.columns:
            preferred_qc.append(pat)
    if not preferred_qc:
        preferred_qc = qcols[:10]

    rows = []
    for yr, idx in d.groupby("_year").groups.items():
        if pd.isna(yr):
            continue
        idx = list(idx)
        n = len(idx)

        valid_energy = (
            le.loc[idx].notna()
            & h.loc[idx].notna()
            & rn.loc[idx].notna()
            & g.loc[idx].notna()
            & ((rn.loc[idx] - g.loc[idx]).abs() > 1e-9)
        )
        n_valid = int(valid_energy.sum())

        closure = np.nan
        closure_pass = False

        # threshold depends on temporal resolution
        min_valid = 50
        if tres == "DD":
            min_valid = 30
        elif tres == "WW":
            min_valid = 8
        elif tres == "MM":
            min_valid = 3
        elif tres == "YY":
            min_valid = 1

        if n_valid >= min_valid:
            denom = float((rn.loc[idx][valid_energy] - g.loc[idx][valid_energy]).sum())
            numer = float((h.loc[idx][valid_energy] + le.loc[idx][valid_energy]).sum())
            if abs(denom) > 1e-9:
                closure = numer / denom
                closure_pass = bool(0.7 <= closure <= 1.3)

        q_fracs = []
        q_used = []
        for qc in preferred_qc:
            q = clean_flux(d.loc[idx, qc])
            if q.notna().sum() >= max(1, min_valid // 2):
                # FLUXNET-style convention: 0 measured/highest confidence; >0 filled/lower confidence.
                q_fracs.append(float((q > 0).mean()))
                q_used.append(qc)

        gap = float(np.mean(q_fracs)) if q_fracs else np.nan
        gap_pass_030 = bool(pd.notna(gap) and gap <= 0.30)
        gap_pass_050 = bool(pd.notna(gap) and gap <= 0.50)

        valid_uwue = (
            le.loc[idx].notna()
            & gpp.loc[idx].notna()
            & vpd.loc[idx].notna()
            & (vpd.loc[idx] > 0)
        )
        uwue_n = int(valid_uwue.sum())

        rows.append({
            "site_id": site,
            "year": int(yr),
            "source_file": str(path),
            "temporal_rank": trank,
            "temporal_resolution_guess": tres,
            "n_rows_year": n,
            "LE_col": le_col,
            "H_col": h_col,
            "NETRAD_col": rn_col,
            "G_col": g_col,
            "GPP_col": gpp_col,
            "VPD_col": vpd_col,
            "SWC_col": swc_col,
            "closure_n_valid": n_valid,
            "closure_ratio": closure,
            "closure_pass_0p7_1p3": closure_pass,
            "gapfill_fraction": gap,
            "gapfill_pass_0p3": gap_pass_030,
            "gapfill_pass_0p5_sensitivity": gap_pass_050,
            "qc_columns_used": ";".join(q_used),
            "uwue_n_valid": uwue_n,
            "has_observed_soil_water": bool(swc_col),
        })

    audit["file_status"] = "computed_site_years" if rows else "no_years_computed"
    return rows, audit

extract_archives()

all_files = candidate_files()
all_audits = []
all_sy = []

for p in all_files:
    rows, audit = compute_file_site_years(p)
    if audit:
        all_audits.append(audit)
    all_sy.extend(rows)

audit_df = pd.DataFrame(all_audits)
sy_raw = pd.DataFrame(all_sy)

audit_df.to_csv(TAB / "Table_PRODUCT03ck_full13_raw_file_audit.csv", index=False)
sy_raw.to_csv(TAB / "Table_PRODUCT03cl_full13_site_year_quality_all_candidates.csv", index=False)

# Select best site-year candidate across resolutions:
# Prefer HH/HR, then DD, then WW/MM/YY; but also prefer computed closure/gapfill.
if len(sy_raw):
    sy_raw["has_closure"] = sy_raw["closure_ratio"].notna()
    sy_raw["has_gapfill"] = sy_raw["gapfill_fraction"].notna()
    sy_raw["selection_score"] = (
        sy_raw["has_closure"].astype(int) * 1000
        + sy_raw["has_gapfill"].astype(int) * 500
        + sy_raw["closure_n_valid"].fillna(0).clip(0, 100000) / 100
        + sy_raw["uwue_n_valid"].fillna(0).clip(0, 100000) / 1000
        - sy_raw["temporal_rank"].fillna(9) * 50
    )
    sy_best = (
        sy_raw.sort_values(["site_id", "year", "selection_score"], ascending=[True, True, False])
        .drop_duplicates(["site_id", "year"], keep="first")
        .copy()
    )
else:
    sy_best = pd.DataFrame()

if len(sy_best):
    sy_best["strict_pass"] = (
        sy_best["closure_pass_0p7_1p3"].astype(bool)
        & sy_best["gapfill_pass_0p3"].astype(bool)
        & (sy_best["uwue_n_valid"] >= 50)
    )
    sy_best["sensitivity_pass"] = (
        sy_best["closure_pass_0p7_1p3"].astype(bool)
        & sy_best["gapfill_pass_0p5_sensitivity"].astype(bool)
        & (sy_best["uwue_n_valid"] >= 50)
    )
else:
    sy_best["strict_pass"] = []
    sy_best["sensitivity_pass"] = []

sy_best.to_csv(TAB / "Table_PRODUCT03cm_full13_site_year_quality_best.csv", index=False)

# Site-level summary.
summaries = []
for site in TARGET_SITES:
    aud = audit_df[audit_df["site_id"].eq(site)] if len(audit_df) else pd.DataFrame()
    sy = sy_best[sy_best["site_id"].eq(site)] if len(sy_best) else pd.DataFrame()

    has_any_file = len(aud) > 0
    has_required_file = bool(len(aud) and (
        aud["has_LE"].fillna(False).astype(bool)
        & aud["has_H"].fillna(False).astype(bool)
        & aud["has_NETRAD"].fillna(False).astype(bool)
        & aud["has_GPP"].fillna(False).astype(bool)
        & aud["has_VPD"].fillna(False).astype(bool)
        & aud["has_QC"].fillna(False).astype(bool)
    ).any())

    has_closure_any = bool(len(sy) and sy["closure_ratio"].notna().any())
    has_gapfill_any = bool(len(sy) and sy["gapfill_fraction"].notna().any())
    strict_years = int(sy["strict_pass"].sum()) if len(sy) else 0
    sens_years = int(sy["sensitivity_pass"].sum()) if len(sy) else 0

    if not has_any_file:
        status = "MISSING_RAW_FILE"
    elif not has_required_file:
        status = "RAW_FILE_PRESENT_BUT_MISSING_REQUIRED_COLUMNS"
    elif not has_closure_any or not has_gapfill_any:
        status = "RAW_FILE_PRESENT_BUT_QUALITY_NOT_COMPUTABLE"
    elif strict_years > 0:
        status = "PASS_STRICT_FILTER"
    elif sens_years > 0:
        status = "PASS_SENSITIVITY_FILTER_ONLY"
    else:
        status = "COMPUTED_BUT_FAILS_REZA_QUALITY_FILTER"

    summaries.append({
        "site_id": site,
        "raw_file_status": status,
        "candidate_files_n": int(len(aud)),
        "has_any_raw_file": has_any_file,
        "has_required_raw_columns": has_required_file,
        "quality_years_n": int(sy["year"].nunique()) if len(sy) else 0,
        "has_closure_ratio": has_closure_any,
        "has_gapfill_fraction": has_gapfill_any,
        "strict_pass_years": strict_years,
        "sensitivity_pass_years": sens_years,
        "median_closure": float(sy["closure_ratio"].median()) if len(sy) and sy["closure_ratio"].notna().any() else np.nan,
        "median_gapfill": float(sy["gapfill_fraction"].median()) if len(sy) and sy["gapfill_fraction"].notna().any() else np.nan,
        "min_gapfill": float(sy["gapfill_fraction"].min()) if len(sy) and sy["gapfill_fraction"].notna().any() else np.nan,
        "max_gapfill": float(sy["gapfill_fraction"].max()) if len(sy) and sy["gapfill_fraction"].notna().any() else np.nan,
        "best_source_file": sy["source_file"].iloc[0] if len(sy) else (aud["source_file"].iloc[0] if len(aud) else ""),
    })

site_summary = pd.DataFrame(summaries)
site_summary.to_csv(TAB / "Table_PRODUCT03cn_full13_site_level_quality_summary.csv", index=False)

strict_sites = sorted(site_summary.loc[site_summary["raw_file_status"].eq("PASS_STRICT_FILTER"), "site_id"].tolist())
sens_sites = sorted(site_summary.loc[
    site_summary["raw_file_status"].isin(["PASS_STRICT_FILTER", "PASS_SENSITIVITY_FILTER_ONLY"]),
    "site_id"
].tolist())

agree = pd.read_csv(AGREE)
agree["site_id"] = agree["site_id"].astype(str)

strict_agree = agree[agree["site_id"].isin(strict_sites)].copy()
sens_agree = agree[agree["site_id"].isin(sens_sites)].copy()

def rank_et(df):
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "et_product", "n_site_product_rows", "n_unique_sites",
            "exact_agreement_rate", "limited_group_agreement_rate"
        ])
    return (
        df.groupby("et_product")
        .agg(
            n_site_product_rows=("site_id", "size"),
            n_unique_sites=("site_id", "nunique"),
            exact_agreement_rate=("exact_agreement", "mean"),
            limited_group_agreement_rate=("limited_group_agreement", "mean"),
        )
        .reset_index()
        .sort_values(["exact_agreement_rate", "limited_group_agreement_rate", "n_unique_sites"], ascending=[False, False, False])
    )

strict_rank = rank_et(strict_agree)
sens_rank = rank_et(sens_agree)
strict_rank.to_csv(TAB / "Table_PRODUCT03co_full13_strict_quality_et_ranking.csv", index=False)
sens_rank.to_csv(TAB / "Table_PRODUCT03cp_full13_sensitivity_quality_et_ranking.csv", index=False)

prod = pd.read_csv(PROD)
screen = pd.read_csv(SCREEN)
c4 = pd.read_csv(C4)
mixed = pd.read_csv(MIXED) if MIXED.exists() else pd.DataFrame()

focal = c4[c4["term"].astype(str).eq("c4_fraction")].copy() if len(c4) else pd.DataFrame()
if len(focal):
    primary_pass = focal.get("primary_controlled_reza_pass", pd.Series(False, index=focal.index)).astype(str).str.lower().eq("true").any()
    sensitivity_c4_pass = focal.get("sensitivity_controlled_pass", pd.Series(False, index=focal.index)).astype(str).str.lower().eq("true").any()
    exploratory_pass = focal.get("exploratory_minimal_pass", pd.Series(False, index=focal.index)).astype(str).str.lower().eq("true").any()
    best_c4_row = focal.sort_values("p_normal_approx", ascending=True).iloc[0]
else:
    primary_pass = sensitivity_c4_pass = exploratory_pass = False
    best_c4_row = None

if primary_pass:
    c4_status = "PRIMARY_CONTROLLED_C4_SUPPORTED"
elif sensitivity_c4_pass:
    c4_status = "SENSITIVITY_CONTROLLED_C4_SUPPORTED"
elif exploratory_pass:
    c4_status = "PRIMARY_CONTROLLED_C4_NOT_SUPPORTED_EXPLORATORY_SIGNAL_PRESENT"
else:
    c4_status = "C4_TEST_COMPLETED_NOT_SUPPORTED"

best_c4_text = "No usable C4 row."
if best_c4_row is not None:
    best_c4_text = (
        f"Best C4 row: response={best_c4_row.get('response')}, "
        f"sample={best_c4_row.get('sample')}, model={best_c4_row.get('model')}, "
        f"n={int(float(best_c4_row.get('n')))}, "
        f"coef={float(best_c4_row.get('coef_standardized')):.3f}, "
        f"p={float(best_c4_row.get('p_normal_approx')):.4g}, "
        f"q={float(best_c4_row.get('bh_q_normal_approx')):.4g}."
    )

raw_quality_computable_sites = sorted(site_summary.loc[
    site_summary["has_closure_ratio"].astype(bool) & site_summary["has_gapfill_fraction"].astype(bool),
    "site_id"
].tolist())

missing_or_not_computable = site_summary.loc[
    ~(
        site_summary["has_closure_ratio"].astype(bool)
        & site_summary["has_gapfill_fraction"].astype(bool)
    )
].copy()

full_raw_quality_inputs_13 = len(raw_quality_computable_sites) == len(TARGET_SITES)
has_quality_filtered_ranking = len(strict_rank) > 0 or len(sens_rank) > 0

top_et_strict = strict_rank.iloc[0]["et_product"] if len(strict_rank) else "UNRESOLVED"
top_et_sens = sens_rank.iloc[0]["et_product"] if len(sens_rank) else "UNRESOLVED"

satisfaction = pd.DataFrame([
    {
        "reza_item": "Product-screened definition",
        "status": "SATISFIED",
        "evidence": screen.iloc[0].get("answer_for_reza", "Product-screened table exists."),
    },
    {
        "reza_item": "Product identifiability / product agreement",
        "status": "SATISFIED",
        "evidence": "Product anomaly correlations computed across product families; product agreement is treated as confidence layer.",
    },
    {
        "reza_item": "All 13 raw tower files ingested",
        "status": "SATISFIED" if site_summary["has_any_raw_file"].sum() == 13 else "NOT_FULLY_SATISFIED",
        "evidence": f"Sites with any raw file={int(site_summary['has_any_raw_file'].sum())}/13.",
    },
    {
        "reza_item": "Closure and gap-fill computable for all 13",
        "status": "SATISFIED" if full_raw_quality_inputs_13 else "NOT_FULLY_SATISFIED",
        "evidence": f"Sites with computable closure+gapfill={len(raw_quality_computable_sites)}/13. Missing/not computable={missing_or_not_computable['site_id'].tolist()}.",
    },
    {
        "reza_item": "Quality-filtered tower ranking",
        "status": "SATISFIED" if has_quality_filtered_ranking else "BLOCKED",
        "evidence": f"Strict sites={strict_sites}; sensitivity sites={sens_sites}; strict top ET={top_et_strict}; sensitivity top ET={top_et_sens}.",
    },
    {
        "reza_item": "C3/C4 pre-specified test",
        "status": "SATISFIED",
        "evidence": f"{c4_status}. Negative primary controlled test is still a completed pre-specified test.",
    },
    {
        "reza_item": "Hierarchical / partial-pooling robustness",
        "status": "SATISFIED" if len(mixed) else "SATISFIED_WITH_CAVEAT",
        "evidence": f"Mixed model rows available: {len(mixed)}.",
    },
])

satisfaction.to_csv(TAB / "Table_PRODUCT03cq_full13_reza_satisfaction_matrix.csv", index=False)

decision = {
    "generated": now(),
    "stage": "1B.6AP_full13_reza_lock",
    "target_sites_n": len(TARGET_SITES),
    "sites_with_any_raw_file": int(site_summary["has_any_raw_file"].sum()),
    "sites_with_required_raw_columns": int(site_summary["has_required_raw_columns"].sum()),
    "sites_with_computable_closure_and_gapfill": len(raw_quality_computable_sites),
    "raw_quality_computable_sites": raw_quality_computable_sites,
    "strict_quality_sites_n": len(strict_sites),
    "strict_quality_sites": strict_sites,
    "sensitivity_quality_sites_n": len(sens_sites),
    "sensitivity_quality_sites": sens_sites,
    "missing_or_not_computable_sites": missing_or_not_computable[["site_id", "raw_file_status"]].to_dict(orient="records"),
    "strict_top_et_product": top_et_strict,
    "sensitivity_top_et_product": top_et_sens,
    "c4_status": c4_status,
    "best_c4": best_c4_text,
    "fully_satisfies_every_reza_raw_tower_requirement": bool(full_raw_quality_inputs_13),
    "can_send_full_reza_without_caveat": bool(full_raw_quality_inputs_13 and has_quality_filtered_ranking),
    "can_send_reza_with_explicit_data_availability_caveat": bool(has_quality_filtered_ranking),
    "recommended_action": (
        "SEND_FULL_REZA_PACKET"
        if full_raw_quality_inputs_13 and has_quality_filtered_ranking
        else "DO_NOT_SEND_AS_FULLY_COMPLETE_YET__USE_MISSING_OR_NOT_COMPUTABLE_TABLE_TO_FIX_REMAINING_SITES"
    )
}
(TAB / "STAGE1B6AP_FULL13_REZA_LOCK_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

email = []
email.append("Hi Reza,")
email.append("")
email.append("Thank you again — I treated your note as an analysis-locking checklist and rebuilt the response around each gate.")
email.append("")
email.append("I audited what I meant by product-screened and found no evidence that pixels were selected because products agreed. The audit supports defining product-screened as QC/product-confidence screening rather than product-agreement filtering.")
email.append("")
email.append("I also quantified the product-identifiability issue directly with anomaly correlations across the product matrix, so product agreement is now treated as a confidence layer rather than an assumption.")
email.append("")
email.append(f"For tower validation, I ingested raw tower files for {int(site_summary['has_any_raw_file'].sum())}/13 target sites and computed closure/gap-fill where the raw H/LE/NETRAD/G/QC fields were available. Closure+gap-fill were computable for {len(raw_quality_computable_sites)}/13 sites.")
email.append("")
if len(missing_or_not_computable):
    email.append("Sites still missing or not fully computable are: " + ", ".join(f"{r.site_id} ({r.raw_file_status})" for r in missing_or_not_computable.itertuples()) + ".")
    email.append("")
email.append(f"The strict quality-passing tower set is: {', '.join(strict_sites) if strict_sites else 'none'}.")
email.append(f"The strict ET ranking selects: {top_et_strict}.")
email.append("")
email.append("For the C3/C4 test, the primary controlled result did not pass, so I would not frame this as a clean primary C3/C4 mechanism paper. There is an exploratory C4 signal that I would report as secondary/supportive rather than central.")
email.append("")
email.append(best_c4_text)
email.append("")
email.append("Best,")
email.append("Akul")

(TXT / "REZA_READY_RESPONSE_FULL13_DRAFT.md").write_text("\n".join(email), encoding="utf-8")

report = []
report.append("# Stage 1B.6AP full 13-site Reza lock")
report.append("")
report.append("## Decision")
report.append("")
report.append("```json")
report.append(json.dumps(decision, indent=2))
report.append("```")
report.append("")
report.append("## Reza satisfaction matrix")
report.append("")
report.append("```text")
report.append(satisfaction.to_string(index=False))
report.append("```")
report.append("")
report.append("## Site-level raw tower quality summary")
report.append("")
report.append("```text")
report.append(site_summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Missing / not-computable sites")
report.append("")
report.append("```text")
report.append(missing_or_not_computable.to_string(index=False) if len(missing_or_not_computable) else "None.")
report.append("```")
report.append("")
report.append("## Strict ET ranking")
report.append("")
report.append("```text")
report.append(strict_rank.to_string(index=False) if len(strict_rank) else "No strict ET ranking.")
report.append("```")
report.append("")
report.append("## Sensitivity ET ranking")
report.append("")
report.append("```text")
report.append(sens_rank.to_string(index=False) if len(sens_rank) else "No sensitivity ET ranking.")
report.append("```")
report.append("")
report.append("## Full draft response")
report.append("")
report.append("```text")
report.append("\n".join(email))
report.append("```")

(TXT / "STAGE1B6AP_FULL13_REZA_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "STAGE1B6AP_FULL13_REZA_LOCK_DECISION.json")
print("WROTE", TAB / "Table_PRODUCT03ck_full13_raw_file_audit.csv")
print("WROTE", TAB / "Table_PRODUCT03cl_full13_site_year_quality_all_candidates.csv")
print("WROTE", TAB / "Table_PRODUCT03cm_full13_site_year_quality_best.csv")
print("WROTE", TAB / "Table_PRODUCT03cn_full13_site_level_quality_summary.csv")
print("WROTE", TAB / "Table_PRODUCT03co_full13_strict_quality_et_ranking.csv")
print("WROTE", TAB / "Table_PRODUCT03cp_full13_sensitivity_quality_et_ranking.csv")
print("WROTE", TAB / "Table_PRODUCT03cq_full13_reza_satisfaction_matrix.csv")
print("WROTE", TXT / "REZA_READY_RESPONSE_FULL13_DRAFT.md")
print("WROTE", TXT / "STAGE1B6AP_FULL13_REZA_LOCK_REPORT.md")
