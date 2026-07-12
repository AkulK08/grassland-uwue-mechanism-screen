from pathlib import Path
from datetime import datetime
import json
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

OUT = Path("results/stage1b6al_tower_quality_completion")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

FINAL_TOWER = Path("results/stage1b6ak_project_complete_resolution_packet/tables/Table_PRODUCT03bf_project_final_tower_validation_table.csv")
AGREEMENT_LONG = Path("results/stage1b6ak_project_complete_resolution_packet/tables/Table_PRODUCT03bg_project_tower_satellite_agreement_long.csv")

if not FINAL_TOWER.exists():
    raise FileNotFoundError(f"Missing final tower table: {FINAL_TOWER}. Run stage1b6ak first.")

tower = pd.read_csv(FINAL_TOWER)
target_sites = sorted(tower["site_id"].dropna().astype(str).unique())

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def read_csv_safe(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", nrows=nrows)
        except Exception:
            try:
                return pd.read_csv(path, sep="\t", nrows=nrows)
            except Exception:
                return None

def first_col(df, candidates):
    lut = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None

def cols_matching(df, patterns):
    out = []
    for c in df.columns:
        lc = norm(c)
        for pat in patterns:
            if re.search(pat, lc):
                out.append(c)
                break
    return out

def infer_site_col(df):
    return first_col(df, [
        "site_id", "SITE_ID", "site", "Site", "tower_id", "Tower_ID",
        "site_name", "SITE", "id"
    ])

def infer_time_col(df):
    return first_col(df, [
        "TIMESTAMP", "TIMESTAMP_START", "TIMESTAMP_END", "date", "datetime",
        "time", "Date", "DATE", "year", "YEAR"
    ])

def extract_year(series):
    s = series.astype(str)
    # Supports YYYY, YYYYMMDD, YYYY-MM-DD, timestamp strings.
    y = s.str.extract(r"(\d{4})")[0]
    return to_num(y)

def find_flux_files():
    roots = [Path("data/raw"), Path("data/processed"), Path("results")]
    suffixes = {".csv", ".txt", ".tsv"}
    files = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in suffixes and p.stat().st_size > 0:
                    s = str(p).lower()
                    if any(k in s for k in ["tower", "fluxnet", "ameriflux", "icos", "ozflux", "project"]):
                        files.append(p)
            except Exception:
                pass
    return sorted(set(files))

def candidate_score(path, df):
    cols = " ".join([norm(c) for c in df.columns])
    s = str(path).lower()
    score = 0
    for k in ["le", "latent", "h_f", "sensible", "netrad", "rnet", "g_f", "soil_heat", "gpp", "vpd", "swc", "qc"]:
        if k in cols:
            score += 2
    for k in ["fluxnet", "ameriflux", "icos", "tower"]:
        if k in s:
            score += 1
    return score

def pick_col(df, role):
    if role == "LE":
        candidates = [
            "LE_F_MDS", "LE_CORR", "LE", "LE_F_MDS_CORR", "LE_PI_F",
            "LE_F", "LE_1_1_1", "LE_f", "latent_heat", "latent_energy"
        ]
        pats = [r"^le(_|$)", r"latent"]
    elif role == "H":
        candidates = [
            "H_F_MDS", "H_CORR", "H", "H_F_MDS_CORR", "H_PI_F",
            "H_F", "H_1_1_1", "sensible_heat"
        ]
        pats = [r"^h(_|$)", r"sensible"]
    elif role == "NETRAD":
        candidates = [
            "NETRAD", "NETRAD_F", "Rn", "RNET", "NETRAD_PI_F",
            "NETRAD_1_1_1", "net_radiation", "netrad"
        ]
        pats = [r"netrad", r"rnet", r"net_radiation", r"^rn$"]
    elif role == "G":
        candidates = [
            "G_F_MDS", "G", "G_1_1_1", "G_PI_F", "G_F",
            "soil_heat_flux", "ground_heat_flux"
        ]
        pats = [r"^g(_|$)", r"soil_heat", r"ground_heat"]
    elif role == "GPP":
        candidates = ["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP", "GPP_NT", "GPP_DT"]
        pats = [r"gpp"]
    elif role == "VPD":
        candidates = ["VPD_F", "VPD", "VPD_PI_F", "vpd", "mean_vpd"]
        pats = [r"vpd"]
    else:
        candidates, pats = [], []
    col = first_col(df, candidates)
    if col:
        return col
    matches = cols_matching(df, pats)
    return matches[0] if matches else None

def qc_cols_for(df, base_role):
    role_patterns = {
        "LE": [r"le.*qc", r"qc.*le"],
        "H": [r"^h.*qc", r"qc.*h"],
        "GPP": [r"gpp.*qc", r"qc.*gpp"],
        "VPD": [r"vpd.*qc", r"qc.*vpd"],
        "NEE": [r"nee.*qc", r"qc.*nee"],
    }
    return cols_matching(df, role_patterns.get(base_role, []))

def compute_quality_for_site(df, site_id, source_path):
    d = df.copy()

    site_col = infer_site_col(d)
    if site_col:
        d = d[d[site_col].astype(str).eq(site_id)].copy()
    else:
        # If no site column and file name includes site ID, allow it.
        if site_id not in str(source_path):
            return None

    if len(d) < 20:
        return None

    time_col = infer_time_col(d)
    if time_col:
        years = extract_year(d[time_col])
    else:
        years = pd.Series(np.nan, index=d.index)

    le_col = pick_col(d, "LE")
    h_col = pick_col(d, "H")
    rn_col = pick_col(d, "NETRAD")
    g_col = pick_col(d, "G")
    gpp_col = pick_col(d, "GPP")
    vpd_col = pick_col(d, "VPD")

    row = {
        "site_id": site_id,
        "source_file": str(source_path),
        "n_rows_for_site": int(len(d)),
        "year_min": float(years.min()) if years.notna().any() else np.nan,
        "year_max": float(years.max()) if years.notna().any() else np.nan,
        "site_years": int(years.nunique(dropna=True)) if years.notna().any() else np.nan,
        "LE_col": le_col,
        "H_col": h_col,
        "NETRAD_col": rn_col,
        "G_col": g_col,
        "GPP_col": gpp_col,
        "VPD_col": vpd_col,
    }

    # Closure ratio: (H + LE) / (NETRAD - G)
    if le_col and h_col and rn_col:
        le = to_num(d[le_col])
        h = to_num(d[h_col])
        rn = to_num(d[rn_col])
        g = to_num(d[g_col]) if g_col else pd.Series(0.0, index=d.index)

        valid = le.notna() & h.notna() & rn.notna() & g.notna() & ((rn - g).abs() > 1e-9)
        row["closure_n_valid"] = int(valid.sum())
        if valid.sum() >= 20:
            closure = float((h[valid] + le[valid]).sum() / (rn[valid] - g[valid]).sum())
            row["closure_ratio"] = closure
            row["closure_pass_0p7_1p3"] = bool(0.7 <= closure <= 1.3)
        else:
            row["closure_ratio"] = np.nan
            row["closure_pass_0p7_1p3"] = False
    else:
        row["closure_n_valid"] = 0
        row["closure_ratio"] = np.nan
        row["closure_pass_0p7_1p3"] = False

    # Gap-fill fraction: use QC columns if present.
    qc_used = []
    gap_fracs = []
    for role in ["LE", "H", "GPP", "VPD", "NEE"]:
        for qc in qc_cols_for(d, role):
            q = to_num(d[qc])
            if q.notna().sum() >= 20:
                # FLUXNET convention: 0 measured/high quality; >0 lower/gapfilled.
                gap_fracs.append(float((q > 0).mean()))
                qc_used.append(qc)

    row["qc_columns_used"] = ";".join(sorted(set(qc_used)))
    row["gapfill_fraction_estimated"] = float(np.mean(gap_fracs)) if gap_fracs else np.nan
    row["gapfill_pass_0p3"] = bool(pd.notna(row["gapfill_fraction_estimated"]) and row["gapfill_fraction_estimated"] <= 0.3)

    # WUE calculability audit.
    row["can_compute_tower_wue"] = bool(gpp_col and le_col)
    row["can_compute_tower_uwue"] = bool(gpp_col and le_col and vpd_col)
    row["has_energy_balance_columns"] = bool(le_col and h_col and rn_col)
    row["has_gapfill_qc_columns"] = bool(len(qc_used) > 0)

    return row

# Inventory raw files.
files = find_flux_files()
inventory_rows = []
usable_rows = []

iterator = tqdm(files, desc="Inspecting tower/raw flux files", unit="file") if tqdm else files
for path in iterator:
    d0 = read_csv_safe(path, nrows=50)
    if d0 is None or len(d0) == 0:
        continue

    score = candidate_score(path, d0)
    site_col = infer_site_col(d0)
    sites_preview = []
    if site_col:
        try:
            full_site_col = read_csv_safe(path, nrows=5000)
            if full_site_col is not None and site_col in full_site_col.columns:
                sites_preview = sorted(full_site_col[site_col].dropna().astype(str).unique())[:30]
        except Exception:
            pass

    inv = {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1e6, 3),
        "n_preview_cols": len(d0.columns),
        "candidate_score": score,
        "site_col": site_col,
        "sites_preview": ";".join(sites_preview),
        "has_LE": bool(pick_col(d0, "LE")),
        "has_H": bool(pick_col(d0, "H")),
        "has_NETRAD": bool(pick_col(d0, "NETRAD")),
        "has_G": bool(pick_col(d0, "G")),
        "has_GPP": bool(pick_col(d0, "GPP")),
        "has_VPD": bool(pick_col(d0, "VPD")),
        "qc_like_cols_preview": ";".join(cols_matching(d0, [r"qc"])[:30]),
        "columns_preview": ";".join(list(d0.columns)[:120]),
    }
    inventory_rows.append(inv)

    # Only fully read likely files.
    if score < 4:
        continue

    d = read_csv_safe(path)
    if d is None or len(d) == 0:
        continue

    for site in target_sites:
        r = compute_quality_for_site(d, site, path)
        if r is not None:
            usable_rows.append(r)

inventory = pd.DataFrame(inventory_rows).sort_values("candidate_score", ascending=False)
quality = pd.DataFrame(usable_rows)

inventory.to_csv(TAB / "Table_PRODUCT03bo_tower_raw_file_inventory.csv", index=False)
quality.to_csv(TAB / "Table_PRODUCT03bp_tower_quality_candidates_by_site_file.csv", index=False)

# Select best quality row per target site.
if len(quality):
    quality["score"] = (
        quality["closure_pass_0p7_1p3"].fillna(False).astype(int) * 5
        + quality["gapfill_pass_0p3"].fillna(False).astype(int) * 5
        + quality["can_compute_tower_uwue"].fillna(False).astype(int) * 3
        + quality["can_compute_tower_wue"].fillna(False).astype(int) * 2
        + quality["closure_n_valid"].fillna(0).clip(0, 10000) / 10000
    )
    best_quality = quality.sort_values(["site_id", "score", "n_rows_for_site"], ascending=[True, False, False]).drop_duplicates("site_id", keep="first")
else:
    best_quality = pd.DataFrame(columns=["site_id"])

# Merge into final tower table.
merged = tower.copy()
if len(best_quality):
    keep_cols = [
        "site_id", "source_file", "site_years", "closure_ratio", "closure_pass_0p7_1p3",
        "gapfill_fraction_estimated", "gapfill_pass_0p3", "LE_col", "H_col", "NETRAD_col",
        "G_col", "GPP_col", "VPD_col", "qc_columns_used", "can_compute_tower_wue",
        "can_compute_tower_uwue", "has_energy_balance_columns", "has_gapfill_qc_columns"
    ]
    keep_cols = [c for c in keep_cols if c in best_quality.columns]
    merged = merged.merge(best_quality[keep_cols], on="site_id", how="left", suffixes=("", "_quality"))

merged.to_csv(TAB / "Table_PRODUCT03bq_project_tower_table_with_quality_completed.csv", index=False)

# Missing manifest.
manifest_rows = []
for site in target_sites:
    row = merged[merged["site_id"].astype(str).eq(site)]
    if len(row):
        r = row.iloc[0]
        needs = []
        if pd.isna(r.get("closure_ratio", np.nan)):
            needs += ["H", "LE", "NETRAD", "G"]
        if pd.isna(r.get("gapfill_fraction_estimated", np.nan)):
            needs += ["LE_QC/GPP_QC/VPD_QC or FLUXNET quality flags"]
        if not bool(r.get("can_compute_tower_uwue", False)):
            needs += ["GPP", "LE", "VPD"]
    else:
        needs = ["raw tower flux file for site"]

    if needs:
        network_guess = "AmeriFlux BASE" if site.startswith("US-") or site.startswith("CA-") else "FLUXNET2015 FULLSET / ICOS / regional network"
        manifest_rows.append({
            "site_id": site,
            "network_guess": network_guess,
            "required_missing_items": "; ".join(sorted(set(needs))),
            "minimum_required_columns": "TIMESTAMP/site_id, GPP, LE, VPD, H, NETRAD, G, LE_QC/GPP_QC/VPD_QC",
            "purpose": "compute energy-balance closure, gap-fill fraction, tower WUE/uWUE, and project-quality tower filter",
        })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(TAB / "Table_PRODUCT03br_missing_tower_raw_download_manifest.csv", index=False)

# Final status.
closure_n = int(merged["closure_ratio"].notna().sum()) if "closure_ratio" in merged.columns else 0
gap_n = int(merged["gapfill_fraction_estimated"].notna().sum()) if "gapfill_fraction_estimated" in merged.columns else 0
uwue_n = int(merged["can_compute_tower_uwue"].fillna(False).sum()) if "can_compute_tower_uwue" in merged.columns else 0

status = {
    "generated": now(),
    "stage": "1B.6AL_tower_quality_completion",
    "target_sites": len(target_sites),
    "raw_files_inspected": int(len(inventory)),
    "site_file_quality_candidates": int(len(quality)),
    "sites_with_closure_ratio": closure_n,
    "sites_with_gapfill_fraction": gap_n,
    "sites_with_tower_uwue_inputs": uwue_n,
    "fully_resolved_for_project": bool(closure_n == len(target_sites) and gap_n == len(target_sites)),
    "missing_sites_count": int(len(manifest)),
    "next_action": (
        "Tower quality filters are complete; send project the completed tower table."
        if closure_n == len(target_sites) and gap_n == len(target_sites)
        else
        "Download/add raw tower exports listed in Table_PRODUCT03br_missing_tower_raw_download_manifest.csv, then rerun this script."
    ),
}
(TAB / "STAGE1B6AL_TOWER_QUALITY_COMPLETION_DECISION.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

# Report.
report = []
report.append("# Stage 1B.6AL tower quality completion")
report.append("")
report.append(f"Generated: {status['generated']}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```json")
report.append(json.dumps(status, indent=2))
report.append("```")
report.append("")
report.append("## Completed tower quality table")
report.append("")
show_cols = [
    "site_id", "igbp_class", "site_years", "tower_response_class",
    "closure_ratio", "closure_pass_0p7_1p3",
    "gapfill_fraction_estimated", "gapfill_pass_0p3",
    "can_compute_tower_wue", "can_compute_tower_uwue",
    "source_file"
]
show_cols = [c for c in show_cols if c in merged.columns]
report.append("```text")
report.append(merged[show_cols].to_string(index=False))
report.append("```")
report.append("")
report.append("## Missing raw tower download manifest")
report.append("")
report.append("```text")
report.append(manifest.to_string(index=False) if len(manifest) else "No missing tower raw files/columns.")
report.append("```")
report.append("")
report.append("## Raw file inventory preview")
report.append("")
report.append("```text")
inv_cols = ["candidate_score", "has_LE", "has_H", "has_NETRAD", "has_G", "has_GPP", "has_VPD", "site_col", "sites_preview", "path"]
report.append(inventory[inv_cols].head(80).to_string(index=False) if len(inventory) else "No raw files inspected.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
if status["fully_resolved_for_project"]:
    report.append("Tower closure and gap-fill are now fully resolved for the clean tower table.")
else:
    report.append("Tower closure and gap-fill are still not fully resolved from local files. This is not a coding issue anymore unless the raw files are present under different names; it requires adding raw tower network exports with the columns listed in the manifest.")

(TXT / "STAGE1B6AL_TOWER_QUALITY_COMPLETION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT03bo_tower_raw_file_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT03bp_tower_quality_candidates_by_site_file.csv")
print("WROTE", TAB / "Table_PRODUCT03bq_project_tower_table_with_quality_completed.csv")
print("WROTE", TAB / "Table_PRODUCT03br_missing_tower_raw_download_manifest.csv")
print("WROTE", TAB / "STAGE1B6AL_TOWER_QUALITY_COMPLETION_DECISION.json")
print("WROTE", TXT / "STAGE1B6AL_TOWER_QUALITY_COMPLETION_REPORT.md")
