from pathlib import Path
from datetime import datetime
import json
import re
import zipfile
import tarfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

OUT = Path("results/stage1b6am_raw_tower_download_ingest")
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

FINAL_TOWER = Path("results/stage1b6ak_reza_complete_resolution_packet/tables/Table_PRODUCT03bf_reza_final_tower_validation_table.csv")
if not FINAL_TOWER.exists():
    raise FileNotFoundError("Missing Stage 1B.6AK tower table. Run 1B.6AK first.")

tower = pd.read_csv(FINAL_TOWER)
target_sites = sorted(tower["site_id"].dropna().astype(str).unique())

RAW_ROOTS = [
    Path("data/raw/towers/_reza_raw_exports"),
    Path("data/raw/towers/_reza_raw_exports/ameriflux_base"),
    Path("data/raw/towers/_reza_raw_exports/manual_fluxnet"),
    Path("data/raw/towers/_reza_raw_exports/extracted"),
]

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace("(", "").replace(")", "")

def read_csv_safe(path, nrows=None):
    seps = [",", "\t", ";"]
    for sep in seps:
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

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def first_col(df, candidates):
    lut = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None

def col_contains(df, patterns):
    out = []
    for c in df.columns:
        lc = norm(c)
        for pat in patterns:
            if re.search(pat, lc):
                out.append(c)
                break
    return out

def pick_col(df, role):
    role_map = {
        "LE": ["LE_F_MDS", "LE_F", "LE", "LE_CORR", "LE_F_MDS_CORR", "LE_PI_F", "LE_1_1_1"],
        "H": ["H_F_MDS", "H_F", "H", "H_CORR", "H_F_MDS_CORR", "H_PI_F", "H_1_1_1"],
        "NETRAD": ["NETRAD", "NETRAD_F", "RNET", "Rn", "NETRAD_PI_F", "NETRAD_1_1_1"],
        "G": ["G_F_MDS", "G_F", "G", "G_1_1_1", "G_PI_F"],
        "GPP": ["GPP_NT_VUT_REF", "GPP_DT_VUT_REF", "GPP_NT", "GPP_DT", "GPP"],
        "VPD": ["VPD_F", "VPD", "VPD_PI_F"],
        "TA": ["TA_F_MDS", "TA", "TA_F"],
        "SWC": ["SWC_F_MDS", "SWC", "SWC_1", "SWC_1_1_1"],
    }
    c = first_col(df, role_map.get(role, []))
    if c:
        return c

    pats = {
        "LE": [r"^le($|_)", r"latent"],
        "H": [r"^h($|_)", r"sensible"],
        "NETRAD": [r"netrad", r"rnet", r"net_radiation", r"^rn$"],
        "G": [r"^g($|_)", r"soil_heat", r"ground_heat"],
        "GPP": [r"gpp"],
        "VPD": [r"vpd"],
        "TA": [r"^ta($|_)", r"air_temp"],
        "SWC": [r"swc", r"soil_water", r"soil_moisture"],
    }.get(role, [])
    m = col_contains(df, pats)
    return m[0] if m else None

def qc_cols(df):
    q = []
    for c in df.columns:
        lc = norm(c)
        if "qc" in lc or lc.endswith("_qf") or "quality" in lc:
            q.append(c)
    return q

def infer_site_col(df):
    return first_col(df, ["site_id", "SITE_ID", "site", "SITE", "tower_id", "Tower_ID", "site_name", "id"])

def infer_timestamp_cols(df):
    start = first_col(df, ["TIMESTAMP_START", "timestamp_start"])
    end = first_col(df, ["TIMESTAMP_END", "timestamp_end"])
    stamp = first_col(df, ["TIMESTAMP", "timestamp", "date", "datetime", "time"])
    return start, end, stamp

def extract_site_from_path(path):
    s = str(path)
    m = re.search(r"([A-Z]{2}-[A-Za-z0-9]{2,3})", s)
    return m.group(1) if m else None

def parse_year(series):
    y = series.astype(str).str.extract(r"(\d{4})")[0]
    return to_num(y)

def timestamp_interval_seconds(df, start_col, end_col, stamp_col):
    # FLUXNET HH/HR uses YYYYMMDDHHMM start/end.
    if start_col and end_col:
        st = pd.to_datetime(df[start_col].astype(str), format="%Y%m%d%H%M", errors="coerce")
        en = pd.to_datetime(df[end_col].astype(str), format="%Y%m%d%H%M", errors="coerce")
        sec = (en - st).dt.total_seconds()
        if sec.notna().sum() > 0 and sec.median() > 0:
            return sec.fillna(sec.median())
    if stamp_col:
        ss = df[stamp_col].astype(str)
        # daily timestamp like YYYYMMDD
        if ss.str.len().median() <= 8:
            return pd.Series(86400.0, index=df.index)
        return pd.Series(1800.0, index=df.index)
    return pd.Series(86400.0, index=df.index)

def extract_archives():
    extract_dir = Path("data/raw/towers/_reza_raw_exports/extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)
    archives = []
    for root in RAW_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".zip", ".tar", ".gz", ".tgz"]:
                archives.append(p)

    rows = []
    for z in tqdm(archives, desc="Extracting tower archives", unit="archive") if tqdm else archives:
        try:
            if zipfile.is_zipfile(z):
                with zipfile.ZipFile(z) as zz:
                    members = [m for m in zz.namelist() if m.lower().endswith((".csv", ".txt"))]
                    relevant = [m for m in members if any(site in m for site in target_sites) or any(site in str(z) for site in target_sites)]
                    if not relevant:
                        relevant = members[:20]
                    for m in relevant:
                        out = extract_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", z.stem + "__" + Path(m).name)
                        if not out.exists():
                            with zz.open(m) as src, open(out, "wb") as dst:
                                dst.write(src.read())
                        rows.append({"archive": str(z), "member": m, "extracted_path": str(out), "status": "extracted"})
            elif tarfile.is_tarfile(z):
                with tarfile.open(z) as tt:
                    members = [m for m in tt.getmembers() if m.name.lower().endswith((".csv", ".txt"))]
                    relevant = [m for m in members if any(site in m.name for site in target_sites) or any(site in str(z) for site in target_sites)]
                    if not relevant:
                        relevant = members[:20]
                    for m in relevant:
                        out = extract_dir / re.sub(r"[^A-Za-z0-9_.-]+", "_", z.stem + "__" + Path(m.name).name)
                        if not out.exists():
                            src = tt.extractfile(m)
                            if src:
                                with open(out, "wb") as dst:
                                    dst.write(src.read())
                        rows.append({"archive": str(z), "member": m.name, "extracted_path": str(out), "status": "extracted"})
        except Exception as e:
            rows.append({"archive": str(z), "member": "", "extracted_path": "", "status": "failed", "error": repr(e)})
    return pd.DataFrame(rows)

extract_log = extract_archives()
extract_log.to_csv(TAB / "Table_PRODUCT03bs_raw_archive_extract_log.csv", index=False)

def find_candidate_csvs():
    files = []
    for root in RAW_ROOTS + [Path("data/raw/towers/_reza_raw_exports/extracted")]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".csv", ".txt", ".tsv"]:
                try:
                    if p.stat().st_size > 0:
                        files.append(p)
                except Exception:
                    pass
    return sorted(set(files))

def score_file(path, df):
    score = 0
    cols = " ".join(norm(c) for c in df.columns)
    s = str(path)
    for role in ["LE", "H", "NETRAD", "G", "GPP", "VPD"]:
        if pick_col(df, role):
            score += 3
    if qc_cols(df):
        score += 3
    if infer_site_col(df):
        score += 2
    if any(site in s for site in target_sites):
        score += 2
    return score

def compute_site_quality(df, site_id, path):
    d = df.copy()
    site_col = infer_site_col(d)
    if site_col:
        d = d[d[site_col].astype(str).eq(site_id)].copy()
    else:
        if site_id not in str(path):
            return None

    if len(d) < 10:
        return None

    le_col = pick_col(d, "LE")
    h_col = pick_col(d, "H")
    rn_col = pick_col(d, "NETRAD")
    g_col = pick_col(d, "G")
    gpp_col = pick_col(d, "GPP")
    vpd_col = pick_col(d, "VPD")
    swc_col = pick_col(d, "SWC")

    start_col, end_col, stamp_col = infer_timestamp_cols(d)
    years = parse_year(d[start_col]) if start_col else parse_year(d[stamp_col]) if stamp_col else pd.Series(np.nan, index=d.index)

    row = {
        "site_id": site_id,
        "source_file": str(path),
        "n_rows": int(len(d)),
        "year_min": float(years.min()) if years.notna().any() else np.nan,
        "year_max": float(years.max()) if years.notna().any() else np.nan,
        "site_years_raw": int(years.nunique(dropna=True)) if years.notna().any() else np.nan,
        "LE_col": le_col,
        "H_col": h_col,
        "NETRAD_col": rn_col,
        "G_col": g_col,
        "GPP_col": gpp_col,
        "VPD_col": vpd_col,
        "SWC_col": swc_col,
        "timestamp_start_col": start_col,
        "timestamp_end_col": end_col,
        "timestamp_col": stamp_col,
    }

    # Closure.
    if le_col and h_col and rn_col:
        le = to_num(d[le_col])
        h = to_num(d[h_col])
        rn = to_num(d[rn_col])
        g = to_num(d[g_col]) if g_col else pd.Series(0.0, index=d.index)
        valid = le.notna() & h.notna() & rn.notna() & g.notna() & ((rn - g).abs() > 1e-9)
        row["closure_n_valid"] = int(valid.sum())
        if valid.sum() >= 10:
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

    # Gap-fill from QC flags.
    qcols = qc_cols(d)
    used_q = []
    gap_fracs = []
    for qc in qcols:
        lc = norm(qc)
        if any(v in lc for v in ["le", "gpp", "vpd", "h_", "ta", "nee"]):
            q = to_num(d[qc])
            if q.notna().sum() >= 10:
                used_q.append(qc)
                gap_fracs.append(float((q > 0).mean()))
    row["qc_columns_used"] = ";".join(used_q)
    row["gapfill_fraction_estimated"] = float(np.mean(gap_fracs)) if gap_fracs else np.nan
    row["gapfill_pass_0p3"] = bool(pd.notna(row["gapfill_fraction_estimated"]) and row["gapfill_fraction_estimated"] <= 0.30)

    # ET from LE.
    if le_col:
        le = to_num(d[le_col])
        seconds = timestamp_interval_seconds(d, start_col, end_col, stamp_col)
        et_mm = le * seconds / 2.45e6
        row["et_from_le_n"] = int(et_mm.notna().sum())
        row["et_from_le_total_mm"] = float(et_mm.sum(skipna=True)) if et_mm.notna().any() else np.nan
    else:
        row["et_from_le_n"] = 0
        row["et_from_le_total_mm"] = np.nan

    # uWUE possible.
    row["can_compute_wue"] = bool(gpp_col and le_col)
    row["can_compute_iwue"] = bool(gpp_col and le_col and vpd_col)
    row["can_compute_uwue"] = bool(gpp_col and le_col and vpd_col)
    row["has_observed_soil_water"] = bool(swc_col)

    return row

csvs = find_candidate_csvs()
inventory_rows = []
quality_rows = []

for p in tqdm(csvs, desc="Scanning raw tower CSV/TXT files", unit="file") if tqdm else csvs:
    d0 = read_csv_safe(p, nrows=200)
    if d0 is None or len(d0.columns) <= 1:
        continue
    score = score_file(p, d0)
    site_col = infer_site_col(d0)
    preview_sites = []
    if site_col:
        preview_sites = sorted(d0[site_col].dropna().astype(str).unique())[:40]
    inventory_rows.append({
        "path": str(p),
        "size_mb": round(p.stat().st_size / 1e6, 3),
        "score": score,
        "site_col": site_col,
        "sites_preview": ";".join(preview_sites),
        "has_LE": bool(pick_col(d0, "LE")),
        "has_H": bool(pick_col(d0, "H")),
        "has_NETRAD": bool(pick_col(d0, "NETRAD")),
        "has_G": bool(pick_col(d0, "G")),
        "has_GPP": bool(pick_col(d0, "GPP")),
        "has_VPD": bool(pick_col(d0, "VPD")),
        "has_QC": bool(qc_cols(d0)),
        "qc_cols_preview": ";".join(qc_cols(d0)[:30]),
        "columns_preview": ";".join(list(d0.columns)[:120]),
    })

    if score < 8:
        continue

    d = read_csv_safe(p)
    if d is None:
        continue

    for site in target_sites:
        q = compute_site_quality(d, site, p)
        if q is not None:
            quality_rows.append(q)

inventory = pd.DataFrame(inventory_rows).sort_values("score", ascending=False) if inventory_rows else pd.DataFrame()
quality = pd.DataFrame(quality_rows)

inventory.to_csv(TAB / "Table_PRODUCT03bt_raw_tower_file_inventory_after_download.csv", index=False)
quality.to_csv(TAB / "Table_PRODUCT03bu_raw_tower_quality_candidates.csv", index=False)

if len(quality):
    quality["score"] = (
        quality["closure_pass_0p7_1p3"].fillna(False).astype(int) * 10
        + quality["gapfill_pass_0p3"].fillna(False).astype(int) * 10
        + quality["can_compute_uwue"].fillna(False).astype(int) * 5
        + quality["has_observed_soil_water"].fillna(False).astype(int) * 2
        + quality["closure_n_valid"].fillna(0).clip(0, 100000) / 100000
    )
    best = quality.sort_values(["site_id", "score", "n_rows"], ascending=[True, False, False]).drop_duplicates("site_id", keep="first")
else:
    best = pd.DataFrame(columns=["site_id"])

merged = tower.copy()
if len(best):
    keep = [
        "site_id", "source_file", "site_years_raw", "closure_ratio", "closure_pass_0p7_1p3",
        "gapfill_fraction_estimated", "gapfill_pass_0p3", "LE_col", "H_col", "NETRAD_col", "G_col",
        "GPP_col", "VPD_col", "SWC_col", "qc_columns_used", "can_compute_wue", "can_compute_iwue",
        "can_compute_uwue", "has_observed_soil_water", "et_from_le_n", "et_from_le_total_mm"
    ]
    keep = [c for c in keep if c in best.columns]
    merged = merged.merge(best[keep], on="site_id", how="left")

merged.to_csv(TAB / "Table_PRODUCT03bv_reza_tower_table_raw_quality_completed.csv", index=False)

missing_rows = []
for site in target_sites:
    r = merged[merged["site_id"].astype(str).eq(site)]
    missing = []
    if len(r) == 0:
        missing = ["all raw tower export files"]
    else:
        rr = r.iloc[0]
        if pd.isna(rr.get("closure_ratio", np.nan)):
            missing += ["H", "LE", "NETRAD", "G"]
        if pd.isna(rr.get("gapfill_fraction_estimated", np.nan)):
            missing += ["LE_QC/GPP_QC/VPD_QC or FLUXNET quality flags"]
        if not bool(rr.get("can_compute_uwue", False)):
            missing += ["GPP", "LE", "VPD"]
    if missing:
        network = "AmeriFlux BASE-BADM" if site.startswith(("US-", "CA-")) else "FLUXNET2015 FULLSET / current FLUXNET Data System / ICOS"
        missing_rows.append({
            "site_id": site,
            "network_or_portal": network,
            "download_product_needed": "BASE-BADM HH/HR/Daily raw flux-met export" if site.startswith(("US-", "CA-")) else "FULLSET HH/HR or DD raw flux-met export",
            "missing_items": "; ".join(sorted(set(missing))),
            "required_columns": "TIMESTAMP_START/TIMESTAMP_END or TIMESTAMP, GPP, LE, VPD, H, NETRAD, G, QC flags",
            "place_download_here": "data/raw/towers/_reza_raw_exports/manual_fluxnet/",
        })

missing = pd.DataFrame(missing_rows)
missing.to_csv(TAB / "Table_PRODUCT03bw_remaining_raw_tower_download_manifest.csv", index=False)

closure_n = int(merged["closure_ratio"].notna().sum()) if "closure_ratio" in merged.columns else 0
gap_n = int(merged["gapfill_fraction_estimated"].notna().sum()) if "gapfill_fraction_estimated" in merged.columns else 0
uwue_n = int(merged["can_compute_uwue"].fillna(False).sum()) if "can_compute_uwue" in merged.columns else 0

status = {
    "generated": now(),
    "stage": "1B.6AM_raw_tower_download_ingest",
    "target_sites": len(target_sites),
    "archives_extracted": int(len(extract_log)),
    "raw_csvs_scanned": int(len(inventory)),
    "quality_candidate_rows": int(len(quality)),
    "sites_with_closure_ratio": closure_n,
    "sites_with_gapfill_fraction": gap_n,
    "sites_with_raw_uwue_inputs": uwue_n,
    "fully_satisfies_reza_tower_quality": bool(closure_n == len(target_sites) and gap_n == len(target_sites)),
    "remaining_missing_sites": int(len(missing)),
    "next_action": (
        "Reza tower-quality requirement is closed. Rerun Stage 1B.6AK and send final packet."
        if closure_n == len(target_sites) and gap_n == len(target_sites)
        else
        "Download the remaining raw tower exports listed in Table_PRODUCT03bw_remaining_raw_tower_download_manifest.csv into data/raw/towers/_reza_raw_exports/manual_fluxnet/, then rerun this same script."
    )
}

(TAB / "STAGE1B6AM_RAW_TOWER_DOWNLOAD_INGEST_DECISION.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

manual = []
manual.append("# Manual downloads still needed if Stage 1B.6AM is not fully resolved")
manual.append("")
manual.append("Put downloaded zip/csv/txt files here:")
manual.append("")
manual.append("```text")
manual.append("data/raw/towers/_reza_raw_exports/manual_fluxnet/")
manual.append("```")
manual.append("")
manual.append("Required fields:")
manual.append("")
manual.append("```text")
manual.append("TIMESTAMP_START/TIMESTAMP_END or TIMESTAMP")
manual.append("GPP")
manual.append("LE")
manual.append("VPD")
manual.append("H")
manual.append("NETRAD")
manual.append("G")
manual.append("QC flags such as LE_QC, GPP_QC, VPD_QC, or FLUXNET quality flags")
manual.append("```")
manual.append("")
manual.append("Sites still missing:")
manual.append("")
manual.append(missing.to_string(index=False) if len(missing) else "None.")
(TXT / "MANUAL_RAW_TOWER_DOWNLOAD_INSTRUCTIONS.md").write_text("\n".join(manual), encoding="utf-8")

report = []
report.append("# Stage 1B.6AM raw tower download + ingest")
report.append("")
report.append(f"Generated: {status['generated']}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```json")
report.append(json.dumps(status, indent=2))
report.append("```")
report.append("")
report.append("## Reza tower table with raw quality fields")
report.append("")
show = [
    "site_id", "igbp_class", "site_years", "tower_response_class",
    "closure_ratio", "closure_pass_0p7_1p3",
    "gapfill_fraction_estimated", "gapfill_pass_0p3",
    "can_compute_uwue", "has_observed_soil_water",
    "LE_col", "H_col", "NETRAD_col", "G_col", "GPP_col", "VPD_col",
    "source_file"
]
show = [c for c in show if c in merged.columns]
report.append("```text")
report.append(merged[show].to_string(index=False))
report.append("```")
report.append("")
report.append("## Remaining raw tower downloads")
report.append("")
report.append("```text")
report.append(missing.to_string(index=False) if len(missing) else "No remaining raw tower downloads needed.")
report.append("```")
report.append("")
report.append("## Best raw quality candidates")
report.append("")
report.append("```text")
report.append(best.head(40).to_string(index=False) if len(best) else "No raw quality candidates found.")
report.append("```")
report.append("")
report.append("## Raw file inventory preview")
report.append("")
inv_show = ["score", "has_LE", "has_H", "has_NETRAD", "has_G", "has_GPP", "has_VPD", "has_QC", "site_col", "sites_preview", "path"]
inv_show = [c for c in inv_show if c in inventory.columns]
report.append("```text")
report.append(inventory[inv_show].head(80).to_string(index=False) if len(inventory) else "No raw files scanned.")
report.append("```")
report.append("")
report.append("## Bottom line")
report.append("")
if status["fully_satisfies_reza_tower_quality"]:
    report.append("Tower closure and gap-fill are now fully populated for the 13-site Reza tower table.")
else:
    report.append("Tower closure/gap-fill are still not fully populated. This means the required raw tower energy-balance/QC exports are still missing for the listed sites. Download them into the manual_fluxnet folder and rerun this script.")

(TXT / "STAGE1B6AM_RAW_TOWER_DOWNLOAD_INGEST_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT03bs_raw_archive_extract_log.csv")
print("WROTE", TAB / "Table_PRODUCT03bt_raw_tower_file_inventory_after_download.csv")
print("WROTE", TAB / "Table_PRODUCT03bu_raw_tower_quality_candidates.csv")
print("WROTE", TAB / "Table_PRODUCT03bv_reza_tower_table_raw_quality_completed.csv")
print("WROTE", TAB / "Table_PRODUCT03bw_remaining_raw_tower_download_manifest.csv")
print("WROTE", TAB / "STAGE1B6AM_RAW_TOWER_DOWNLOAD_INGEST_DECISION.json")
print("WROTE", TXT / "MANUAL_RAW_TOWER_DOWNLOAD_INSTRUCTIONS.md")
print("WROTE", TXT / "STAGE1B6AM_RAW_TOWER_DOWNLOAD_INGEST_REPORT.md")
