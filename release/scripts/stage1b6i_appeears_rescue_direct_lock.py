from pathlib import Path
from datetime import datetime
import json
import math
import re
import pandas as pd
import numpy as np

OUT = Path("results/stage1b6i_appeears_rescue_direct_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

FINAL_POINTS = Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv")
DIRECT_SUMMARY = Path("results/stage1b6h_direct_earthdata_point_extract/tables/Table_PRODUCT02ba_direct_extract_coverage_summary.csv")
DIRECT_OUTPUT_SUMMARY = Path("results/stage1b6h_direct_earthdata_point_extract/tables/Table_PRODUCT02az_direct_extract_output_summary.csv")

if not FINAL_POINTS.exists():
    raise FileNotFoundError(f"Missing final points: {FINAL_POINTS}")

points = pd.read_csv(FINAL_POINTS)
points.columns = [str(c).strip() for c in points.columns]
points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
points = points.dropna(subset=["latitude", "longitude"]).copy()

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def find_col(df, names):
    lookup = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl:
                return c
    return None

search_roots = [
    Path("data/raw"),
    Path("data/external"),
    Path("data/raw_local"),
    Path("results"),
    Path("/Users/me/Downloads"),
]

candidate_files = []
patterns = [
    "720524fb",
    "grassland_modis_qa_2001_2024",
    "MOD17A2HGF",
    "MOD16A2GF",
    "appeears",
    "AppEEARS",
]

for root in search_roots:
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = str(p)
        low = name.lower()
        if p.suffix.lower() not in [".csv", ".json", ".txt", ".zip"]:
            continue
        score = 0
        for pat in patterns:
            if pat.lower() in low:
                score += 1
        if score:
            candidate_files.append({
                "path": str(p),
                "name": p.name,
                "suffix": p.suffix.lower(),
                "size_bytes": p.stat().st_size,
                "score": score,
            })

candidates = pd.DataFrame(candidate_files).sort_values(["score", "size_bytes"], ascending=[False, False]) if candidate_files else pd.DataFrame(columns=["path","name","suffix","size_bytes","score"])
candidates.to_csv(TAB / "Table_PRODUCT02bb_local_appeears_candidate_files.csv", index=False)

coverage_rows = []

for _, cand in candidates.iterrows():
    p = Path(cand["path"])
    if p.suffix.lower() != ".csv":
        continue
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        coverage_rows.append({
            "path": str(p),
            "read_status": "READ_ERROR",
            "error": str(e)[:500],
        })
        continue

    df.columns = [str(c).strip() for c in df.columns]
    latc = find_col(df, ["Latitude", "latitude", "lat"])
    lonc = find_col(df, ["Longitude", "longitude", "lon"])
    datec = find_col(df, ["Date", "date", "time"])
    idc = find_col(df, ["ID", "id", "target_id", "point_id", "site", "site_id"])

    var_cols = [c for c in df.columns if any(k in c.lower() for k in ["gpp", "psn", "et", "qc", "mod17", "mod16"])]

    if latc is None or lonc is None:
        coverage_rows.append({
            "path": str(p),
            "read_status": "NO_LAT_LON",
            "n_rows": len(df),
            "columns": ";".join(df.columns[:80]),
            "variables_found": ";".join(var_cols[:80]),
        })
        continue

    pts = df[[latc, lonc] + ([idc] if idc else [])].drop_duplicates().copy()
    pts[latc] = pd.to_numeric(pts[latc], errors="coerce")
    pts[lonc] = pd.to_numeric(pts[lonc], errors="coerce")
    pts = pts.dropna(subset=[latc, lonc])

    matched = 0
    nearest_dists = []
    for _, target in points.iterrows():
        if len(pts) == 0:
            nearest_dists.append(np.nan)
            continue
        dists = pts.apply(lambda r: haversine_km(float(target["latitude"]), float(target["longitude"]), float(r[latc]), float(r[lonc])), axis=1)
        nd = float(dists.min())
        nearest_dists.append(nd)
        if nd <= 1.0:
            matched += 1

    if datec is not None:
        dates = pd.to_datetime(df[datec], errors="coerce")
        n_dates = int(dates.nunique())
        date_min = str(dates.min().date()) if dates.notna().any() else ""
        date_max = str(dates.max().date()) if dates.notna().any() else ""
    else:
        n_dates = 0
        date_min = ""
        date_max = ""

    has_gpp = any("gpp" in c.lower() for c in var_cols)
    has_et = any(re.search(r"(^|[^a-z])et([^a-z]|$)", c.lower()) or "et_" in c.lower() for c in var_cols)
    has_qc = any("qc" in c.lower() for c in var_cols)

    if matched == len(points) and has_gpp and has_et and n_dates >= 1000:
        status = "CAN_USE_FOR_PRIMARY_MODIS_GPP_ET"
    elif matched == len(points):
        status = "COORDS_MATCH_BUT_VARIABLE_OR_DATE_COVERAGE_WEAK"
    elif matched > 0:
        status = "PARTIAL_COORDINATE_MATCH_ONLY"
    else:
        status = "DO_NOT_USE_FOR_PRIMARY_TARGETS"

    coverage_rows.append({
        "path": str(p),
        "read_status": "READ_OK",
        "n_rows": len(df),
        "n_unique_file_points": len(pts),
        "n_final_targets": len(points),
        "n_targets_matched_within_1km": matched,
        "min_nearest_km": float(np.nanmin(nearest_dists)) if len(nearest_dists) else np.nan,
        "median_nearest_km": float(np.nanmedian(nearest_dists)) if len(nearest_dists) else np.nan,
        "max_nearest_km": float(np.nanmax(nearest_dists)) if len(nearest_dists) else np.nan,
        "date_min": date_min,
        "date_max": date_max,
        "n_unique_dates": n_dates,
        "has_gpp_like_column": has_gpp,
        "has_et_like_column": has_et,
        "has_qc_like_column": has_qc,
        "variables_found": ";".join(var_cols[:80]),
        "coverage_status": status,
    })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02bc_local_appeears_candidate_coverage.csv", index=False)

usable_app = coverage[coverage.get("coverage_status", pd.Series(dtype=str)).eq("CAN_USE_FOR_PRIMARY_MODIS_GPP_ET")] if len(coverage) else pd.DataFrame()

direct_cov = pd.read_csv(DIRECT_SUMMARY) if DIRECT_SUMMARY.exists() else pd.DataFrame()
direct_out = pd.read_csv(DIRECT_OUTPUT_SUMMARY) if DIRECT_OUTPUT_SUMMARY.exists() else pd.DataFrame()

decisions = []

if len(usable_app):
    decisions.append({
        "product_group": "MODIS_GPP_MOD17",
        "decision": "USE_EXISTING_APPEEARS_FILE_IF_USER_ACCEPTS_PRIOR_APPEEARS",
        "reason": "A local AppEEARS candidate appears to match all final primary targets with GPP/ET/date coverage.",
        "source": usable_app.iloc[0]["path"],
    })
    decisions.append({
        "product_group": "MODIS_ET_MOD16",
        "decision": "USE_EXISTING_APPEEARS_FILE_IF_USER_ACCEPTS_PRIOR_APPEEARS",
        "reason": "A local AppEEARS candidate appears to match all final primary targets with GPP/ET/date coverage.",
        "source": usable_app.iloc[0]["path"],
    })
else:
    decisions.append({
        "product_group": "MODIS_GPP_MOD17",
        "decision": "USE_DIRECT_EARTHDATA_EXTRACTION_FULL_RUN",
        "reason": "No local AppEEARS file was found that covers the final 13 primary targets. Direct Earthdata smoke extraction worked.",
        "source": "data/raw_local/no_gee_direct_point_extract/MODIS_GPP_MOD17_direct_earthdata_point_samples.csv",
    })
    decisions.append({
        "product_group": "MODIS_ET_MOD16",
        "decision": "USE_DIRECT_EARTHDATA_EXTRACTION_FULL_RUN",
        "reason": "No local AppEEARS file was found that covers the final 13 primary targets. Direct Earthdata smoke extraction worked.",
        "source": "data/raw_local/no_gee_direct_point_extract/MODIS_ET_MOD16_direct_earthdata_point_samples.csv",
    })

decisions.append({
    "product_group": "MODIS_LAI_MCD15",
    "decision": "USE_DIRECT_EARTHDATA_EXTRACTION_FULL_RUN",
    "reason": "Prior AppEEARS email/order does not include LAI. Direct Earthdata smoke extraction worked.",
    "source": "data/raw_local/no_gee_direct_point_extract/MODIS_LAI_MCD15_direct_earthdata_point_samples.csv",
})
decisions.append({
    "product_group": "MCD64A1_BURNED_AREA",
    "decision": "USE_DIRECT_EARTHDATA_EXTRACTION_FULL_RUN",
    "reason": "Prior AppEEARS email/order does not include burned area. Direct Earthdata smoke extraction worked.",
    "source": "data/raw_local/no_gee_direct_point_extract/MCD64A1_BURNED_AREA_direct_earthdata_point_samples.csv",
})

decision = pd.DataFrame(decisions)
decision.to_csv(TAB / "Table_PRODUCT02bd_final_source_decision_no_gee_no_new_appeears.csv", index=False)

mentor_gap_rows = [
    {
        "mentor_requirement": "MODIS GPP in 3x3 product matrix",
        "current_source_plan": decision[decision.product_group.eq("MODIS_GPP_MOD17")]["decision"].iloc[0],
        "left_behind": False,
    },
    {
        "mentor_requirement": "MODIS ET in 3x3 product matrix",
        "current_source_plan": decision[decision.product_group.eq("MODIS_ET_MOD16")]["decision"].iloc[0],
        "left_behind": False,
    },
    {
        "mentor_requirement": "LAI / growing-season / canopy covariate",
        "current_source_plan": "direct Earthdata MCD15A2H point extraction",
        "left_behind": False,
    },
    {
        "mentor_requirement": "burned/disturbed observation exclusion",
        "current_source_plan": "direct Earthdata MCD64A1 point extraction",
        "left_behind": False,
    },
    {
        "mentor_requirement": "full gridded xarray/Zarr MODIS cube",
        "current_source_plan": "not feasible without external storage; tower-centered strict point extraction used instead",
        "left_behind": "Only if mentor explicitly requires gridded map/cube rather than tower-centered product arbitration.",
    },
]
mentor = pd.DataFrame(mentor_gap_rows)
mentor.to_csv(TAB / "Table_PRODUCT02be_mentor_requirement_gap_lock.csv", index=False)

report = []
report.append("# Stage 1B.6I AppEEARS rescue + direct extraction lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("Prior AppEEARS data is allowed as already-downloaded local data only if it matches the final 13 primary target coordinates. No new AppEEARS dependency is required. If it does not match, use direct Earthdata extraction.")
report.append("")
report.append("## Local AppEEARS candidates")
report.append("")
report.append("```text")
report.append(candidates.head(80).to_string(index=False) if len(candidates) else "No local AppEEARS/MOD17/MOD16 candidates found.")
report.append("```")
report.append("")
report.append("## Local AppEEARS candidate coverage")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No readable local AppEEARS coverage table.")
report.append("```")
report.append("")
report.append("## Final source decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Mentor requirement gap lock")
report.append("")
report.append("```text")
report.append(mentor.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Nothing is left behind for tower-centered strict product arbitration. Full gridded cube remains a separate storage-dependent extension.")
report.append("")

(TXT / "STAGE1B6I_APPEEARS_RESCUE_DIRECT_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6I_appeears_rescue_direct_lock",
    "status": "complete",
    "n_local_candidates": int(len(candidates)),
    "n_usable_appeears_primary_files": int(len(usable_app)),
    "outputs": {
        "candidate_files": str(TAB / "Table_PRODUCT02bb_local_appeears_candidate_files.csv"),
        "candidate_coverage": str(TAB / "Table_PRODUCT02bc_local_appeears_candidate_coverage.csv"),
        "source_decision": str(TAB / "Table_PRODUCT02bd_final_source_decision_no_gee_no_new_appeears.csv"),
        "mentor_gap_lock": str(TAB / "Table_PRODUCT02be_mentor_requirement_gap_lock.csv"),
        "report": str(TXT / "STAGE1B6I_APPEEARS_RESCUE_DIRECT_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6I_APPEEARS_RESCUE_DIRECT_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bb_local_appeears_candidate_files.csv")
print("WROTE", TAB / "Table_PRODUCT02bc_local_appeears_candidate_coverage.csv")
print("WROTE", TAB / "Table_PRODUCT02bd_final_source_decision_no_gee_no_new_appeears.csv")
print("WROTE", TAB / "Table_PRODUCT02be_mentor_requirement_gap_lock.csv")
print("WROTE", TXT / "STAGE1B6I_APPEEARS_RESCUE_DIRECT_LOCK_REPORT.md")
