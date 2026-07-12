from pathlib import Path
from datetime import datetime
import json
import math
import pandas as pd
import numpy as np

OUT = Path("results/stage1b6f_strict_point_narrowing_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
REQ = Path("data/raw_local/no_gee_point_requests")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
REQ.mkdir(parents=True, exist_ok=True)

COORD_FILES = {
    "main13": Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv"),
    "strict_gra5": Path("results/tower_satellite_extraction_targets_FINAL/SENSITIVITY_strict_GRA_coordinates.csv"),
    "all49": Path("results/tower_satellite_extraction_targets_FINAL/CONTRAST_all_49_tower_coordinates.csv"),
}

MODIS_POINT_FILES = {
    "MODIS_GPP_MOD17": Path("data/raw/appeears_modis_qa/grassland-modis-qa-2001-2024-MOD17A2HGF-061-results.csv"),
    "MODIS_ET_MOD16": Path("data/raw/appeears_modis_qa/grassland-modis-qa-2001-2024-MOD16A2GF-061-results.csv"),
}

OPTIONAL_TABLES = [
    Path("results/final_nonwriting_lock/tables/Table_FINAL_site_mismatch_diagnostics.csv"),
    Path("results/tower_centered_phase19_no_gee/tables/Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv"),
    Path("results/tower_grassland_spatial_trait_lock/tables/Table101_tower_landcover_spatial_trait_annotation.csv"),
    Path("results/tower_validation_broad_inventory/tables/Table_tower_response_by_site.csv"),
]

def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_col(df, options):
    lookup = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in lookup:
            return lookup[o.lower()]
    for c in df.columns:
        cl = c.lower()
        for o in options:
            if o.lower() in cl:
                return c
    return None

def load_coords(path, scope):
    if not path.exists():
        return pd.DataFrame(columns=["target_id", "lat", "lon", "scope"])
    df = norm_cols(pd.read_csv(path))
    lat = find_col(df, ["latitude", "lat"])
    lon = find_col(df, ["longitude", "lon", "long"])
    tid = find_col(df, ["target_id", "tower_id", "site_id", "site", "id", "point_id"])
    if lat is None or lon is None:
        return pd.DataFrame(columns=["target_id", "lat", "lon", "scope"])
    if tid is None:
        df["target_id"] = [f"{scope}_{i:03d}" for i in range(len(df))]
        tid = "target_id"
    out = df[[tid, lat, lon]].rename(columns={tid: "target_id", lat: "lat", lon: "lon"})
    out["scope"] = scope
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    return out

coord_parts = []
for scope, path in COORD_FILES.items():
    coord_parts.append(load_coords(path, scope))

coords = pd.concat(coord_parts, ignore_index=True).drop_duplicates(subset=["target_id", "lat", "lon", "scope"])
coords.to_csv(TAB / "Table_PRODUCT02ag_all_candidate_target_points.csv", index=False)

# Build "interesting" points. At minimum this is main13 + strict GRA + all49.
# If prior mismatch/breakdown tables exist, add flags.
interesting = coords.copy()
interesting["include_reason"] = interesting["scope"]

for p in OPTIONAL_TABLES:
    if not p.exists():
        continue
    try:
        df = norm_cols(pd.read_csv(p))
    except Exception:
        continue

    id_col = find_col(df, ["target_id", "tower_id", "site", "site_id"])
    if id_col is None:
        continue

    useful_cols = [id_col]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["class", "breakdown", "saturation", "enhancement", "mismatch", "agreement", "post_slope", "slope_change"]):
            useful_cols.append(c)

    tmp = df[useful_cols].copy().rename(columns={id_col: "target_id"})
    tmp["source_table"] = str(p)

    # Add a broad interesting flag if the row mentions mismatch/breakdown/saturation.
    text = tmp.astype(str).agg(" ".join, axis=1).str.lower()
    tmp["extra_interest_flag"] = np.where(
        text.str.contains("breakdown|mismatch|false|saturation|disagree|weak|negative", regex=True),
        "prior_response_or_mismatch_interest",
        ""
    )

    flags = tmp[tmp["extra_interest_flag"].ne("")][["target_id", "extra_interest_flag"]].drop_duplicates()
    if len(flags):
        interesting = interesting.merge(flags, on="target_id", how="left")
        interesting["include_reason"] = np.where(
            interesting["extra_interest_flag"].fillna("").ne(""),
            interesting["include_reason"] + "; " + interesting["extra_interest_flag"].fillna(""),
            interesting["include_reason"]
        )
        interesting = interesting.drop(columns=["extra_interest_flag"])

# For strict mentor completion, keep all49 as the full tower arbitration set.
# main13 remains the primary manuscript scope.
interesting["strict_priority"] = np.where(interesting["scope"].eq("main13"), 1,
                                  np.where(interesting["scope"].eq("strict_gra5"), 2, 3))
interesting = interesting.sort_values(["strict_priority", "target_id"]).drop_duplicates(subset=["target_id", "lat", "lon"])

interesting.to_csv(TAB / "Table_PRODUCT02ah_strict_interesting_points.csv", index=False)

# AppEEARS-style point request CSV.
appeears = interesting[["target_id", "lat", "lon"]].copy()
appeears = appeears.rename(columns={"target_id": "id", "lat": "latitude", "lon": "longitude"})
appeears.to_csv(REQ / "strict_tower_interesting_points_for_appeears.csv", index=False)
appeears.to_csv(TAB / "Table_PRODUCT02ai_strict_point_request_csv.csv", index=False)

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

coverage_rows = []

for product_group, path in MODIS_POINT_FILES.items():
    if not path.exists():
        for _, pt in interesting.iterrows():
            coverage_rows.append({
                "product_group": product_group,
                "target_id": pt["target_id"],
                "target_lat": pt["lat"],
                "target_lon": pt["lon"],
                "point_file_exists": False,
                "matched": False,
                "nearest_id": "",
                "nearest_distance_km": None,
                "n_rows_for_nearest": 0,
                "date_min": "",
                "date_max": "",
                "n_unique_dates": 0,
                "variables_found": "",
                "coverage_status": "POINT_FILE_MISSING",
            })
        continue

    try:
        df = norm_cols(pd.read_csv(path, low_memory=False))
    except Exception as e:
        for _, pt in interesting.iterrows():
            coverage_rows.append({
                "product_group": product_group,
                "target_id": pt["target_id"],
                "target_lat": pt["lat"],
                "target_lon": pt["lon"],
                "point_file_exists": True,
                "matched": False,
                "nearest_id": "",
                "nearest_distance_km": None,
                "n_rows_for_nearest": 0,
                "date_min": "",
                "date_max": "",
                "n_unique_dates": 0,
                "variables_found": "",
                "coverage_status": "POINT_FILE_READ_ERROR",
                "error": str(e)[:500],
            })
        continue

    lat_col = find_col(df, ["Latitude", "lat", "latitude"])
    lon_col = find_col(df, ["Longitude", "lon", "longitude"])
    id_col = find_col(df, ["ID", "id", "target_id", "point_id"])
    date_col = find_col(df, ["Date", "date", "time"])

    var_cols = [c for c in df.columns if any(k in c.lower() for k in ["gpp", "psn", "et", "qc", "lai", "fpar", "burn"])]

    if lat_col is None or lon_col is None:
        for _, pt in interesting.iterrows():
            coverage_rows.append({
                "product_group": product_group,
                "target_id": pt["target_id"],
                "target_lat": pt["lat"],
                "target_lon": pt["lon"],
                "point_file_exists": True,
                "matched": False,
                "nearest_id": "",
                "nearest_distance_km": None,
                "n_rows_for_nearest": 0,
                "date_min": "",
                "date_max": "",
                "n_unique_dates": 0,
                "variables_found": ";".join(var_cols[:50]),
                "coverage_status": "NO_LAT_LON_COLUMNS_IN_POINT_FILE",
            })
        continue

    pts = df[[lat_col, lon_col] + ([id_col] if id_col else [])].drop_duplicates().copy()
    pts[lat_col] = pd.to_numeric(pts[lat_col], errors="coerce")
    pts[lon_col] = pd.to_numeric(pts[lon_col], errors="coerce")
    pts = pts.dropna(subset=[lat_col, lon_col])

    for _, pt in interesting.iterrows():
        dists = pts.apply(lambda r: haversine_km(float(pt["lat"]), float(pt["lon"]), float(r[lat_col]), float(r[lon_col])), axis=1)
        if len(dists) == 0:
            nearest_idx = None
        else:
            nearest_idx = dists.idxmin()

        if nearest_idx is None:
            nearest_dist = None
            nearest_id = ""
            sub = pd.DataFrame()
        else:
            nearest = pts.loc[nearest_idx]
            nearest_dist = float(dists.loc[nearest_idx])
            nearest_id = str(nearest[id_col]) if id_col else ""
            if id_col:
                sub = df[df[id_col].astype(str).eq(nearest_id)].copy()
            else:
                sub = df[
                    np.isclose(pd.to_numeric(df[lat_col], errors="coerce"), float(nearest[lat_col]))
                    & np.isclose(pd.to_numeric(df[lon_col], errors="coerce"), float(nearest[lon_col]))
                ].copy()

        matched = nearest_dist is not None and nearest_dist <= 1.0

        if date_col and len(sub):
            dates = pd.to_datetime(sub[date_col], errors="coerce")
            date_min = str(dates.min().date()) if dates.notna().any() else ""
            date_max = str(dates.max().date()) if dates.notna().any() else ""
            n_dates = int(dates.nunique())
        else:
            date_min = ""
            date_max = ""
            n_dates = 0

        if matched and n_dates >= 500:
            status = "COVERED_POINT_TIMESERIES"
        elif matched:
            status = "MATCHED_BUT_DATE_COVERAGE_WEAK"
        else:
            status = "NO_NEARBY_MATCH_WITHIN_1KM"

        coverage_rows.append({
            "product_group": product_group,
            "target_id": pt["target_id"],
            "target_lat": pt["lat"],
            "target_lon": pt["lon"],
            "point_file_exists": True,
            "matched": matched,
            "nearest_id": nearest_id,
            "nearest_distance_km": nearest_dist,
            "n_rows_for_nearest": int(len(sub)),
            "date_min": date_min,
            "date_max": date_max,
            "n_unique_dates": n_dates,
            "variables_found": ";".join(var_cols[:80]),
            "coverage_status": status,
        })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02aj_existing_modis_point_csv_coverage.csv", index=False)

coverage_summary = coverage.groupby("product_group").agg(
    n_targets=("target_id", "nunique"),
    n_covered=("coverage_status", lambda s: int((s == "COVERED_POINT_TIMESERIES").sum())),
    n_matched=("matched", "sum"),
    min_nearest_distance_km=("nearest_distance_km", "min"),
    median_nearest_distance_km=("nearest_distance_km", "median"),
    max_nearest_distance_km=("nearest_distance_km", "max"),
    min_unique_dates=("n_unique_dates", "min"),
    median_unique_dates=("n_unique_dates", "median"),
    max_unique_dates=("n_unique_dates", "max"),
).reset_index()

coverage_summary["strict_point_csv_acceptable"] = coverage_summary["n_covered"].eq(coverage_summary["n_targets"])
coverage_summary.to_csv(TAB / "Table_PRODUCT02ak_existing_point_csv_coverage_summary.csv", index=False)

# Decision table.
decision_rows = []

for _, r in coverage_summary.iterrows():
    if bool(r["strict_point_csv_acceptable"]):
        decision = "USE_EXISTING_POINT_CSV_FOR_STRICT_TOWER_PRODUCT_MATRIX"
        action = "No raw full-tile download needed for this product for tower-centered strict product arbitration."
    else:
        decision = "NEED_NEW_POINT_EXTRACTION_OR_RAW_DOWNLOAD"
        action = "Use the generated point request CSV in AppEEARS/LP DAAC point extraction or download raw tiles to external storage."
    decision_rows.append({
        "product_group": r["product_group"],
        "strict_point_csv_acceptable": bool(r["strict_point_csv_acceptable"]),
        "decision": decision,
        "action": action,
    })

# LAI/burned area decisions.
decision_rows.append({
    "product_group": "MODIS_LAI_MCD15",
    "strict_point_csv_acceptable": False,
    "decision": "NEED_NEW_POINT_EXTRACTION_OR_RAW_DOWNLOAD",
    "action": "Use generated point request CSV to request MCD15A2H Lai_500m/FparLai_QC by point, or download raw HDFs to external storage."
})
decision_rows.append({
    "product_group": "MCD64A1_BURNED_AREA",
    "strict_point_csv_acceptable": False,
    "decision": "NEED_NEW_POINT_EXTRACTION_OR_RAW_DOWNLOAD",
    "action": "Use generated point request CSV to request MCD64A1 Burn Date/QA by point, or download raw HDFs to external storage."
})
decision_rows.append({
    "product_group": "SOILGRIDS_TEXTURE",
    "strict_point_csv_acceptable": True,
    "decision": "USE_EXISTING_POINT_CSV_FOR_STRICT_TOWER_TRAIT_CONTROLS",
    "action": "Existing SoilGrids point table is acceptable for tower-centered trait causal model; raw rasters only needed for gridded map."
})

decision = pd.DataFrame(decision_rows)
decision.to_csv(TAB / "Table_PRODUCT02al_strict_narrowing_decision.csv", index=False)

report = []
report.append("# Stage 1B.6F strict point/narrowing lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Meaning")
report.append("")
report.append("This stage decides whether the strict analysis can avoid full MODIS tile downloads by using tower/interesting-point extraction. This is still no-GEE. It is valid for tower-centered product arbitration and trait causal analysis, but not for making a full gridded global xarray cube.")
report.append("")
report.append("## Candidate target points")
report.append("")
report.append("```text")
report.append(interesting.to_string(index=False))
report.append("```")
report.append("")
report.append("## Existing MODIS point CSV coverage summary")
report.append("")
report.append("```text")
report.append(coverage_summary.to_string(index=False) if len(coverage_summary) else "No MODIS point coverage summary available.")
report.append("```")
report.append("")
report.append("## Strict narrowing decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Generated point request CSV")
report.append("")
report.append(str(REQ / "strict_tower_interesting_points_for_appeears.csv"))
report.append("")
report.append("## Strict completion rule")
report.append("")
report.append("Stage 1B.6F is complete when each limited product is either covered by existing point CSVs or has a generated no-GEE point request file for AppEEARS/LP DAAC extraction.")
report.append("")

(TXT / "STAGE1B6F_STRICT_POINT_NARROWING_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6F_strict_point_narrowing_lock",
    "status": "complete",
    "n_target_points": int(len(interesting)),
    "outputs": {
        "candidate_points": str(TAB / "Table_PRODUCT02ag_all_candidate_target_points.csv"),
        "interesting_points": str(TAB / "Table_PRODUCT02ah_strict_interesting_points.csv"),
        "point_request_csv": str(REQ / "strict_tower_interesting_points_for_appeears.csv"),
        "existing_modis_point_coverage": str(TAB / "Table_PRODUCT02aj_existing_modis_point_csv_coverage.csv"),
        "coverage_summary": str(TAB / "Table_PRODUCT02ak_existing_point_csv_coverage_summary.csv"),
        "decision": str(TAB / "Table_PRODUCT02al_strict_narrowing_decision.csv"),
        "report": str(TXT / "STAGE1B6F_STRICT_POINT_NARROWING_LOCK_REPORT.md"),
    }
}

(TAB / "STAGE1B6F_STRICT_POINT_NARROWING_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02ag_all_candidate_target_points.csv")
print("WROTE", TAB / "Table_PRODUCT02ah_strict_interesting_points.csv")
print("WROTE", REQ / "strict_tower_interesting_points_for_appeears.csv")
print("WROTE", TAB / "Table_PRODUCT02aj_existing_modis_point_csv_coverage.csv")
print("WROTE", TAB / "Table_PRODUCT02ak_existing_point_csv_coverage_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02al_strict_narrowing_decision.csv")
print("WROTE", TXT / "STAGE1B6F_STRICT_POINT_NARROWING_LOCK_REPORT.md")
