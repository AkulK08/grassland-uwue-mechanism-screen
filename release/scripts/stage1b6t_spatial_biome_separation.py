from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6t_spatial_biome_separation")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6t")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

ARB = Path("data/processed/stage1b6s/tower_arbitration_prep_strict2x2.csv")

TARGET_CANDIDATES = [
    Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv"),
    Path("results/stage1b6g_scientific_target_lock/tables/Table_PRODUCT02aq_FINAL_no_gee_product_point_request.csv"),
    Path("data/raw/tower_centered_phase19/phase19_main_13_tower_points_for_export.csv"),
]

GEO_CANDIDATES = [
    Path("results/paper_point_geography_thesis_lock/tables/Table72_high_vpd_point_geography.csv"),
    Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
    Path("data/external/aridity_by_point.csv"),
    Path("data/external/soilgrids_texture_by_point.csv"),
]

def find_existing(cands):
    for p in cands:
        if p.exists():
            return p
    return None

def normalize_targets(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = next((cols[k] for k in ["id", "point_id", "site_id", "site", "tower_id"] if k in cols), None)
    lat_col = next((cols[k] for k in ["lat", "latitude"] if k in cols), None)
    lon_col = next((cols[k] for k in ["lon", "longitude"] if k in cols), None)
    if not id_col or not lat_col or not lon_col:
        return None
    return pd.DataFrame({
        "point_id": df[id_col].astype(str),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    }).drop_duplicates("point_id")

def broad_region_from_point(pid):
    pid = str(pid)
    if pid.startswith("US-Ne"):
        return "US_Great_Plains_Nebraska"
    if pid in ["US-CMW", "US-Cop"]:
        return "US_Southwest_Dryland"
    if pid in ["US-Ton", "US-Var"]:
        return "California_Savanna_Grassland"
    if pid in ["US-Dk1", "US-SP1"]:
        return "US_Eastern_Southeast"
    if pid.startswith("CA-"):
        return "Canada_Boreal_Grassland"
    if pid.startswith("CN-"):
        return "China_Qinghai_Tibetan"
    if pid.startswith("NL-"):
        return "Europe_Temperate"
    if pid.startswith("RU-"):
        return "Russia_High_Latitude"
    return "Other"

def us_vs_nonus(pid):
    return "US" if str(pid).startswith("US-") else "Non_US"

def west_vs_east(lon):
    if pd.isna(lon):
        return "unknown"
    if lon < -100:
        return "western_or_central_NA"
    if lon < -70:
        return "eastern_NA"
    if lon < 40:
        return "Europe_Africa_sector"
    return "Asia_sector"

def latitude_band(lat):
    if pd.isna(lat):
        return "unknown"
    if lat < 35:
        return "low_mid_lat_lt35"
    if lat < 50:
        return "mid_lat_35_50"
    return "high_lat_ge50"

def classify_strength(frac):
    if pd.isna(frac):
        return "unknown"
    if frac >= 0.60:
        return "strong_limitation"
    if frac >= 0.40:
        return "moderate_limitation"
    if frac >= 0.20:
        return "weak_or_mixed_limitation"
    return "low_limitation"

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def simple_binary_diff(df, group_col, value_col, flag_col):
    rows = []
    sub = df[[group_col, value_col, flag_col]].dropna(subset=[group_col, value_col])
    if sub.empty:
        return rows
    counts = sub[group_col].value_counts()
    usable_levels = counts[counts >= 2].index.tolist()
    sub = sub[sub[group_col].isin(usable_levels)].copy()
    if sub[group_col].nunique() < 2:
        return rows

    group_stats = sub.groupby(group_col).agg(
        n=(value_col, "size"),
        mean_value=(value_col, "mean"),
        median_value=(value_col, "median"),
        n_flag=(flag_col, "sum"),
    ).reset_index()
    group_stats["flag_fraction"] = group_stats["n_flag"] / group_stats["n"]

    hi = group_stats.sort_values("mean_value", ascending=False).iloc[0]
    lo = group_stats.sort_values("mean_value", ascending=True).iloc[0]

    rows.append({
        "group_col": group_col,
        "value_col": value_col,
        "flag_col": flag_col,
        "n_groups_used": int(group_stats.shape[0]),
        "n_rows_used": int(sub.shape[0]),
        "highest_group": hi[group_col],
        "highest_group_n": int(hi["n"]),
        "highest_group_mean": float(hi["mean_value"]),
        "highest_group_flag_fraction": float(hi["flag_fraction"]),
        "lowest_group": lo[group_col],
        "lowest_group_n": int(lo["n"]),
        "lowest_group_mean": float(lo["mean_value"]),
        "lowest_group_flag_fraction": float(lo["flag_fraction"]),
        "mean_difference_high_minus_low": float(hi["mean_value"] - lo["mean_value"]),
        "interpretation": "exploratory_large_gap" if float(hi["mean_value"] - lo["mean_value"]) >= 0.25 else "exploratory_small_or_moderate_gap",
    })
    return rows

def corr_row(df, x_col, y_col):
    sub = df[[x_col, y_col]].copy()
    sub[x_col] = safe_numeric(sub[x_col])
    sub[y_col] = safe_numeric(sub[y_col])
    sub = sub.dropna()
    if len(sub) < 5 or sub[x_col].nunique() < 2:
        return None
    r = float(sub[x_col].corr(sub[y_col], method="spearman"))
    return {
        "x_col": x_col,
        "y_col": y_col,
        "n": int(len(sub)),
        "spearman_r": r,
        "abs_spearman_r": abs(r),
        "interpretation": "exploratory_strong_association" if abs(r) >= 0.5 else "exploratory_weak_to_moderate_association",
    }

if not ARB.exists():
    raise FileNotFoundError(f"Missing arbitration table: {ARB}")

arb = pd.read_csv(ARB)
arb["point_id"] = arb["point_id"].astype(str)

# Attach target lat/lon.
target_path = find_existing(TARGET_CANDIDATES)
target = None
if target_path is not None:
    target = normalize_targets(pd.read_csv(target_path))

if target is not None:
    arb = arb.merge(target, on="point_id", how="left", suffixes=("", "_target"))

# Attach geography/covariate files.
source_rows = []
for p in GEO_CANDIDATES:
    if not p.exists():
        source_rows.append({"path": str(p), "exists": False, "used": False, "reason": "missing"})
        continue
    try:
        g = pd.read_csv(p)
        if "point_id" not in g.columns:
            source_rows.append({"path": str(p), "exists": True, "used": False, "reason": "no_point_id"})
            continue
        g["point_id"] = g["point_id"].astype(str)
        cols_to_add = ["point_id"]
        for c in g.columns:
            if c == "point_id":
                continue
            if c in arb.columns:
                continue
            # Keep likely useful covariates.
            cl = c.lower()
            if any(k in cl for k in [
                "lat", "lon", "aridity", "vpd", "soil", "lai", "rooting",
                "p50", "iso", "geo_", "eco_", "biome", "realm", "region",
                "hydro", "country", "continent", "precip", "temperature"
            ]):
                cols_to_add.append(c)
        if len(cols_to_add) > 1:
            before_cols = len(arb.columns)
            arb = arb.merge(g[cols_to_add].drop_duplicates("point_id"), on="point_id", how="left")
            source_rows.append({
                "path": str(p),
                "exists": True,
                "used": True,
                "n_cols_added": len(arb.columns) - before_cols,
                "cols_added": ";".join([c for c in cols_to_add if c != "point_id"]),
            })
        else:
            source_rows.append({"path": str(p), "exists": True, "used": False, "reason": "no_new_useful_cols"})
    except Exception as e:
        source_rows.append({"path": str(p), "exists": True, "used": False, "reason": repr(e)})

# Derived spatial groups.
arb["broad_region_handbuilt"] = arb["point_id"].apply(broad_region_from_point)
arb["us_vs_nonus"] = arb["point_id"].apply(us_vs_nonus)

lat_col = "lat" if "lat" in arb.columns else None
lon_col = "lon" if "lon" in arb.columns else None
if lat_col:
    arb["latitude_band_handbuilt"] = arb[lat_col].apply(latitude_band)
else:
    arb["latitude_band_handbuilt"] = "unknown"

if lon_col:
    arb["longitude_sector_handbuilt"] = arb[lon_col].apply(west_vs_east)
else:
    arb["longitude_sector_handbuilt"] = "unknown"

# Define outcome flags.
arb["satellite_limitation_mean_fraction"] = safe_numeric(arb["satellite_limitation_mean_fraction"])
arb["satellite_moderate_or_strong_limitation"] = arb["satellite_limitation_mean_fraction"] >= 0.40
arb["satellite_strong_limitation"] = arb["satellite_limitation_mean_fraction"] >= 0.60
arb["satellite_strength_class_recomputed"] = arb["satellite_limitation_mean_fraction"].apply(classify_strength)

if "tower_limitation_like" in arb.columns:
    arb["tower_limitation_like"] = arb["tower_limitation_like"].astype(str).str.lower().isin(["true", "1", "yes"])
if "tower_satellite_limitation_agreement_strict2x2" in arb.columns:
    arb["tower_satellite_limitation_agreement_strict2x2"] = arb["tower_satellite_limitation_agreement_strict2x2"].astype(str).str.lower().isin(["true", "1", "yes"])

arb.to_csv(DATA / "spatial_biome_heterogeneity_input_strict2x2.csv", index=False)

# Candidate grouping variables.
candidate_groups = [
    "broad_region_handbuilt",
    "us_vs_nonus",
    "latitude_band_handbuilt",
    "longitude_sector_handbuilt",
    "geo_continent",
    "geo_subregion",
    "geo_country",
    "eco_biome",
    "eco_realm",
    "eco_ecoregion",
    "hydroclimatic_regime",
    "aridity_quartile",
    "mean_vpd_quartile",
    "dryland_class_unep_if_ai",
    "latitude_band",
]

candidate_groups = [c for c in candidate_groups if c in arb.columns]

group_rows = []
for g in candidate_groups:
    vals = arb[g].dropna().astype(str)
    if vals.nunique() < 2:
        continue
    if vals.nunique() > 8:
        continue
    group_rows.extend(simple_binary_diff(
        arb.assign(**{g: arb[g].astype(str)}),
        g,
        "satellite_limitation_mean_fraction",
        "satellite_moderate_or_strong_limitation",
    ))

group_df = pd.DataFrame(group_rows).sort_values(
    ["mean_difference_high_minus_low", "n_rows_used"],
    ascending=[False, False]
) if group_rows else pd.DataFrame()
group_df.to_csv(TAB / "Table_PRODUCT02da_spatial_biome_group_signal_scan.csv", index=False)

# Continuous covariate correlations.
candidate_cont = [
    "lat", "lon", "mean_vpd", "mean_soil_moisture", "mean_annual_temperature",
    "mean_annual_precipitation", "mean_lai", "growing_season_mean_lai",
    "soil_sand", "soil_clay", "soil_silt", "rooting_depth", "p50",
    "isohydricity", "aridity_index", "abs_lat",
]
candidate_cont = [c for c in candidate_cont if c in arb.columns]

corrs = []
for x in candidate_cont:
    r = corr_row(arb, x, "satellite_limitation_mean_fraction")
    if r:
        corrs.append(r)

corr_df = pd.DataFrame(corrs).sort_values("abs_spearman_r", ascending=False) if corrs else pd.DataFrame()
corr_df.to_csv(TAB / "Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv", index=False)

# Site ranking.
rank = arb[[
    "point_id",
    "sat_limitation_fraction_log_uwue",
    "sat_limitation_fraction_log_wue",
    "satellite_limitation_mean_fraction",
    "satellite_strength_class_recomputed",
    "tower_response_class" if "tower_response_class" in arb.columns else "point_id",
    "satellite_response_class" if "satellite_response_class" in arb.columns else "point_id",
    "tower_satellite_limitation_agreement_strict2x2" if "tower_satellite_limitation_agreement_strict2x2" in arb.columns else "point_id",
    "broad_region_handbuilt",
    "us_vs_nonus",
    "latitude_band_handbuilt",
    "longitude_sector_handbuilt",
]].copy()

rank = rank.loc[:, ~rank.columns.duplicated()]
rank = rank.sort_values("satellite_limitation_mean_fraction", ascending=False)
rank.to_csv(TAB / "Table_PRODUCT02dc_site_spatial_limitation_ranking.csv", index=False)

# Decision.
best_group_gap = float(group_df["mean_difference_high_minus_low"].max()) if len(group_df) else np.nan
best_corr = float(corr_df["abs_spearman_r"].max()) if len(corr_df) else np.nan

if len(group_df) and best_group_gap >= 0.25:
    verdict = "SPATIAL_BIOME_HETEROGENEITY_SIGNAL_PRESENT_EXPLORATORY"
    blocking_next = False
elif len(corr_df) and best_corr >= 0.50:
    verdict = "CONTINUOUS_SPATIAL_TRAIT_SIGNAL_PRESENT_EXPLORATORY"
    blocking_next = False
else:
    verdict = "NO_STRONG_SPATIAL_BIOME_SIGNAL_FOUND_YET"
    blocking_next = False

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_sites": int(arb["point_id"].nunique()),
    "n_group_variables_scanned": int(len(candidate_groups)),
    "n_continuous_variables_scanned": int(len(candidate_cont)),
    "best_group_mean_difference": best_group_gap,
    "best_abs_spearman": best_corr,
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "TRAIT_CLIMATE_SOIL_MODEL_WITH_SPATIAL_BIOME_TERMS",
}])
decision.to_csv(TAB / "Table_PRODUCT02dd_spatial_biome_separation_decision.csv", index=False)

source_df = pd.DataFrame(source_rows)
source_df.to_csv(TAB / "Table_PRODUCT02de_spatial_biome_source_inventory.csv", index=False)

report = []
report.append("# Stage 1B.6T spatial/biome separation scout")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Site ranking")
report.append("")
report.append("```text")
report.append(rank.to_string(index=False))
report.append("```")
report.append("")
report.append("## Group signal scan")
report.append("")
report.append("```text")
report.append(group_df.head(60).to_string(index=False) if len(group_df) else "No group signal rows.")
report.append("```")
report.append("")
report.append("## Continuous covariate signal scan")
report.append("")
report.append("```text")
report.append(corr_df.head(60).to_string(index=False) if len(corr_df) else "No continuous covariate signal rows.")
report.append("```")
report.append("")
report.append("## Source inventory")
report.append("")
report.append("```text")
report.append(source_df.to_string(index=False) if len(source_df) else "No source rows.")
report.append("```")
report.append("")
report.append("## Outputs")
report.append("")
report.append("- Heterogeneity input: `data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv`")
report.append("- Group scan: `results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02da_spatial_biome_group_signal_scan.csv`")
report.append("- Continuous scan: `results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv`")
report.append("- Site ranking: `results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02dc_site_spatial_limitation_ranking.csv`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Treat spatial/biome separation as heterogeneity analysis, not proof of a causal biome mechanism. With n=13, use it to motivate stratified/partial-pooling models and to explain tower-satellite disagreement.")
report.append("")

(TXT / "STAGE1B6T_SPATIAL_BIOME_SEPARATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6T_spatial_biome_separation",
    "status": str(decision["verdict"].iloc[0]),
    "outputs": {
        "input": "data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv",
        "group_scan": str(TAB / "Table_PRODUCT02da_spatial_biome_group_signal_scan.csv"),
        "continuous_scan": str(TAB / "Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv"),
        "site_ranking": str(TAB / "Table_PRODUCT02dc_site_spatial_limitation_ranking.csv"),
        "decision": str(TAB / "Table_PRODUCT02dd_spatial_biome_separation_decision.csv"),
        "report": str(TXT / "STAGE1B6T_SPATIAL_BIOME_SEPARATION_REPORT.md"),
    }
}
(TAB / "STAGE1B6T_SPATIAL_BIOME_SEPARATION_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "spatial_biome_heterogeneity_input_strict2x2.csv")
print("WROTE", TAB / "Table_PRODUCT02da_spatial_biome_group_signal_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02dc_site_spatial_limitation_ranking.csv")
print("WROTE", TAB / "Table_PRODUCT02dd_spatial_biome_separation_decision.csv")
print("WROTE", TXT / "STAGE1B6T_SPATIAL_BIOME_SEPARATION_REPORT.md")
