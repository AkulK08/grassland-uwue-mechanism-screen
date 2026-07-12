from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

KEY = ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"
PHASE8_POINT = ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv"
PHASE8_OBS = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"

df = pd.read_csv(KEY, low_memory=False).replace([np.inf, -np.inf], np.nan)
latent = pd.read_csv(PHASE8_POINT, low_memory=False).replace([np.inf, -np.inf], np.nan)
obs = pd.read_csv(PHASE8_OBS, low_memory=False).replace([np.inf, -np.inf], np.nan)

# -----------------------------
# Core provenance table
# -----------------------------

prov = df.copy()
prov["_row_index"] = prov.index

required = ["point_id", "lat", "lon", "latent_slope_change", "mean_vpd", "c4_fraction"]
missing = [c for c in required if c not in prov.columns]
if missing:
    raise SystemExit(f"Missing required columns in key table: {missing}")

prov["has_lat_lon"] = prov[["lat", "lon"]].notna().all(axis=1)
prov["has_latent_response"] = prov["latent_slope_change"].notna()
prov["has_mean_vpd"] = prov["mean_vpd"].notna()
prov["has_c4_fraction"] = prov["c4_fraction"].notna()
prov["has_rooting_depth"] = prov["rooting_depth"].notna() if "rooting_depth" in prov.columns else False

prov["sahel_broad"] = prov["lat"].between(10, 20) & prov["lon"].between(-20, 40)
prov["sahel_core"] = prov["lat"].between(12, 18) & prov["lon"].between(-17, 35)

prov["lat_band"] = pd.cut(
    prov["lat"],
    bins=[-90, -60, -30, 0, 15, 30, 45, 60, 90],
    labels=["-90:-60", "-60:-30", "-30:0", "0:15", "15:30", "30:45", "45:60", "60:90"],
    include_lowest=True,
)

prov["vpd_band"] = pd.cut(
    prov["mean_vpd"],
    bins=[-np.inf, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
    labels=["<=0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"],
)

prov["mat_band"] = pd.cut(
    prov["mean_annual_temperature"],
    bins=[-np.inf, -5, 0, 5, 10, 15, 20, 25, np.inf],
    labels=["<=-5", "-5:0", "0:5", "5:10", "10:15", "15:20", "20:25", ">25"],
) if "mean_annual_temperature" in prov.columns else np.nan

# -----------------------------
# Phase8 observation counts per point
# -----------------------------

obs_counts = obs.groupby("point_id", dropna=False).agg(
    n_phase8_observations=("point_id", "size"),
    n_gpp_products=("gpp_product", lambda x: x.nunique(dropna=True) if "gpp_product" in obs.columns else np.nan),
    n_et_products=("et_product", lambda x: x.nunique(dropna=True) if "et_product" in obs.columns else np.nan),
    n_product_combos=("product_combo", lambda x: x.nunique(dropna=True) if "product_combo" in obs.columns else np.nan),
    n_metrics=("metric", lambda x: x.nunique(dropna=True) if "metric" in obs.columns else np.nan),
    n_stress_defs=("stress_definition", lambda x: x.nunique(dropna=True) if "stress_definition" in obs.columns else np.nan),
    n_growing_seasons=("growing_season", lambda x: x.nunique(dropna=True) if "growing_season" in obs.columns else np.nan),
    mean_obs_vpd=("mean_vpd", "mean"),
    mean_obs_soil_moisture=("mean_soil_moisture", "mean"),
).reset_index()

prov = prov.merge(obs_counts, on="point_id", how="left", validate="one_to_one")

prov["has_phase8_observations"] = prov["n_phase8_observations"].notna()

# -----------------------------
# Duplicate / identity checks
# -----------------------------

dupe_point_ids = prov["point_id"].duplicated(keep=False)
dupe_coords = prov.duplicated(subset=["lat", "lon"], keep=False)

prov["duplicate_point_id"] = dupe_point_ids
prov["duplicate_lat_lon"] = dupe_coords

# -----------------------------
# C4 missingness classification
# -----------------------------

def classify_missing_c4(row):
    if row["has_c4_fraction"]:
        return "has_c4"
    if not row["has_lat_lon"]:
        return "missing_coordinates"
    lat = row["lat"]
    mat = row.get("mean_annual_temperature", np.nan)
    vpd = row.get("mean_vpd", np.nan)

    # Heuristic categories to diagnose whether missing C4 looks like outside-warm-C4-domain.
    if pd.notna(lat) and abs(lat) >= 45:
        return "missing_c4_high_latitude_likely_outside_c4_domain"
    if pd.notna(mat) and mat <= 5:
        return "missing_c4_cold_climate_likely_outside_c4_domain"
    if pd.notna(vpd) and vpd <= 0.5:
        return "missing_c4_low_vpd_likely_outside_c4_domain"
    return "missing_c4_possible_join_or_raster_mask_issue"

prov["c4_missingness_class"] = prov.apply(classify_missing_c4, axis=1)

# -----------------------------
# Summary tables
# -----------------------------

summary = []

def add_summary(stage, mask):
    d = prov[mask]
    summary.append({
        "stage": stage,
        "n": len(d),
        "rate": len(d) / len(prov) if len(prov) else np.nan,
        "mean_lat": d["lat"].mean(),
        "mean_vpd": d["mean_vpd"].mean(),
        "mean_mat": d["mean_annual_temperature"].mean() if "mean_annual_temperature" in d.columns else np.nan,
        "mean_latent_slope_change": d["latent_slope_change"].mean(),
        "n_sahel_broad": int(d["sahel_broad"].sum()) if len(d) else 0,
    })

add_summary("raw_key_table", pd.Series(True, index=prov.index))
add_summary("has_latent_response", prov["has_latent_response"])
add_summary("has_vpd", prov["has_mean_vpd"])
add_summary("has_c4_fraction", prov["has_c4_fraction"])
add_summary("missing_c4_fraction", ~prov["has_c4_fraction"])
add_summary("has_c4_and_response_and_vpd", prov["has_c4_fraction"] & prov["has_latent_response"] & prov["has_mean_vpd"])
add_summary("strict_plus_rooting_depth", prov["has_c4_fraction"] & prov["has_latent_response"] & prov["has_mean_vpd"] & prov["has_rooting_depth"])

pd.DataFrame(summary).to_csv(TAB / "point_provenance_stage_summary.csv", index=False)

prov["c4_missingness_class"].value_counts(dropna=False).reset_index().rename(
    columns={"index": "c4_missingness_class", "c4_missingness_class": "n"}
).to_csv(TAB / "c4_missingness_class_counts.csv", index=False)

pd.crosstab(prov["lat_band"], prov["has_c4_fraction"], dropna=False).to_csv(
    TAB / "c4_availability_by_lat_band.csv"
)

pd.crosstab(prov["vpd_band"], prov["has_c4_fraction"], dropna=False).to_csv(
    TAB / "c4_availability_by_vpd_band.csv"
)

if "mean_annual_temperature" in prov.columns:
    pd.crosstab(prov["mat_band"], prov["has_c4_fraction"], dropna=False).to_csv(
        TAB / "c4_availability_by_temperature_band.csv"
    )

# Environmental comparison table.
compare_cols = [
    "lat", "lon", "mean_vpd", "mean_annual_temperature",
    "mean_annual_precipitation", "aridity",
    "growing_season_mean_lai", "mean_soil_moisture",
    "rooting_depth", "latent_slope_change",
    "n_phase8_observations", "n_product_combos",
]

rows = []
for c in [x for x in compare_cols if x in prov.columns]:
    for flag, label in [(True, "has_c4"), (False, "missing_c4")]:
        vals = pd.to_numeric(prov.loc[prov["has_c4_fraction"].eq(flag), c], errors="coerce")
        rows.append({
            "variable": c,
            "group": label,
            "n": int(vals.notna().sum()),
            "mean": float(vals.mean()) if vals.notna().any() else np.nan,
            "median": float(vals.median()) if vals.notna().any() else np.nan,
            "sd": float(vals.std()) if vals.notna().any() else np.nan,
            "min": float(vals.min()) if vals.notna().any() else np.nan,
            "max": float(vals.max()) if vals.notna().any() else np.nan,
        })

pd.DataFrame(rows).to_csv(TAB / "has_c4_vs_missing_c4_full_comparison.csv", index=False)

# Save exact rows.
front_cols = [
    "_row_index", "point_id", "lat", "lon",
    "has_c4_fraction", "c4_missingness_class",
    "c4_fraction_raw", "c4_fraction", "c4_sample_distance_deg",
    "latent_slope_change", "mean_vpd", "mean_annual_temperature",
    "aridity", "growing_season_mean_lai", "mean_soil_moisture",
    "rooting_depth", "n_phase8_observations", "n_product_combos",
    "lat_band", "vpd_band", "mat_band",
]
front_cols = [c for c in front_cols if c in prov.columns]
prov[front_cols + [c for c in prov.columns if c not in front_cols]].to_csv(
    TAB / "FULL_POINT_PROVENANCE_TABLE.csv",
    index=False,
)

prov[~prov["has_c4_fraction"]][front_cols].to_csv(
    TAB / "EXACT_57_MISSING_C4_POINTS_CLASSIFIED.csv",
    index=False,
)

prov[prov["c4_missingness_class"].eq("missing_c4_possible_join_or_raster_mask_issue")][front_cols].to_csv(
    TAB / "POSSIBLE_TRUE_C4_JOIN_OR_MASK_FAILURE_POINTS.csv",
    index=False,
)

# -----------------------------
# Check whether missing C4 could be imputed as zero and what it would do
# -----------------------------

analysis_cols = ["latent_slope_change", "mean_vpd", "c4_fraction"]
main = prov[["latent_slope_change", "mean_vpd", "c4_fraction"]].copy()
main["c4_zero_if_missing"] = main["c4_fraction"].fillna(0.0)

import statsmodels.formula.api as smf

def z(d, c):
    s = pd.to_numeric(d[c], errors="coerce")
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return np.nan
    return (s - s.mean()) / sd

model_rows = []
coef_rows = []

for sample_name, d0 in [
    ("current_has_c4_only_n142", main.dropna(subset=["latent_slope_change", "mean_vpd", "c4_fraction"]).copy()),
    ("zero_impute_missing_c4_n199", main.dropna(subset=["latent_slope_change", "mean_vpd", "c4_zero_if_missing"]).copy()),
]:
    d = d0.copy()
    c4_col = "c4_fraction" if sample_name == "current_has_c4_only_n142" else "c4_zero_if_missing"

    d["y_z"] = z(d, "latent_slope_change")
    d["vpd_z"] = z(d, "mean_vpd")
    d["c4_z"] = z(d, c4_col)

    for model, formula in [
        ("C4_only", "y_z ~ c4_z"),
        ("VPD_only", "y_z ~ vpd_z"),
        ("C4_plus_VPD", "y_z ~ c4_z + vpd_z"),
        ("C4xVPD", "y_z ~ c4_z * vpd_z"),
    ]:
        dd = d.dropna(subset=["y_z", "vpd_z", "c4_z"])
        fit = smf.ols(formula, data=dd).fit(cov_type="HC3")
        model_rows.append({
            "sample": sample_name,
            "model": model,
            "n": int(fit.nobs),
            "r2": fit.rsquared,
            "aic": fit.aic,
            "bic": fit.bic,
            "formula": formula,
        })
        ci = fit.conf_int()
        for term in fit.params.index:
            coef_rows.append({
                "sample": sample_name,
                "model": model,
                "term": term,
                "coef": fit.params[term],
                "se_hc3": fit.bse[term],
                "p": fit.pvalues[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
            })

pd.DataFrame(model_rows).to_csv(TAB / "zero_impute_c4_sensitivity_model_fits.csv", index=False)
pd.DataFrame(coef_rows).to_csv(TAB / "zero_impute_c4_sensitivity_coefficients.csv", index=False)

# -----------------------------
# Memo
# -----------------------------

def show(path, n=40):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    return pd.read_csv(p).head(n).to_string(index=False)

memo = []
memo.append("Stage1B6AZ point provenance and C4 missingness audit")
memo.append("=" * 80)
memo.append("")
memo.append("Point provenance stage summary:")
memo.append(show("point_provenance_stage_summary.csv"))
memo.append("")
memo.append("C4 missingness classes:")
memo.append(show("c4_missingness_class_counts.csv"))
memo.append("")
memo.append("Has-C4 vs missing-C4 comparison:")
memo.append(show("has_c4_vs_missing_c4_full_comparison.csv", 60))
memo.append("")
memo.append("Zero-impute missing C4 sensitivity model fits:")
memo.append(show("zero_impute_c4_sensitivity_model_fits.csv", 40))
memo.append("")
memo.append("Zero-impute missing C4 sensitivity key coefficients:")
coefs = pd.read_csv(TAB / "zero_impute_c4_sensitivity_coefficients.csv")
key = coefs[coefs["term"].str.contains("c4|vpd|:", case=False, na=False)]
memo.append(key.to_string(index=False))
memo.append("")
memo.append("Important files:")
for f in [
    "FULL_POINT_PROVENANCE_TABLE.csv",
    "EXACT_57_MISSING_C4_POINTS_CLASSIFIED.csv",
    "POSSIBLE_TRUE_C4_JOIN_OR_MASK_FAILURE_POINTS.csv",
    "c4_availability_by_lat_band.csv",
    "c4_availability_by_vpd_band.csv",
    "c4_availability_by_temperature_band.csv",
    "zero_impute_c4_sensitivity_model_fits.csv",
    "zero_impute_c4_sensitivity_coefficients.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_point_provenance_and_c4_missingness.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_point_provenance_and_c4_missingness.txt'}")
