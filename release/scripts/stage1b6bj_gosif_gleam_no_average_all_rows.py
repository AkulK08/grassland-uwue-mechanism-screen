from pathlib import Path
import re
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bj_gosif_gleam_no_average_all_rows"
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv"
OBS = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"
LC = ROOT / "results/stage1b6be_FULL_STRICT_lai_artifact_screen/tables/POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv"

if not POINT.exists():
    raise SystemExit(f"Missing point table: {POINT}")
if not OBS.exists():
    raise SystemExit(f"Missing observation table: {OBS}")

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def first_existing(cols, *names):
    for n in names:
        if n in cols:
            return n
    low = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in low:
            return low[norm(n)]
    return None

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def formula_vars(formula):
    toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
    bad = {"C", "I", "Q", "Treatment", "reference"}
    return sorted(set(t for t in toks if t not in bad))

def fit_cluster(data, formula, focal, weight_col=None):
    vars_needed = formula_vars(formula)
    extra = ["point_id"]
    if weight_col:
        extra.append(weight_col)

    missing = [v for v in vars_needed if v not in data.columns]
    if missing:
        return {
            "status": f"MISSING_VARS: {missing}",
            "n_rows": 0,
            "n_points": 0,
            "coef": np.nan,
            "p_cluster": np.nan,
        }

    use_cols = sorted(set(vars_needed + extra))
    use = data[use_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(use) < max(50, len(vars_needed) + 10) or use["point_id"].nunique() < 20:
        return {
            "status": "N_TOO_SMALL",
            "n_rows": len(use),
            "n_points": use["point_id"].nunique() if "point_id" in use.columns else 0,
            "coef": np.nan,
            "p_cluster": np.nan,
        }

    try:
        if weight_col:
            model = smf.wls(formula, data=use, weights=use[weight_col])
        else:
            model = smf.ols(formula, data=use)

        fit = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": use["point_id"], "use_correction": True},
        )

        ci = fit.conf_int()
        return {
            "status": "FIT_OK",
            "n_rows": int(fit.nobs),
            "n_points": int(use["point_id"].nunique()),
            "coef": float(fit.params.get(focal, np.nan)),
            "se_cluster": float(fit.bse.get(focal, np.nan)),
            "p_cluster": float(fit.pvalues.get(focal, np.nan)),
            "ci_low": float(ci.loc[focal, 0]) if focal in ci.index else np.nan,
            "ci_high": float(ci.loc[focal, 1]) if focal in ci.index else np.nan,
            "r2": float(fit.rsquared),
            "aic": float(fit.aic),
            "n_params": int(len(fit.params)),
        }

    except Exception as e:
        return {
            "status": f"FIT_FAIL: {e}",
            "n_rows": len(use),
            "n_points": int(use["point_id"].nunique()) if "point_id" in use.columns else 0,
            "coef": np.nan,
            "p_cluster": np.nan,
        }

# -------------------------
# Load point-level predictors
# -------------------------

raw = pd.read_csv(POINT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()
cols = list(raw.columns)

src = {
    "point_id": first_existing(cols, "point_id"),
    "latent_y": first_existing(cols, "latent_slope_change"),
    "lai": first_existing(cols, "growing_season_mean_lai", "mean_lai"),
    "vpd": first_existing(cols, "mean_vpd", "mean_obs_vpd"),
    "mat": first_existing(cols, "mean_annual_temperature", "mean_temperature"),
    "map": first_existing(cols, "mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing(cols, "aridity"),
    "sm": first_existing(cols, "mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing(cols, "lat", "latitude"),
    "lon": first_existing(cols, "lon", "longitude"),
    "c4": first_existing(cols, "c4_fraction", "c4_fraction_raw"),
    "sand": first_existing(cols, "soil_sand", "sand", "sand_fraction"),
    "clay": first_existing(cols, "soil_clay", "clay", "clay_fraction"),
    "silt": first_existing(cols, "soil_silt", "silt", "silt_fraction"),
}

missing = [k for k in ["point_id", "lai", "vpd", "mat", "map", "arid", "sm", "lat", "lon"] if src[k] is None]
if missing:
    raise SystemExit(f"Missing point variables: {missing}")

pt = pd.DataFrame()
for k, c in src.items():
    if c is None:
        continue
    if k == "point_id":
        pt[k] = raw[c].astype(str)
    else:
        pt[k] = pd.to_numeric(raw[c], errors="coerce")

# soil texture PC1
if all(c in pt.columns for c in ["sand", "clay", "silt"]):
    soil = pt[["sand", "clay", "silt"]].apply(zscore)
    use = soil.dropna()
    pc1 = pd.Series(np.nan, index=pt.index)
    if len(use) >= 40:
        X = use.values
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        scores = X @ vt[0, :]
        pc1.loc[use.index] = scores
        if pc1.corr(pt["clay"]) < 0:
            pc1 = -pc1
        pt["soil_texture_pc1"] = pc1

# landcover flags
if LC.exists():
    lc = pd.read_csv(LC, low_memory=False)
    lc = lc.loc[:, ~lc.columns.duplicated()].copy()
    if "point_id" in lc.columns:
        keep = ["point_id"]
        for c in [
            "any_cropland_managed_irrigation_flag",
            "any_natural_grassland_indicator",
            "n_cropland_managed_irrigation_flags",
            "n_natural_grassland_indicators",
        ]:
            if c in lc.columns:
                keep.append(c)
        lc = lc[keep].copy()
        lc["point_id"] = lc["point_id"].astype(str)
        lc = lc.groupby("point_id", dropna=False).first().reset_index()
        pt = pt.merge(lc, on="point_id", how="left")

if "any_cropland_managed_irrigation_flag" not in pt.columns:
    pt["any_cropland_managed_irrigation_flag"] = False
if "any_natural_grassland_indicator" not in pt.columns:
    pt["any_natural_grassland_indicator"] = True

pt["any_cropland_managed_irrigation_flag"] = pt["any_cropland_managed_irrigation_flag"].fillna(False).astype(bool)
pt["any_natural_grassland_indicator"] = pt["any_natural_grassland_indicator"].fillna(False).astype(bool)
pt["sahel_broad"] = pt["lat"].between(10, 20) & pt["lon"].between(-20, 40)
pt["abs_lat"] = pt["lat"].abs()
pt["has_c4"] = pt["c4"].notna() if "c4" in pt.columns else False

for c in list(pt.columns):
    if c in [
        "point_id",
        "any_cropland_managed_irrigation_flag",
        "any_natural_grassland_indicator",
        "sahel_broad",
        "has_c4",
    ]:
        continue
    if pd.api.types.is_numeric_dtype(pt[c]):
        pt[c + "_z"] = zscore(pt[c])

# -------------------------
# Load observation rows: exact GOSIF x GLEAM only, no hidden-definition filtering
# -------------------------

obs = pd.read_csv(OBS, low_memory=False).replace([np.inf, -np.inf], np.nan)
obs = obs.loc[:, ~obs.columns.duplicated()].copy()
oc = list(obs.columns)

point_col = first_existing(oc, "point_id")
gpp_col = first_existing(oc, "gpp_product", "gpp")
et_col = first_existing(oc, "et_product", "et")
slope_col = first_existing(oc, "slope_change", "latent_slope_change")

if point_col is None or gpp_col is None or et_col is None or slope_col is None:
    raise SystemExit(f"Missing observation columns: point={point_col}, gpp={gpp_col}, et={et_col}, slope={slope_col}")

obs[point_col] = obs[point_col].astype(str)
obs[gpp_col] = obs[gpp_col].astype(str)
obs[et_col] = obs[et_col].astype(str)
obs[slope_col] = pd.to_numeric(obs[slope_col], errors="coerce")

combo_counts = obs.groupby([gpp_col, et_col], dropna=False).size().reset_index(name="n_rows")
combo_counts.to_csv(TAB / "PRODUCT_COMBO_ROW_COUNTS.csv", index=False)

gg = obs[
    obs[gpp_col].str.lower().str.fullmatch("gosif", na=False)
    & obs[et_col].str.lower().str.fullmatch("gleam", na=False)
].copy()

# fallback if product strings are not exact
if len(gg) == 0:
    gg = obs[
        obs[gpp_col].str.lower().str.contains("gosif|sif", regex=True, na=False)
        & obs[et_col].str.lower().str.contains("gleam", regex=True, na=False)
    ].copy()

if len(gg) == 0:
    raise SystemExit("No GOSIF x GLEAM rows found.")

gg = gg.rename(columns={point_col: "point_id", slope_col: "y_raw"})
gg["point_id"] = gg["point_id"].astype(str)

# Keep every hidden definition row. Do not filter on these columns.
possible_definition_cols = [
    "metric",
    "growing_season",
    "stress_definition",
    "co2_version",
    "primary_metric",
    "metric_et_interaction",
    "stress_et_interaction",
    "accepted_transition",
    "response_class_4way",
    "response_class_original",
    "satbreak_indicator",
]
definition_cols = [c for c in possible_definition_cols if c in gg.columns]

# Convert categorical definition columns to strings for Patsy.
for c in definition_cols:
    gg[c] = gg[c].astype(str).fillna("NA")

# Merge point-level predictors onto every row.
dat = gg.merge(pt, on="point_id", how="left", suffixes=("", "_point"))

# Row-level outcome z-score. This is NOT point averaging.
dat["y_z"] = zscore(dat["y_raw"])

# Equal-point weights: each point contributes total weight ~1.
row_counts = dat.groupby("point_id").size().rename("rows_per_point").reset_index()
dat = dat.merge(row_counts, on="point_id", how="left")
dat["w_equal_point"] = 1.0 / dat["rows_per_point"]

# Save raw row-level table head and counts.
dat.head(200).to_csv(TAB / "ROW_LEVEL_GOSIF_GLEAM_HEAD_200_ROWS.csv", index=False)
row_counts.to_csv(TAB / "ROWS_PER_POINT.csv", index=False)

# Hidden definition counts, no exclusion.
hidden_count_rows = []
for c in definition_cols:
    vc = dat[c].value_counts(dropna=False)
    for val, n in vc.items():
        hidden_count_rows.append({"column": c, "value": val, "n_rows": int(n)})
pd.DataFrame(hidden_count_rows).to_csv(TAB / "HIDDEN_DEFINITION_COUNTS_INCLUDED_NOT_FILTERED.csv", index=False)

# -------------------------
# Models
# -------------------------

full_controls = [
    "vpd_z",
    "arid_z",
    "mat_z",
    "map_z",
    "sm_z",
    "soil_texture_pc1_z",
    "lat_z",
    "lon_z",
]
full_controls = [c for c in full_controls if c in dat.columns and dat[c].notna().sum() >= 100]

core_controls = [
    "vpd_z",
    "arid_z",
    "mat_z",
    "map_z",
    "sm_z",
]
core_controls = [c for c in core_controls if c in dat.columns and dat[c].notna().sum() >= 100]

controls_no_mat = [c for c in full_controls if c != "mat_z"]

# Definition fixed effects: these are controls, NOT exclusions.
# Do not include response_class or accepted_transition in the main FE because those are outcome-derived/post-hoc labels.
safe_fe_cols = [
    c for c in ["metric", "growing_season", "stress_definition", "co2_version", "primary_metric"]
    if c in dat.columns and dat[c].nunique(dropna=True) > 1
]
safe_fe = " + ".join([f"C({c})" for c in safe_fe_cols])

# Broader FE including interaction labels but still no exclusion.
broad_fe_cols = [
    c for c in [
        "metric",
        "growing_season",
        "stress_definition",
        "co2_version",
        "primary_metric",
        "metric_et_interaction",
        "stress_et_interaction",
        "satbreak_indicator",
    ]
    if c in dat.columns and dat[c].nunique(dropna=True) > 1
]
broad_fe = " + ".join([f"C({c})" for c in broad_fe_cols])

samples = {
    "ALL_ROWS_NO_EXCLUSION": pd.Series(True, index=dat.index),
    "CROPLAND_CLEAN_ROWS_NO_METRIC_EXCLUSION": ~dat["any_cropland_managed_irrigation_flag"],
    "NATURAL_GRASSLAND_INDICATOR_ROWS_NO_METRIC_EXCLUSION": dat["any_natural_grassland_indicator"],
    "NO_BROAD_SAHEL_ROWS_NO_METRIC_EXCLUSION": ~dat["sahel_broad"],
    "CROPLAND_CLEAN_NO_BROAD_SAHEL_ROWS_NO_METRIC_EXCLUSION": (~dat["any_cropland_managed_irrigation_flag"]) & (~dat["sahel_broad"]),
}

model_specs = []

model_specs.append({
    "model": "M0_bivariate_no_definition_FE",
    "formula": "y_z ~ lai_z",
    "focal": "lai_z",
})

model_specs.append({
    "model": "M1_VPD_only_no_definition_FE",
    "formula": "y_z ~ lai_z + vpd_z",
    "focal": "lai_z",
})

if core_controls:
    model_specs.append({
        "model": "M2_core_controls_no_definition_FE",
        "formula": "y_z ~ lai_z + " + " + ".join(core_controls),
        "focal": "lai_z",
    })

if full_controls:
    model_specs.append({
        "model": "M3_FULL_STRICT_controls_no_definition_FE",
        "formula": "y_z ~ lai_z + " + " + ".join(full_controls),
        "focal": "lai_z",
    })

if full_controls and safe_fe:
    model_specs.append({
        "model": "M4_FULL_STRICT_controls_plus_safe_definition_FE",
        "formula": "y_z ~ lai_z + " + " + ".join(full_controls) + " + " + safe_fe,
        "focal": "lai_z",
    })

if full_controls and broad_fe:
    model_specs.append({
        "model": "M5_FULL_STRICT_controls_plus_broad_definition_FE",
        "formula": "y_z ~ lai_z + " + " + ".join(full_controls) + " + " + broad_fe,
        "focal": "lai_z",
    })

if controls_no_mat:
    model_specs.append({
        "model": "I1_LAI_x_MAT_FULL_STRICT_no_definition_FE",
        "formula": "y_z ~ lai_z * mat_z + " + " + ".join(controls_no_mat),
        "focal": "lai_z:mat_z",
    })

if controls_no_mat and safe_fe:
    model_specs.append({
        "model": "I2_LAI_x_MAT_FULL_STRICT_plus_safe_definition_FE",
        "formula": "y_z ~ lai_z * mat_z + " + " + ".join(controls_no_mat) + " + " + safe_fe,
        "focal": "lai_z:mat_z",
    })

if controls_no_mat and broad_fe:
    model_specs.append({
        "model": "I3_LAI_x_MAT_FULL_STRICT_plus_broad_definition_FE",
        "formula": "y_z ~ lai_z * mat_z + " + " + ".join(controls_no_mat) + " + " + broad_fe,
        "focal": "lai_z:mat_z",
    })

rows = []

for sample_name, mask in samples.items():
    sub = dat.loc[mask].copy()

    for spec in model_specs:
        for weighting, weight_col in [
            ("unweighted_rows_clustered_by_point", None),
            ("equal_point_weighted_rows_clustered_by_point", "w_equal_point"),
        ]:
            r = fit_cluster(sub, spec["formula"], spec["focal"], weight_col=weight_col)
            r.update({
                "sample": sample_name,
                "model": spec["model"],
                "focal": spec["focal"],
                "weighting": weighting,
                "formula": spec["formula"],
                "n_hidden_definition_columns_included": len(definition_cols),
                "safe_definition_FE_cols": ",".join(safe_fe_cols),
                "broad_definition_FE_cols": ",".join(broad_fe_cols),
            })
            rows.append(r)

res = pd.DataFrame(rows)

# Add BH q within each sample x weighting across all model rows as a descriptive check.
res["p_for_q"] = pd.to_numeric(res["p_cluster"], errors="coerce")
res["bh_q_within_sample_weighting"] = np.nan
for (sample, weighting), idx in res.groupby(["sample", "weighting"]).groups.items():
    pvals = res.loc[idx, "p_for_q"].astype(float)
    valid = pvals.notna()
    if valid.sum() > 0:
        order = np.argsort(pvals[valid].values)
        pv = pvals[valid].values[order]
        m = len(pv)
        q = np.empty(m)
        prev = 1.0
        for i in range(m - 1, -1, -1):
            prev = min(prev, pv[i] * m / (i + 1))
            q[i] = prev
        out = np.empty(m)
        out[order] = q
        res.loc[pvals[valid].index, "bh_q_within_sample_weighting"] = out

res.to_csv(TAB / "ROW_LEVEL_NO_AVERAGE_ALL_GOSIF_GLEAM_MODELS.csv", index=False)

# Compact main table.
main_keep = res[
    res["sample"].isin([
        "ALL_ROWS_NO_EXCLUSION",
        "CROPLAND_CLEAN_ROWS_NO_METRIC_EXCLUSION",
        "NO_BROAD_SAHEL_ROWS_NO_METRIC_EXCLUSION",
        "CROPLAND_CLEAN_NO_BROAD_SAHEL_ROWS_NO_METRIC_EXCLUSION",
    ])
    & res["model"].isin([
        "M3_FULL_STRICT_controls_no_definition_FE",
        "M4_FULL_STRICT_controls_plus_safe_definition_FE",
        "M5_FULL_STRICT_controls_plus_broad_definition_FE",
        "I1_LAI_x_MAT_FULL_STRICT_no_definition_FE",
        "I2_LAI_x_MAT_FULL_STRICT_plus_safe_definition_FE",
        "I3_LAI_x_MAT_FULL_STRICT_plus_broad_definition_FE",
    ])
].copy()
main_keep.to_csv(TAB / "MAIN_ROW_LEVEL_NO_AVERAGE_RESULTS.csv", index=False)

# -------------------------
# Sanity checks
# -------------------------

sanity = {
    "point_input": str(POINT),
    "obs_input": str(OBS),
    "n_gosif_gleam_rows_used_no_averaging": int(len(dat)),
    "n_unique_points": int(dat["point_id"].nunique()),
    "rows_per_point_min": int(dat["rows_per_point"].min()),
    "rows_per_point_median": float(dat["rows_per_point"].median()),
    "rows_per_point_max": int(dat["rows_per_point"].max()),
    "definition_columns_included_not_filtered": definition_cols,
    "safe_definition_FE_cols": safe_fe_cols,
    "broad_definition_FE_cols": broad_fe_cols,
    "full_controls": full_controls,
    "core_controls": core_controls,
    "note": "All GOSIF x GLEAM rows are used. No averaging to point level. No filtering by metric, stress definition, growing season, CO2 version, primary_metric, accepted_transition, or response class. SE are clustered by point_id.",
}
(TAB / "RUN_SANITY.json").write_text(json.dumps(sanity, indent=2))

# -------------------------
# Memo
# -------------------------

def show_csv(path, n=80, cols=None):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if cols:
        cols = [c for c in cols if c in x.columns]
        x = x[cols]
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BJ GOSIF x GLEAM no-average all-row test")
memo.append("=" * 95)
memo.append("")
memo.append("Design:")
memo.append("- Exact GOSIF x GLEAM rows only.")
memo.append("- No point-level averaging.")
memo.append("- No filtering by metric, stress_definition, growing_season, co2_version, primary_metric, accepted_transition, response_class, or satbreak labels.")
memo.append("- Repeated rows are handled with cluster-robust SE by point_id.")
memo.append("- Equal-point-weighted versions are also run so each point contributes equal total weight.")
memo.append("- Definition fixed effects are controls, not exclusions.")
memo.append("")
memo.append("Sanity:")
memo.append(json.dumps(sanity, indent=2))
memo.append("")
memo.append("Product combo row counts:")
memo.append(show_csv("PRODUCT_COMBO_ROW_COUNTS.csv", 30))
memo.append("")
memo.append("Definition counts included, not filtered:")
memo.append(show_csv("HIDDEN_DEFINITION_COUNTS_INCLUDED_NOT_FILTERED.csv", 80))
memo.append("")
memo.append("Rows per point summary:")
rp = pd.read_csv(TAB / "ROWS_PER_POINT.csv")
memo.append(rp["rows_per_point"].describe().to_string())
memo.append("")
memo.append("MAIN ROW-LEVEL NO-AVERAGE RESULTS:")
cols = [
    "sample",
    "model",
    "weighting",
    "status",
    "n_rows",
    "n_points",
    "focal",
    "coef",
    "se_cluster",
    "p_cluster",
    "bh_q_within_sample_weighting",
    "ci_low",
    "ci_high",
    "r2",
    "aic",
]
memo.append(show_csv("MAIN_ROW_LEVEL_NO_AVERAGE_RESULTS.csv", 200, cols=cols))
memo.append("")
memo.append("All model results file:")
memo.append(str(TAB / "ROW_LEVEL_NO_AVERAGE_ALL_GOSIF_GLEAM_MODELS.csv"))
memo.append("")
memo.append("Important files:")
for f in [
    "RUN_SANITY.json",
    "PRODUCT_COMBO_ROW_COUNTS.csv",
    "HIDDEN_DEFINITION_COUNTS_INCLUDED_NOT_FILTERED.csv",
    "ROWS_PER_POINT.csv",
    "ROW_LEVEL_GOSIF_GLEAM_HEAD_200_ROWS.csv",
    "MAIN_ROW_LEVEL_NO_AVERAGE_RESULTS.csv",
    "ROW_LEVEL_NO_AVERAGE_ALL_GOSIF_GLEAM_MODELS.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_gosif_gleam_no_average_all_rows.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_gosif_gleam_no_average_all_rows.txt'}")
