from pathlib import Path
import re
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bh_gosif_gleam_deep_test"
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"
for p in [TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT_CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]

POINT_INPUT = next((p for p in POINT_CANDIDATES if p.exists()), None)
OBS_INPUT = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"
LC_FLAGS_INPUT = ROOT / "results/stage1b6be_FULL_STRICT_lai_artifact_screen/tables/POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv"

if POINT_INPUT is None:
    raise SystemExit("No point-level input table found.")
if not OBS_INPUT.exists():
    raise SystemExit(f"Missing observation table: {OBS_INPUT}")

# -----------------------------
# Utility functions
# -----------------------------

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def first_existing(cols, *names):
    cols = list(cols)
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
    return sorted(set(t for t in toks if t not in {"C", "I", "Q"}))

def fit_model(data, formula):
    vars_needed = formula_vars(formula)
    missing = [v for v in vars_needed if v not in data.columns]
    if missing:
        return None, pd.DataFrame(), f"MISSING_VARS: {missing}"

    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(25, len(vars_needed) + 8):
        return None, use, "N_TOO_SMALL"

    try:
        fit = smf.ols(formula, data=use).fit(cov_type="HC3")
        return fit, use, "FIT_OK"
    except Exception as e:
        return None, use, f"FIT_FAIL: {e}"

def compare_model(data, test_label, outcome, full_formula, reduced_formula, focal_term, sample_note=""):
    full_fit, use, status = fit_model(data, full_formula)
    red_fit, _, red_status = fit_model(data, reduced_formula)

    row = {
        "test_label": test_label,
        "outcome": outcome,
        "status": status,
        "reduced_status": red_status,
        "focal_term": focal_term,
        "n": len(use),
        "sample_note": sample_note,
        "full_formula": full_formula,
        "reduced_formula": reduced_formula,
    }

    if full_fit is None or red_fit is None:
        return row

    try:
        full_nr = smf.ols(full_formula, data=use).fit()
        red_same = smf.ols(reduced_formula, data=use).fit()
        nested_p = float(full_nr.compare_f_test(red_same)[1])
    except Exception:
        nested_p = np.nan

    ci = full_fit.conf_int()

    row.update({
        "coef": full_fit.params.get(focal_term, np.nan),
        "se_hc3": full_fit.bse.get(focal_term, np.nan),
        "p": full_fit.pvalues.get(focal_term, np.nan),
        "ci_low": ci.loc[focal_term, 0] if focal_term in ci.index else np.nan,
        "ci_high": ci.loc[focal_term, 1] if focal_term in ci.index else np.nan,
        "ci_excludes_zero": bool(ci.loc[focal_term, 0] * ci.loc[focal_term, 1] > 0) if focal_term in ci.index else False,
        "full_r2": full_fit.rsquared,
        "reduced_r2": red_fit.rsquared,
        "delta_r2": full_fit.rsquared - red_fit.rsquared,
        "full_aic": full_fit.aic,
        "reduced_aic": red_fit.aic,
        "delta_aic_full_minus_reduced": full_fit.aic - red_fit.aic,
        "nested_f_p": nested_p,
    })
    return row

def bootstrap_coef(data, formula, focal_term, n_boot=1000, seed=123):
    vars_needed = formula_vars(formula)
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(25, len(vars_needed) + 8):
        return {"boot_n": 0, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_sign_stability": np.nan}

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(use), len(use))
        boot = use.iloc[idx].copy()
        try:
            fit = smf.ols(formula, data=boot).fit()
            vals.append(fit.params.get(focal_term, np.nan))
        except Exception:
            vals.append(np.nan)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"boot_n": 0, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_sign_stability": np.nan}

    med_sign = np.sign(np.nanmedian(vals))
    sign_stab = float(np.mean(np.sign(vals) == med_sign)) if med_sign != 0 else np.nan

    return {
        "boot_n": int(len(vals)),
        "boot_ci_low": float(np.nanpercentile(vals, 2.5)),
        "boot_ci_high": float(np.nanpercentile(vals, 97.5)),
        "boot_sign_stability": sign_stab,
    }

def leave_region_out(data, formula, focal_term):
    vars_needed = formula_vars(formula) + ["region_block"]
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 40 or use["region_block"].nunique() < 3:
        return pd.DataFrame(), {"lro_n_regions": 0, "lro_sign_stability": np.nan, "lro_min_coef": np.nan, "lro_max_coef": np.nan}

    full, _, status = fit_model(use, formula)
    if full is None:
        return pd.DataFrame(), {"lro_n_regions": 0, "lro_sign_stability": np.nan, "lro_min_coef": np.nan, "lro_max_coef": np.nan}

    base_sign = np.sign(full.params.get(focal_term, np.nan))
    rows = []
    for reg in sorted(use["region_block"].dropna().unique()):
        sub = use[use["region_block"] != reg].copy()
        fit, subuse, st = fit_model(sub, formula)
        coef = np.nan if fit is None else fit.params.get(focal_term, np.nan)
        p = np.nan if fit is None else fit.pvalues.get(focal_term, np.nan)
        rows.append({
            "left_out_region": reg,
            "status": st,
            "n": len(subuse),
            "coef": coef,
            "p": p,
            "same_sign_as_full": bool(np.sign(coef) == base_sign) if np.isfinite(coef) and np.isfinite(base_sign) else False,
        })

    lro = pd.DataFrame(rows)
    ok = lro[np.isfinite(lro["coef"])]
    summary = {
        "lro_n_regions": int(len(ok)),
        "lro_sign_stability": float(ok["same_sign_as_full"].mean()) if len(ok) else np.nan,
        "lro_min_coef": float(ok["coef"].min()) if len(ok) else np.nan,
        "lro_max_coef": float(ok["coef"].max()) if len(ok) else np.nan,
    }
    return lro, summary

def corr_pair(x, y):
    z = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(z) < 4:
        return len(z), np.nan, np.nan, np.nan, np.nan
    pear = stats.pearsonr(z["x"], z["y"])
    spear = stats.spearmanr(z["x"], z["y"])
    return len(z), float(pear.statistic), float(pear.pvalue), float(spear.correlation), float(spear.pvalue)

# -----------------------------
# Read point-level table
# -----------------------------

raw = pd.read_csv(POINT_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()
cols = list(raw.columns)

src = {
    "point_id": first_existing(cols, "point_id"),
    "latent_y": first_existing(cols, "latent_slope_change"),
    "vpd": first_existing(cols, "mean_vpd", "mean_obs_vpd"),
    "lai": first_existing(cols, "growing_season_mean_lai", "mean_lai"),
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

missing = [k for k in ["point_id", "latent_y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"] if src[k] is None]
if missing:
    raise SystemExit(f"Missing canonical variables in point table: {missing}")

d = pd.DataFrame(index=raw.index)
for k, c in src.items():
    if c is None:
        continue
    if k == "point_id":
        d[k] = raw[c].astype(str)
    else:
        d[k] = pd.to_numeric(raw[c], errors="coerce")

# Soil texture PC1.
if all(c in d.columns for c in ["sand", "clay", "silt"]):
    soil_std = d[["sand", "clay", "silt"]].apply(zscore)
    use_soil = soil_std.dropna()
    pc1 = pd.Series(np.nan, index=d.index)
    if len(use_soil) >= 40:
        X = use_soil.values
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        scores = X @ vt[0, :]
        pc1.loc[use_soil.index] = scores
        if pc1.corr(d["clay"]) < 0:
            pc1 = -pc1
        d["soil_texture_pc1"] = pc1

# Land-cover flags.
if LC_FLAGS_INPUT.exists():
    lc = pd.read_csv(LC_FLAGS_INPUT, low_memory=False)
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
        d = d.merge(lc, on="point_id", how="left")

if "any_cropland_managed_irrigation_flag" not in d.columns:
    d["any_cropland_managed_irrigation_flag"] = False
if "any_natural_grassland_indicator" not in d.columns:
    d["any_natural_grassland_indicator"] = True

d["any_cropland_managed_irrigation_flag"] = d["any_cropland_managed_irrigation_flag"].fillna(False).astype(bool)
d["any_natural_grassland_indicator"] = d["any_natural_grassland_indicator"].fillna(False).astype(bool)

d["abs_lat"] = d["lat"].abs()
d["sahel_broad"] = d["lat"].between(10, 20) & d["lon"].between(-20, 40)
d["sahel_core"] = d["lat"].between(12, 18) & d["lon"].between(-17, 35)
d["has_c4"] = d["c4"].notna() if "c4" in d.columns else False
d["region_block"] = (
    pd.cut(d["lat"], [-90, -30, 0, 30, 60, 90], labels=False).astype(str)
    + "_"
    + pd.cut(d["lon"], [-180, -90, 0, 90, 180], labels=False).astype(str)
)

# Standardized columns.
for c in list(d.columns):
    if c in [
        "point_id",
        "region_block",
        "sahel_broad",
        "sahel_core",
        "has_c4",
        "any_cropland_managed_irrigation_flag",
        "any_natural_grassland_indicator",
    ]:
        continue
    if pd.api.types.is_numeric_dtype(d[c]):
        d[c + "_z"] = zscore(d[c])

# -----------------------------
# Read exact GOSIF × GLEAM outcome
# -----------------------------

obs = pd.read_csv(OBS_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
obs = obs.loc[:, ~obs.columns.duplicated()].copy()
obs_cols = list(obs.columns)

point_col = first_existing(obs_cols, "point_id")
gpp_col = first_existing(obs_cols, "gpp_product", "gpp")
et_col = first_existing(obs_cols, "et_product", "et")
slope_col = first_existing(obs_cols, "slope_change", "latent_slope_change")

if point_col is None or gpp_col is None or et_col is None or slope_col is None:
    raise SystemExit(f"Missing observation columns. point={point_col}, gpp={gpp_col}, et={et_col}, slope={slope_col}")

obs[point_col] = obs[point_col].astype(str)
obs[gpp_col] = obs[gpp_col].astype(str)
obs[et_col] = obs[et_col].astype(str)
obs[slope_col] = pd.to_numeric(obs[slope_col], errors="coerce")

# Print available product combos.
combo_counts = (
    obs.groupby([gpp_col, et_col], dropna=False)
       .size()
       .reset_index(name="n_obs_rows")
       .sort_values([gpp_col, et_col])
)
combo_counts.to_csv(TAB / "AVAILABLE_PRODUCT_COMBOS.csv", index=False)

gg = obs[
    obs[gpp_col].str.lower().str.contains("gosif|sif", regex=True, na=False)
    & obs[et_col].str.lower().str.contains("gleam", regex=True, na=False)
].copy()

if len(gg) == 0:
    raise SystemExit("No exact GOSIF/SIF × GLEAM rows found in observation table.")

gg_point = (
    gg.dropna(subset=[point_col, slope_col])
      .groupby(point_col, dropna=False)[slope_col]
      .mean()
      .reset_index()
      .rename(columns={point_col: "point_id", slope_col: "gosif_gleam_y"})
)

d = d.merge(gg_point, on="point_id", how="left")
d["gosif_gleam_y_z"] = zscore(d["gosif_gleam_y"])
d["latent_y_z"] = zscore(d["latent_y"])

# Common complete case across all product combos.
combo_wide = None
common_product_complete_ids = set()
try:
    tmp = (
        obs.dropna(subset=[point_col, gpp_col, et_col, slope_col])
           .groupby([point_col, gpp_col, et_col], dropna=False)[slope_col]
           .mean()
           .reset_index()
    )
    tmp["combo"] = tmp[gpp_col].astype(str) + "__X__" + tmp[et_col].astype(str)
    combo_wide = tmp.pivot(index=point_col, columns="combo", values=slope_col)
    combo_wide["all_product_combos_complete"] = combo_wide.notna().all(axis=1)
    common_product_complete_ids = set(combo_wide[combo_wide["all_product_combos_complete"]].index.astype(str))
except Exception:
    common_product_complete_ids = set()

d["all_product_combos_complete"] = d["point_id"].astype(str).isin(common_product_complete_ids)

# -----------------------------
# Fixed controls and masks
# -----------------------------

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
full_controls = [c for c in full_controls if c in d.columns and d[c].notna().sum() >= 40]

core_controls = [
    "vpd_z",
    "arid_z",
    "mat_z",
    "map_z",
    "sm_z",
]
core_controls = [c for c in core_controls if c in d.columns and d[c].notna().sum() >= 40]

full_controls_no_mat = [c for c in full_controls if c != "mat_z"]

masks = {
    "all_available": pd.Series(True, index=d.index),
    "cropland_clean": ~d["any_cropland_managed_irrigation_flag"],
    "natural_grassland_indicator": d["any_natural_grassland_indicator"],
    "exclude_broad_sahel": ~d["sahel_broad"],
    "cropland_clean_no_broad_sahel": (~d["any_cropland_managed_irrigation_flag"]) & (~d["sahel_broad"]),
    "warm_mat_gt_0C": d["mat"] > 0,
    "warm_mat_gt_2p08C": d["mat"] > 2.08,
    "abs_lat_le_48": d["abs_lat"] <= 48,
    "c4_covered_domain": d["has_c4"],
    "common_complete_all_product_combos": d["all_product_combos_complete"],
    "cropland_clean_common_complete_all_product_combos": (~d["any_cropland_managed_irrigation_flag"]) & d["all_product_combos_complete"],
}

# -----------------------------
# Models
# -----------------------------

outcome = "gosif_gleam_y_z"

model_specs = []

model_specs.append({
    "model": "bivariate_LAI_only",
    "full": f"{outcome} ~ lai_z",
    "reduced": f"{outcome} ~ 1",
    "focal": "lai_z",
})

model_specs.append({
    "model": "VPD_only_control",
    "full": f"{outcome} ~ lai_z + vpd_z",
    "reduced": f"{outcome} ~ vpd_z",
    "focal": "lai_z",
})

if len(core_controls):
    model_specs.append({
        "model": "core_hydroclimate_controls",
        "full": f"{outcome} ~ lai_z + " + " + ".join(core_controls),
        "reduced": f"{outcome} ~ " + " + ".join(core_controls),
        "focal": "lai_z",
    })

if len(full_controls):
    model_specs.append({
        "model": "FULL_STRICT_controls",
        "full": f"{outcome} ~ lai_z + " + " + ".join(full_controls),
        "reduced": f"{outcome} ~ " + " + ".join(full_controls),
        "focal": "lai_z",
    })

if "mat_z" in d.columns and len(full_controls_no_mat):
    model_specs.append({
        "model": "LAI_x_MAT_FULL_STRICT_controls",
        "full": f"{outcome} ~ lai_z * mat_z + " + " + ".join(full_controls_no_mat),
        "reduced": f"{outcome} ~ lai_z + mat_z + " + " + ".join(full_controls_no_mat),
        "focal": "lai_z:mat_z",
    })

rows = []
boot_rows = []
lro_all = []

for mask_name, mask in masks.items():
    sub = d.loc[mask].copy()
    sample_note = mask_name

    for spec in model_specs:
        row = compare_model(
            sub,
            f"{mask_name}__{spec['model']}",
            outcome,
            spec["full"],
            spec["reduced"],
            spec["focal"],
            sample_note=sample_note,
        )
        row["mask"] = mask_name
        row["model"] = spec["model"]
        rows.append(row)

        if spec["model"] == "FULL_STRICT_controls":
            boot = bootstrap_coef(sub, spec["full"], spec["focal"], n_boot=1000, seed=123)
            boot["mask"] = mask_name
            boot["model"] = spec["model"]
            boot_rows.append(boot)

            lro, lro_summary = leave_region_out(sub, spec["full"], spec["focal"])
            for k, v in lro_summary.items():
                row[k] = v
            if len(lro):
                lro["mask"] = mask_name
                lro["model"] = spec["model"]
                lro_all.append(lro)

results = pd.DataFrame(rows)
boot_df = pd.DataFrame(boot_rows)

if len(boot_df):
    results = results.merge(boot_df, on=["mask", "model"], how="left")

results.to_csv(TAB / "GOSIF_GLEAM_FIXED_MASK_MODEL_TESTS.csv", index=False)

if len(lro_all):
    pd.concat(lro_all, ignore_index=True).to_csv(TAB / "GOSIF_GLEAM_LEAVE_REGION_OUT.csv", index=False)
else:
    pd.DataFrame([{"status": "NO_LRO_RESULTS"}]).to_csv(TAB / "GOSIF_GLEAM_LEAVE_REGION_OUT.csv", index=False)

# -----------------------------
# Correlation diagnostics
# -----------------------------

corr_rows = []
for mask_name, mask in masks.items():
    sub = d.loc[mask].copy()

    n, pear_r, pear_p, spear_r, spear_p = corr_pair(sub["gosif_gleam_y_z"], sub["latent_y_z"])
    corr_rows.append({
        "mask": mask_name,
        "x": "gosif_gleam_y_z",
        "y": "latent_y_z",
        "n": n,
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_r": spear_r,
        "spearman_p": spear_p,
    })

    n, pear_r, pear_p, spear_r, spear_p = corr_pair(sub["lai_z"], sub["gosif_gleam_y_z"])
    corr_rows.append({
        "mask": mask_name,
        "x": "lai_z",
        "y": "gosif_gleam_y_z",
        "n": n,
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_r": spear_r,
        "spearman_p": spear_p,
    })

    n, pear_r, pear_p, spear_r, spear_p = corr_pair(sub["lai_z"], sub["latent_y_z"])
    corr_rows.append({
        "mask": mask_name,
        "x": "lai_z",
        "y": "latent_y_z",
        "n": n,
        "pearson_r": pear_r,
        "pearson_p": pear_p,
        "spearman_r": spear_r,
        "spearman_p": spear_p,
    })

corr = pd.DataFrame(corr_rows)
corr.to_csv(TAB / "GOSIF_GLEAM_CORRELATION_DIAGNOSTICS.csv", index=False)

# -----------------------------
# Outcome summaries and point table
# -----------------------------

summary_rows = []
for mask_name, mask in masks.items():
    sub = d.loc[mask].copy()
    x = sub["gosif_gleam_y"].dropna()
    y = sub["latent_y"].dropna()
    summary_rows.append({
        "mask": mask_name,
        "n_mask_raw": int(mask.sum()),
        "n_gosif_gleam_outcome": int(x.notna().sum()),
        "gosif_gleam_mean": float(x.mean()) if len(x) else np.nan,
        "gosif_gleam_sd": float(x.std()) if len(x) else np.nan,
        "gosif_gleam_min": float(x.min()) if len(x) else np.nan,
        "gosif_gleam_max": float(x.max()) if len(x) else np.nan,
        "n_latent_outcome": int(y.notna().sum()),
        "latent_mean": float(y.mean()) if len(y) else np.nan,
        "latent_sd": float(y.std()) if len(y) else np.nan,
        "latent_min": float(y.min()) if len(y) else np.nan,
        "latent_max": float(y.max()) if len(y) else np.nan,
    })

pd.DataFrame(summary_rows).to_csv(TAB / "GOSIF_GLEAM_OUTCOME_SUMMARY_BY_MASK.csv", index=False)

point_cols = [
    "point_id",
    "lat",
    "lon",
    "lai",
    "mat",
    "vpd",
    "arid",
    "sm",
    "latent_y",
    "gosif_gleam_y",
    "latent_y_z",
    "gosif_gleam_y_z",
    "any_cropland_managed_irrigation_flag",
    "any_natural_grassland_indicator",
    "sahel_broad",
    "has_c4",
    "all_product_combos_complete",
]
point_cols = [c for c in point_cols if c in d.columns]
d[point_cols].to_csv(TAB / "GOSIF_GLEAM_POINT_LEVEL_VALUES.csv", index=False)

# -----------------------------
# Partial residual / scatter plots
# -----------------------------

try:
    plot_sub = d[["lai_z", "gosif_gleam_y_z", "latent_y_z"] + full_controls].dropna().copy()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(plot_sub["lai_z"], plot_sub["gosif_gleam_y_z"], alpha=0.65)
    if len(plot_sub) >= 3:
        b = np.polyfit(plot_sub["lai_z"], plot_sub["gosif_gleam_y_z"], 1)
        xs = np.linspace(plot_sub["lai_z"].min(), plot_sub["lai_z"].max(), 100)
        ax.plot(xs, b[0] * xs + b[1])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("LAI z")
    ax.set_ylabel("GOSIF × GLEAM slope-change z")
    ax.set_title("GOSIF × GLEAM outcome vs LAI")
    fig.tight_layout()
    fig.savefig(FIG / "FIG_gosif_gleam_vs_lai_bivariate.png", dpi=220)
    plt.close(fig)

    # Full-control partial residual plot.
    controls_formula = " + ".join(full_controls)
    if controls_formula:
        y_res = smf.ols(f"gosif_gleam_y_z ~ {controls_formula}", data=plot_sub).fit().resid
        x_res = smf.ols(f"lai_z ~ {controls_formula}", data=plot_sub).fit().resid
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_res, y_res, alpha=0.65)
        if len(x_res) >= 3:
            b = np.polyfit(x_res, y_res, 1)
            xs = np.linspace(np.min(x_res), np.max(x_res), 100)
            ax.plot(xs, b[0] * xs + b[1])
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("LAI residual after full controls")
        ax.set_ylabel("GOSIF × GLEAM outcome residual after full controls")
        ax.set_title("Full-control partial relationship")
        fig.tight_layout()
        fig.savefig(FIG / "FIG_gosif_gleam_partial_residual_full_controls.png", dpi=220)
        plt.close(fig)

except Exception as e:
    (TXT / "PLOT_ERROR.txt").write_text(str(e))

# -----------------------------
# Compact readme
# -----------------------------

def show(path, n=80):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

readme = []
readme.append("Stage1B6BH GOSIF x GLEAM deep test")
readme.append("=" * 90)
readme.append("")
readme.append(f"Point input: {POINT_INPUT}")
readme.append(f"Observation input: {OBS_INPUT}")
readme.append(f"GOSIF/SIF x GLEAM observation rows: {len(gg)}")
readme.append(f"GOSIF/SIF x GLEAM point-level rows: {len(gg_point)}")
readme.append(f"Full controls used: {full_controls}")
readme.append("")
readme.append("Available product combos:")
readme.append(show("AVAILABLE_PRODUCT_COMBOS.csv", 30))
readme.append("")
readme.append("GOSIF x GLEAM model tests:")
model_df = pd.read_csv(TAB / "GOSIF_GLEAM_FIXED_MASK_MODEL_TESTS.csv")
cols = [c for c in [
    "mask", "model", "status", "n", "focal_term", "coef", "se_hc3", "p",
    "ci_low", "ci_high", "delta_r2", "delta_aic_full_minus_reduced",
    "nested_f_p", "boot_ci_low", "boot_ci_high", "boot_sign_stability",
    "lro_sign_stability"
] if c in model_df.columns]
readme.append(model_df[cols].to_string(index=False))
readme.append("")
readme.append("Correlation diagnostics:")
readme.append(show("GOSIF_GLEAM_CORRELATION_DIAGNOSTICS.csv", 80))
readme.append("")
readme.append("Outcome summary by mask:")
readme.append(show("GOSIF_GLEAM_OUTCOME_SUMMARY_BY_MASK.csv", 80))
readme.append("")
readme.append("Important files:")
for f in [
    "AVAILABLE_PRODUCT_COMBOS.csv",
    "GOSIF_GLEAM_FIXED_MASK_MODEL_TESTS.csv",
    "GOSIF_GLEAM_CORRELATION_DIAGNOSTICS.csv",
    "GOSIF_GLEAM_OUTCOME_SUMMARY_BY_MASK.csv",
    "GOSIF_GLEAM_POINT_LEVEL_VALUES.csv",
    "GOSIF_GLEAM_LEAVE_REGION_OUT.csv",
]:
    readme.append(f"- {TAB / f}")
for f in [
    "FIG_gosif_gleam_vs_lai_bivariate.png",
    "FIG_gosif_gleam_partial_residual_full_controls.png",
]:
    readme.append(f"- {FIG / f}")

(TXT / "READ_ME_gosif_gleam_deep_test.txt").write_text("\n".join(readme))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_gosif_gleam_deep_test.txt'}")
