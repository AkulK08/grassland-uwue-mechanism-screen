#!/usr/bin/env python
from pathlib import Path
import json
import warnings
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

# =============================================================================
# PHASE 5: causal-adjusted trait models
# =============================================================================

DATASET = Path("results/trait_framework/trait_model_dataset.csv")
OUTDIR = Path("results/trait_framework/phase5")
OUTDIR.mkdir(parents=True, exist_ok=True)

OLS_OUT = OUTDIR / "table_trait_ols_models.csv"
ML_OUT = OUTDIR / "table_trait_ml_importance.csv"
DML_OUT = OUTDIR / "table_trait_causal_adjusted_effects.csv"
SENS_OUT = OUTDIR / "table_trait_sensitivity_by_product_family.csv"
MANIFEST_OUT = OUTDIR / "phase5_model_manifest.json"
README_OUT = OUTDIR / "README_phase5_causal_trait_models.md"

RANDOM_SEED = 42
MIN_N_OLS = 20
MIN_N_ML = 25
MIN_N_DML = 25
MAX_CV = 5

PRIMARY_OUTCOMES = [
    "consensus_slope_change_independent",
    "consensus_slope_change_all",
]

SECONDARY_OUTCOMES = [
    "consensus_post_slope_independent",
    "consensus_post_slope_all",
]

SENSITIVITY_OUTCOMES = [
    "satbreak_fraction_all",
    "satbreak_fraction_independent",
    "negative_slope_fraction_all",
    "negative_slope_fraction_independent",
]

PRODUCT_FAMILY_OUTCOMES = [
    "consensus_slope_change_all",
    "consensus_slope_change_independent",
    "consensus_slope_change_pml_containing",
    "consensus_slope_change_gosif_gpp",
    "consensus_slope_change_gleam_et",
]

CORE_TRAITS = [
    "p50",
    "rooting_depth",
]

LIMITED_TRAITS = [
    "isohydricity",
]

TRAIT_ALIASES = {
    "p50": "P50 / xylem vulnerability",
    "rooting_depth": "rooting depth / rooting-zone storage",
    "isohydricity": "isohydricity / stomatal strategy",
}

# =============================================================================
# Utilities
# =============================================================================

def die(msg):
    raise SystemExit("\nERROR: " + str(msg) + "\n")

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return make_unique_columns(df)

def make_unique_columns(df):
    df = df.copy()
    seen = {}
    cols = []
    for c in df.columns:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
    df.columns = cols
    return df

def to_num(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def finite_n(df, col):
    if col not in df.columns:
        return 0
    return int(to_num(df[col]).notna().sum())

def has_col(df, col, min_n=1):
    return col in df.columns and finite_n(df, col) >= min_n

def zscore(s):
    x = to_num(s)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd

def direction(x):
    if pd.isna(x):
        return "NA"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_unique_columns(df)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def infer_available(df, candidates, min_n=10):
    return [c for c in candidates if has_col(df, c, min_n=min_n)]

def choose_precip_col(df):
    for c in ["mean_annual_precipitation", "mean_precipitation"]:
        if has_col(df, c, min_n=10):
            return c
    return None

def choose_temp_col(df):
    for c in ["mean_annual_temperature", "mean_temperature"]:
        if has_col(df, c, min_n=10):
            return c
    return None

def choose_lai_col(df):
    for c in ["growing_season_mean_lai", "mean_lai"]:
        if has_col(df, c, min_n=10):
            return c
    return None

def choose_soil_cols(df):
    # Avoid full sand+silt+clay compositional collinearity by preferring sand + clay.
    out = []
    if has_col(df, "soil_sand", min_n=10):
        out.append("soil_sand")
    if has_col(df, "soil_clay", min_n=10):
        out.append("soil_clay")
    if not out and has_col(df, "soil_silt", min_n=10):
        out.append("soil_silt")
    return out

def choose_region_cols(df):
    return [c for c in ["region_label", "lat_band", "lon_region", "aridity_quartile", "biome_label"] if c in df.columns]

def standardize_numeric_cols(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = zscore(out[c])
    return out

def build_design(df, numeric_cols, categorical_cols=None, standardize=True, add_intercept=True):
    categorical_cols = categorical_cols or []

    use = df.copy()
    actual_numeric = [c for c in numeric_cols if c in use.columns]
    actual_cats = [c for c in categorical_cols if c in use.columns]

    for c in actual_numeric:
        use[c] = to_num(use[c])

    X_num = use[actual_numeric].copy()

    if standardize:
        for c in actual_numeric:
            X_num[c] = zscore(X_num[c])

    X_parts = [X_num]

    if actual_cats:
        cat = use[actual_cats].astype(str).fillna("missing")
        X_cat = pd.get_dummies(cat, columns=actual_cats, drop_first=True, dtype=float)
        X_parts.append(X_cat)

    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=use.index)

    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop all-null columns and zero-variance columns.
    keep_cols = []
    for c in X.columns:
        x = to_num(X[c])
        if x.notna().sum() == 0:
            continue
        if x.nunique(dropna=True) <= 1:
            continue
        keep_cols.append(c)

    X = X[keep_cols].copy()

    if X.shape[1] > 0:
        X = X.fillna(X.median(numeric_only=True))
    else:
        X = pd.DataFrame(index=use.index)

    if add_intercept:
        X = sm.add_constant(X, has_constant="add")

    return X

def complete_case(df, outcome, numeric_cols, categorical_cols=None):
    categorical_cols = categorical_cols or []

    needed = [outcome] + [c for c in numeric_cols if c in df.columns]
    d = df.copy()

    for c in needed:
        if c not in d.columns:
            return pd.DataFrame()
        d[c] = to_num(d[c])

    ok = d[outcome].notna()
    for c in numeric_cols:
        if c in d.columns:
            ok &= d[c].notna()

    for c in categorical_cols:
        if c in d.columns:
            ok &= d[c].notna()

    return d.loc[ok].copy()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def p_value_from_t(t, df_resid):
    try:
        return float(2 * (1 - stats.t.cdf(abs(t), df=df_resid)))
    except Exception:
        return np.nan

# =============================================================================
# Load and inspect dataset
# =============================================================================

def load_dataset():
    if not DATASET.exists():
        die(
            f"Missing Phase 4 dataset: {DATASET}\n"
            "Run Phase 4 first."
        )

    df = clean_columns(pd.read_csv(DATASET, low_memory=False))

    if "point_id" not in df.columns:
        die("trait_model_dataset.csv is missing point_id.")

    df["point_id"] = df["point_id"].astype(str)

    # Coerce important numeric columns.
    for c in df.columns:
        if c.startswith("z_"):
            df[c] = to_num(df[c])

    for c in [
        "p50",
        "xylem_vulnerability_p50",
        "rooting_depth",
        "rooting_zone_storage_rooting_depth",
        "isohydricity",
        "stomatal_strategy_isohydricity",
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        "mean_annual_precipitation",
        "mean_precipitation",
        "mean_annual_temperature",
        "mean_temperature",
        "growing_season_mean_lai",
        "mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "lat",
        "lon",
        "abs_lat",
    ] + PRIMARY_OUTCOMES + SECONDARY_OUTCOMES + SENSITIVITY_OUTCOMES + PRODUCT_FAMILY_OUTCOMES:
        if c in df.columns:
            df[c] = to_num(df[c])

    return df

# =============================================================================
# Define variables / model sets
# =============================================================================

def define_variable_sets(df):
    precip = choose_precip_col(df)
    temp = choose_temp_col(df)
    lai = choose_lai_col(df)
    soil = choose_soil_cols(df)
    region_cols = choose_region_cols(df)

    climate = []
    for c in ["aridity", "mean_vpd", "mean_soil_moisture", precip, temp, lai]:
        if c is not None and has_col(df, c, min_n=10):
            climate.append(c)

    # Keep aridity as explicit main climate context.
    aridity_only = ["aridity"] if has_col(df, "aridity", min_n=10) else []

    spatial_numeric = []
    for c in ["lat", "lon", "abs_lat"]:
        if has_col(df, c, min_n=10):
            spatial_numeric.append(c)

    outcomes = {
        "primary": infer_available(df, PRIMARY_OUTCOMES, min_n=10),
        "secondary": infer_available(df, SECONDARY_OUTCOMES, min_n=10),
        "sensitivity": infer_available(df, SENSITIVITY_OUTCOMES, min_n=10),
        "product_family": infer_available(df, PRODUCT_FAMILY_OUTCOMES, min_n=10),
    }

    # Prefer independent primary if usable.
    if "consensus_slope_change_independent" in outcomes["primary"]:
        preferred_primary = "consensus_slope_change_independent"
    elif outcomes["primary"]:
        preferred_primary = outcomes["primary"][0]
    else:
        preferred_primary = None

    traits_core = [c for c in CORE_TRAITS if has_col(df, c, min_n=10)]
    traits_limited = [c for c in LIMITED_TRAITS if has_col(df, c, min_n=10)]

    sets = {
        "traits_core": traits_core,
        "traits_limited": traits_limited,
        "aridity_only": aridity_only,
        "climate_controls": climate,
        "soil_controls": soil,
        "spatial_numeric": spatial_numeric,
        "region_cols": region_cols,
        "outcomes": outcomes,
        "preferred_primary": preferred_primary,
        "precip_col": precip,
        "temp_col": temp,
        "lai_col": lai,
    }

    return sets

# =============================================================================
# OLS / robust linear models
# =============================================================================

def fit_ols_single(df, outcome, predictors, categorical_cols, model_name, sample_flag, model_role, robust_kind="HC3"):
    rows = []

    numeric_cols = list(dict.fromkeys([c for c in predictors if c in df.columns]))
    categorical_cols = list(dict.fromkeys([c for c in categorical_cols if c in df.columns]))

    d = complete_case(df, outcome, numeric_cols, categorical_cols)

    if d.empty or len(d) < MIN_N_OLS:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "OLS_HC3",
            "sample_flag": sample_flag,
            "n": int(len(d)) if not d.empty else 0,
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": "skipped_low_n",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })
        return rows

    y = zscore(d[outcome])
    X = build_design(d, numeric_cols, categorical_cols, standardize=True, add_intercept=True)

    ok = y.notna()
    X = X.loc[ok]
    y = y.loc[ok]

    if len(y) < MIN_N_OLS or X.shape[1] < 2:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "OLS_HC3",
            "sample_flag": sample_flag,
            "n": int(len(y)),
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": "skipped_low_n_or_no_predictors",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })
        return rows

    try:
        model = sm.OLS(y, X).fit()
        fit = model.get_robustcov_results(cov_type=robust_kind)

        params = pd.Series(fit.params, index=X.columns)
        bse = pd.Series(fit.bse, index=X.columns)
        tvals = pd.Series(fit.tvalues, index=X.columns)
        pvals = pd.Series(fit.pvalues, index=X.columns)
        conf = pd.DataFrame(fit.conf_int(), index=X.columns, columns=["ci_low", "ci_high"])

        for term in X.columns:
            rows.append({
                "outcome": outcome,
                "model_name": model_name,
                "model_role": model_role,
                "model_family": "OLS_HC3",
                "sample_flag": sample_flag,
                "n": int(len(y)),
                "term": term,
                "estimate": float(params.loc[term]),
                "std_error": float(bse.loc[term]),
                "t_stat": float(tvals.loc[term]),
                "p_value": float(pvals.loc[term]),
                "ci_low": float(conf.loc[term, "ci_low"]),
                "ci_high": float(conf.loc[term, "ci_high"]),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "status": "ok",
                "model_formula": f"z({outcome}) ~ z({') + z('.join(numeric_cols)})" + (f" + FE({','.join(categorical_cols)})" if categorical_cols else ""),
            })

    except Exception as e:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "OLS_HC3",
            "sample_flag": sample_flag,
            "n": int(len(y)),
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": f"failed: {type(e).__name__}: {e}",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })

    return rows

def fit_rlm_single(df, outcome, predictors, categorical_cols, model_name, sample_flag, model_role):
    rows = []

    numeric_cols = list(dict.fromkeys([c for c in predictors if c in df.columns]))
    categorical_cols = list(dict.fromkeys([c for c in categorical_cols if c in df.columns]))

    d = complete_case(df, outcome, numeric_cols, categorical_cols)

    if d.empty or len(d) < MIN_N_OLS:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "RLM_Huber",
            "sample_flag": sample_flag,
            "n": int(len(d)) if not d.empty else 0,
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": "skipped_low_n",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })
        return rows

    y = zscore(d[outcome])
    X = build_design(d, numeric_cols, categorical_cols, standardize=True, add_intercept=True)

    ok = y.notna()
    X = X.loc[ok]
    y = y.loc[ok]

    if len(y) < MIN_N_OLS or X.shape[1] < 2:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "RLM_Huber",
            "sample_flag": sample_flag,
            "n": int(len(y)),
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": "skipped_low_n_or_no_predictors",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })
        return rows

    try:
        fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()

        # pseudo R2 from fitted values.
        yhat = fit.fittedvalues
        pseudo_r2 = r2_score(y, yhat) if len(y) > 2 else np.nan

        params = fit.params
        bse = fit.bse
        tvals = fit.tvalues
        pvals = fit.pvalues

        for term in X.columns:
            est = float(params.loc[term])
            se = float(bse.loc[term]) if term in bse.index else np.nan
            tval = float(tvals.loc[term]) if term in tvals.index else np.nan
            pval = float(pvals.loc[term]) if term in pvals.index else np.nan
            ci_low = est - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = est + 1.96 * se if np.isfinite(se) else np.nan

            rows.append({
                "outcome": outcome,
                "model_name": model_name,
                "model_role": model_role,
                "model_family": "RLM_Huber",
                "sample_flag": sample_flag,
                "n": int(len(y)),
                "term": term,
                "estimate": est,
                "std_error": se,
                "t_stat": tval,
                "p_value": pval,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "r2": float(pseudo_r2),
                "adj_r2": np.nan,
                "status": "ok",
                "model_formula": f"z({outcome}) ~ z({') + z('.join(numeric_cols)})" + (f" + FE({','.join(categorical_cols)})" if categorical_cols else ""),
            })

    except Exception as e:
        rows.append({
            "outcome": outcome,
            "model_name": model_name,
            "model_role": model_role,
            "model_family": "RLM_Huber",
            "sample_flag": sample_flag,
            "n": int(len(y)),
            "term": "__MODEL_STATUS__",
            "estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "status": f"failed: {type(e).__name__}: {e}",
            "model_formula": f"{outcome} ~ {' + '.join(numeric_cols + categorical_cols)}",
        })

    return rows

def run_ols_framework(df, sets):
    rows = []

    traits = sets["traits_core"]
    aridity = sets["aridity_only"]
    climate = sets["climate_controls"]
    soil = sets["soil_controls"]
    spatial_numeric = sets["spatial_numeric"]
    region_cols = sets["region_cols"]

    outcomes = []
    for k in ["primary", "secondary", "sensitivity"]:
        outcomes.extend(sets["outcomes"][k])
    outcomes = list(dict.fromkeys(outcomes))

    for outcome in outcomes:
        model_role = (
            "primary" if outcome in sets["outcomes"]["primary"]
            else "secondary" if outcome in sets["outcomes"]["secondary"]
            else "sensitivity"
        )

        specs = [
            ("M1_traits_only", traits, [], "all_available_complete_case"),
            ("M2_traits_plus_aridity", traits + aridity, [], "all_available_complete_case"),
            ("M3_traits_plus_aridity_climate", traits + climate, [], "core_trait_climate_model_ready"),
            ("M4_traits_plus_aridity_climate_soil", traits + climate + soil, [], "core_trait_climate_soil_model_ready"),
            ("M5_traits_plus_aridity_climate_soil_region", traits + climate + soil + spatial_numeric, region_cols[:1], "core_trait_climate_soil_model_ready_region"),
        ]

        for model_name, predictors, cats, sample_flag in specs:
            rows.extend(fit_ols_single(df, outcome, predictors, cats, model_name, sample_flag, model_role, robust_kind="HC3"))
            rows.extend(fit_rlm_single(df, outcome, predictors, cats, model_name, sample_flag, model_role))

    # Limited-coverage isohydricity model.
    if "isohydricity" in sets["traits_limited"]:
        for outcome in sets["outcomes"]["primary"]:
            predictors = traits + ["isohydricity"] + climate + soil
            rows.extend(fit_ols_single(
                df,
                outcome,
                predictors,
                [],
                "M6_limited_isohydricity_traits_plus_controls",
                "full_trait_with_isohydricity_ready",
                "limited_coverage_isohydricity",
                robust_kind="HC3",
            ))
            rows.extend(fit_rlm_single(
                df,
                outcome,
                predictors,
                [],
                "M6_limited_isohydricity_traits_plus_controls",
                "full_trait_with_isohydricity_ready",
                "limited_coverage_isohydricity",
            ))

    return pd.DataFrame(rows)

# =============================================================================
# ML importance models
# =============================================================================

def prepare_ml_matrix(df, outcome, numeric_predictors, categorical_predictors):
    needed_num = [c for c in numeric_predictors if c in df.columns]
    needed_cat = [c for c in categorical_predictors if c in df.columns]

    d = df.copy()

    if outcome not in d.columns:
        return None, None, None, None

    y = to_num(d[outcome])
    ok = y.notna()

    for c in needed_num:
        d[c] = to_num(d[c])
        ok &= d[c].notna()

    for c in needed_cat:
        ok &= d[c].notna()

    d = d.loc[ok].copy()
    y = to_num(d[outcome])

    if len(d) < MIN_N_ML:
        return None, None, None, None

    # Standardize outcome for comparability.
    y = zscore(y)

    X_num = d[needed_num].copy()
    for c in needed_num:
        X_num[c] = zscore(X_num[c])

    X_parts = [X_num]

    if needed_cat:
        X_cat = pd.get_dummies(d[needed_cat].astype(str).fillna("missing"), drop_first=True, dtype=float)
        X_parts.append(X_cat)

    X = pd.concat(X_parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop all-null/zero-var columns.
    keep = []
    for c in X.columns:
        x = to_num(X[c])
        if x.notna().sum() == 0:
            continue
        if x.nunique(dropna=True) <= 1:
            continue
        keep.append(c)

    X = X[keep].copy()

    if X.shape[1] == 0:
        return None, None, None, None

    X = X.fillna(X.median(numeric_only=True))

    # Map encoded features back to original predictor where possible.
    feature_to_predictor = {}
    for c in X.columns:
        assigned = None
        for p in needed_num:
            if c == p:
                assigned = p
                break
        if assigned is None:
            for cat in needed_cat:
                if c.startswith(cat + "_"):
                    assigned = cat
                    break
        feature_to_predictor[c] = assigned if assigned is not None else c

    return X, y, d, feature_to_predictor

def run_single_ml_model(df, outcome, predictors, categorical, predictor_set, model_family, model):
    rows = []

    X, y, d, feature_to_predictor = prepare_ml_matrix(df, outcome, predictors, categorical)

    if X is None:
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "predictor_set": predictor_set,
            "predictor": "__MODEL_STATUS__",
            "encoded_feature": "__MODEL_STATUS__",
            "permutation_importance_mean": np.nan,
            "permutation_importance_sd": np.nan,
            "rank": np.nan,
            "cv_r2_mean": np.nan,
            "cv_r2_sd": np.nan,
            "cv_rmse_mean": np.nan,
            "cv_rmse_sd": np.nan,
            "n": 0,
            "status": "skipped_low_n_or_no_predictors",
        })
        return rows

    n = len(y)
    k = min(MAX_CV, n)
    if k < 3:
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "predictor_set": predictor_set,
            "predictor": "__MODEL_STATUS__",
            "encoded_feature": "__MODEL_STATUS__",
            "permutation_importance_mean": np.nan,
            "permutation_importance_sd": np.nan,
            "rank": np.nan,
            "cv_r2_mean": np.nan,
            "cv_r2_sd": np.nan,
            "cv_rmse_mean": np.nan,
            "cv_rmse_sd": np.nan,
            "n": int(n),
            "status": "skipped_too_few_folds",
        })
        return rows

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    r2s = []
    rmses = []
    importances = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        try:
            m = model
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)

            r2s.append(r2_score(y_te, pred))
            rmses.append(rmse(y_te, pred))

            pi = permutation_importance(
                m,
                X_te,
                y_te,
                n_repeats=20,
                random_state=RANDOM_SEED + fold,
                scoring="r2",
            )

            imp = pd.DataFrame({
                "encoded_feature": X.columns,
                "importance": pi.importances_mean,
                "importance_sd": pi.importances_std,
            })
            imp["fold"] = fold
            importances.append(imp)

        except Exception as e:
            rows.append({
                "outcome": outcome,
                "model_family": model_family,
                "predictor_set": predictor_set,
                "predictor": "__MODEL_STATUS__",
                "encoded_feature": "__MODEL_STATUS__",
                "permutation_importance_mean": np.nan,
                "permutation_importance_sd": np.nan,
                "rank": np.nan,
                "cv_r2_mean": np.nan,
                "cv_r2_sd": np.nan,
                "cv_rmse_mean": np.nan,
                "cv_rmse_sd": np.nan,
                "n": int(n),
                "status": f"fold_failed: {type(e).__name__}: {e}",
            })

    if not importances:
        return rows

    imp_all = pd.concat(importances, ignore_index=True)
    imp_all["predictor"] = imp_all["encoded_feature"].map(feature_to_predictor)

    # Aggregate encoded features back to original predictor.
    pred_imp = (
        imp_all
        .groupby("predictor", dropna=False)
        .agg(
            permutation_importance_mean=("importance", "mean"),
            permutation_importance_sd=("importance", "std"),
        )
        .reset_index()
    )

    pred_imp = pred_imp.sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    pred_imp["rank"] = np.arange(1, len(pred_imp) + 1)

    for _, r in pred_imp.iterrows():
        rows.append({
            "outcome": outcome,
            "model_family": model_family,
            "predictor_set": predictor_set,
            "predictor": r["predictor"],
            "encoded_feature": "__aggregated_to_predictor__",
            "permutation_importance_mean": float(r["permutation_importance_mean"]),
            "permutation_importance_sd": float(r["permutation_importance_sd"]) if pd.notna(r["permutation_importance_sd"]) else np.nan,
            "rank": int(r["rank"]),
            "cv_r2_mean": float(np.nanmean(r2s)) if r2s else np.nan,
            "cv_r2_sd": float(np.nanstd(r2s)) if r2s else np.nan,
            "cv_rmse_mean": float(np.nanmean(rmses)) if rmses else np.nan,
            "cv_rmse_sd": float(np.nanstd(rmses)) if rmses else np.nan,
            "n": int(n),
            "status": "ok",
        })

    return rows

def run_ml_framework(df, sets):
    rows = []

    traits = sets["traits_core"]
    climate = sets["climate_controls"]
    soil = sets["soil_controls"]
    spatial = sets["spatial_numeric"]
    cats = sets["region_cols"][:1]

    outcomes = []
    for k in ["primary", "secondary", "sensitivity"]:
        outcomes.extend(sets["outcomes"][k])
    outcomes = list(dict.fromkeys(outcomes))

    predictor_sets = [
        ("traits_only", traits, []),
        ("controls_only", climate + soil + spatial, cats),
        ("traits_plus_controls", traits + climate + soil + spatial, cats),
    ]

    model_specs = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=500,
                random_state=RANDOM_SEED,
                min_samples_leaf=3,
                max_features="sqrt",
            ),
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(
                random_state=RANDOM_SEED,
                n_estimators=300,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.8,
            ),
        ),
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(
                random_state=RANDOM_SEED,
                max_iter=300,
                learning_rate=0.03,
                l2_regularization=0.1,
            ),
        ),
    ]

    for outcome in outcomes:
        for predictor_set, preds, cat_cols in predictor_sets:
            for model_family, model in model_specs:
                rows.extend(run_single_ml_model(
                    df=df,
                    outcome=outcome,
                    predictors=preds,
                    categorical=cat_cols,
                    predictor_set=predictor_set,
                    model_family=model_family,
                    model=model,
                ))

    return pd.DataFrame(rows)

# =============================================================================
# DML / residualized causal-adjusted trait effects
# =============================================================================

def prepare_controls_matrix(df, controls_numeric, controls_categorical):
    X = build_design(df, controls_numeric, controls_categorical, standardize=True, add_intercept=False)
    return X

def crossfit_predict(model_factory, X, y, n_splits):
    preds = pd.Series(np.nan, index=y.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr, te) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr = y.iloc[tr]

        m = model_factory()
        m.fit(X_tr, y_tr)
        preds.iloc[te] = m.predict(X_te)

    return preds

def default_nuisance_model():
    return RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        min_samples_leaf=3,
        max_features="sqrt",
    )

def dml_single(df, outcome, treatments, controls_numeric, controls_categorical, control_set_name, sample_flag, model_role):
    rows = []

    treatments = [t for t in treatments if t in df.columns]
    controls_numeric = [c for c in controls_numeric if c in df.columns and c not in treatments]
    controls_categorical = [c for c in controls_categorical if c in df.columns]

    needed = [outcome] + treatments + controls_numeric
    d = df.copy()

    if outcome not in d.columns or not treatments:
        rows.append({
            "outcome": outcome,
            "treatment_trait": ",".join(treatments) if treatments else "none",
            "model_role": model_role,
            "n": 0,
            "control_set": control_set_name,
            "learner_y": "RandomForestRegressor",
            "learner_t": "RandomForestRegressor",
            "effect_estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2_y_nuisance": np.nan,
            "r2_t_nuisance": np.nan,
            "sample_flag": sample_flag,
            "term": "__MODEL_STATUS__",
            "status": "missing_outcome_or_treatment",
            "interpretation": "Skipped.",
        })
        return rows

    # Complete-case.
    ok = to_num(d[outcome]).notna()
    for c in treatments + controls_numeric:
        ok &= to_num(d[c]).notna()
    for c in controls_categorical:
        ok &= d[c].notna()

    d = d.loc[ok].copy()

    if len(d) < MIN_N_DML:
        rows.append({
            "outcome": outcome,
            "treatment_trait": ",".join(treatments),
            "model_role": model_role,
            "n": int(len(d)),
            "control_set": control_set_name,
            "learner_y": "RandomForestRegressor",
            "learner_t": "RandomForestRegressor",
            "effect_estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2_y_nuisance": np.nan,
            "r2_t_nuisance": np.nan,
            "sample_flag": sample_flag,
            "term": "__MODEL_STATUS__",
            "status": "skipped_low_n",
            "interpretation": "Skipped due to insufficient complete-case sample size.",
        })
        return rows

    y = zscore(d[outcome])
    tmat = pd.DataFrame({t: zscore(d[t]) for t in treatments}, index=d.index)

    X = prepare_controls_matrix(d, controls_numeric, controls_categorical)

    if X.shape[1] == 0:
        # If no controls, use intercept-only residualization.
        y_hat = pd.Series(y.mean(), index=y.index)
        t_hat = pd.DataFrame({t: tmat[t].mean() for t in treatments}, index=d.index)
        r2_y = 0.0
        r2_t_mean = 0.0
    else:
        n_splits = min(MAX_CV, len(d))
        if n_splits < 3:
            n_splits = 3

        try:
            y_hat = crossfit_predict(default_nuisance_model, X, y, n_splits)
            r2_y = r2_score(y, y_hat)

            t_hat_cols = {}
            r2_ts = []
            for t in treatments:
                pred_t = crossfit_predict(default_nuisance_model, X, tmat[t], n_splits)
                t_hat_cols[t] = pred_t
                try:
                    r2_ts.append(r2_score(tmat[t], pred_t))
                except Exception:
                    pass
            t_hat = pd.DataFrame(t_hat_cols, index=d.index)
            r2_t_mean = float(np.nanmean(r2_ts)) if r2_ts else np.nan

        except Exception as e:
            rows.append({
                "outcome": outcome,
                "treatment_trait": ",".join(treatments),
                "model_role": model_role,
                "n": int(len(d)),
                "control_set": control_set_name,
                "learner_y": "RandomForestRegressor",
                "learner_t": "RandomForestRegressor",
                "effect_estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2_y_nuisance": np.nan,
                "r2_t_nuisance": np.nan,
                "sample_flag": sample_flag,
                "term": "__MODEL_STATUS__",
                "status": f"nuisance_failed: {type(e).__name__}: {e}",
                "interpretation": "Skipped due to nuisance-model failure.",
            })
            return rows

    y_res = y - y_hat
    t_res = tmat - t_hat

    final_X = sm.add_constant(t_res, has_constant="add")

    try:
        fit = sm.OLS(y_res, final_X).fit()
        rob = fit.get_robustcov_results(cov_type="HC3")

        params = pd.Series(rob.params, index=final_X.columns)
        bse = pd.Series(rob.bse, index=final_X.columns)
        tvals = pd.Series(rob.tvalues, index=final_X.columns)
        pvals = pd.Series(rob.pvalues, index=final_X.columns)
        conf = pd.DataFrame(rob.conf_int(), index=final_X.columns, columns=["ci_low", "ci_high"])

        for t in treatments:
            rows.append({
                "outcome": outcome,
                "treatment_trait": t,
                "treatment_label": TRAIT_ALIASES.get(t, t),
                "model_role": model_role,
                "n": int(len(d)),
                "control_set": control_set_name,
                "learner_y": "RandomForestRegressor",
                "learner_t": "RandomForestRegressor",
                "effect_estimate": float(params.loc[t]),
                "std_error": float(bse.loc[t]),
                "t_stat": float(tvals.loc[t]),
                "p_value": float(pvals.loc[t]),
                "ci_low": float(conf.loc[t, "ci_low"]),
                "ci_high": float(conf.loc[t, "ci_high"]),
                "r2_y_nuisance": float(r2_y),
                "r2_t_nuisance": float(r2_t_mean),
                "sample_flag": sample_flag,
                "term": t,
                "status": "ok",
                "interpretation": "DML-style cross-fitted residualized association after flexible adjustment for controls. Observational, not experimental.",
            })

    except Exception as e:
        rows.append({
            "outcome": outcome,
            "treatment_trait": ",".join(treatments),
            "model_role": model_role,
            "n": int(len(d)),
            "control_set": control_set_name,
            "learner_y": "RandomForestRegressor",
            "learner_t": "RandomForestRegressor",
            "effect_estimate": np.nan,
            "std_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2_y_nuisance": float(r2_y) if "r2_y" in locals() else np.nan,
            "r2_t_nuisance": float(r2_t_mean) if "r2_t_mean" in locals() else np.nan,
            "sample_flag": sample_flag,
            "term": "__MODEL_STATUS__",
            "status": f"final_stage_failed: {type(e).__name__}: {e}",
            "interpretation": "Skipped due to final-stage failure.",
        })

    return rows

def run_dml_framework(df, sets):
    rows = []

    outcomes = []
    for k in ["primary", "secondary", "sensitivity"]:
        outcomes.extend(sets["outcomes"][k])
    outcomes = list(dict.fromkeys(outcomes))

    traits = sets["traits_core"]
    limited = sets["traits_limited"]

    climate = sets["climate_controls"]
    soil = sets["soil_controls"]
    spatial = sets["spatial_numeric"]
    cats = sets["region_cols"][:1]

    # DML control set: aridity + climate + soil + spatial/categorical, excluding treatment.
    controls_numeric = climate + soil + spatial
    controls_categorical = cats

    for outcome in outcomes:
        model_role = (
            "primary" if outcome in sets["outcomes"]["primary"]
            else "secondary" if outcome in sets["outcomes"]["secondary"]
            else "sensitivity"
        )

        # Individual trait treatments.
        for t in traits:
            other_traits = [x for x in traits if x != t]
            rows.extend(dml_single(
                df=df,
                outcome=outcome,
                treatments=[t],
                controls_numeric=controls_numeric + other_traits,
                controls_categorical=controls_categorical,
                control_set_name="climate_soil_spatial_plus_other_trait",
                sample_flag="core_trait_climate_or_soil_model_ready",
                model_role=model_role,
            ))

        # Multi-treatment P50 + rooting depth.
        if len(traits) >= 2:
            rows.extend(dml_single(
                df=df,
                outcome=outcome,
                treatments=traits,
                controls_numeric=controls_numeric,
                controls_categorical=controls_categorical,
                control_set_name="climate_soil_spatial_multitreatment",
                sample_flag="core_trait_climate_or_soil_model_ready",
                model_role=model_role,
            ))

        # Limited isohydricity.
        for t in limited:
            rows.extend(dml_single(
                df=df,
                outcome=outcome,
                treatments=[t],
                controls_numeric=controls_numeric + traits,
                controls_categorical=controls_categorical,
                control_set_name="limited_isohydricity_climate_soil_spatial_plus_core_traits",
                sample_flag="full_trait_with_isohydricity_ready",
                model_role="limited_coverage_isohydricity",
            ))

    return pd.DataFrame(rows)

# =============================================================================
# Product family sensitivity
# =============================================================================

def run_product_family_sensitivity(df, sets, ols_df, dml_df):
    rows = []

    traits = sets["traits_core"]
    climate = sets["climate_controls"]
    soil = sets["soil_controls"]
    spatial = sets["spatial_numeric"]
    cats = sets["region_cols"][:1]

    primary_outcome = sets["preferred_primary"]

    # OLS sensitivity: run the same adjusted model for each product-family outcome.
    for outcome in sets["outcomes"]["product_family"]:
        if outcome not in df.columns:
            continue

        product_family = outcome.replace("consensus_slope_change_", "")
        predictors = traits + climate + soil + spatial

        ols_rows = fit_ols_single(
            df=df,
            outcome=outcome,
            predictors=predictors,
            categorical_cols=cats,
            model_name="product_family_adjusted_ols",
            sample_flag="product_family_complete_case",
            model_role="product_family_sensitivity",
            robust_kind="HC3",
        )

        for r in ols_rows:
            if r.get("term") in traits:
                rows.append({
                    "outcome": outcome,
                    "product_family": product_family,
                    "model_family": "OLS_HC3",
                    "trait": r.get("term"),
                    "estimate_or_importance": r.get("estimate"),
                    "p_value_or_rank": r.get("p_value"),
                    "n": r.get("n"),
                    "direction": direction(r.get("estimate")),
                    "same_direction_as_primary": np.nan,
                    "robustness_label": "pending",
                    "status": r.get("status"),
                })

        # DML sensitivity.
        for t in traits:
            dml_rows = dml_single(
                df=df,
                outcome=outcome,
                treatments=[t],
                controls_numeric=climate + soil + spatial + [x for x in traits if x != t],
                controls_categorical=cats,
                control_set_name="product_family_climate_soil_spatial_plus_other_trait",
                sample_flag="product_family_complete_case",
                model_role="product_family_sensitivity",
            )

            for r in dml_rows:
                if r.get("term") == t:
                    rows.append({
                        "outcome": outcome,
                        "product_family": product_family,
                        "model_family": "DML_residualized_RF",
                        "trait": t,
                        "estimate_or_importance": r.get("effect_estimate"),
                        "p_value_or_rank": r.get("p_value"),
                        "n": r.get("n"),
                        "direction": direction(r.get("effect_estimate")),
                        "same_direction_as_primary": np.nan,
                        "robustness_label": "pending",
                        "status": r.get("status"),
                    })

    sens = pd.DataFrame(rows)

    if sens.empty:
        return pd.DataFrame(columns=[
            "outcome", "product_family", "model_family", "trait",
            "estimate_or_importance", "p_value_or_rank", "n", "direction",
            "same_direction_as_primary", "robustness_label", "status",
        ])

    # Define primary direction using preferred primary outcome, DML if possible, otherwise OLS.
    primary_dirs = {}
    for trait in traits:
        d = sens[
            (sens["outcome"].eq(primary_outcome)) &
            (sens["model_family"].eq("DML_residualized_RF")) &
            (sens["trait"].eq(trait)) &
            (sens["status"].eq("ok"))
        ].copy()

        if d.empty:
            d = sens[
                (sens["outcome"].eq(primary_outcome)) &
                (sens["model_family"].eq("OLS_HC3")) &
                (sens["trait"].eq(trait)) &
                (sens["status"].eq("ok"))
            ].copy()

        if not d.empty:
            primary_dirs[trait] = direction(d["estimate_or_importance"].iloc[0])

    same = []
    label = []

    for _, r in sens.iterrows():
        tr = r["trait"]
        d = r["direction"]
        pdirection = primary_dirs.get(tr, None)

        if pdirection is None or d == "NA":
            same.append(np.nan)
            label.append("primary_direction_unavailable")
        else:
            is_same = d == pdirection
            same.append(bool(is_same))
            if is_same and r["status"] == "ok":
                label.append("same_direction_as_primary")
            elif r["status"] == "ok":
                label.append("opposite_direction_from_primary")
            else:
                label.append("model_not_ok")

    sens["same_direction_as_primary"] = same
    sens["robustness_label"] = label

    return sens

# =============================================================================
# Figures
# =============================================================================

def plot_ols_coefficients(ols_df):
    if ols_df.empty:
        return None

    d = ols_df[
        (ols_df["model_family"].eq("OLS_HC3")) &
        (ols_df["status"].eq("ok")) &
        (ols_df["term"].isin(CORE_TRAITS + LIMITED_TRAITS)) &
        (ols_df["model_name"].isin([
            "M3_traits_plus_aridity_climate",
            "M4_traits_plus_aridity_climate_soil",
            "M5_traits_plus_aridity_climate_soil_region",
            "M6_limited_isohydricity_traits_plus_controls",
        ]))
    ].copy()

    if d.empty:
        return None

    d["label"] = d["outcome"] + "\n" + d["model_name"] + "\n" + d["term"]
    d = d.sort_values(["outcome", "model_name", "term"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(d))))
    y = np.arange(len(d))
    est = d["estimate"].to_numpy()
    lo = d["ci_low"].to_numpy()
    hi = d["ci_high"].to_numpy()
    xerr = np.vstack([est - lo, hi - est])

    ax.errorbar(est, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Standardized coefficient with HC3 95% CI")
    ax.set_title("Figure 4. Trait coefficients from adjusted OLS models")
    fig.tight_layout()

    out = OUTDIR / "Figure4_trait_coefficients_ols.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out

def plot_ml_importance(ml_df):
    if ml_df.empty:
        return None

    d = ml_df[
        (ml_df["status"].eq("ok")) &
        (ml_df["predictor_set"].eq("traits_plus_controls")) &
        (ml_df["model_family"].isin(["RandomForestRegressor", "GradientBoostingRegressor"])) &
        (~ml_df["predictor"].eq("__MODEL_STATUS__"))
    ].copy()

    if d.empty:
        return None

    # Keep top 20 by mean importance across outcomes/models.
    agg = (
        d.groupby("predictor", dropna=False)
        .agg(importance=("permutation_importance_mean", "mean"))
        .reset_index()
        .sort_values("importance", ascending=False)
        .head(20)
    )
    keep = agg["predictor"].tolist()
    d = d[d["predictor"].isin(keep)].copy()

    plot_data = (
        d.groupby("predictor", dropna=False)
        .agg(
            importance=("permutation_importance_mean", "mean"),
            importance_sd=("permutation_importance_mean", "std"),
        )
        .reset_index()
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot_data))))
    ax.barh(plot_data["predictor"], plot_data["importance"])
    ax.set_xlabel("Mean permutation importance across ML models")
    ax.set_title("Figure 5. Trait and control importance in nonlinear models")
    fig.tight_layout()

    out = OUTDIR / "Figure5_trait_ml_importance.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out

def plot_dml_effects(dml_df):
    if dml_df.empty:
        return None

    d = dml_df[
        (dml_df["status"].eq("ok")) &
        (dml_df["treatment_trait"].isin(CORE_TRAITS + LIMITED_TRAITS)) &
        (dml_df["model_role"].isin(["primary", "secondary", "sensitivity", "limited_coverage_isohydricity"]))
    ].copy()

    if d.empty:
        return None

    d["label"] = d["outcome"] + "\n" + d["treatment_trait"]
    d = d.sort_values(["outcome", "treatment_trait"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(d))))
    y = np.arange(len(d))
    est = d["effect_estimate"].to_numpy()
    lo = d["ci_low"].to_numpy()
    hi = d["ci_high"].to_numpy()
    xerr = np.vstack([est - lo, hi - est])

    ax.errorbar(est, y, xerr=xerr, fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("DML residualized standardized effect with 95% CI")
    ax.set_title("Figure 6. Causal-adjusted trait effects")
    fig.tight_layout()

    out = OUTDIR / "Figure6_dml_trait_effects.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out

def plot_product_family_sensitivity(sens_df):
    if sens_df.empty:
        return None

    d = sens_df[
        (sens_df["status"].eq("ok")) &
        (sens_df["trait"].isin(CORE_TRAITS)) &
        (sens_df["model_family"].isin(["OLS_HC3", "DML_residualized_RF"]))
    ].copy()

    if d.empty:
        return None

    d["label"] = d["product_family"] + "\n" + d["model_family"] + "\n" + d["trait"]
    d = d.sort_values(["trait", "model_family", "product_family"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(d))))
    y = np.arange(len(d))
    x = d["estimate_or_importance"].to_numpy()

    ax.scatter(x, y)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Estimate")
    ax.set_title("Figure 7. Product-family sensitivity of trait effects")
    fig.tight_layout()

    out = OUTDIR / "Figure7_product_family_sensitivity.png"
    fig.savefig(out, dpi=300)
    fig.savefig(str(out).replace(".png", ".pdf"))
    plt.close(fig)
    return out

# =============================================================================
# README / manifest
# =============================================================================

def make_readme(df, sets, ols_df, ml_df, dml_df, sens_df, figs, manifest):
    lines = []
    lines.append("# Phase 5: Causal-adjusted trait models")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Estimate trait-associated differences in WUE stress response while holding climate and soil context fixed.")
    lines.append("")
    lines.append("## Conceptual causal model")
    lines.append("")
    lines.append("Traits are not interpreted in isolation. Trait distributions are climatically and edaphically structured, so P50/rooting/isohydricity are tested after adjustment for aridity, VPD, soil moisture, precipitation, temperature, LAI, soil texture, and spatial/region labels where available.")
    lines.append("")
    lines.append("## Primary model")
    lines.append("")
    lines.append("`consensus_slope_change ~ P50 + rooting_depth + aridity + climate_controls + soil_controls`")
    lines.append("")
    lines.append("## Secondary model")
    lines.append("")
    lines.append("`consensus_post_slope ~ P50 + rooting_depth + aridity + climate_controls + soil_controls`")
    lines.append("")
    lines.append("## Sensitivity model")
    lines.append("")
    lines.append("`satbreak_fraction ~ P50 + rooting_depth + aridity + climate_controls + soil_controls`")
    lines.append("")
    lines.append("## Limited-coverage model")
    lines.append("")
    lines.append("`consensus_slope_change ~ P50 + rooting_depth + isohydricity + aridity + controls`, only where isohydricity exists.")
    lines.append("")
    lines.append("## Model families run")
    lines.append("")
    lines.append("1. OLS with HC3 robust standard errors and robust linear model with Huber loss.")
    lines.append("2. Random forest / gradient boosting / histogram gradient boosting with cross-validation and permutation importance.")
    lines.append("3. DML-style cross-fitted residualized trait-effect models using random-forest nuisance learners.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for p in [
        OLS_OUT,
        ML_OUT,
        DML_OUT,
        SENS_OUT,
        MANIFEST_OUT,
    ]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name, p in figs.items():
        if p is not None:
            lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Variable sets")
    lines.append("")
    lines.append(json.dumps(sets, indent=2, default=str))
    lines.append("")
    lines.append("## Result table shapes")
    lines.append("")
    lines.append(f"- OLS/RLM table: {ols_df.shape}")
    lines.append(f"- ML importance table: {ml_df.shape}")
    lines.append(f"- DML causal-adjusted table: {dml_df.shape}")
    lines.append(f"- Product-family sensitivity table: {sens_df.shape}")
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("These models support trait-associated differences after adjustment, not experimental proof that traits cause WUE response. A strong result requires P50/rooting effects to be directionally stable across OLS/RLM, ML importance, DML residualization, and product-family sensitivity.")
    lines.append("")
    lines.append("## Manifest")
    lines.append("")
    lines.append(json.dumps(manifest, indent=2, default=str))

    README_OUT.write_text("\n".join(lines))
    print(f"WROTE {README_OUT}")

# =============================================================================
# Main
# =============================================================================

def main():
    print("PHASE 5 START")

    df = load_dataset()
    sets = define_variable_sets(df)

    print("Loaded dataset:", DATASET, df.shape)
    print("Variable sets:")
    print(json.dumps(sets, indent=2, default=str))

    # Main model outputs.
    ols_df = run_ols_framework(df, sets)
    save_csv(ols_df, OLS_OUT)

    ml_df = run_ml_framework(df, sets)
    save_csv(ml_df, ML_OUT)

    dml_df = run_dml_framework(df, sets)
    save_csv(dml_df, DML_OUT)

    sens_df = run_product_family_sensitivity(df, sets, ols_df, dml_df)
    save_csv(sens_df, SENS_OUT)

    # Figures.
    fig4 = plot_ols_coefficients(ols_df)
    fig5 = plot_ml_importance(ml_df)
    fig6 = plot_dml_effects(dml_df)
    fig7 = plot_product_family_sensitivity(sens_df)

    figs = {
        "Figure4_trait_coefficients_ols": str(fig4) if fig4 is not None else None,
        "Figure5_trait_ml_importance": str(fig5) if fig5 is not None else None,
        "Figure6_dml_trait_effects": str(fig6) if fig6 is not None else None,
        "Figure7_product_family_sensitivity": str(fig7) if fig7 is not None else None,
    }

    # Summary counts.
    manifest = {
        "phase": "Phase 5: Define causal adjustment framework and estimate trait-associated effects",
        "input_dataset": str(DATASET),
        "output_tables": {
            "table_trait_ols_models": str(OLS_OUT),
            "table_trait_ml_importance": str(ML_OUT),
            "table_trait_causal_adjusted_effects": str(DML_OUT),
            "table_trait_sensitivity_by_product_family": str(SENS_OUT),
        },
        "figures": figs,
        "n_points_total": int(df["point_id"].nunique()) if "point_id" in df.columns else int(len(df)),
        "dataset_shape": list(df.shape),
        "variable_sets": sets,
        "model_family_counts": {
            "ols_rows": int(len(ols_df)),
            "ml_rows": int(len(ml_df)),
            "dml_rows": int(len(dml_df)),
            "sensitivity_rows": int(len(sens_df)),
        },
        "successful_models": {
            "ols_ok_rows": int((ols_df["status"] == "ok").sum()) if "status" in ols_df.columns else 0,
            "ml_ok_rows": int((ml_df["status"] == "ok").sum()) if "status" in ml_df.columns else 0,
            "dml_ok_rows": int((dml_df["status"] == "ok").sum()) if "status" in dml_df.columns else 0,
            "sensitivity_ok_rows": int((sens_df["status"] == "ok").sum()) if "status" in sens_df.columns else 0,
        },
        "interpretation_guardrail": "Report trait-associated differences after climate/soil/spatial adjustment, not experimental causal proof.",
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"WROTE {MANIFEST_OUT}")

    make_readme(df, sets, ols_df, ml_df, dml_df, sens_df, figs, manifest)

    print("")
    print("DONE Phase 5.")
    print("")
    print("OUTPUT TABLES:")
    print(OLS_OUT)
    print(ML_OUT)
    print(DML_OUT)
    print(SENS_OUT)
    print("")
    print("FIGURES:")
    for k, v in figs.items():
        print(k, v)
    print("")
    print("MANIFEST:")
    print(json.dumps(manifest, indent=2, default=str))

if __name__ == "__main__":
    main()
