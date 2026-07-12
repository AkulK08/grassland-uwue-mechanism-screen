#!/usr/bin/env python
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# =============================================================================
# PHASE 6: Robustness and falsification tests
# PHASE 7: Physiological mechanism interpretation
# =============================================================================

BOOT_RAW = Path("results/project_final_nature_boot50/fullspec_response_results_raw.csv")
BOOT_CO2 = Path("results/project_final_nature_boot50/fullspec_response_results_co2corrected.csv")
TRAIT_DATASET = Path("results/trait_framework/trait_model_dataset.csv")
PHASE3_CONSENSUS = Path("results/trait_framework/point_product_consensus_response.csv")
PHASE3_COMBO = Path("results/trait_framework/phase3/point_product_combo_level_response.csv")

OUT6 = Path("results/trait_framework/phase6")
OUT7 = Path("results/trait_framework/phase7")
OUT6.mkdir(parents=True, exist_ok=True)
OUT7.mkdir(parents=True, exist_ok=True)

ALL_COEFF_OUT = OUT6 / "table_phase6_all_robustness_coefficients.csv"
RAW_CO2_OUT = OUT6 / "table_robustness_raw_vs_co2.csv"
METRIC_OUT = OUT6 / "table_robustness_metric_comparison.csv"
PRODUCT_OUT = OUT6 / "table_robustness_product_consensus_vs_specific.csv"
STRESS_SEASON_OUT = OUT6 / "table_robustness_stress_growing_season.csv"
ARIDITY_OUT = OUT6 / "table_robustness_aridity_stratified.csv"
NEGATIVE_OUT = OUT6 / "table_robustness_negative_controls.csv"
SETTING_MANIFEST_OUT = OUT6 / "table_phase6_robustness_settings_manifest.csv"

FIG3_OUT = OUT6 / "Figure3_robustness_coefficient_plot.png"
FIG3_PDF = OUT6 / "Figure3_robustness_coefficient_plot.pdf"

PHASE6_MANIFEST_OUT = OUT6 / "phase6_robustness_manifest.json"
PHASE6_README_OUT = OUT6 / "README_phase6_robustness_falsification.md"
PHASE7_INTERPRETATION_OUT = OUT7 / "physiological_mechanism_interpretation.md"
PHASE7_MANIFEST_OUT = OUT7 / "phase7_interpretation_manifest.json"

RANDOM_SEED = 42
MIN_N_OLS = 18
MIN_N_DML = 25
MAX_CV = 5

TRAITS = ["p50", "rooting_depth"]

SATBREAK_CLASSES = {"saturation", "breakdown"}

METRIC_ALIASES = {
    "uwue": "uWUE",
    "raw_wue": "raw WUE",
    "wue": "raw WUE",
    "iwue": "iWUE",
}

STRESS_DEFS_REQUIRED = [
    "zscore",
    "percentile_joint",
    "copula_joint",
    "interaction_surface",
]

SEASONS_REQUIRED = [
    "month_fixed",
    "gpp_threshold",
    "climate_common",
]

NEGATIVE_CONTROLS = [
    "raw_vs_co2_stability_all",
    "product_agreement_all",
    "n_product_combos_all",
]

PRODUCT_SPECIFIC_PREFIXES = [
    "slope_change_",
    "post_slope_",
    "satbreak_fraction_",
]

# =============================================================================
# Utility functions
# =============================================================================

def die(msg):
    raise SystemExit("\nERROR: " + str(msg) + "\n")

def make_unique_columns(df):
    df = df.copy()
    seen = {}
    cols = []
    for c in df.columns:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            cols.append(c)
        else:
            seen[c] += 1
            cols.append(f"{c}__dup{seen[c]}")
    df.columns = cols
    return df

def clean_df(df):
    return make_unique_columns(df)

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

def safe_median(s):
    x = to_num(s).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())

def safe_mean(s):
    x = to_num(s).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())

def direction(x):
    if pd.isna(x):
        return "NA"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"

def sign_value(x):
    if pd.isna(x):
        return np.nan
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_unique_columns(df)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def slug(s):
    return str(s).lower().replace("/", "_").replace(" ", "_").replace("-", "_")

def pval_from_t(t, df_resid):
    try:
        return float(2 * (1 - stats.t.cdf(abs(t), df=df_resid)))
    except Exception:
        return np.nan

def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))

# =============================================================================
# Load inputs
# =============================================================================

def load_trait_dataset():
    if not TRAIT_DATASET.exists():
        die(
            f"Missing Phase 4 trait model dataset:\n{TRAIT_DATASET}\n\n"
            "Run Phase 4 before Phase 6."
        )

    df = clean_df(pd.read_csv(TRAIT_DATASET, low_memory=False))

    if "point_id" not in df.columns:
        die("trait_model_dataset.csv is missing point_id.")

    df["point_id"] = df["point_id"].astype(str)

    for c in df.columns:
        if c.startswith("z_"):
            df[c] = to_num(df[c])

    numeric_candidates = [
        "p50",
        "rooting_depth",
        "isohydricity",
        "aridity",
        "mean_vpd",
        "p90_vpd",
        "mean_soil_moisture",
        "p10_soil_moisture",
        "mean_annual_precipitation",
        "mean_precipitation",
        "mean_annual_temperature",
        "mean_temperature",
        "mean_lai",
        "growing_season_mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "lat",
        "lon",
        "abs_lat",
        "raw_vs_co2_stability_all",
        "product_agreement_all",
        "n_product_combos_all",
    ]

    for c in numeric_candidates:
        if c in df.columns:
            df[c] = to_num(df[c])

    return df

def normalize_boot_columns(df):
    df = clean_df(df)

    aliases = {
        "class": "response_class_strict",
        "response_class": "response_class_strict",
        "response_class_final": "response_class_strict",
        "strict_class": "response_class_strict",
        "pre_slope_estimate": "pre_slope",
        "post_slope_estimate": "post_slope",
        "delta_slope": "slope_change",
        "slope_delta": "slope_change",
    }

    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df = df.rename(columns={old: new})

    required = ["point_id", "metric", "gpp_product", "et_product"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"BOOT response table missing required columns: {missing}")

    if "response_class_strict" not in df.columns:
        df["response_class_strict"] = "inconclusive"

    if "slope_change" not in df.columns:
        if "pre_slope" in df.columns and "post_slope" in df.columns:
            df["slope_change"] = to_num(df["post_slope"]) - to_num(df["pre_slope"])
        else:
            die("BOOT response table missing slope_change and cannot compute from pre_slope/post_slope.")

    if "post_slope" not in df.columns:
        df["post_slope"] = np.nan

    if "pre_slope" not in df.columns:
        df["pre_slope"] = np.nan

    df["point_id"] = df["point_id"].astype(str)
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["gpp_product"] = df["gpp_product"].astype(str).str.strip().str.upper()
    df["et_product"] = df["et_product"].astype(str).str.strip().str.upper()
    df["product_combo"] = df["gpp_product"] + "/" + df["et_product"]
    df["response_class_strict"] = df["response_class_strict"].astype(str).str.strip()

    if "stress_definition" in df.columns:
        df["stress_definition"] = df["stress_definition"].astype(str).str.strip()
    else:
        df["stress_definition"] = "unknown"

    if "growing_season" in df.columns:
        df["growing_season"] = df["growing_season"].astype(str).str.strip()
    else:
        df["growing_season"] = "unknown"

    for c in ["pre_slope", "post_slope", "slope_change"]:
        df[c] = to_num(df[c])

    df["sat_or_breakdown"] = df["response_class_strict"].isin(SATBREAK_CLASSES).astype(float)

    return df

def load_boot_results():
    if not BOOT_RAW.exists():
        die(f"Missing raw response file: {BOOT_RAW}")
    if not BOOT_CO2.exists():
        die(f"Missing CO2-corrected response file: {BOOT_CO2}")

    raw = normalize_boot_columns(pd.read_csv(BOOT_RAW, low_memory=False))
    raw["version"] = "raw"

    co2 = normalize_boot_columns(pd.read_csv(BOOT_CO2, low_memory=False))
    co2["version"] = "co2corrected"

    all_df = pd.concat([raw, co2], ignore_index=True)
    all_df = clean_df(all_df)

    return all_df

# =============================================================================
# Controls / design matrices
# =============================================================================

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
    out = []
    if has_col(df, "soil_sand", min_n=10):
        out.append("soil_sand")
    if has_col(df, "soil_clay", min_n=10):
        out.append("soil_clay")
    if not out and has_col(df, "soil_silt", min_n=10):
        out.append("soil_silt")
    return out

def choose_control_cols(trait_df):
    controls = []

    for c in [
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        choose_precip_col(trait_df),
        choose_temp_col(trait_df),
        choose_lai_col(trait_df),
    ]:
        if c is not None and has_col(trait_df, c, min_n=10) and c not in controls:
            controls.append(c)

    for c in choose_soil_cols(trait_df):
        if c not in controls:
            controls.append(c)

    for c in ["lat", "lon", "abs_lat"]:
        if has_col(trait_df, c, min_n=10) and c not in controls:
            controls.append(c)

    return controls

def build_design(df, numeric_cols, categorical_cols=None, standardize=True, add_intercept=True):
    categorical_cols = categorical_cols or []

    d = df.copy()

    numeric_cols = [c for c in numeric_cols if c in d.columns]
    categorical_cols = [c for c in categorical_cols if c in d.columns]

    X_parts = []

    if numeric_cols:
        Xn = pd.DataFrame(index=d.index)
        for c in numeric_cols:
            Xn[c] = to_num(d[c])
            if standardize:
                Xn[c] = zscore(Xn[c])
        X_parts.append(Xn)

    if categorical_cols:
        cat = d[categorical_cols].astype(str).fillna("missing")
        Xc = pd.get_dummies(cat, drop_first=True, dtype=float)
        X_parts.append(Xc)

    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=d.index)

    X = X.replace([np.inf, -np.inf], np.nan)

    keep = []
    for c in X.columns:
        x = to_num(X[c])
        if x.notna().sum() == 0:
            continue
        if x.nunique(dropna=True) <= 1:
            continue
        keep.append(c)

    X = X[keep].copy()

    if X.shape[1] > 0:
        X = X.fillna(X.median(numeric_only=True))

    if add_intercept:
        X = sm.add_constant(X, has_constant="add")

    return X

def complete_case(df, outcome, numeric_cols, categorical_cols=None):
    categorical_cols = categorical_cols or []

    if outcome not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    ok = to_num(d[outcome]).notna()

    for c in numeric_cols:
        if c not in d.columns:
            return pd.DataFrame()
        d[c] = to_num(d[c])
        ok &= d[c].notna()

    for c in categorical_cols:
        if c not in d.columns:
            return pd.DataFrame()
        ok &= d[c].notna()

    return d.loc[ok].copy()

# =============================================================================
# Aggregate response phenotypes for robustness settings
# =============================================================================

def aggregate_boot_response(boot_df, version=None, metric=None, product_combo=None, stress_definition=None, growing_season=None):
    df = boot_df.copy()

    if version is not None:
        df = df[df["version"].eq(version)]
    if metric is not None:
        df = df[df["metric"].eq(metric)]
    if product_combo is not None:
        df = df[df["product_combo"].eq(product_combo)]
    if stress_definition is not None:
        df = df[df["stress_definition"].eq(stress_definition)]
    if growing_season is not None:
        df = df[df["growing_season"].eq(growing_season)]

    if df.empty:
        return pd.DataFrame()

    # First collapse point × product_combo, so product combinations receive equal weight.
    combo = (
        df.groupby(["point_id", "product_combo"], dropna=False)
        .agg(
            slope_change_combo=("slope_change", safe_median),
            post_slope_combo=("post_slope", safe_median),
            satbreak_combo=("sat_or_breakdown", safe_mean),
            n_rows_combo=("point_id", "size"),
        )
        .reset_index()
    )

    # Then collapse across product combinations.
    point = (
        combo.groupby("point_id", dropna=False)
        .agg(
            slope_change=("slope_change_combo", safe_median),
            post_slope=("post_slope_combo", safe_median),
            satbreak_fraction=("satbreak_combo", safe_mean),
            n_product_combos=("product_combo", "nunique"),
            n_rows=("n_rows_combo", "sum"),
        )
        .reset_index()
    )

    return point

def merge_response_with_traits(response, trait_df):
    keep_trait_cols = ["point_id"] + [c for c in trait_df.columns if c != "point_id"]
    merged = response.merge(trait_df[keep_trait_cols], on="point_id", how="left")
    merged = clean_df(merged)
    return merged

def build_aridity_classes(trait_df):
    out = trait_df[["point_id"]].copy()

    if "aridity" in trait_df.columns and finite_n(trait_df, "aridity") >= 10:
        ar = to_num(trait_df["aridity"])

        try:
            out["aridity_class"] = pd.qcut(
                ar,
                q=3,
                labels=["dry", "semi_arid", "mesic"],
                duplicates="drop",
            ).astype(str)
        except Exception:
            out["aridity_class"] = "unknown"
    elif "aridity_quartile" in trait_df.columns:
        q = trait_df["aridity_quartile"].astype(str)
        out["aridity_class"] = np.where(
            q.str.contains("Q1", na=False),
            "dry",
            np.where(q.str.contains("Q2|Q3", regex=True, na=False), "semi_arid", "mesic"),
        )
    else:
        out["aridity_class"] = "unknown"

    out.loc[out["aridity_class"].isin(["nan", "NaN", "None", ""]), "aridity_class"] = "unknown"

    return out

# =============================================================================
# OLS and DML models for each robustness setting
# =============================================================================

def fit_ols_effects(model_df, outcome, setting_info, trait_df):
    rows = []

    controls = choose_control_cols(trait_df)
    traits = [t for t in TRAITS if t in model_df.columns and finite_n(model_df, t) >= 5]
    cats = []

    numeric = traits + controls
    numeric = list(dict.fromkeys([c for c in numeric if c in model_df.columns]))

    d = complete_case(model_df, outcome, numeric, cats)

    if d.empty or len(d) < MIN_N_OLS or not traits:
        for t in TRAITS:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": t,
                "model_family": "OLS_HC3",
                "n": int(len(d)) if not d.empty else 0,
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2": np.nan,
                "direction": "NA",
                "status": "skipped_low_n_or_missing_trait",
                "controls_used": ",".join(controls),
            })
        return rows

    y = zscore(d[outcome])
    X = build_design(d, numeric, cats, standardize=True, add_intercept=True)

    ok = y.notna()
    y = y.loc[ok]
    X = X.loc[ok]

    if len(y) < MIN_N_OLS or X.shape[1] < 2:
        for t in TRAITS:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": t,
                "model_family": "OLS_HC3",
                "n": int(len(y)),
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2": np.nan,
                "direction": "NA",
                "status": "skipped_low_n_or_no_design",
                "controls_used": ",".join(controls),
            })
        return rows

    try:
        fit0 = sm.OLS(y, X).fit()
        fit = fit0.get_robustcov_results(cov_type="HC3")

        params = pd.Series(fit.params, index=X.columns)
        bse = pd.Series(fit.bse, index=X.columns)
        tvals = pd.Series(fit.tvalues, index=X.columns)
        pvals = pd.Series(fit.pvalues, index=X.columns)
        ci = pd.DataFrame(fit.conf_int(), index=X.columns, columns=["ci_low", "ci_high"])

        for t in TRAITS:
            if t in params.index:
                est = float(params.loc[t])
                rows.append({
                    **setting_info,
                    "outcome": outcome,
                    "trait": t,
                    "model_family": "OLS_HC3",
                    "n": int(len(y)),
                    "estimate": est,
                    "std_error": float(bse.loc[t]),
                    "t_stat": float(tvals.loc[t]),
                    "p_value": float(pvals.loc[t]),
                    "ci_low": float(ci.loc[t, "ci_low"]),
                    "ci_high": float(ci.loc[t, "ci_high"]),
                    "r2": float(fit0.rsquared),
                    "direction": direction(est),
                    "status": "ok",
                    "controls_used": ",".join(controls),
                })
            else:
                rows.append({
                    **setting_info,
                    "outcome": outcome,
                    "trait": t,
                    "model_family": "OLS_HC3",
                    "n": int(len(y)),
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r2": float(fit0.rsquared),
                    "direction": "NA",
                    "status": "trait_not_in_design",
                    "controls_used": ",".join(controls),
                })

    except Exception as e:
        for t in TRAITS:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": t,
                "model_family": "OLS_HC3",
                "n": int(len(y)),
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2": np.nan,
                "direction": "NA",
                "status": f"failed: {type(e).__name__}: {e}",
                "controls_used": ",".join(controls),
            })

    return rows

def crossfit_pred_rf(X, y):
    n = len(y)
    k = min(MAX_CV, n)
    if k < 3:
        return pd.Series(np.nan, index=y.index), np.nan

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    pred = pd.Series(np.nan, index=y.index, dtype=float)

    for fold, (tr, te) in enumerate(kf.split(X)):
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED + fold,
            min_samples_leaf=3,
            max_features="sqrt",
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = model.predict(X.iloc[te])

    try:
        r2 = float(r2_score(y, pred))
    except Exception:
        r2 = np.nan

    return pred, r2

def fit_dml_effects(model_df, outcome, setting_info, trait_df):
    rows = []

    controls = choose_control_cols(trait_df)

    for trait in TRAITS:
        if trait not in model_df.columns:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": trait,
                "model_family": "DML_RF",
                "n": 0,
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2_y_nuisance": np.nan,
                "r2_t_nuisance": np.nan,
                "direction": "NA",
                "status": "trait_missing",
                "controls_used": ",".join(controls),
            })
            continue

        other_traits = [t for t in TRAITS if t != trait and t in model_df.columns]
        controls_use = list(dict.fromkeys(controls + other_traits))
        numeric = [trait] + controls_use

        d = complete_case(model_df, outcome, numeric, [])

        if d.empty or len(d) < MIN_N_DML:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": trait,
                "model_family": "DML_RF",
                "n": int(len(d)) if not d.empty else 0,
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2_y_nuisance": np.nan,
                "r2_t_nuisance": np.nan,
                "direction": "NA",
                "status": "skipped_low_n",
                "controls_used": ",".join(controls_use),
            })
            continue

        try:
            y = zscore(d[outcome])
            t = zscore(d[trait])
            X = build_design(d, controls_use, [], standardize=True, add_intercept=False)

            if X.shape[1] == 0:
                y_hat = pd.Series(y.mean(), index=y.index)
                t_hat = pd.Series(t.mean(), index=t.index)
                r2y = 0.0
                r2t = 0.0
            else:
                y_hat, r2y = crossfit_pred_rf(X, y)
                t_hat, r2t = crossfit_pred_rf(X, t)

            y_res = y - y_hat
            t_res = t - t_hat

            final_X = sm.add_constant(pd.DataFrame({trait: t_res}, index=d.index), has_constant="add")
            final_fit0 = sm.OLS(y_res, final_X).fit()
            final_fit = final_fit0.get_robustcov_results(cov_type="HC3")

            params = pd.Series(final_fit.params, index=final_X.columns)
            bse = pd.Series(final_fit.bse, index=final_X.columns)
            tvals = pd.Series(final_fit.tvalues, index=final_X.columns)
            pvals = pd.Series(final_fit.pvalues, index=final_X.columns)
            ci = pd.DataFrame(final_fit.conf_int(), index=final_X.columns, columns=["ci_low", "ci_high"])

            est = float(params.loc[trait])

            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": trait,
                "model_family": "DML_RF",
                "n": int(len(d)),
                "estimate": est,
                "std_error": float(bse.loc[trait]),
                "t_stat": float(tvals.loc[trait]),
                "p_value": float(pvals.loc[trait]),
                "ci_low": float(ci.loc[trait, "ci_low"]),
                "ci_high": float(ci.loc[trait, "ci_high"]),
                "r2_y_nuisance": float(r2y),
                "r2_t_nuisance": float(r2t),
                "direction": direction(est),
                "status": "ok",
                "controls_used": ",".join(controls_use),
            })

        except Exception as e:
            rows.append({
                **setting_info,
                "outcome": outcome,
                "trait": trait,
                "model_family": "DML_RF",
                "n": int(len(d)),
                "estimate": np.nan,
                "std_error": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2_y_nuisance": np.nan,
                "r2_t_nuisance": np.nan,
                "direction": "NA",
                "status": f"failed: {type(e).__name__}: {e}",
                "controls_used": ",".join(controls_use),
            })

    return rows

def run_models_for_setting(setting_df, setting_info, trait_df, outcomes):
    rows = []

    for outcome in outcomes:
        if outcome not in setting_df.columns:
            continue

        rows.extend(fit_ols_effects(setting_df, outcome, setting_info, trait_df))
        rows.extend(fit_dml_effects(setting_df, outcome, setting_info, trait_df))

    return rows

# =============================================================================
# Build robustness settings
# =============================================================================

def setting_info(group, setting, detail="", version="", metric="", product_combo="", stress_definition="", growing_season="", aridity_class="", outcome_source=""):
    return {
        "robustness_group": group,
        "setting": setting,
        "setting_detail": detail,
        "version": version,
        "metric": metric,
        "product_combo": product_combo,
        "stress_definition": stress_definition,
        "growing_season": growing_season,
        "aridity_class": aridity_class,
        "outcome_source": outcome_source,
    }

def add_available_setting(settings_manifest, group, setting, description, n_points, notes=""):
    settings_manifest.append({
        "robustness_group": group,
        "setting": setting,
        "description": description,
        "n_points_available": int(n_points) if pd.notna(n_points) else 0,
        "notes": notes,
    })

def build_boot_setting_dataset(boot_df, trait_df, info, version=None, metric=None, product_combo=None, stress_definition=None, growing_season=None):
    resp = aggregate_boot_response(
        boot_df,
        version=version,
        metric=metric,
        product_combo=product_combo,
        stress_definition=stress_definition,
        growing_season=growing_season,
    )

    if resp.empty:
        return pd.DataFrame()

    merged = merge_response_with_traits(resp, trait_df)
    return merged

def build_negative_control_dataset(trait_df, outcome_col):
    if outcome_col not in trait_df.columns:
        return pd.DataFrame()

    cols = ["point_id", outcome_col] + [c for c in trait_df.columns if c != "point_id" and c != outcome_col]
    d = trait_df[cols].copy()
    d = d.rename(columns={outcome_col: "negative_control"})
    return d

def build_phase6_coefficients(boot_df, trait_df):
    rows = []
    settings_manifest = []

    # ---------------------------------------------------------------------
    # 1. Raw vs CO2-corrected response.
    # ---------------------------------------------------------------------
    for version in ["raw", "co2corrected"]:
        info = setting_info(
            group="raw_vs_co2",
            setting=f"{version}_uwue_consensus",
            detail=f"{version} uWUE product-consensus response",
            version=version,
            metric="uwue",
            outcome_source="boot_response_product_consensus",
        )
        d = build_boot_setting_dataset(
            boot_df,
            trait_df,
            info,
            version=version,
            metric="uwue",
        )
        add_available_setting(settings_manifest, "raw_vs_co2", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
        rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # ---------------------------------------------------------------------
    # 2. uWUE vs raw WUE vs iWUE.
    # ---------------------------------------------------------------------
    available_metrics = sorted(boot_df["metric"].dropna().unique().tolist())
    target_metrics = []
    for m in ["uwue", "raw_wue", "wue", "iwue"]:
        if m in available_metrics and m not in target_metrics:
            target_metrics.append(m)

    for metric in target_metrics:
        for version in ["co2corrected", "raw"]:
            if len(boot_df[(boot_df["metric"].eq(metric)) & (boot_df["version"].eq(version))]) == 0:
                continue

            label = METRIC_ALIASES.get(metric, metric)
            info = setting_info(
                group="metric_comparison",
                setting=f"{version}_{metric}_consensus",
                detail=f"{version} {label} product-consensus response",
                version=version,
                metric=metric,
                outcome_source="boot_response_product_consensus",
            )
            d = build_boot_setting_dataset(
                boot_df,
                trait_df,
                info,
                version=version,
                metric=metric,
            )
            add_available_setting(settings_manifest, "metric_comparison", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
            rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # ---------------------------------------------------------------------
    # 3. Product-consensus vs product-specific response.
    # ---------------------------------------------------------------------
    # Product consensus main.
    info = setting_info(
        group="product_consensus_vs_specific",
        setting="co2_uwue_product_consensus_all",
        detail="CO2-corrected uWUE all-product consensus",
        version="co2corrected",
        metric="uwue",
        outcome_source="boot_response_product_consensus",
    )
    d = build_boot_setting_dataset(boot_df, trait_df, info, version="co2corrected", metric="uwue")
    add_available_setting(settings_manifest, "product_consensus_vs_specific", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
    rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # Product-specific combos.
    combos = sorted(boot_df.loc[(boot_df["version"].eq("co2corrected")) & (boot_df["metric"].eq("uwue")), "product_combo"].dropna().unique().tolist())
    for combo in combos:
        info = setting_info(
            group="product_consensus_vs_specific",
            setting=f"co2_uwue_product_specific_{slug(combo)}",
            detail=f"CO2-corrected uWUE product-specific response: {combo}",
            version="co2corrected",
            metric="uwue",
            product_combo=combo,
            outcome_source="boot_response_product_specific",
        )
        d = build_boot_setting_dataset(
            boot_df,
            trait_df,
            info,
            version="co2corrected",
            metric="uwue",
            product_combo=combo,
        )
        add_available_setting(settings_manifest, "product_consensus_vs_specific", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
        rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # ---------------------------------------------------------------------
    # 4. Stress definition robustness.
    # ---------------------------------------------------------------------
    available_stress = sorted(boot_df.loc[(boot_df["version"].eq("co2corrected")) & (boot_df["metric"].eq("uwue")), "stress_definition"].dropna().unique().tolist())
    for stress_def in STRESS_DEFS_REQUIRED:
        if stress_def not in available_stress:
            add_available_setting(settings_manifest, "stress_definition", f"missing_{stress_def}", f"Stress definition missing: {stress_def}", 0, notes="missing in BOOT table")
            continue

        info = setting_info(
            group="stress_definition",
            setting=f"stress_{stress_def}",
            detail=f"CO2-corrected uWUE response using stress definition: {stress_def}",
            version="co2corrected",
            metric="uwue",
            stress_definition=stress_def,
            outcome_source="boot_response_stress_filtered",
        )
        d = build_boot_setting_dataset(
            boot_df,
            trait_df,
            info,
            version="co2corrected",
            metric="uwue",
            stress_definition=stress_def,
        )
        add_available_setting(settings_manifest, "stress_definition", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
        rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # ---------------------------------------------------------------------
    # 5. Growing-season robustness.
    # ---------------------------------------------------------------------
    available_seasons = sorted(boot_df.loc[(boot_df["version"].eq("co2corrected")) & (boot_df["metric"].eq("uwue")), "growing_season"].dropna().unique().tolist())
    for season in SEASONS_REQUIRED:
        if season not in available_seasons:
            add_available_setting(settings_manifest, "growing_season", f"missing_{season}", f"Growing season missing: {season}", 0, notes="missing in BOOT table")
            continue

        info = setting_info(
            group="growing_season",
            setting=f"season_{season}",
            detail=f"CO2-corrected uWUE response using growing season: {season}",
            version="co2corrected",
            metric="uwue",
            growing_season=season,
            outcome_source="boot_response_season_filtered",
        )
        d = build_boot_setting_dataset(
            boot_df,
            trait_df,
            info,
            version="co2corrected",
            metric="uwue",
            growing_season=season,
        )
        add_available_setting(settings_manifest, "growing_season", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
        rows.extend(run_models_for_setting(d, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

    # ---------------------------------------------------------------------
    # 6. Aridity-stratified models.
    # ---------------------------------------------------------------------
    aridity_classes = build_aridity_classes(trait_df)
    main_resp = aggregate_boot_response(boot_df, version="co2corrected", metric="uwue")
    if not main_resp.empty:
        main_merged = merge_response_with_traits(main_resp, trait_df)
        main_merged = main_merged.merge(aridity_classes, on="point_id", how="left")

        for klass in ["dry", "semi_arid", "mesic"]:
            sub = main_merged[main_merged["aridity_class"].eq(klass)].copy()
            info = setting_info(
                group="aridity_stratified",
                setting=f"aridity_{klass}",
                detail=f"CO2-corrected uWUE response within aridity class: {klass}",
                version="co2corrected",
                metric="uwue",
                aridity_class=klass,
                outcome_source="boot_response_product_consensus_aridity_stratum",
            )
            add_available_setting(settings_manifest, "aridity_stratified", info["setting"], info["setting_detail"], sub["point_id"].nunique() if not sub.empty else 0)
            rows.extend(run_models_for_setting(sub, info, trait_df, ["slope_change", "post_slope", "satbreak_fraction"]))

        # Interaction model: trait × aridity numeric / groups. OLS only, explicit test.
        if "aridity" in main_merged.columns:
            interaction_rows = fit_aridity_interaction_models(main_merged, trait_df)
            rows.extend(interaction_rows)

    # ---------------------------------------------------------------------
    # 7. Negative-control outcomes.
    # ---------------------------------------------------------------------
    for nc in NEGATIVE_CONTROLS:
        if nc not in trait_df.columns:
            add_available_setting(settings_manifest, "negative_control", f"missing_{nc}", f"Negative control missing: {nc}", 0, notes="missing in trait dataset")
            continue

        d = build_negative_control_dataset(trait_df, nc)
        info = setting_info(
            group="negative_control",
            setting=f"negative_control_{nc}",
            detail=f"Negative-control outcome: {nc}",
            outcome_source="trait_model_dataset_negative_control",
        )
        add_available_setting(settings_manifest, "negative_control", info["setting"], info["setting_detail"], d["point_id"].nunique() if not d.empty else 0)
        rows.extend(run_models_for_setting(d, info, trait_df, ["negative_control"]))

    coeff = pd.DataFrame(rows)
    settings_df = pd.DataFrame(settings_manifest)

    return coeff, settings_df

def fit_aridity_interaction_models(model_df, trait_df):
    rows = []
    controls = choose_control_cols(trait_df)
    controls_no_aridity = [c for c in controls if c != "aridity"]

    for outcome in ["slope_change", "post_slope", "satbreak_fraction"]:
        if outcome not in model_df.columns:
            continue

        needed = [outcome, "aridity"] + [t for t in TRAITS if t in model_df.columns] + [c for c in controls_no_aridity if c in model_df.columns]
        d = complete_case(model_df, outcome, needed, [])

        if d.empty or len(d) < MIN_N_OLS:
            for trait in TRAITS:
                rows.append({
                    **setting_info(
                        group="aridity_interaction",
                        setting="trait_x_aridity_interaction",
                        detail="Trait × aridity interaction model",
                        version="co2corrected",
                        metric="uwue",
                        outcome_source="boot_response_aridity_interaction",
                    ),
                    "outcome": outcome,
                    "trait": trait,
                    "model_family": "OLS_HC3_interaction",
                    "n": int(len(d)) if not d.empty else 0,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r2": np.nan,
                    "direction": "NA",
                    "status": "skipped_low_n",
                    "controls_used": ",".join(controls),
                })
            continue

        try:
            # Standardize variables.
            work = d.copy()
            work[outcome] = zscore(work[outcome])
            work["aridity"] = zscore(work["aridity"])

            terms = []
            for trait in TRAITS:
                if trait in work.columns:
                    work[trait] = zscore(work[trait])
                    inter = f"{trait}_x_aridity"
                    work[inter] = work[trait] * work["aridity"]
                    terms.extend([trait, inter])

            for c in controls_no_aridity:
                if c in work.columns:
                    work[c] = zscore(work[c])
                    terms.append(c)

            terms = list(dict.fromkeys([t for t in terms if t in work.columns]))
            X = build_design(work, terms + ["aridity"], [], standardize=False, add_intercept=True)
            y = to_num(work[outcome])

            ok = y.notna()
            X = X.loc[ok]
            y = y.loc[ok]

            fit0 = sm.OLS(y, X).fit()
            fit = fit0.get_robustcov_results(cov_type="HC3")

            params = pd.Series(fit.params, index=X.columns)
            bse = pd.Series(fit.bse, index=X.columns)
            tvals = pd.Series(fit.tvalues, index=X.columns)
            pvals = pd.Series(fit.pvalues, index=X.columns)
            ci = pd.DataFrame(fit.conf_int(), index=X.columns, columns=["ci_low", "ci_high"])

            for trait in TRAITS:
                inter = f"{trait}_x_aridity"
                for term in [trait, inter]:
                    if term in params.index:
                        est = float(params.loc[term])
                        rows.append({
                            **setting_info(
                                group="aridity_interaction",
                                setting="trait_x_aridity_interaction",
                                detail="Trait × continuous aridity interaction model",
                                version="co2corrected",
                                metric="uwue",
                                outcome_source="boot_response_aridity_interaction",
                            ),
                            "outcome": outcome,
                            "trait": term,
                            "model_family": "OLS_HC3_interaction",
                            "n": int(len(y)),
                            "estimate": est,
                            "std_error": float(bse.loc[term]),
                            "t_stat": float(tvals.loc[term]),
                            "p_value": float(pvals.loc[term]),
                            "ci_low": float(ci.loc[term, "ci_low"]),
                            "ci_high": float(ci.loc[term, "ci_high"]),
                            "r2": float(fit0.rsquared),
                            "direction": direction(est),
                            "status": "ok",
                            "controls_used": ",".join(controls),
                        })

        except Exception as e:
            for trait in TRAITS:
                rows.append({
                    **setting_info(
                        group="aridity_interaction",
                        setting="trait_x_aridity_interaction",
                        detail="Trait × continuous aridity interaction model",
                        version="co2corrected",
                        metric="uwue",
                        outcome_source="boot_response_aridity_interaction",
                    ),
                    "outcome": outcome,
                    "trait": trait,
                    "model_family": "OLS_HC3_interaction",
                    "n": int(len(d)),
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r2": np.nan,
                    "direction": "NA",
                    "status": f"failed: {type(e).__name__}: {e}",
                    "controls_used": ",".join(controls),
                })

    return rows

# =============================================================================
# Robustness summaries and Phase 7 interpretation
# =============================================================================

def add_direction_consistency(coeff):
    coeff = coeff.copy()

    # Main reference: OLS, CO2 uWUE product consensus, slope_change.
    ref = coeff[
        (coeff["model_family"].eq("OLS_HC3")) &
        (coeff["status"].eq("ok")) &
        (coeff["setting"].eq("co2_uwue_product_consensus_all")) &
        (coeff["outcome"].eq("slope_change")) &
        (coeff["trait"].isin(TRAITS))
    ].copy()

    ref_dir = {}
    for _, r in ref.iterrows():
        ref_dir[r["trait"]] = direction(r["estimate"])

    same = []
    ref_direction = []

    for _, r in coeff.iterrows():
        trait = r["trait"]
        d = direction(r["estimate"])
        rd = ref_dir.get(trait, "NA")
        ref_direction.append(rd)
        if rd == "NA" or d == "NA":
            same.append(np.nan)
        else:
            same.append(bool(d == rd))

    coeff["reference_direction"] = ref_direction
    coeff["same_direction_as_main_co2_uwue_consensus"] = same

    return coeff

def summarize_robustness(coeff):
    ok = coeff[
        (coeff["status"].eq("ok")) &
        (coeff["model_family"].eq("OLS_HC3")) &
        (coeff["trait"].isin(TRAITS)) &
        (~coeff["robustness_group"].eq("negative_control")) &
        (~coeff["robustness_group"].eq("aridity_interaction"))
    ].copy()

    rows = []

    for trait in TRAITS:
        d = ok[ok["trait"].eq(trait)].copy()
        if d.empty:
            rows.append({
                "trait": trait,
                "n_robustness_coefficients": 0,
                "direction_consistency_fraction": np.nan,
                "median_abs_effect": np.nan,
                "n_nominal_p_lt_0p10": 0,
                "status": "no_ok_coefficients",
            })
            continue

        sc = d["same_direction_as_main_co2_uwue_consensus"]
        sc = sc.dropna()

        rows.append({
            "trait": trait,
            "n_robustness_coefficients": int(len(d)),
            "direction_consistency_fraction": float(sc.mean()) if len(sc) else np.nan,
            "median_abs_effect": float(d["estimate"].abs().median()),
            "n_nominal_p_lt_0p10": int((d["p_value"] < 0.10).sum()),
            "status": "ok",
        })

    return pd.DataFrame(rows)

def detect_negative_control_warning(coeff):
    nc = coeff[
        (coeff["robustness_group"].eq("negative_control")) &
        (coeff["status"].eq("ok")) &
        (coeff["model_family"].eq("OLS_HC3")) &
        (coeff["trait"].isin(TRAITS))
    ].copy()

    if nc.empty:
        return {
            "warning": False,
            "reason": "No successful negative-control OLS coefficients.",
            "n_warning_coefficients": 0,
        }

    warning_rows = nc[(nc["estimate"].abs() >= 0.30) & (nc["p_value"] < 0.10)].copy()

    return {
        "warning": bool(len(warning_rows) > 0),
        "reason": "Traits predict negative-control outcomes with |standardized effect| >= 0.30 and p < 0.10." if len(warning_rows) else "No strong negative-control warning by threshold.",
        "n_warning_coefficients": int(len(warning_rows)),
        "warning_rows": warning_rows[[
            "setting",
            "outcome",
            "trait",
            "estimate",
            "p_value",
            "n",
            "status",
        ]].to_dict(orient="records"),
    }

def classify_support(summary, negative_warning):
    if summary.empty:
        return "weak_support"

    vals = summary["direction_consistency_fraction"].dropna()
    if len(vals) == 0:
        return "weak_support"

    median_consistency = float(vals.median())

    if negative_warning.get("warning", False):
        if median_consistency >= 0.67:
            return "moderate_support_with_negative_control_warning"
        return "weak_support_with_negative_control_warning"

    if median_consistency >= 0.70:
        return "strong_support"
    if median_consistency >= 0.50:
        return "moderate_support"
    return "weak_support"

def write_phase7_interpretation(coeff, summary, negative_warning, support_level):
    lines = []
    lines.append("# Phase 7: Physiological mechanism interpretation")
    lines.append("")
    lines.append("## Mechanistic question")
    lines.append("")
    lines.append("Do hydraulic and rooting traits explain spatial variation in the satellite-derived WUE stress-response phenotype after climate, soil, and spatial adjustment?")
    lines.append("")
    lines.append("## Main expected mechanism")
    lines.append("")
    lines.append("If the trait models are robust, the biological interpretation is that more drought-resistant xylem and deeper rooting systems maintain WUE sensitivity under high compound atmospheric-soil moisture stress, while more vulnerable or shallow-rooted systems show stronger weakening of the uWUE response at high stress.")
    lines.append("")
    lines.append("## Acceptable claim")
    lines.append("")
    lines.append("The satellite-derived WUE response phenotype is statistically consistent with a hydraulic/rooting mechanism after climate and soil adjustment.")
    lines.append("")
    lines.append("## Claim to avoid")
    lines.append("")
    lines.append("Do not claim that xylem vulnerability causally proves WUE breakdown. Without tower validation, the result remains a satellite-derived, trait-adjusted association consistent with a physiological mechanism.")
    lines.append("")
    lines.append("## Robustness classification")
    lines.append("")
    lines.append(f"`{support_level}`")
    lines.append("")
    lines.append("## Trait robustness summary")
    lines.append("")
    if summary.empty:
        lines.append("No successful robustness coefficients were available.")
    else:
        lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("## Negative-control check")
    lines.append("")
    lines.append(json.dumps(negative_warning, indent=2, default=str))
    lines.append("")
    lines.append("## Interpretation selected from robustness results")
    lines.append("")

    if support_level == "strong_support":
        lines.append("Across product, metric, stress-definition, growing-season, and aridity robustness settings, the trait effects were directionally stable and the negative-control tests did not strongly warn of spatial confounding. This supports the interpretation that the satellite-derived WUE stress-response phenotype behaves like an ecosystem-scale physiological phenotype shaped by plant water-transport and rooting strategy.")
    elif support_level == "moderate_support":
        lines.append("Trait associations were directionally coherent across a meaningful fraction of robustness settings, but not uniformly stable. This supports a cautious hydraulic/rooting interpretation, while emphasizing that product uncertainty, stress definition, growing-season choice, and sample size remain important limits.")
    elif support_level == "moderate_support_with_negative_control_warning":
        lines.append("Trait associations showed some directional robustness, but negative-control tests produced warnings. The physiological interpretation should be cautious: the signal is compatible with hydraulic/rooting mechanisms, but residual spatial or product confounding may remain.")
    elif support_level == "weak_support_with_negative_control_warning":
        lines.append("Trait associations were not robust and negative-control tests warned of possible confounding. The apparent trait signal should not be interpreted as strong physiological evidence.")
    else:
        lines.append("Trait associations were not stable across robustness settings. This suggests that the apparent trait signal may reflect spatial/product confounding, limitations of global trait maps, incomplete product support, or metric/stress-definition sensitivity rather than a robust hydraulic mechanism.")

    lines.append("")
    lines.append("## Final safe manuscript wording")
    lines.append("")
    if support_level in ["strong_support", "moderate_support", "moderate_support_with_negative_control_warning"]:
        lines.append("After climate, soil, and spatial adjustment, hydraulic and rooting traits showed trait-associated differences in the product-consensus WUE response phenotype. These associations are consistent with a hydraulic/rooting interpretation of grassland WUE stress response, but tower validation remains necessary for direct physiological confirmation.")
    else:
        lines.append("After robustness and falsification testing, the current satellite-derived trait associations are not sufficiently stable to support a strong hydraulic/rooting mechanism claim. The analysis should be framed as a test of whether current global trait maps can explain WUE response heterogeneity, rather than as confirmation of a physiological mechanism.")

    PHASE7_INTERPRETATION_OUT.write_text("\n".join(lines))
    print(f"WROTE {PHASE7_INTERPRETATION_OUT}")

    manifest = {
        "support_level": support_level,
        "negative_control_warning": negative_warning,
        "summary": summary.to_dict(orient="records"),
        "interpretation_file": str(PHASE7_INTERPRETATION_OUT),
        "acceptable_claim": "The satellite-derived WUE response phenotype is statistically consistent with a hydraulic/rooting mechanism after climate and soil adjustment.",
        "claim_to_avoid": "Xylem vulnerability causally proves WUE breakdown.",
    }

    with open(PHASE7_MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"WROTE {PHASE7_MANIFEST_OUT}")

    return manifest

# =============================================================================
# Figures
# =============================================================================

def plot_figure3(coeff):
    d = coeff[
        (coeff["status"].eq("ok")) &
        (coeff["model_family"].eq("OLS_HC3")) &
        (coeff["trait"].isin(TRAITS)) &
        (coeff["outcome"].isin(["slope_change", "post_slope", "satbreak_fraction"])) &
        (~coeff["robustness_group"].eq("negative_control")) &
        (~coeff["robustness_group"].eq("aridity_interaction"))
    ].copy()

    if d.empty:
        return None

    # Make concise labels.
    d["x_label"] = d["setting"].astype(str)

    # Limit label length but keep uniqueness.
    d["x_label"] = d["x_label"].str.replace("co2corrected", "co2", regex=False)
    d["x_label"] = d["x_label"].str.replace("co2_uwue_", "", regex=False)
    d["x_label"] = d["x_label"].str.replace("product_specific_", "prod_", regex=False)
    d["x_label"] = d["x_label"].str.replace("product_consensus_", "cons_", regex=False)
    d["x_label"] = d["x_label"].str.replace("stress_", "stress:", regex=False)
    d["x_label"] = d["x_label"].str.replace("season_", "season:", regex=False)
    d["x_label"] = d["x_label"].str.replace("aridity_", "arid:", regex=False)
    d["x_label"] = d["x_label"].str.replace("metric_", "", regex=False)

    outcomes = ["slope_change", "post_slope", "satbreak_fraction"]
    traits = TRAITS

    fig, axes = plt.subplots(len(outcomes), 1, figsize=(18, 13), sharex=False)

    if len(outcomes) == 1:
        axes = [axes]

    for ax, outcome in zip(axes, outcomes):
        sub = d[d["outcome"].eq(outcome)].copy()

        if sub.empty:
            ax.set_title(outcome + " (no successful models)")
            ax.axhline(0, linestyle="--", linewidth=1)
            continue

        # Order settings by robustness group then setting.
        setting_order = (
            sub[["robustness_group", "setting", "x_label"]]
            .drop_duplicates()
            .sort_values(["robustness_group", "setting"])
        )
        labels = setting_order["x_label"].tolist()
        label_to_x = {lab: i for i, lab in enumerate(labels)}
        setting_to_label = dict(zip(setting_order["setting"], setting_order["x_label"]))

        for j, trait in enumerate(traits):
            ss = sub[sub["trait"].eq(trait)].copy()
            if ss.empty:
                continue

            xs = []
            ys = []
            yerr_low = []
            yerr_high = []

            for _, r in ss.iterrows():
                lab = setting_to_label.get(r["setting"], r["setting"])
                x = label_to_x.get(lab, np.nan)
                if pd.isna(x):
                    continue
                est = r["estimate"]
                lo = r["ci_low"]
                hi = r["ci_high"]
                if pd.isna(est):
                    continue
                xs.append(x + (j - 0.5) * 0.18)
                ys.append(est)
                yerr_low.append(est - lo if pd.notna(lo) else 0)
                yerr_high.append(hi - est if pd.notna(hi) else 0)

            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=np.vstack([yerr_low, yerr_high]),
                    fmt="o",
                    capsize=2,
                    label=trait,
                )

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_ylabel("standardized effect")
        ax.set_title(outcome)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2, frameon=False)
    fig.suptitle("Figure 3. Robustness of P50/rooting-depth effects across falsification settings", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG3_OUT, dpi=300)
    fig.savefig(FIG3_PDF)
    plt.close(fig)

    return FIG3_OUT

# =============================================================================
# README / manifest
# =============================================================================

def write_readme(coeff, settings_df, summary, negative_warning, phase7_manifest):
    lines = []
    lines.append("# Phase 6: Robustness and falsification tests")
    lines.append("")
    lines.append("## Goal")
    lines.append("")
    lines.append("Show that trait associations are not artifacts of product choice, CO2 correction, WUE metric, stress definition, growing-season definition, aridity structure, or negative-control confounding.")
    lines.append("")
    lines.append("## Robustness tests executed")
    lines.append("")
    lines.append("1. Raw vs CO2-corrected response.")
    lines.append("2. uWUE vs raw WUE vs iWUE.")
    lines.append("3. Product-consensus vs product-specific response.")
    lines.append("4. Stress definition robustness: zscore, percentile_joint, copula_joint, interaction_surface.")
    lines.append("5. Growing-season robustness: month_fixed, gpp_threshold, climate_common.")
    lines.append("6. Aridity-stratified models: dry, semi_arid, mesic.")
    lines.append("7. Trait × aridity interaction models.")
    lines.append("8. Negative-control outcomes: raw_vs_co2_stability_all, product_agreement_all, n_product_combos_all.")
    lines.append("")
    lines.append("## Output tables")
    lines.append("")
    for p in [
        ALL_COEFF_OUT,
        RAW_CO2_OUT,
        METRIC_OUT,
        PRODUCT_OUT,
        STRESS_SEASON_OUT,
        ARIDITY_OUT,
        NEGATIVE_OUT,
        SETTING_MANIFEST_OUT,
    ]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Output figures")
    lines.append("")
    lines.append(f"- `{FIG3_OUT}`")
    lines.append(f"- `{FIG3_PDF}`")
    lines.append("")
    lines.append("## Phase 7 interpretation")
    lines.append("")
    lines.append(f"- `{PHASE7_INTERPRETATION_OUT}`")
    lines.append("")
    lines.append("## Robustness settings manifest")
    lines.append("")
    lines.append(settings_df.to_string(index=False))
    lines.append("")
    lines.append("## Trait robustness summary")
    lines.append("")
    lines.append(summary.to_string(index=False) if not summary.empty else "No summary available.")
    lines.append("")
    lines.append("## Negative-control warning")
    lines.append("")
    lines.append(json.dumps(negative_warning, indent=2, default=str))
    lines.append("")
    lines.append("## Phase 7 classification")
    lines.append("")
    lines.append(json.dumps(phase7_manifest, indent=2, default=str))

    PHASE6_README_OUT.write_text("\n".join(lines))
    print(f"WROTE {PHASE6_README_OUT}")

def build_manifest(coeff, settings_df, summary, negative_warning, fig_path, phase7_manifest):
    manifest = {
        "phase": "Phase 6 robustness + Phase 7 physiological interpretation",
        "inputs": {
            "boot_raw": str(BOOT_RAW),
            "boot_co2": str(BOOT_CO2),
            "trait_dataset": str(TRAIT_DATASET),
            "phase3_consensus": str(PHASE3_CONSENSUS),
            "phase3_combo": str(PHASE3_COMBO),
        },
        "outputs": {
            "all_coefficients": str(ALL_COEFF_OUT),
            "raw_vs_co2": str(RAW_CO2_OUT),
            "metric_comparison": str(METRIC_OUT),
            "product_consensus_vs_specific": str(PRODUCT_OUT),
            "stress_growing_season": str(STRESS_SEASON_OUT),
            "aridity_stratified": str(ARIDITY_OUT),
            "negative_controls": str(NEGATIVE_OUT),
            "settings_manifest": str(SETTING_MANIFEST_OUT),
            "figure3": str(fig_path) if fig_path is not None else None,
            "phase7_interpretation": str(PHASE7_INTERPRETATION_OUT),
        },
        "n_coefficients_total": int(len(coeff)),
        "n_coefficients_ok": int((coeff["status"] == "ok").sum()) if "status" in coeff.columns else 0,
        "settings_count": int(len(settings_df)),
        "robustness_summary": summary.to_dict(orient="records"),
        "negative_control_warning": negative_warning,
        "phase7": phase7_manifest,
    }

    with open(PHASE6_MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"WROTE {PHASE6_MANIFEST_OUT}")
    return manifest

# =============================================================================
# Main
# =============================================================================

def main():
    print("PHASE 6/7 START")

    trait_df = load_trait_dataset()
    boot_df = load_boot_results()

    print("Loaded trait dataset:", TRAIT_DATASET, trait_df.shape)
    print("Loaded BOOT response rows:", boot_df.shape)
    print("Available metrics:", sorted(boot_df["metric"].dropna().unique().tolist()))
    print("Available stress definitions:", sorted(boot_df["stress_definition"].dropna().unique().tolist()))
    print("Available growing seasons:", sorted(boot_df["growing_season"].dropna().unique().tolist()))
    print("Available product combos:", sorted(boot_df["product_combo"].dropna().unique().tolist()))

    coeff, settings_df = build_phase6_coefficients(boot_df, trait_df)
    coeff = add_direction_consistency(coeff)

    save_csv(coeff, ALL_COEFF_OUT)
    save_csv(settings_df, SETTING_MANIFEST_OUT)

    # Required output tables.
    save_csv(coeff[coeff["robustness_group"].eq("raw_vs_co2")].copy(), RAW_CO2_OUT)
    save_csv(coeff[coeff["robustness_group"].eq("metric_comparison")].copy(), METRIC_OUT)
    save_csv(coeff[coeff["robustness_group"].eq("product_consensus_vs_specific")].copy(), PRODUCT_OUT)
    save_csv(coeff[coeff["robustness_group"].isin(["stress_definition", "growing_season"])].copy(), STRESS_SEASON_OUT)
    save_csv(coeff[coeff["robustness_group"].isin(["aridity_stratified", "aridity_interaction"])].copy(), ARIDITY_OUT)
    save_csv(coeff[coeff["robustness_group"].eq("negative_control")].copy(), NEGATIVE_OUT)

    # Figure 3.
    fig_path = plot_figure3(coeff)
    if fig_path is not None:
        print(f"WROTE {fig_path}")
        print(f"WROTE {FIG3_PDF}")
    else:
        print("Figure 3 not produced because no successful OLS robustness coefficients were available.")

    # Phase 7 interpretation.
    summary = summarize_robustness(coeff)
    negative_warning = detect_negative_control_warning(coeff)
    support_level = classify_support(summary, negative_warning)
    phase7_manifest = write_phase7_interpretation(coeff, summary, negative_warning, support_level)

    # README / manifest.
    write_readme(coeff, settings_df, summary, negative_warning, phase7_manifest)
    manifest = build_manifest(coeff, settings_df, summary, negative_warning, fig_path, phase7_manifest)

    print("")
    print("DONE Phase 6/7.")
    print("")
    print("ROBUSTNESS SUMMARY:")
    print(summary.to_string(index=False))
    print("")
    print("NEGATIVE CONTROL WARNING:")
    print(json.dumps(negative_warning, indent=2, default=str))
    print("")
    print("PHASE 7 SUPPORT LEVEL:")
    print(support_level)
    print("")
    print("MANIFEST:")
    print(json.dumps(manifest, indent=2, default=str))

if __name__ == "__main__":
    main()
