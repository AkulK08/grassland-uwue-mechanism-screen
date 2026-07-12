#!/usr/bin/env python
from pathlib import Path
import argparse
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

import statsmodels.api as sm
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

RANDOM_SEED = 42

RAW_RESPONSE = Path("results/project_final_nature_boot50/fullspec_response_results_raw.csv")
CO2_RESPONSE = Path("results/project_final_nature_boot50/fullspec_response_results_co2corrected.csv")

RAW_SURFACE = Path("results/project_final_nature_boot50/fullspec_vpd_sm_surface_raw.csv")
CO2_SURFACE = Path("results/project_final_nature_boot50/fullspec_vpd_sm_surface_co2corrected.csv")

RAW_METRIC_MATRIX_CANDIDATES = [
    Path("data/processed/project_metric_matrix_raw.csv"),
    Path("data/raw/agents/merged_full_matrix_raw.csv"),
]
CO2_METRIC_MATRIX_CANDIDATES = [
    Path("data/processed/project_metric_matrix_co2corrected.csv"),
    Path("data/raw/agents/merged_full_matrix_co2corrected.csv"),
]

TRAIT_DATASET = Path("results/trait_framework/trait_model_dataset.csv")
PHASE3_CONSENSUS = Path("results/trait_framework/point_product_consensus_response.csv")
PHASE3_COMBO = Path("results/trait_framework/phase3/point_product_combo_level_response.csv")

OUT = Path("results/trait_framework/phase8")
OUT.mkdir(parents=True, exist_ok=True)

OBS_OUT = OUT / "table_latent_model_observations.csv"
LATENT_OUT = OUT / "table_latent_response_by_point.csv"
CLASS_PROB_OUT = OUT / "table_latent_response_class_probabilities.csv"

PRODUCT_BIAS_OUT = OUT / "table_product_bias_effects.csv"
GPP_ET_BIAS_OUT = OUT / "table_gpp_vs_et_product_bias_decomposition.csv"
METRIC_BIAS_OUT = OUT / "table_metric_bias_effects.csv"
STRESS_BIAS_OUT = OUT / "table_stress_bias_effects.csv"
SEASON_BIAS_OUT = OUT / "table_season_bias_effects.csv"
CO2_BIAS_OUT = OUT / "table_co2_bias_effects.csv"

LOFO_OUT = OUT / "table_leave_one_product_family_out.csv"
WEIGHT_SENS_OUT = OUT / "table_independence_weight_sensitivity.csv"

FLUX_CLASS_OUT = OUT / "table_flux_decomposition_by_response_class.csv"
FLUX_PRODUCT_OUT = OUT / "table_flux_decomposition_by_product_family.csv"
FLUX_POINT_OUT = OUT / "table_flux_decomposition_by_point_product.csv"

EXTERNAL_REF_OUT = OUT / "table_external_flux_reference_comparison.csv"
EXTERNAL_REF_INVENTORY_OUT = OUT / "table_external_flux_reference_inventory.csv"

TRAIT_LATENT_OUT = OUT / "table_phase8_trait_effects_on_latent_response.csv"
TRAIT_ML_OUT = OUT / "table_phase8_trait_ml_importance_on_latent_response.csv"
TRAIT_DML_OUT = OUT / "table_phase8_trait_dml_effects_on_latent_response.csv"

FIG8 = OUT / "Figure8_latent_response_map.png"
FIG8_PDF = OUT / "Figure8_latent_response_map.pdf"
FIG9 = OUT / "Figure9_product_bias_forestplot.png"
FIG9_PDF = OUT / "Figure9_product_bias_forestplot.pdf"
FIG10 = OUT / "Figure10_flux_decomposition_by_class.png"
FIG10_PDF = OUT / "Figure10_flux_decomposition_by_class.pdf"
FIG11 = OUT / "Figure11_leave_one_family_out_robustness.png"
FIG11_PDF = OUT / "Figure11_leave_one_family_out_robustness.pdf"
FIG12 = OUT / "Figure12_external_flux_reference_comparison.png"
FIG12_PDF = OUT / "Figure12_external_flux_reference_comparison.pdf"
FIG13 = OUT / "Figure13_trait_effects_on_latent_response.png"
FIG13_PDF = OUT / "Figure13_trait_effects_on_latent_response.pdf"

README_OUT = OUT / "README_phase8_latent_ecosystem_response.md"
MANIFEST_OUT = OUT / "phase8_latent_ecosystem_response_manifest.json"

CLASS_LEVELS = ["enhancement", "saturation", "breakdown", "inconclusive"]

EFFECT_FAMILIES = [
    "gpp_product",
    "et_product",
    "metric",
    "stress_definition",
    "growing_season",
    "co2_version",
    "gpp_et_interaction",
    "metric_et_interaction",
    "stress_et_interaction",
]

CORE_CONTINUOUS_OUTCOMES = [
    "slope_change",
    "post_slope",
    "satbreak_indicator",
]

TRAITS = ["p50", "rooting_depth", "isohydricity"]

PRIMARY_TRAITS = ["p50", "rooting_depth"]

INDEPENDENCE_WEIGHTS = {
    "GOSIF/GLEAM": 1.00,
    "GOSIF/MODIS": 0.75,
    "GOSIF/PML": 0.70,
    "MODIS/GLEAM": 0.75,
    "PML/GLEAM": 0.70,
    "MODIS/MODIS": 0.45,
    "MODIS/PML": 0.45,
    "PML/MODIS": 0.45,
    "PML/PML": 0.35,
}

def die(msg):
    raise SystemExit("\nERROR: " + str(msg) + "\n")

def make_unique_columns(df):
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df.columns = new_cols
    return df

def to_num(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_unique_columns(df)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def read_csv(path, label, required=True, usecols=None):
    if not path.exists():
        if required:
            die(f"Missing required {label}: {path}")
        return pd.DataFrame()
    if usecols is not None:
        df = pd.read_csv(path, low_memory=False, usecols=usecols)
    else:
        df = pd.read_csv(path, low_memory=False)
    return make_unique_columns(df)

def read_existing_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def finite_n(df, col):
    if col not in df.columns:
        return 0
    return int(to_num(df[col]).notna().sum())

def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))

def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())

def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
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

def ci_width(low, high):
    if pd.isna(low) or pd.isna(high):
        return np.nan
    return float(high - low)

def classify_response_class(raw):
    s = str(raw).strip().lower()
    if s == "breakdown":
        return "breakdown"
    if s == "saturation":
        return "saturation"
    if s in ["enhancement", "enhancement_no_accepted_breakpoint"]:
        return "enhancement"
    return "inconclusive"

def satbreak_from_class(raw):
    s = str(raw).strip().lower()
    return 1.0 if s in ["saturation", "breakdown"] else 0.0

def normalize_product_name(x):
    return str(x).strip().upper()

def normalize_combo(gpp, et):
    return normalize_product_name(gpp) + "/" + normalize_product_name(et)

def product_independence_weight(combo):
    combo = str(combo).strip().upper()
    return float(INDEPENDENCE_WEIGHTS.get(combo, 0.50))

def downweight_same_family(combo):
    combo = str(combo).strip().upper()
    if combo in ["MODIS/MODIS", "PML/PML"]:
        return 0.35
    return 1.0

def is_independent_combo(combo):
    combo = str(combo).strip().upper()
    return combo in ["GOSIF/GLEAM", "GOSIF/MODIS", "MODIS/GLEAM", "GOSIF/PML", "PML/GLEAM"]

def standardize(s):
    x = to_num(s)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean(skipna=True)) / sd

def detect_lat_lon(df):
    lat_col = None
    lon_col = None
    for c in ["lat", "latitude", "LAT", "Latitude"]:
        if c in df.columns:
            lat_col = c
            break
    for c in ["lon", "longitude", "LON", "Longitude"]:
        if c in df.columns:
            lon_col = c
            break
    return lat_col, lon_col

# =============================================================================
# Step 8A: observation table
# =============================================================================

def normalize_response_table(df, co2_version):
    df = make_unique_columns(df.copy())

    aliases = {
        "class": "response_class_strict",
        "response_class": "response_class_strict",
        "response_class_final": "response_class_strict",
        "delta_slope": "slope_change",
        "slope_delta": "slope_change",
    }

    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df = df.rename(columns={old: new})

    required = [
        "point_id",
        "gpp_product",
        "et_product",
        "metric",
        "stress_definition",
        "growing_season",
        "pre_slope",
        "post_slope",
        "slope_change",
        "response_class_strict",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"Response table missing required columns {missing}")

    df["point_id"] = df["point_id"].astype(str)
    df["gpp_product"] = df["gpp_product"].astype(str).str.upper()
    df["et_product"] = df["et_product"].astype(str).str.upper()
    df["product_combo"] = df["gpp_product"] + "/" + df["et_product"]
    df["metric"] = df["metric"].astype(str).str.lower()
    df["stress_definition"] = df["stress_definition"].astype(str)
    df["growing_season"] = df["growing_season"].astype(str)
    df["co2_version"] = co2_version

    if "combo" not in df.columns:
        df["combo"] = df["product_combo"]

    for c in [
        "n",
        "linear_slope",
        "breakpoint",
        "pre_slope",
        "post_slope",
        "slope_change",
        "breakpoint_block_ci_low",
        "breakpoint_block_ci_high",
        "pre_slope_block_ci_low",
        "pre_slope_block_ci_high",
        "post_slope_block_ci_low",
        "post_slope_block_ci_high",
        "slope_change_block_ci_low",
        "slope_change_block_ci_high",
        "delta_bic_linear_minus_segmented",
        "supf_permutation_p",
    ]:
        if c in df.columns:
            df[c] = to_num(df[c])

    if "accepted_transition" in df.columns:
        df["accepted_transition"] = df["accepted_transition"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        df["accepted_transition"] = np.nan

    df["response_class_original"] = df["response_class_strict"].astype(str)
    df["response_class_4way"] = df["response_class_strict"].map(classify_response_class)
    df["satbreak_indicator"] = df["response_class_strict"].map(satbreak_from_class)

    df["gpp_et_interaction"] = df["gpp_product"] + ":" + df["et_product"]
    df["metric_et_interaction"] = df["metric"] + ":" + df["et_product"]
    df["stress_et_interaction"] = df["stress_definition"] + ":" + df["et_product"]

    df["independence_weight"] = df["product_combo"].map(product_independence_weight)
    df["downweight_same_family_weight"] = df["product_combo"].map(downweight_same_family)
    df["independent_combo_flag"] = df["product_combo"].map(is_independent_combo).astype(int)

    cols = [
        "point_id",
        "gpp_product",
        "et_product",
        "product_combo",
        "combo",
        "metric",
        "primary_metric",
        "stress_definition",
        "growing_season",
        "co2_version",
        "n",
        "linear_slope",
        "breakpoint",
        "pre_slope",
        "post_slope",
        "slope_change",
        "response_class_original",
        "response_class_4way",
        "satbreak_indicator",
        "accepted_transition",
        "breakpoint_block_ci_low",
        "breakpoint_block_ci_high",
        "pre_slope_block_ci_low",
        "pre_slope_block_ci_high",
        "post_slope_block_ci_low",
        "post_slope_block_ci_high",
        "slope_change_block_ci_low",
        "slope_change_block_ci_high",
        "delta_bic_linear_minus_segmented",
        "supf_permutation_p",
        "gpp_et_interaction",
        "metric_et_interaction",
        "stress_et_interaction",
        "independence_weight",
        "downweight_same_family_weight",
        "independent_combo_flag",
    ]

    return df[[c for c in cols if c in df.columns]].copy()

def build_observation_table():
    raw = read_csv(RAW_RESPONSE, "raw fullspec response")
    co2 = read_csv(CO2_RESPONSE, "CO2-corrected fullspec response")

    raw = normalize_response_table(raw, "raw")
    co2 = normalize_response_table(co2, "co2corrected")

    obs = pd.concat([raw, co2], ignore_index=True)
    obs = make_unique_columns(obs)

    if TRAIT_DATASET.exists():
        trait = pd.read_csv(TRAIT_DATASET, low_memory=False)
        trait = make_unique_columns(trait)
        if "point_id" in trait.columns:
            trait["point_id"] = trait["point_id"].astype(str)
            lat_col, lon_col = detect_lat_lon(trait)
            add_cols = ["point_id"]
            if lat_col:
                add_cols.append(lat_col)
            if lon_col:
                add_cols.append(lon_col)
            add_cols += [c for c in ["aridity", "mean_vpd", "mean_soil_moisture"] if c in trait.columns]
            meta = trait[add_cols].drop_duplicates("point_id")
            rename = {}
            if lat_col and lat_col != "lat":
                rename[lat_col] = "lat"
            if lon_col and lon_col != "lon":
                rename[lon_col] = "lon"
            meta = meta.rename(columns=rename)
            obs = obs.merge(meta, on="point_id", how="left")

    return obs

# =============================================================================
# Latent additive response model
# =============================================================================

def _group_weighted_mean(values, weights, groups):
    tmp = pd.DataFrame({
        "group": pd.Series(groups).astype(str).values,
        "value": np.asarray(values, dtype=float),
        "weight": np.asarray(weights, dtype=float),
    })

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["group", "value", "weight"])
    tmp = tmp[tmp["weight"] > 0]

    if tmp.empty:
        return pd.Series(dtype=float)

    tmp["wv"] = tmp["value"] * tmp["weight"]
    g = tmp.groupby("group", dropna=False).agg(wv=("wv", "sum"), w=("weight", "sum"))
    return g["wv"] / g["w"]

def fit_latent_additive(
    df,
    y_col,
    point_col="point_id",
    effect_families=None,
    weight_col=None,
    n_iter=30,
    min_obs_per_point=1,
):
    effect_families = effect_families or EFFECT_FAMILIES

    needed = [point_col, y_col] + [c for c in effect_families if c in df.columns]
    if weight_col and weight_col in df.columns:
        needed.append(weight_col)

    work = df[needed].copy()
    work[point_col] = work[point_col].astype(str)
    work[y_col] = to_num(work[y_col])

    if weight_col and weight_col in work.columns:
        work["_w"] = to_num(work[weight_col])
        work["_w"] = work["_w"].fillna(1.0)
        work.loc[work["_w"] <= 0, "_w"] = 1.0
    else:
        work["_w"] = 1.0

    work = work.dropna(subset=[point_col, y_col]).copy()

    if work.empty:
        return {
            "theta": pd.Series(dtype=float),
            "effects": {},
            "bias_table": pd.DataFrame(),
            "diagnostics": {"status": "empty", "n": 0},
        }

    point_counts = work.groupby(point_col).size()
    valid_points = point_counts[point_counts >= min_obs_per_point].index
    work = work[work[point_col].isin(valid_points)].copy()

    if work.empty:
        return {
            "theta": pd.Series(dtype=float),
            "effects": {},
            "bias_table": pd.DataFrame(),
            "diagnostics": {"status": "empty_after_min_obs", "n": 0},
        }

    y = work[y_col].astype(float).values
    w = work["_w"].astype(float).values

    families = [f for f in effect_families if f in work.columns]
    for f in families:
        work[f] = work[f].astype(str).fillna("missing")

    theta = _group_weighted_mean(y, w, work[point_col])
    effects = {}
    for f in families:
        levels = sorted(work[f].dropna().astype(str).unique())
        effects[f] = pd.Series(0.0, index=levels, dtype=float)

    for iteration in range(n_iter):
        sum_effects = np.zeros(len(work), dtype=float)
        for f in families:
            mapped = work[f].map(effects[f]).fillna(0.0).astype(float).values
            sum_effects += mapped

        resid_theta = y - sum_effects
        theta = _group_weighted_mean(resid_theta, w, work[point_col])

        theta_mapped = work[point_col].map(theta).fillna(0.0).astype(float).values

        for f in families:
            other = theta_mapped.copy()
            for other_f in families:
                if other_f == f:
                    continue
                other += work[other_f].map(effects[other_f]).fillna(0.0).astype(float).values

            resid_f = y - other
            ef = _group_weighted_mean(resid_f, w, work[f])

            mapped_ef = work[f].map(ef).fillna(0.0).astype(float).values
            center = weighted_mean(mapped_ef, w)
            if pd.isna(center):
                center = 0.0
            ef = ef - center
            effects[f] = ef

    fitted = work[point_col].map(theta).fillna(0.0).astype(float).values
    for f in families:
        fitted += work[f].map(effects[f]).fillna(0.0).astype(float).values

    resid = y - fitted
    sigma = float(np.sqrt(np.nanmean(resid ** 2)))

    bias_rows = []
    for f in families:
        ef = effects[f]
        for level, value in ef.items():
            obs_n = int((work[f].astype(str) == str(level)).sum())
            bias_rows.append({
                "outcome": y_col,
                "effect_family": f,
                "level": level,
                "bias_effect": float(value),
                "n_observations": obs_n,
            })

    bias_table = pd.DataFrame(bias_rows)

    point_obs = work.groupby(point_col).size().rename("n_observations").reset_index()
    theta_df = theta.rename("latent_value").reset_index().rename(columns={"group": point_col})
    theta_df.columns = [point_col, "latent_value"]
    theta_df = theta_df.merge(point_obs, on=point_col, how="left")

    return {
        "theta": theta_df,
        "effects": effects,
        "bias_table": bias_table,
        "diagnostics": {
            "status": "ok",
            "n": int(len(work)),
            "n_points": int(work[point_col].nunique()),
            "sigma": sigma,
            "n_families": int(len(families)),
            "families": families,
            "n_iter": int(n_iter),
        },
    }

def bootstrap_latent(
    df,
    y_col,
    point_col="point_id",
    effect_families=None,
    weight_col=None,
    n_boot=100,
    n_iter=20,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    effect_families = effect_families or EFFECT_FAMILIES

    df = df.dropna(subset=[y_col, point_col]).copy()
    df[point_col] = df[point_col].astype(str)

    points = sorted(df[point_col].unique())
    if len(points) == 0:
        return pd.DataFrame(), pd.DataFrame()

    point_to_indices = {
        p: np.where(df[point_col].values == p)[0]
        for p in points
    }

    theta_records = []
    bias_records = []

    for b in range(n_boot):
        boot_idx = []
        for p, idx in point_to_indices.items():
            if len(idx) == 0:
                continue
            boot_idx.extend(rng.choice(idx, size=len(idx), replace=True).tolist())

        boot = df.iloc[boot_idx].copy()

        fit = fit_latent_additive(
            boot,
            y_col=y_col,
            point_col=point_col,
            effect_families=effect_families,
            weight_col=weight_col,
            n_iter=n_iter,
        )

        theta = fit["theta"].copy()
        if not theta.empty:
            theta["boot"] = b
            theta["outcome"] = y_col
            theta_records.append(theta)

        bt = fit["bias_table"].copy()
        if not bt.empty:
            bt["boot"] = b
            bias_records.append(bt)

    theta_boot = pd.concat(theta_records, ignore_index=True) if theta_records else pd.DataFrame()
    bias_boot = pd.concat(bias_records, ignore_index=True) if bias_records else pd.DataFrame()

    return theta_boot, bias_boot

def summarize_bootstrap_theta(theta_main, theta_boot, outcome_name, prefix):
    main = theta_main.copy()
    if main.empty:
        return pd.DataFrame()

    main = main.rename(columns={"latent_value": f"{prefix}"})
    if theta_boot.empty:
        main[f"{prefix}_ci_low"] = np.nan
        main[f"{prefix}_ci_high"] = np.nan
        main[f"{prefix}_posterior_sd"] = np.nan
        return main

    b = theta_boot[theta_boot["outcome"].eq(outcome_name)].copy()
    if b.empty:
        main[f"{prefix}_ci_low"] = np.nan
        main[f"{prefix}_ci_high"] = np.nan
        main[f"{prefix}_posterior_sd"] = np.nan
        return main

    summ = (
        b.groupby("point_id")
        .agg(
            ci_low=("latent_value", lambda x: float(np.nanpercentile(x, 2.5))),
            ci_high=("latent_value", lambda x: float(np.nanpercentile(x, 97.5))),
            posterior_sd=("latent_value", "std"),
        )
        .reset_index()
        .rename(columns={
            "ci_low": f"{prefix}_ci_low",
            "ci_high": f"{prefix}_ci_high",
            "posterior_sd": f"{prefix}_posterior_sd",
        })
    )

    out = main.merge(summ, on="point_id", how="left")
    return out

def summarize_bootstrap_bias(bias_main, bias_boot):
    if bias_main.empty:
        return pd.DataFrame()

    out = bias_main.copy()

    if bias_boot.empty:
        out["ci_low"] = np.nan
        out["ci_high"] = np.nan
        out["posterior_sd"] = np.nan
        return out

    summ = (
        bias_boot.groupby(["outcome", "effect_family", "level"])
        .agg(
            ci_low=("bias_effect", lambda x: float(np.nanpercentile(x, 2.5))),
            ci_high=("bias_effect", lambda x: float(np.nanpercentile(x, 97.5))),
            posterior_sd=("bias_effect", "std"),
        )
        .reset_index()
    )

    out = out.merge(summ, on=["outcome", "effect_family", "level"], how="left")
    return out

def fit_latent_suite(obs, weight_col=None, n_boot=100, n_iter=30, boot_iter=20, label="equal"):
    work = obs.copy()

    main_fits = {}
    theta_tables = []
    bias_tables = []
    theta_boot_all = []
    bias_boot_all = []

    outcomes = CORE_CONTINUOUS_OUTCOMES.copy()

    for cl in CLASS_LEVELS:
        col = f"class_indicator_{cl}"
        work[col] = (work["response_class_4way"].eq(cl)).astype(float)
        outcomes.append(col)

    for outcome in outcomes:
        print(f"Fitting latent model: {label} / {outcome}")
        fit = fit_latent_additive(
            work,
            y_col=outcome,
            point_col="point_id",
            effect_families=EFFECT_FAMILIES,
            weight_col=weight_col,
            n_iter=n_iter,
        )
        main_fits[outcome] = fit

        theta = fit["theta"].copy()
        if not theta.empty:
            theta["outcome"] = outcome
            theta["weighting_scheme"] = label
            theta_tables.append(theta)

        bt = fit["bias_table"].copy()
        if not bt.empty:
            bt["weighting_scheme"] = label
            bias_tables.append(bt)

        if n_boot > 0:
            theta_boot, bias_boot = bootstrap_latent(
                work,
                y_col=outcome,
                point_col="point_id",
                effect_families=EFFECT_FAMILIES,
                weight_col=weight_col,
                n_boot=n_boot,
                n_iter=boot_iter,
                seed=RANDOM_SEED + abs(hash(label + outcome)) % 100000,
            )
            if not theta_boot.empty:
                theta_boot["weighting_scheme"] = label
                theta_boot_all.append(theta_boot)
            if not bias_boot.empty:
                bias_boot["weighting_scheme"] = label
                bias_boot_all.append(bias_boot)

    theta_main = pd.concat(theta_tables, ignore_index=True) if theta_tables else pd.DataFrame()
    bias_main = pd.concat(bias_tables, ignore_index=True) if bias_tables else pd.DataFrame()
    theta_boot = pd.concat(theta_boot_all, ignore_index=True) if theta_boot_all else pd.DataFrame()
    bias_boot = pd.concat(bias_boot_all, ignore_index=True) if bias_boot_all else pd.DataFrame()

    return {
        "work": work,
        "fits": main_fits,
        "theta_main": theta_main,
        "bias_main": bias_main,
        "theta_boot": theta_boot,
        "bias_boot": bias_boot,
    }

def build_latent_response_tables(obs, suite):
    fits = suite["fits"]
    theta_boot = suite["theta_boot"]
    bias_main = suite["bias_main"]
    bias_boot = suite["bias_boot"]

    point_ids = sorted(obs["point_id"].astype(str).unique())
    latent = pd.DataFrame({"point_id": point_ids})

    if "lat" in obs.columns:
        meta = obs.groupby("point_id", as_index=False).agg(lat=("lat", "first"))
        latent = latent.merge(meta, on="point_id", how="left")
    if "lon" in obs.columns:
        meta = obs.groupby("point_id", as_index=False).agg(lon=("lon", "first"))
        latent = latent.merge(meta, on="point_id", how="left")
    if "aridity" in obs.columns:
        meta = obs.groupby("point_id", as_index=False).agg(aridity=("aridity", "first"))
        latent = latent.merge(meta, on="point_id", how="left")

    for outcome, prefix in [
        ("slope_change", "latent_slope_change"),
        ("post_slope", "latent_post_slope"),
        ("satbreak_indicator", "latent_satbreak_probability_direct"),
    ]:
        theta = fits[outcome]["theta"].copy()
        summ = summarize_bootstrap_theta(theta, theta_boot, outcome, prefix)
        keep = ["point_id"] + [c for c in summ.columns if c.startswith(prefix)]
        latent = latent.merge(summ[keep], on="point_id", how="left")

    class_prob = pd.DataFrame({"point_id": point_ids})

    for cl in CLASS_LEVELS:
        outcome = f"class_indicator_{cl}"
        prefix = f"p_{cl}"
        theta = fits[outcome]["theta"].copy()
        summ = summarize_bootstrap_theta(theta, theta_boot, outcome, prefix)
        keep = ["point_id"] + [c for c in summ.columns if c.startswith(prefix)]
        class_prob = class_prob.merge(summ[keep], on="point_id", how="left")

    for cl in CLASS_LEVELS:
        c = f"p_{cl}"
        if c not in class_prob.columns:
            class_prob[c] = 0.0
        class_prob[c] = to_num(class_prob[c]).clip(lower=0.0, upper=1.0)

    prob_cols = [f"p_{cl}" for cl in CLASS_LEVELS]
    total = class_prob[prob_cols].sum(axis=1).replace(0, np.nan)
    for c in prob_cols:
        class_prob[c] = class_prob[c] / total

    class_prob["p_satbreak"] = class_prob["p_saturation"].fillna(0) + class_prob["p_breakdown"].fillna(0)
    class_prob["p_threshold_like"] = class_prob["p_satbreak"]

    class_prob["max_class_probability"] = class_prob[prob_cols].max(axis=1)
    class_prob["latent_response_class"] = class_prob[prob_cols].idxmax(axis=1).str.replace("p_", "", regex=False)
    class_prob.loc[class_prob["max_class_probability"] < 0.45, "latent_response_class"] = "inconclusive"

    latent = latent.merge(class_prob, on="point_id", how="left")

    if "latent_satbreak_probability_direct" in latent.columns:
        latent["latent_satbreak_probability"] = latent["p_satbreak"].where(
            latent["p_satbreak"].notna(),
            latent["latent_satbreak_probability_direct"],
        )
    else:
        latent["latent_satbreak_probability"] = latent["p_satbreak"]

    uncertainty = compute_uncertainty_components(obs)
    latent = latent.merge(uncertainty, on="point_id", how="left")

    bias = summarize_bootstrap_bias(bias_main, bias_boot)

    return latent, class_prob, bias

def compute_uncertainty_components(obs):
    rows = []
    dimensions = [
        ("product_uncertainty", "product_combo"),
        ("gpp_product_uncertainty", "gpp_product"),
        ("et_product_uncertainty", "et_product"),
        ("metric_uncertainty", "metric"),
        ("stress_definition_uncertainty", "stress_definition"),
        ("season_uncertainty", "growing_season"),
        ("co2_uncertainty", "co2_version"),
    ]

    for point_id, d in obs.groupby("point_id"):
        row = {"point_id": point_id}
        for prefix, dim in dimensions:
            if dim not in d.columns:
                row[f"{prefix}_slope_change_range"] = np.nan
                row[f"{prefix}_post_slope_range"] = np.nan
                row[f"{prefix}_satbreak_range"] = np.nan
                continue

            g = (
                d.groupby(dim)
                .agg(
                    slope_change=("slope_change", safe_mean),
                    post_slope=("post_slope", safe_mean),
                    satbreak_indicator=("satbreak_indicator", safe_mean),
                )
            )

            for out_col, base_col in [
                (f"{prefix}_slope_change_range", "slope_change"),
                (f"{prefix}_post_slope_range", "post_slope"),
                (f"{prefix}_satbreak_range", "satbreak_indicator"),
            ]:
                vals = to_num(g[base_col]).dropna()
                if len(vals) <= 1:
                    row[out_col] = 0.0 if len(vals) == 1 else np.nan
                else:
                    row[out_col] = float(vals.max() - vals.min())

        rows.append(row)

    return pd.DataFrame(rows)

# =============================================================================
# Bias decomposition tables
# =============================================================================

def split_bias_tables(bias):
    product = bias[bias["effect_family"].isin(["gpp_product", "et_product", "gpp_et_interaction"])].copy()
    gpp_et = product.copy()
    metric = bias[bias["effect_family"].eq("metric")].copy()
    stress = bias[bias["effect_family"].eq("stress_definition")].copy()
    season = bias[bias["effect_family"].eq("growing_season")].copy()
    co2 = bias[bias["effect_family"].eq("co2_version")].copy()

    return product, gpp_et, metric, stress, season, co2

# =============================================================================
# Leave-one-product-family-out tests
# =============================================================================

def run_leave_one_family_out(obs, main_latent, n_iter=25):
    tests = [
        ("exclude_MODIS_GPP", "gpp_product", "MODIS"),
        ("exclude_GOSIF_GPP", "gpp_product", "GOSIF"),
        ("exclude_PML_GPP", "gpp_product", "PML"),
        ("exclude_MODIS_ET", "et_product", "MODIS"),
        ("exclude_GLEAM_ET", "et_product", "GLEAM"),
        ("exclude_PML_ET", "et_product", "PML"),
    ]

    rows = []

    main = main_latent[[
        "point_id",
        "latent_slope_change",
        "latent_post_slope",
        "latent_satbreak_probability",
        "latent_response_class",
    ]].copy()

    for name, col, value in tests:
        print(f"Leave-one-family-out: {name}")
        sub = obs[~obs[col].astype(str).str.upper().eq(value)].copy()

        if sub.empty:
            rows.append({
                "test": name,
                "excluded_axis": col,
                "excluded_family": value,
                "status": "empty_after_exclusion",
                "n_observations": 0,
                "n_points": 0,
            })
            continue

        suite = fit_latent_suite(
            sub,
            weight_col=None,
            n_boot=0,
            n_iter=n_iter,
            boot_iter=10,
            label=name,
        )

        latent, class_prob, bias = build_latent_response_tables(sub, suite)

        merged = main.merge(
            latent[[
                "point_id",
                "latent_slope_change",
                "latent_post_slope",
                "latent_satbreak_probability",
                "latent_response_class",
            ]],
            on="point_id",
            how="inner",
            suffixes=("_main", "_lofo"),
        )

        if merged.empty:
            rows.append({
                "test": name,
                "excluded_axis": col,
                "excluded_family": value,
                "status": "no_overlap",
                "n_observations": int(len(sub)),
                "n_points": int(sub["point_id"].nunique()),
            })
            continue

        def corr(a, b):
            aa = to_num(merged[a])
            bb = to_num(merged[b])
            ok = aa.notna() & bb.notna()
            if ok.sum() < 3:
                return np.nan
            return float(np.corrcoef(aa[ok], bb[ok])[0, 1])

        rows.append({
            "test": name,
            "excluded_axis": col,
            "excluded_family": value,
            "status": "ok",
            "n_observations": int(len(sub)),
            "n_points": int(sub["point_id"].nunique()),
            "slope_change_correlation_with_main": corr("latent_slope_change_main", "latent_slope_change_lofo"),
            "post_slope_correlation_with_main": corr("latent_post_slope_main", "latent_post_slope_lofo"),
            "satbreak_probability_correlation_with_main": corr("latent_satbreak_probability_main", "latent_satbreak_probability_lofo"),
            "median_abs_slope_change_difference": float((to_num(merged["latent_slope_change_main"]) - to_num(merged["latent_slope_change_lofo"])).abs().median(skipna=True)),
            "median_abs_post_slope_difference": float((to_num(merged["latent_post_slope_main"]) - to_num(merged["latent_post_slope_lofo"])).abs().median(skipna=True)),
            "median_abs_satbreak_probability_difference": float((to_num(merged["latent_satbreak_probability_main"]) - to_num(merged["latent_satbreak_probability_lofo"])).abs().median(skipna=True)),
            "class_match_fraction": float((merged["latent_response_class_main"] == merged["latent_response_class_lofo"]).mean()),
            "main_satbreak_fraction": float((merged["latent_response_class_main"].isin(["saturation", "breakdown"])).mean()),
            "lofo_satbreak_fraction": float((merged["latent_response_class_lofo"].isin(["saturation", "breakdown"])).mean()),
        })

    return pd.DataFrame(rows)

# =============================================================================
# Algorithmic independence weighting sensitivity
# =============================================================================

def run_weight_sensitivity(obs, main_latent, n_iter=25):
    schemes = [
        ("equal_weights", None, obs.copy()),
        ("independence_weights", "independence_weight", obs.copy()),
        ("downweight_same_family", "downweight_same_family_weight", obs.copy()),
        ("only_independent_products", None, obs[obs["independent_combo_flag"].eq(1)].copy()),
    ]

    rows = []

    main = main_latent[[
        "point_id",
        "latent_slope_change",
        "latent_post_slope",
        "latent_satbreak_probability",
        "latent_response_class",
    ]].copy()

    for scheme, weight_col, sub in schemes:
        print(f"Weight sensitivity: {scheme}")
        if sub.empty:
            rows.append({
                "weighting_scheme": scheme,
                "status": "empty",
                "n_observations": 0,
                "n_points": 0,
            })
            continue

        suite = fit_latent_suite(
            sub,
            weight_col=weight_col,
            n_boot=0,
            n_iter=n_iter,
            boot_iter=10,
            label=scheme,
        )

        latent, class_prob, bias = build_latent_response_tables(sub, suite)

        merged = main.merge(
            latent[[
                "point_id",
                "latent_slope_change",
                "latent_post_slope",
                "latent_satbreak_probability",
                "latent_response_class",
            ]],
            on="point_id",
            how="inner",
            suffixes=("_main", "_scheme"),
        )

        if merged.empty:
            rows.append({
                "weighting_scheme": scheme,
                "status": "no_overlap",
                "n_observations": int(len(sub)),
                "n_points": int(sub["point_id"].nunique()),
            })
            continue

        def corr(a, b):
            aa = to_num(merged[a])
            bb = to_num(merged[b])
            ok = aa.notna() & bb.notna()
            if ok.sum() < 3:
                return np.nan
            return float(np.corrcoef(aa[ok], bb[ok])[0, 1])

        rows.append({
            "weighting_scheme": scheme,
            "status": "ok",
            "n_observations": int(len(sub)),
            "n_points": int(sub["point_id"].nunique()),
            "slope_change_correlation_with_equal_main": corr("latent_slope_change_main", "latent_slope_change_scheme"),
            "post_slope_correlation_with_equal_main": corr("latent_post_slope_main", "latent_post_slope_scheme"),
            "satbreak_probability_correlation_with_equal_main": corr("latent_satbreak_probability_main", "latent_satbreak_probability_scheme"),
            "median_abs_slope_change_difference": float((to_num(merged["latent_slope_change_main"]) - to_num(merged["latent_slope_change_scheme"])).abs().median(skipna=True)),
            "median_abs_post_slope_difference": float((to_num(merged["latent_post_slope_main"]) - to_num(merged["latent_post_slope_scheme"])).abs().median(skipna=True)),
            "median_abs_satbreak_probability_difference": float((to_num(merged["latent_satbreak_probability_main"]) - to_num(merged["latent_satbreak_probability_scheme"])).abs().median(skipna=True)),
            "class_match_fraction": float((merged["latent_response_class_main"] == merged["latent_response_class_scheme"]).mean()),
            "scheme_satbreak_fraction": float((merged["latent_response_class_scheme"].isin(["saturation", "breakdown"])).mean()),
        })

    return pd.DataFrame(rows)

# =============================================================================
# Flux decomposition
# =============================================================================

def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def read_metric_matrix_columns(path):
    if path is None or not path.exists():
        return []
    return list(pd.read_csv(path, nrows=0).columns)

def pick_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def product_col_candidates(prefix, product):
    p = str(product).lower()
    return [
        f"{prefix}_{p}",
        f"log_{prefix}_{p}",
        f"{prefix}_{p}_log",
        f"log{prefix}_{p}",
        f"{prefix}{p}",
    ]

def stress_col_candidates(stress):
    s = str(stress).lower()
    if s == "zscore":
        return [
            "stress_zscore",
            "compound_stress_zscore",
            "zscore",
            "z_stress",
            "compound_stress",
            "stress_index",
        ]
    if s == "percentile_joint":
        return [
            "stress_percentile_joint",
            "percentile_joint",
            "joint_percentile_stress",
            "percentile_stress",
        ]
    if s == "copula_joint":
        return [
            "stress_copula_joint",
            "copula_joint",
            "copula_stress",
        ]
    if s == "interaction_surface":
        return [
            "stress_interaction_surface",
            "interaction_surface",
            "compound_stress_zscore",
            "stress_zscore",
            "compound_stress",
        ]
    return [s, f"stress_{s}", f"compound_stress_{s}"]

def season_flag_candidates(season):
    s = str(season).lower()
    return [
        f"in_{s}",
        f"is_{s}",
        f"gs_{s}",
        f"growing_season_{s}",
        f"{s}_flag",
    ]

def simple_group_slope(df, y_col, x_col, group_cols):
    needed = group_cols + [y_col, x_col]
    d = df[needed].copy()
    d[y_col] = to_num(d[y_col])
    d[x_col] = to_num(d[x_col])
    d = d.dropna(subset=[y_col, x_col])

    if d.empty:
        return pd.DataFrame(columns=group_cols + ["slope", "n"])

    rows = []
    for keys, g in d.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = to_num(g[x_col])
        y = to_num(g[y_col])
        ok = x.notna() & y.notna()
        if ok.sum() < 10 or x[ok].var() == 0:
            slope = np.nan
        else:
            slope = float(np.cov(x[ok], y[ok], ddof=1)[0, 1] / np.var(x[ok], ddof=1))
        row = dict(zip(group_cols, keys))
        row["slope"] = slope
        row["n"] = int(ok.sum())
        rows.append(row)

    return pd.DataFrame(rows)

def compute_flux_decomposition(obs, latent):
    rows = []
    point_rows = []

    matrix_paths = {
        "raw": find_first_existing(RAW_METRIC_MATRIX_CANDIDATES),
        "co2corrected": find_first_existing(CO2_METRIC_MATRIX_CANDIDATES),
    }

    for version, path in matrix_paths.items():
        if path is None:
            rows.append({
                "co2_version": version,
                "status": "metric_matrix_missing",
                "message": "No local metric matrix found; flux decomposition unavailable for this version.",
            })
            continue

        print(f"Attempting flux decomposition from {path}")

        cols = read_metric_matrix_columns(path)
        if not cols:
            rows.append({
                "co2_version": version,
                "status": "empty_columns",
                "message": str(path),
            })
            continue

        point_col = pick_column(cols, ["point_id", "site_id", "id"])
        if point_col is None:
            rows.append({
                "co2_version": version,
                "status": "missing_point_id",
                "message": str(path),
            })
            continue

        gpp_cols = {}
        et_cols = {}
        for gpp in ["MODIS", "GOSIF", "PML"]:
            col = pick_column(cols, product_col_candidates("gpp", gpp) + product_col_candidates("log_gpp", gpp))
            if col:
                gpp_cols[gpp] = col

        for et in ["MODIS", "GLEAM", "PML"]:
            col = pick_column(cols, product_col_candidates("et", et) + product_col_candidates("log_et", et))
            if col:
                et_cols[et] = col

        if not gpp_cols or not et_cols:
            rows.append({
                "co2_version": version,
                "status": "missing_gpp_or_et_columns",
                "message": f"path={path}; gpp_cols={gpp_cols}; et_cols={et_cols}",
            })
            continue

        stress_cols = {}
        for stress in sorted(obs["stress_definition"].dropna().astype(str).unique()):
            col = pick_column(cols, stress_col_candidates(stress))
            if col:
                stress_cols[stress] = col

        if not stress_cols:
            rows.append({
                "co2_version": version,
                "status": "missing_stress_columns",
                "message": f"path={path}; columns_checked={len(cols)}",
            })
            continue

        needed = [point_col] + sorted(set(list(gpp_cols.values()) + list(et_cols.values()) + list(stress_cols.values())))
        for c in ["date", "system:index"]:
            if c in cols and c not in needed:
                needed.append(c)

        try:
            mat = pd.read_csv(path, low_memory=False, usecols=needed)
        except Exception as e:
            rows.append({
                "co2_version": version,
                "status": "read_failed",
                "message": f"{type(e).__name__}: {e}",
            })
            continue

        mat = make_unique_columns(mat)
        mat = mat.rename(columns={point_col: "point_id"})
        mat["point_id"] = mat["point_id"].astype(str)

        for gpp, gpp_col in gpp_cols.items():
            for et, et_col in et_cols.items():
                combo = normalize_combo(gpp, et)
                for stress, stress_col in stress_cols.items():
                    sub_cols = ["point_id", gpp_col, et_col, stress_col]
                    d = mat[sub_cols].copy()
                    d = d.rename(columns={
                        gpp_col: "log_gpp_or_gpp",
                        et_col: "log_et_or_et",
                        stress_col: "stress_value",
                    })

                    # If these columns are not already logs, log-transform positive raw values.
                    # Heuristic: values > 50 usually raw flux, not log flux.
                    for ycol in ["log_gpp_or_gpp", "log_et_or_et"]:
                        x = to_num(d[ycol])
                        if x.dropna().quantile(0.95) > 50:
                            x = np.where(x > 0, np.log(x), np.nan)
                            d[ycol] = x
                        else:
                            d[ycol] = x

                    gpp_slope = simple_group_slope(d, "log_gpp_or_gpp", "stress_value", ["point_id"])
                    et_slope = simple_group_slope(d, "log_et_or_et", "stress_value", ["point_id"])

                    if gpp_slope.empty or et_slope.empty:
                        continue

                    merged = gpp_slope.merge(et_slope, on="point_id", how="inner", suffixes=("_gpp", "_et"))
                    if merged.empty:
                        continue

                    merged["co2_version"] = version
                    merged["gpp_product"] = gpp
                    merged["et_product"] = et
                    merged["product_combo"] = combo
                    merged["stress_definition"] = stress
                    merged["gpp_stress_slope"] = merged["slope_gpp"]
                    merged["et_stress_slope"] = merged["slope_et"]
                    merged["gpp_contribution_to_log_wue"] = merged["gpp_stress_slope"]
                    merged["et_contribution_to_log_wue"] = -merged["et_stress_slope"]
                    merged["approx_log_wue_stress_slope"] = merged["gpp_contribution_to_log_wue"] + merged["et_contribution_to_log_wue"]

                    def flux_interpretation(r):
                        gpp_c = r["gpp_contribution_to_log_wue"]
                        et_c = r["et_contribution_to_log_wue"]
                        if pd.isna(gpp_c) or pd.isna(et_c):
                            return "unresolved"
                        if et_c > 0 and abs(et_c) > abs(gpp_c):
                            return "ET_suppression_dominates"
                        if gpp_c < 0 and abs(gpp_c) > abs(et_c):
                            return "GPP_weakening_dominates"
                        if et_c > 0 and gpp_c >= 0:
                            return "ET_suppression_plus_GPP_resilience"
                        if et_c <= 0 and gpp_c < 0:
                            return "both_flux_terms_weaken_WUE"
                        return "mixed"

                    merged["flux_interpretation"] = merged.apply(flux_interpretation, axis=1)

                    point_rows.append(merged[[
                        "point_id",
                        "co2_version",
                        "gpp_product",
                        "et_product",
                        "product_combo",
                        "stress_definition",
                        "gpp_stress_slope",
                        "et_stress_slope",
                        "gpp_contribution_to_log_wue",
                        "et_contribution_to_log_wue",
                        "approx_log_wue_stress_slope",
                        "flux_interpretation",
                        "n_gpp",
                        "n_et",
                    ]])

        rows.append({
            "co2_version": version,
            "status": "attempted",
            "message": f"path={path}; gpp_cols={gpp_cols}; et_cols={et_cols}; stress_cols={stress_cols}",
        })

    point_df = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()

    if point_df.empty:
        status = pd.DataFrame(rows)
        return point_df, status, pd.DataFrame(), pd.DataFrame()

    point_df = point_df.merge(
        latent[["point_id", "latent_response_class", "latent_satbreak_probability"]],
        on="point_id",
        how="left",
    )

    class_summary = (
        point_df.groupby(["latent_response_class", "flux_interpretation"], dropna=False)
        .agg(
            n=("point_id", "size"),
            n_points=("point_id", "nunique"),
            median_gpp_contribution=("gpp_contribution_to_log_wue", "median"),
            median_et_contribution=("et_contribution_to_log_wue", "median"),
            median_approx_log_wue_slope=("approx_log_wue_stress_slope", "median"),
        )
        .reset_index()
    )

    product_summary = (
        point_df.groupby(["product_combo", "flux_interpretation"], dropna=False)
        .agg(
            n=("point_id", "size"),
            n_points=("point_id", "nunique"),
            median_gpp_contribution=("gpp_contribution_to_log_wue", "median"),
            median_et_contribution=("et_contribution_to_log_wue", "median"),
            median_approx_log_wue_slope=("approx_log_wue_stress_slope", "median"),
        )
        .reset_index()
    )

    status = pd.DataFrame(rows)

    return point_df, status, class_summary, product_summary

# =============================================================================
# External soft-reference flux comparison
# =============================================================================

def discover_external_reference_files():
    roots = [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/external"),
        Path("results/external_flux"),
        Path("results/soft_reference_flux"),
    ]
    patterns = [
        "*FLUXCOM*",
        "*fluxcom*",
        "*FLUXCOMX*",
        "*fluxcomx*",
        "*FLUXCOM-X*",
        "*gloflux*",
        "*GloFlux*",
        "*FluxSat*",
        "*fluxsat*",
        "*BESS*",
        "*bess*",
        "*pmodel*",
        "*Pmodel*",
        "*P_MODEL*",
    ]

    files = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            files.extend(list(root.rglob(pat)))

    keep = []
    for f in sorted(set(files)):
        if f.is_file() and f.suffix.lower() in [".csv", ".parquet", ".feather"]:
            keep.append(f)

    rows = []
    for f in keep:
        rows.append({
            "path": str(f),
            "suffix": f.suffix,
            "size_bytes": f.stat().st_size if f.exists() else np.nan,
            "detected_reference_family": detect_reference_family(f.name),
            "status": "detected",
        })

    if not rows:
        rows.append({
            "path": "",
            "suffix": "",
            "size_bytes": np.nan,
            "detected_reference_family": "",
            "status": "no_external_reference_files_detected",
        })

    return pd.DataFrame(rows), keep

def detect_reference_family(name):
    s = str(name).lower()
    if "fluxcom" in s:
        return "FLUXCOM_or_FLUXCOMX"
    if "gloflux" in s:
        return "GloFlux"
    if "fluxsat" in s:
        return "FluxSat"
    if "bess" in s:
        return "BESS"
    if "pmodel" in s or "p_model" in s:
        return "P-model"
    return "unknown"

def load_external_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    return pd.DataFrame()

def external_comparison(latent):
    inventory, files = discover_external_reference_files()
    save_csv(inventory, EXTERNAL_REF_INVENTORY_OUT)

    rows = []

    if not files:
        rows.append({
            "reference_family": "",
            "path": "",
            "status": "not_available",
            "message": "No local FLUXCOM/FLUXCOM-X/GloFlux/FluxSat/BESS/P-model reference files detected. Phase 8 records this as not run, not as failed.",
        })
        return pd.DataFrame(rows)

    for path in files:
        family = detect_reference_family(path.name)
        print(f"Attempting external soft-reference comparison: {family} {path}")

        try:
            df = load_external_file(path)
            df = make_unique_columns(df)
        except Exception as e:
            rows.append({
                "reference_family": family,
                "path": str(path),
                "status": "read_failed",
                "message": f"{type(e).__name__}: {e}",
            })
            continue

        if df.empty:
            rows.append({
                "reference_family": family,
                "path": str(path),
                "status": "empty",
                "message": "",
            })
            continue

        point_col = None
        for c in ["point_id", "site_id", "id"]:
            if c in df.columns:
                point_col = c
                break

        if point_col is None:
            rows.append({
                "reference_family": family,
                "path": str(path),
                "status": "missing_point_id",
                "message": f"columns={list(df.columns)[:50]}",
            })
            continue

        df = df.rename(columns={point_col: "point_id"})
        df["point_id"] = df["point_id"].astype(str)

        response_cols = []
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["slope_change", "post_slope", "satbreak", "response_class"]):
                response_cols.append(c)

        if not response_cols:
            rows.append({
                "reference_family": family,
                "path": str(path),
                "status": "no_response_columns",
                "message": "Detected point_id but no response-shape columns. File may be raw flux; run an external response-shape preprocessing step first.",
            })
            continue

        merged = latent.merge(df[["point_id"] + response_cols].drop_duplicates("point_id"), on="point_id", how="inner")

        if merged.empty:
            rows.append({
                "reference_family": family,
                "path": str(path),
                "status": "no_point_overlap",
                "message": f"response_cols={response_cols}",
            })
            continue

        row = {
            "reference_family": family,
            "path": str(path),
            "status": "ok",
            "n_points_overlap": int(merged["point_id"].nunique()),
            "response_columns_used": ",".join(response_cols),
        }

        for col in response_cols:
            lc = col.lower()
            if "slope_change" in lc:
                x = to_num(merged["latent_slope_change"])
                y = to_num(merged[col])
                ok = x.notna() & y.notna()
                if ok.sum() >= 3:
                    row[f"{col}_corr_with_latent_slope_change"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if "post_slope" in lc:
                x = to_num(merged["latent_post_slope"])
                y = to_num(merged[col])
                ok = x.notna() & y.notna()
                if ok.sum() >= 3:
                    row[f"{col}_corr_with_latent_post_slope"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if "satbreak" in lc:
                x = to_num(merged["latent_satbreak_probability"])
                y = to_num(merged[col])
                ok = x.notna() & y.notna()
                if ok.sum() >= 3:
                    row[f"{col}_corr_with_latent_satbreak_probability"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if "response_class" in lc:
                row[f"{col}_class_match_fraction"] = float((merged["latent_response_class"].astype(str) == merged[col].astype(str)).mean())

        rows.append(row)

    return pd.DataFrame(rows)

# =============================================================================
# Trait re-analysis on latent responses
# =============================================================================

def choose_control_cols(df):
    candidates = [
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
        "soil_clay",
        "soil_silt",
        "lat",
        "lon",
        "abs_lat",
    ]

    out = []
    for c in candidates:
        if c in df.columns and finite_n(df, c) >= 20:
            out.append(c)

    return out

def build_model_df(latent):
    if not TRAIT_DATASET.exists():
        return pd.DataFrame()

    trait = pd.read_csv(TRAIT_DATASET, low_memory=False)
    trait = make_unique_columns(trait)
    if "point_id" not in trait.columns:
        return pd.DataFrame()

    trait["point_id"] = trait["point_id"].astype(str)
    model = latent.merge(trait, on="point_id", how="left", suffixes=("", "_trait"))
    model = make_unique_columns(model)

    for c in model.columns:
        if c not in ["point_id", "latent_response_class"]:
            try:
                model[c] = to_num(model[c])
            except Exception:
                pass

    return model

def complete_case(df, cols):
    d = df.copy()
    ok = pd.Series(True, index=d.index)
    for c in cols:
        if c not in d.columns:
            return pd.DataFrame()
        d[c] = to_num(d[c])
        ok &= d[c].notna()
    return d.loc[ok].copy()

def fit_ols_trait_models(model_df):
    rows = []

    if model_df.empty:
        return pd.DataFrame([{"status": "trait_dataset_missing_or_empty"}])

    outcomes = [
        "latent_slope_change",
        "latent_post_slope",
        "latent_satbreak_probability",
        "p_saturation",
        "p_breakdown",
        "p_enhancement",
        "p_inconclusive",
    ]

    controls = choose_control_cols(model_df)

    for outcome in outcomes:
        if outcome not in model_df.columns:
            continue

        available_traits = [t for t in TRAITS if t in model_df.columns and finite_n(model_df, t) >= 5]

        for model_role, trait_set in [
            ("traits_only", available_traits),
            ("climate_soil_spatial_adjusted_core_traits", [t for t in PRIMARY_TRAITS if t in available_traits]),
            ("all_available_traits_adjusted", available_traits),
        ]:
            predictors = trait_set.copy()
            if "adjusted" in model_role:
                predictors += controls
            predictors = list(dict.fromkeys([p for p in predictors if p in model_df.columns]))

            if not predictors:
                rows.append({
                    "outcome": outcome,
                    "model_role": model_role,
                    "status": "no_predictors",
                })
                continue

            d = complete_case(model_df, [outcome] + predictors)
            if len(d) < 20:
                rows.append({
                    "outcome": outcome,
                    "model_role": model_role,
                    "status": "low_n",
                    "n": int(len(d)),
                })
                continue

            y = standardize(d[outcome])
            X = pd.DataFrame(index=d.index)
            for p in predictors:
                X[p] = standardize(d[p])
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X = sm.add_constant(X, has_constant="add")

            try:
                fit0 = sm.OLS(y, X).fit()
                fit = fit0.get_robustcov_results(cov_type="HC3")
                params = pd.Series(fit.params, index=X.columns)
                bse = pd.Series(fit.bse, index=X.columns)
                tvals = pd.Series(fit.tvalues, index=X.columns)
                pvals = pd.Series(fit.pvalues, index=X.columns)
                ci = pd.DataFrame(fit.conf_int(), index=X.columns, columns=["ci_low", "ci_high"])

                for term in predictors:
                    rows.append({
                        "outcome": outcome,
                        "model_role": model_role,
                        "model_family": "OLS_HC3",
                        "n": int(len(d)),
                        "term": term,
                        "estimate": float(params.get(term, np.nan)),
                        "std_error": float(bse.get(term, np.nan)),
                        "t_stat": float(tvals.get(term, np.nan)),
                        "p_value": float(pvals.get(term, np.nan)),
                        "ci_low": float(ci.loc[term, "ci_low"]) if term in ci.index else np.nan,
                        "ci_high": float(ci.loc[term, "ci_high"]) if term in ci.index else np.nan,
                        "r2": float(fit0.rsquared),
                        "status": "ok",
                        "controls_used": ",".join([c for c in controls if c in predictors]),
                    })

            except Exception as e:
                rows.append({
                    "outcome": outcome,
                    "model_role": model_role,
                    "model_family": "OLS_HC3",
                    "n": int(len(d)),
                    "status": f"failed: {type(e).__name__}: {e}",
                })

    return pd.DataFrame(rows)

def fit_ml_trait_importance(model_df):
    rows = []

    if model_df.empty:
        return pd.DataFrame([{"status": "trait_dataset_missing_or_empty"}])

    outcomes = [
        "latent_slope_change",
        "latent_post_slope",
        "latent_satbreak_probability",
        "p_saturation",
        "p_breakdown",
    ]

    controls = choose_control_cols(model_df)

    for outcome in outcomes:
        if outcome not in model_df.columns:
            continue

        available_traits = [t for t in TRAITS if t in model_df.columns and finite_n(model_df, t) >= 5]

        predictor_sets = {
            "traits_only": available_traits,
            "traits_plus_climate_soil_spatial": list(dict.fromkeys(available_traits + controls)),
        }

        for set_name, predictors in predictor_sets.items():
            predictors = [p for p in predictors if p in model_df.columns]
            d = complete_case(model_df, [outcome] + predictors)

            if len(d) < 40 or not predictors:
                rows.append({
                    "outcome": outcome,
                    "predictor_set": set_name,
                    "status": "low_n_or_no_predictors",
                    "n": int(len(d)),
                })
                continue

            y = standardize(d[outcome])
            X = pd.DataFrame(index=d.index)
            for p in predictors:
                X[p] = standardize(d[p])
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            k = min(5, len(d))
            if k < 3:
                rows.append({
                    "outcome": outcome,
                    "predictor_set": set_name,
                    "status": "low_cv_n",
                    "n": int(len(d)),
                })
                continue

            try:
                kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
                importances = []
                scores = []

                for fold, (tr, te) in enumerate(kf.split(X)):
                    model = RandomForestRegressor(
                        n_estimators=400,
                        random_state=RANDOM_SEED + fold,
                        min_samples_leaf=3,
                        max_features="sqrt",
                    )
                    model.fit(X.iloc[tr], y.iloc[tr])
                    pred = model.predict(X.iloc[te])
                    scores.append(r2_score(y.iloc[te], pred))
                    importances.append(model.feature_importances_)

                imp = np.vstack(importances)
                mean_imp = imp.mean(axis=0)
                sd_imp = imp.std(axis=0)
                ranks = pd.Series(-mean_imp, index=predictors).rank(method="dense").astype(int)

                for predictor, mi, si in zip(predictors, mean_imp, sd_imp):
                    rows.append({
                        "outcome": outcome,
                        "predictor_set": set_name,
                        "model_family": "RandomForestRegressor",
                        "predictor": predictor,
                        "importance_mean": float(mi),
                        "importance_sd": float(si),
                        "rank": int(ranks[predictor]),
                        "cv_r2_mean": float(np.nanmean(scores)),
                        "cv_r2_sd": float(np.nanstd(scores)),
                        "n": int(len(d)),
                        "status": "ok",
                    })

            except Exception as e:
                rows.append({
                    "outcome": outcome,
                    "predictor_set": set_name,
                    "model_family": "RandomForestRegressor",
                    "n": int(len(d)),
                    "status": f"failed: {type(e).__name__}: {e}",
                })

    return pd.DataFrame(rows)

def crossfit_rf_pred(X, y):
    n = len(y)
    k = min(5, n)
    if k < 3:
        return pd.Series(np.nan, index=y.index), np.nan

    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    pred = pd.Series(np.nan, index=y.index)

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

def fit_dml_trait_effects(model_df):
    rows = []

    if model_df.empty:
        return pd.DataFrame([{"status": "trait_dataset_missing_or_empty"}])

    outcomes = [
        "latent_slope_change",
        "latent_post_slope",
        "latent_satbreak_probability",
        "p_saturation",
        "p_breakdown",
    ]

    controls = choose_control_cols(model_df)

    for outcome in outcomes:
        if outcome not in model_df.columns:
            continue

        for trait in PRIMARY_TRAITS:
            if trait not in model_df.columns or finite_n(model_df, trait) < 20:
                rows.append({
                    "outcome": outcome,
                    "treatment_trait": trait,
                    "status": "trait_missing_or_low_coverage",
                })
                continue

            other_traits = [t for t in PRIMARY_TRAITS if t != trait and t in model_df.columns]
            controls_use = list(dict.fromkeys(controls + other_traits))
            cols = [outcome, trait] + controls_use
            d = complete_case(model_df, cols)

            if len(d) < 40:
                rows.append({
                    "outcome": outcome,
                    "treatment_trait": trait,
                    "status": "low_n",
                    "n": int(len(d)),
                })
                continue

            try:
                y = standardize(d[outcome])
                t = standardize(d[trait])

                X = pd.DataFrame(index=d.index)
                for c in controls_use:
                    X[c] = standardize(d[c])
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                if X.shape[1] == 0:
                    y_hat = pd.Series(y.mean(), index=y.index)
                    t_hat = pd.Series(t.mean(), index=t.index)
                    r2_y = 0.0
                    r2_t = 0.0
                else:
                    y_hat, r2_y = crossfit_rf_pred(X, y)
                    t_hat, r2_t = crossfit_rf_pred(X, t)

                y_res = y - y_hat
                t_res = t - t_hat

                X_final = sm.add_constant(pd.DataFrame({trait: t_res}, index=d.index), has_constant="add")
                fit0 = sm.OLS(y_res, X_final).fit()
                fit = fit0.get_robustcov_results(cov_type="HC3")

                params = pd.Series(fit.params, index=X_final.columns)
                bse = pd.Series(fit.bse, index=X_final.columns)
                tvals = pd.Series(fit.tvalues, index=X_final.columns)
                pvals = pd.Series(fit.pvalues, index=X_final.columns)
                ci = pd.DataFrame(fit.conf_int(), index=X_final.columns, columns=["ci_low", "ci_high"])

                rows.append({
                    "outcome": outcome,
                    "treatment_trait": trait,
                    "model_family": "DML_RF",
                    "n": int(len(d)),
                    "effect_estimate": float(params[trait]),
                    "std_error": float(bse[trait]),
                    "t_stat": float(tvals[trait]),
                    "p_value": float(pvals[trait]),
                    "ci_low": float(ci.loc[trait, "ci_low"]),
                    "ci_high": float(ci.loc[trait, "ci_high"]),
                    "r2_y_nuisance": float(r2_y),
                    "r2_t_nuisance": float(r2_t),
                    "controls_used": ",".join(controls_use),
                    "status": "ok",
                })

            except Exception as e:
                rows.append({
                    "outcome": outcome,
                    "treatment_trait": trait,
                    "model_family": "DML_RF",
                    "n": int(len(d)),
                    "status": f"failed: {type(e).__name__}: {e}",
                })

    return pd.DataFrame(rows)

# =============================================================================
# Figures
# =============================================================================

def plot_latent_map(latent):
    d = latent.copy()
    if "lat" not in d.columns or "lon" not in d.columns:
        return None

    d["lat"] = to_num(d["lat"])
    d["lon"] = to_num(d["lon"])
    d = d.dropna(subset=["lat", "lon"])

    if d.empty:
        return None

    class_to_color = {
        "enhancement": "tab:blue",
        "saturation": "tab:orange",
        "breakdown": "tab:red",
        "inconclusive": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    for cl, sub in d.groupby("latent_response_class"):
        ax.scatter(
            sub["lon"],
            sub["lat"],
            s=35 + 80 * to_num(sub["latent_satbreak_probability"]).fillna(0),
            alpha=0.75,
            label=cl,
            color=class_to_color.get(cl, "black"),
            edgecolor="none",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 8. Latent product-adjusted response class by point")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG8, dpi=300)
    fig.savefig(FIG8_PDF)
    plt.close(fig)

    return FIG8

def plot_product_bias(bias):
    if bias.empty:
        return None

    d = bias[
        (bias["outcome"].isin(["slope_change", "post_slope", "satbreak_indicator"])) &
        (bias["effect_family"].isin(["gpp_product", "et_product", "gpp_et_interaction"]))
    ].copy()

    if d.empty:
        return None

    d["label"] = d["outcome"] + " | " + d["effect_family"] + " | " + d["level"]
    d = d.sort_values(["outcome", "effect_family", "bias_effect"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.25 * len(d))))
    y = np.arange(len(d))
    ax.errorbar(
        d["bias_effect"],
        y,
        xerr=np.vstack([
            d["bias_effect"] - to_num(d["ci_low"]),
            to_num(d["ci_high"]) - d["bias_effect"],
        ]),
        fmt="o",
        capsize=2,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=7)
    ax.set_xlabel("Bias effect relative to latent response")
    ax.set_title("Figure 9. Product-family bias effects")
    fig.tight_layout()
    fig.savefig(FIG9, dpi=300)
    fig.savefig(FIG9_PDF)
    plt.close(fig)

    return FIG9

def plot_flux_decomposition(class_summary):
    if class_summary.empty or "status" in class_summary.columns:
        return None

    d = class_summary.copy()

    if "latent_response_class" not in d.columns:
        return None

    pivot = (
        d.groupby("latent_response_class")
        .agg(
            median_gpp_contribution=("median_gpp_contribution", "median"),
            median_et_contribution=("median_et_contribution", "median"),
        )
        .reset_index()
    )

    if pivot.empty:
        return None

    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, pivot["median_gpp_contribution"], width, label="GPP contribution")
    ax.bar(x + width/2, pivot["median_et_contribution"], width, label="ET contribution")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["latent_response_class"], rotation=30, ha="right")
    ax.set_ylabel("Contribution to log(WUE) stress slope")
    ax.set_title("Figure 10. Flux decomposition by latent response class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG10, dpi=300)
    fig.savefig(FIG10_PDF)
    plt.close(fig)

    return FIG10

def plot_lofo(lofo):
    if lofo.empty or "status" not in lofo.columns:
        return None

    d = lofo[lofo["status"].eq("ok")].copy()
    if d.empty:
        return None

    metrics = [
        "slope_change_correlation_with_main",
        "post_slope_correlation_with_main",
        "satbreak_probability_correlation_with_main",
        "class_match_fraction",
    ]

    plot_df = d.set_index("test")[[m for m in metrics if m in d.columns]]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
    im = ax.imshow(plot_df.values, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_title("Figure 11. Leave-one-product-family-out robustness")
    fig.colorbar(im, ax=ax, label="robustness score")
    fig.tight_layout()
    fig.savefig(FIG11, dpi=300)
    fig.savefig(FIG11_PDF)
    plt.close(fig)

    return FIG11

def plot_external_ref(external):
    if external.empty:
        return None

    d = external[external["status"].eq("ok")].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No local external flux reference files detected", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG12, dpi=300)
        fig.savefig(FIG12_PDF)
        plt.close(fig)
        return FIG12

    corr_cols = [c for c in d.columns if "corr_with_latent" in c or "class_match_fraction" in c]
    if not corr_cols:
        return None

    long = d.melt(
        id_vars=["reference_family", "path", "status"],
        value_vars=corr_cols,
        var_name="comparison_metric",
        value_name="score",
    ).dropna(subset=["score"])

    if long.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(long))))
    labels = long["reference_family"] + " | " + long["comparison_metric"]
    y = np.arange(len(long))
    ax.barh(y, long["score"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("external-reference agreement score")
    ax.set_title("Figure 12. External soft-reference comparison")
    fig.tight_layout()
    fig.savefig(FIG12, dpi=300)
    fig.savefig(FIG12_PDF)
    plt.close(fig)

    return FIG12

def plot_trait_effects(trait_ols, trait_dml):
    pieces = []

    if not trait_ols.empty:
        d = trait_ols[
            (trait_ols.get("status", "") == "ok") &
            (trait_ols.get("term", "").isin(PRIMARY_TRAITS))
        ].copy()
        if not d.empty:
            d["trait"] = d["term"]
            d["estimate_plot"] = d["estimate"]
            d["source"] = "OLS_HC3"
            pieces.append(d)

    if not trait_dml.empty:
        d = trait_dml[
            (trait_dml.get("status", "") == "ok") &
            (trait_dml.get("treatment_trait", "").isin(PRIMARY_TRAITS))
        ].copy()
        if not d.empty:
            d["trait"] = d["treatment_trait"]
            d["estimate_plot"] = d["effect_estimate"]
            d["source"] = "DML_RF"
            pieces.append(d)

    if not pieces:
        return None

    all_d = pd.concat(pieces, ignore_index=True)
    all_d = all_d[all_d["outcome"].isin(["latent_slope_change", "latent_post_slope", "latent_satbreak_probability"])].copy()
    if all_d.empty:
        return None

    all_d["label"] = all_d["outcome"] + " | " + all_d["source"] + " | " + all_d["trait"]
    all_d = all_d.sort_values(["outcome", "source", "trait"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(all_d))))
    y = np.arange(len(all_d))

    low = to_num(all_d.get("ci_low", pd.Series(np.nan, index=all_d.index)))
    high = to_num(all_d.get("ci_high", pd.Series(np.nan, index=all_d.index)))
    est = to_num(all_d["estimate_plot"])

    ax.errorbar(
        est,
        y,
        xerr=np.vstack([
            est - low,
            high - est,
        ]),
        fmt="o",
        capsize=2,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(all_d["label"], fontsize=7)
    ax.set_xlabel("standardized trait effect")
    ax.set_title("Figure 13. Trait effects on latent product-adjusted response")
    fig.tight_layout()
    fig.savefig(FIG13, dpi=300)
    fig.savefig(FIG13_PDF)
    plt.close(fig)

    return FIG13

# =============================================================================
# README / manifest
# =============================================================================

def write_readme(manifest):
    lines = []
    lines.append("# Phase 8: Latent product-adjusted ecosystem response")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This phase resolves satellite product disagreement statistically by treating each product/metric/stress/season result as a biased observation of a latent product-adjusted satellite response phenotype. It does not claim tower-equivalent validation.")
    lines.append("")
    lines.append("## Implemented framework")
    lines.append("")
    lines.append("1. Build a latent consensus response for slope change, post-transition slope, and saturation/breakdown probability.")
    lines.append("2. Classify latent point-level response using posterior/bootstrap class probabilities.")
    lines.append("3. Decompose carbon-side and water-side disagreement through GPP-product and ET-product bias effects.")
    lines.append("4. Run leave-one-product-family-out tests for MODIS/GOSIF/PML GPP and MODIS/GLEAM/PML ET.")
    lines.append("5. Run algorithmic-independence weighted consensus sensitivity.")
    lines.append("6. Attempt log-flux decomposition into GPP and ET contributions using local metric matrices.")
    lines.append("7. Detect and compare local external soft-reference flux products when available.")
    lines.append("8. Re-run trait analysis using latent response outcomes.")
    lines.append("")
    lines.append("## Safe interpretation")
    lines.append("")
    lines.append("The result is a latent product-adjusted satellite phenotype, not a tower-validated true ecosystem flux response.")
    lines.append("")
    lines.append("## Key outputs")
    lines.append("")
    for p in [
        OBS_OUT,
        LATENT_OUT,
        CLASS_PROB_OUT,
        PRODUCT_BIAS_OUT,
        GPP_ET_BIAS_OUT,
        METRIC_BIAS_OUT,
        STRESS_BIAS_OUT,
        SEASON_BIAS_OUT,
        CO2_BIAS_OUT,
        LOFO_OUT,
        WEIGHT_SENS_OUT,
        FLUX_CLASS_OUT,
        FLUX_PRODUCT_OUT,
        EXTERNAL_REF_OUT,
        TRAIT_LATENT_OUT,
        TRAIT_ML_OUT,
        TRAIT_DML_OUT,
        FIG8,
        FIG9,
        FIG10,
        FIG11,
        FIG12,
        FIG13,
    ]:
        lines.append(f"- `{p}`")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=80)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--boot-iter", type=int, default=20)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    if args.fast:
        args.n_boot = min(args.n_boot, 20)
        args.n_iter = min(args.n_iter, 20)
        args.boot_iter = min(args.boot_iter, 12)

    print("PHASE 8 START")
    print("n_boot:", args.n_boot)
    print("n_iter:", args.n_iter)
    print("boot_iter:", args.boot_iter)

    obs = build_observation_table()
    save_csv(obs, OBS_OUT)

    print("Observation table summary:")
    print("rows:", len(obs))
    print("points:", obs["point_id"].nunique())
    print("product_combos:", sorted(obs["product_combo"].dropna().unique().tolist()))
    print("metrics:", sorted(obs["metric"].dropna().unique().tolist()))
    print("stress_definitions:", sorted(obs["stress_definition"].dropna().unique().tolist()))
    print("growing_seasons:", sorted(obs["growing_season"].dropna().unique().tolist()))
    print("co2_versions:", sorted(obs["co2_version"].dropna().unique().tolist()))

    suite = fit_latent_suite(
        obs,
        weight_col=None,
        n_boot=args.n_boot,
        n_iter=args.n_iter,
        boot_iter=args.boot_iter,
        label="equal_weights_main",
    )

    latent, class_prob, bias = build_latent_response_tables(obs, suite)

    save_csv(latent, LATENT_OUT)
    save_csv(class_prob, CLASS_PROB_OUT)

    product_bias, gpp_et_bias, metric_bias, stress_bias, season_bias, co2_bias = split_bias_tables(bias)
    save_csv(product_bias, PRODUCT_BIAS_OUT)
    save_csv(gpp_et_bias, GPP_ET_BIAS_OUT)
    save_csv(metric_bias, METRIC_BIAS_OUT)
    save_csv(stress_bias, STRESS_BIAS_OUT)
    save_csv(season_bias, SEASON_BIAS_OUT)
    save_csv(co2_bias, CO2_BIAS_OUT)

    lofo = run_leave_one_family_out(obs, latent, n_iter=max(15, args.n_iter - 5))
    save_csv(lofo, LOFO_OUT)

    weight_sens = run_weight_sensitivity(obs, latent, n_iter=max(15, args.n_iter - 5))
    save_csv(weight_sens, WEIGHT_SENS_OUT)

    flux_point, flux_status, flux_class, flux_product = compute_flux_decomposition(obs, latent)
    if flux_point.empty:
        save_csv(flux_status, FLUX_POINT_OUT)
    else:
        save_csv(flux_point, FLUX_POINT_OUT)
    if flux_class.empty:
        save_csv(flux_status, FLUX_CLASS_OUT)
    else:
        save_csv(flux_class, FLUX_CLASS_OUT)
    if flux_product.empty:
        save_csv(flux_status, FLUX_PRODUCT_OUT)
    else:
        save_csv(flux_product, FLUX_PRODUCT_OUT)

    external = external_comparison(latent)
    save_csv(external, EXTERNAL_REF_OUT)

    model_df = build_model_df(latent)
    trait_ols = fit_ols_trait_models(model_df)
    trait_ml = fit_ml_trait_importance(model_df)
    trait_dml = fit_dml_trait_effects(model_df)
    save_csv(trait_ols, TRAIT_LATENT_OUT)
    save_csv(trait_ml, TRAIT_ML_OUT)
    save_csv(trait_dml, TRAIT_DML_OUT)

    fig8 = plot_latent_map(latent)
    fig9 = plot_product_bias(bias)
    fig10 = plot_flux_decomposition(flux_class)
    fig11 = plot_lofo(lofo)
    fig12 = plot_external_ref(external)
    fig13 = plot_trait_effects(trait_ols, trait_dml)

    for fig in [fig8, fig9, fig10, fig11, fig12, fig13]:
        if fig is not None:
            print("WROTE", fig)

    manifest = {
        "phase": "Phase 8: latent product-adjusted ecosystem response",
        "interpretation_guardrail": "This is a latent product-adjusted satellite phenotype, not tower validation and not proof of true ecosystem flux response.",
        "inputs": {
            "raw_response": str(RAW_RESPONSE),
            "co2_response": str(CO2_RESPONSE),
            "raw_surface": str(RAW_SURFACE),
            "co2_surface": str(CO2_SURFACE),
            "trait_dataset": str(TRAIT_DATASET),
            "phase3_consensus": str(PHASE3_CONSENSUS),
            "phase3_combo": str(PHASE3_COMBO),
        },
        "parameters": {
            "n_boot": args.n_boot,
            "n_iter": args.n_iter,
            "boot_iter": args.boot_iter,
            "effect_families": EFFECT_FAMILIES,
            "independence_weights": INDEPENDENCE_WEIGHTS,
        },
        "observation_summary": {
            "n_rows": int(len(obs)),
            "n_points": int(obs["point_id"].nunique()),
            "product_combos": sorted(obs["product_combo"].dropna().unique().tolist()),
            "metrics": sorted(obs["metric"].dropna().unique().tolist()),
            "stress_definitions": sorted(obs["stress_definition"].dropna().unique().tolist()),
            "growing_seasons": sorted(obs["growing_season"].dropna().unique().tolist()),
            "co2_versions": sorted(obs["co2_version"].dropna().unique().tolist()),
        },
        "latent_response_summary": {
            "n_points": int(latent["point_id"].nunique()),
            "class_counts": latent["latent_response_class"].value_counts(dropna=False).to_dict() if "latent_response_class" in latent.columns else {},
            "median_latent_slope_change": safe_median(latent["latent_slope_change"]) if "latent_slope_change" in latent.columns else np.nan,
            "median_latent_post_slope": safe_median(latent["latent_post_slope"]) if "latent_post_slope" in latent.columns else np.nan,
            "median_latent_satbreak_probability": safe_median(latent["latent_satbreak_probability"]) if "latent_satbreak_probability" in latent.columns else np.nan,
        },
        "outputs": {
            "observations": str(OBS_OUT),
            "latent_response_by_point": str(LATENT_OUT),
            "class_probabilities": str(CLASS_PROB_OUT),
            "product_bias": str(PRODUCT_BIAS_OUT),
            "gpp_et_bias_decomposition": str(GPP_ET_BIAS_OUT),
            "metric_bias": str(METRIC_BIAS_OUT),
            "stress_bias": str(STRESS_BIAS_OUT),
            "season_bias": str(SEASON_BIAS_OUT),
            "co2_bias": str(CO2_BIAS_OUT),
            "leave_one_family_out": str(LOFO_OUT),
            "independence_weight_sensitivity": str(WEIGHT_SENS_OUT),
            "flux_decomposition_by_point_product": str(FLUX_POINT_OUT),
            "flux_decomposition_by_response_class": str(FLUX_CLASS_OUT),
            "flux_decomposition_by_product_family": str(FLUX_PRODUCT_OUT),
            "external_flux_reference_inventory": str(EXTERNAL_REF_INVENTORY_OUT),
            "external_flux_reference_comparison": str(EXTERNAL_REF_OUT),
            "trait_effects_ols": str(TRAIT_LATENT_OUT),
            "trait_effects_ml": str(TRAIT_ML_OUT),
            "trait_effects_dml": str(TRAIT_DML_OUT),
            "figures": [str(x) for x in [fig8, fig9, fig10, fig11, fig12, fig13] if x is not None],
        },
        "main_safe_claim": "We infer a latent product-adjusted satellite response phenotype and report product, metric, stress-definition, season, CO2, and external-reference uncertainty.",
        "claim_to_avoid": "The latent response is the true tower-validated ecosystem flux response.",
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"WROTE {MANIFEST_OUT}")

    write_readme(manifest)

    print("")
    print("DONE Phase 8.")
    print("")
    print("LATENT RESPONSE CLASS COUNTS:")
    print(latent["latent_response_class"].value_counts(dropna=False).to_string())
    print("")
    print("LEAVE-ONE-FAMILY-OUT:")
    print(lofo.to_string(index=False))
    print("")
    print("INDEPENDENCE WEIGHT SENSITIVITY:")
    print(weight_sens.to_string(index=False))
    print("")
    print("EXTERNAL REFERENCE COMPARISON:")
    print(external.to_string(index=False))
    print("")
    print("TRAIT EFFECTS ON LATENT RESPONSE, TOP ROWS:")
    print(trait_ols.head(40).to_string(index=False))

if __name__ == "__main__":
    main()
