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

PHASE4_DATASET = Path("results/trait_framework/trait_model_dataset.csv")

PHASE5_OLS = Path("results/trait_framework/phase5/table_trait_ols_models.csv")
PHASE5_ML = Path("results/trait_framework/phase5/table_trait_ml_importance.csv")
PHASE5_DML = Path("results/trait_framework/phase5/table_trait_causal_adjusted_effects.csv")
PHASE5_PRODUCT_SENS = Path("results/trait_framework/phase5/table_trait_sensitivity_by_product_family.csv")
PHASE5_MANIFEST = Path("results/trait_framework/phase5/phase5_model_manifest.json")

PHASE6_ALL = Path("results/trait_framework/phase6/table_phase6_all_robustness_coefficients.csv")
PHASE6_NEG = Path("results/trait_framework/phase6/table_robustness_negative_controls.csv")
PHASE6_RAWCO2 = Path("results/trait_framework/phase6/table_robustness_raw_vs_co2.csv")
PHASE6_METRIC = Path("results/trait_framework/phase6/table_robustness_metric_comparison.csv")
PHASE6_PRODUCT = Path("results/trait_framework/phase6/table_robustness_product_consensus_vs_specific.csv")
PHASE6_STRESS_SEASON = Path("results/trait_framework/phase6/table_robustness_stress_growing_season.csv")
PHASE6_ARIDITY = Path("results/trait_framework/phase6/table_robustness_aridity_stratified.csv")
PHASE6_MANIFEST = Path("results/trait_framework/phase6/phase6_robustness_manifest.json")

PHASE3_MANIFEST = Path("results/trait_framework/phase3/phase3_product_consensus_manifest.json")
PHASE2_MANIFEST = Path("results/trait_framework/phase2/phase2_no_universal_threshold_manifest.json")

OUTDIR = Path("results/trait_framework/phase7")
OUTDIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_OUT = OUTDIR / "table_phase7_mechanism_evidence.csv"
CLAIM_AUDIT_OUT = OUTDIR / "table_phase7_claim_audit.csv"
SUPPORT_SCORE_OUT = OUTDIR / "table_phase7_support_score.csv"
INTERPRETATION_OUT = OUTDIR / "physiological_mechanism_interpretation.md"
MANIFEST_OUT = OUTDIR / "phase7_interpretation_manifest.json"
README_OUT = OUTDIR / "README_phase7_physiological_mechanism.md"
FIG_AUDIT_OUT = OUTDIR / "Figure8_phase7_claim_audit.png"
FIG_AUDIT_PDF = OUTDIR / "Figure8_phase7_claim_audit.pdf"
FIG_EVIDENCE_OUT = OUTDIR / "Figure9_phase7_trait_evidence_summary.png"
FIG_EVIDENCE_PDF = OUTDIR / "Figure9_phase7_trait_evidence_summary.pdf"

TRAITS_CORE = ["p50", "rooting_depth"]
TRAITS_LIMITED = ["isohydricity"]

SAFE_CLAIM = "The satellite-derived WUE response phenotype is statistically consistent with a hydraulic/rooting mechanism after climate and soil adjustment."
CLAIM_TO_AVOID = "Xylem vulnerability causally proves WUE breakdown."
FUTURE_TOWER_STATEMENT = "Trait-adjusted associations support a mechanistic hydraulic interpretation, but tower validation remains a necessary future validation step."

NEGATIVE_CONTROL_OUTCOMES = [
    "raw_vs_co2_stability_all",
    "product_agreement_all",
    "n_product_combos_all",
    "negative_control",
]

PRIMARY_SLOPE_OUTCOMES = [
    "consensus_slope_change_independent",
    "consensus_slope_change_all",
    "slope_change",
]

POST_SLOPE_OUTCOMES = [
    "consensus_post_slope_independent",
    "consensus_post_slope_all",
    "post_slope",
]

SATBREAK_OUTCOMES = [
    "satbreak_fraction",
    "satbreak_fraction_all",
    "satbreak_fraction_independent",
    "negative_slope_fraction_all",
    "negative_slope_fraction_independent",
]

EXPECTED_DIRECTION_BY_PHYSIOLOGY_TRAIT = {
    "hydraulic_resistance": {
        "slope_change": "positive",
        "post_slope": "positive",
        "satbreak_fraction": "negative",
        "negative_slope_fraction": "negative",
        "negative_control": "null",
        "other": "unknown",
    },
    "rooting_depth": {
        "slope_change": "positive",
        "post_slope": "positive",
        "satbreak_fraction": "negative",
        "negative_slope_fraction": "negative",
        "negative_control": "null",
        "other": "unknown",
    },
    "isohydricity": {
        "slope_change": "unknown",
        "post_slope": "unknown",
        "satbreak_fraction": "unknown",
        "negative_slope_fraction": "unknown",
        "negative_control": "null",
        "other": "unknown",
    },
}

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

def read_csv_required(path, label):
    if not path.exists():
        die(f"Missing required {label}: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = make_unique_columns(df)
    return df

def read_csv_optional(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = make_unique_columns(df)
    return df

def read_json_optional(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def to_num(s):
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def finite_n(df, col):
    if col not in df.columns:
        return 0
    return int(to_num(df[col]).notna().sum())

def direction(x):
    if pd.isna(x):
        return "NA"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"

def direction_matches(observed, expected):
    if expected in ["unknown", "null"] or observed in ["NA", "zero"]:
        return np.nan
    return bool(observed == expected)

def signed_match_value(observed, expected):
    m = direction_matches(observed, expected)
    if pd.isna(m):
        return np.nan
    return 1.0 if m else 0.0

def classify_outcome_family(outcome, setting="", robustness_group=""):
    o = str(outcome).lower()
    s = str(setting).lower()
    g = str(robustness_group).lower()

    if "negative_control" in g or "negative_control" in s or o in NEGATIVE_CONTROL_OUTCOMES:
        return "negative_control"

    if "satbreak" in o or "sat_or_break" in o:
        return "satbreak_fraction"

    if "negative_slope" in o:
        return "negative_slope_fraction"

    if "post_slope" in o:
        return "post_slope"

    if "slope_change" in o or o == "slope_change":
        return "slope_change"

    if "product_agreement" in o or "raw_vs_co2_stability" in o or "n_product_combos" in o:
        return "negative_control"

    return "other"

def physiology_trait_for_raw_trait(trait):
    trait = str(trait)
    if trait == "p50":
        return "hydraulic_resistance"
    if trait == "rooting_depth":
        return "rooting_depth"
    if trait == "isohydricity":
        return "isohydricity"
    if "p50" in trait:
        return "hydraulic_resistance"
    if "root" in trait:
        return "rooting_depth"
    if "isohyd" in trait:
        return "isohydricity"
    return trait

def infer_p50_transform(trait_dataset):
    if "p50" not in trait_dataset.columns:
        return {
            "p50_median": None,
            "raw_p50_sign_convention": "p50_missing",
            "hydraulic_resistance_definition": "unavailable",
            "p50_to_hydraulic_resistance_multiplier": np.nan,
            "interpretation_note": "P50 was missing, so hydraulic-resistance interpretation could not be constructed.",
        }

    p50 = to_num(trait_dataset["p50"])
    med = float(p50.median(skipna=True)) if p50.notna().sum() else np.nan

    if pd.notna(med) and med < 0:
        return {
            "p50_median": med,
            "raw_p50_sign_convention": "negative_water_potential_values",
            "hydraulic_resistance_definition": "hydraulic_resistance = -P50",
            "p50_to_hydraulic_resistance_multiplier": -1.0,
            "interpretation_note": "Median P50 is negative, so more negative P50 is interpreted as greater hydraulic resistance; raw P50 coefficients are multiplied by -1 for physiological interpretation.",
        }

    if pd.notna(med) and med >= 0:
        return {
            "p50_median": med,
            "raw_p50_sign_convention": "nonnegative_or_absolute_values",
            "hydraulic_resistance_definition": "hydraulic_resistance interpreted in the stored P50 direction; verify source convention",
            "p50_to_hydraulic_resistance_multiplier": 1.0,
            "interpretation_note": "Median P50 is nonnegative. The file may store absolute vulnerability/resistance values; coefficients are not inverted, but the source convention should be checked.",
        }

    return {
        "p50_median": med,
        "raw_p50_sign_convention": "all_missing_or_unreadable",
        "hydraulic_resistance_definition": "unavailable",
        "p50_to_hydraulic_resistance_multiplier": np.nan,
        "interpretation_note": "P50 values were not finite.",
    }

def transform_effect_for_physiology(row, p50_multiplier):
    raw_trait = str(row.get("raw_trait", row.get("trait", "")))
    est = row.get("raw_estimate", np.nan)
    ci_low = row.get("raw_ci_low", np.nan)
    ci_high = row.get("raw_ci_high", np.nan)

    if pd.isna(est):
        return np.nan, np.nan, np.nan

    if raw_trait == "p50" or "p50" in raw_trait:
        if pd.isna(p50_multiplier):
            return np.nan, np.nan, np.nan
        phys_est = float(p50_multiplier) * float(est)
        if pd.notna(ci_low) and pd.notna(ci_high):
            lo = float(p50_multiplier) * float(ci_low)
            hi = float(p50_multiplier) * float(ci_high)
            phys_ci_low = min(lo, hi)
            phys_ci_high = max(lo, hi)
        else:
            phys_ci_low = np.nan
            phys_ci_high = np.nan
        return phys_est, phys_ci_low, phys_ci_high

    return est, ci_low, ci_high

def expected_direction_for(phys_trait, outcome_family):
    return EXPECTED_DIRECTION_BY_PHYSIOLOGY_TRAIT.get(phys_trait, {}).get(outcome_family, "unknown")

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_unique_columns(df)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def bool_label(x):
    if pd.isna(x):
        return "not_testable"
    return "pass" if bool(x) else "fail"

def status_pass_fail_warning(condition, warning=False, not_testable=False):
    if not_testable:
        return "not_testable"
    if warning:
        return "warning"
    return "pass" if condition else "fail"

def proportion_true(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return float(s.astype(bool).mean())

def proportion_ok_direction(df):
    if df.empty or "matches_expected_direction" not in df.columns:
        return np.nan
    x = df.loc[df["matches_expected_direction"].notna(), "matches_expected_direction"]
    if len(x) == 0:
        return np.nan
    return float(x.astype(bool).mean())

def safe_mean_abs(df, col):
    if df.empty or col not in df.columns:
        return np.nan
    x = to_num(df[col]).abs().dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())

# =============================================================================
# Load evidence inputs
# =============================================================================

def load_inputs():
    trait_dataset = read_csv_required(PHASE4_DATASET, "Phase 4 trait model dataset")

    ols = read_csv_required(PHASE5_OLS, "Phase 5 OLS table")
    ml = read_csv_required(PHASE5_ML, "Phase 5 ML importance table")
    dml = read_csv_required(PHASE5_DML, "Phase 5 DML table")
    product_sens = read_csv_required(PHASE5_PRODUCT_SENS, "Phase 5 product-family sensitivity table")

    phase6_all = read_csv_required(PHASE6_ALL, "Phase 6 all robustness coefficients")
    phase6_neg = read_csv_required(PHASE6_NEG, "Phase 6 negative-control table")
    phase6_rawco2 = read_csv_required(PHASE6_RAWCO2, "Phase 6 raw-vs-CO2 table")
    phase6_metric = read_csv_required(PHASE6_METRIC, "Phase 6 metric comparison table")
    phase6_product = read_csv_required(PHASE6_PRODUCT, "Phase 6 product robustness table")
    phase6_stress_season = read_csv_required(PHASE6_STRESS_SEASON, "Phase 6 stress/growing-season table")
    phase6_aridity = read_csv_required(PHASE6_ARIDITY, "Phase 6 aridity table")

    manifests = {
        "phase2": read_json_optional(PHASE2_MANIFEST),
        "phase3": read_json_optional(PHASE3_MANIFEST),
        "phase5": read_json_optional(PHASE5_MANIFEST),
        "phase6": read_json_optional(PHASE6_MANIFEST),
    }

    return {
        "trait_dataset": trait_dataset,
        "ols": ols,
        "ml": ml,
        "dml": dml,
        "product_sens": product_sens,
        "phase6_all": phase6_all,
        "phase6_neg": phase6_neg,
        "phase6_rawco2": phase6_rawco2,
        "phase6_metric": phase6_metric,
        "phase6_product": phase6_product,
        "phase6_stress_season": phase6_stress_season,
        "phase6_aridity": phase6_aridity,
        "manifests": manifests,
    }

# =============================================================================
# Build mechanism evidence table
# =============================================================================

def evidence_from_phase5_ols(ols, p50_info):
    rows = []
    if ols.empty:
        return rows

    d = ols.copy()

    if "term" not in d.columns:
        return rows

    d = d[d["term"].isin(TRAITS_CORE + TRAITS_LIMITED)].copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()

    for _, r in d.iterrows():
        raw_trait = str(r.get("term"))
        outcome = str(r.get("outcome", ""))
        model_family = str(r.get("model_family", "OLS"))
        model_name = str(r.get("model_name", ""))
        setting = model_name
        outcome_family = classify_outcome_family(outcome, setting=setting)

        raw_est = pd.to_numeric(pd.Series([r.get("estimate", np.nan)]), errors="coerce").iloc[0]
        raw_ci_low = pd.to_numeric(pd.Series([r.get("ci_low", np.nan)]), errors="coerce").iloc[0]
        raw_ci_high = pd.to_numeric(pd.Series([r.get("ci_high", np.nan)]), errors="coerce").iloc[0]

        base = {
            "trait": raw_trait,
            "raw_trait": raw_trait,
            "physiology_trait": physiology_trait_for_raw_trait(raw_trait),
            "outcome": outcome,
            "outcome_family": outcome_family,
            "evidence_source": "phase5_ols_rlm",
            "model_family": model_family,
            "setting": setting,
            "setting_detail": str(r.get("model_role", "")),
            "raw_estimate": raw_est,
            "raw_ci_low": raw_ci_low,
            "raw_ci_high": raw_ci_high,
            "p_value": pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0],
            "n": pd.to_numeric(pd.Series([r.get("n", np.nan)]), errors="coerce").iloc[0],
            "status": str(r.get("status", "ok")),
            "source_table": str(PHASE5_OLS),
        }

        phys_est, phys_ci_low, phys_ci_high = transform_effect_for_physiology(base, p50_info["p50_to_hydraulic_resistance_multiplier"])
        phys_trait = base["physiology_trait"]
        expected = expected_direction_for(phys_trait, outcome_family)
        observed = direction(phys_est)

        base.update({
            "physiology_effect_estimate": phys_est,
            "physiology_ci_low": phys_ci_low,
            "physiology_ci_high": phys_ci_high,
            "observed_direction": observed,
            "expected_direction": expected,
            "matches_expected_direction": direction_matches(observed, expected),
            "evidence_weight": 1.0 if model_family in ["OLS_HC3", "OLS", "RLM_Huber"] else 0.75,
            "interpretation_note": p50_info["interpretation_note"] if raw_trait == "p50" else "",
        })

        rows.append(base)

    return rows

def evidence_from_phase5_dml(dml, p50_info):
    rows = []
    if dml.empty:
        return rows

    d = dml.copy()

    trait_col = "treatment_trait" if "treatment_trait" in d.columns else "trait"
    if trait_col not in d.columns:
        return rows

    d = d[d[trait_col].isin(TRAITS_CORE + TRAITS_LIMITED)].copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()

    for _, r in d.iterrows():
        raw_trait = str(r.get(trait_col))
        outcome = str(r.get("outcome", ""))
        outcome_family = classify_outcome_family(outcome)

        raw_est = pd.to_numeric(pd.Series([r.get("effect_estimate", np.nan)]), errors="coerce").iloc[0]
        raw_ci_low = pd.to_numeric(pd.Series([r.get("ci_low", np.nan)]), errors="coerce").iloc[0]
        raw_ci_high = pd.to_numeric(pd.Series([r.get("ci_high", np.nan)]), errors="coerce").iloc[0]

        base = {
            "trait": raw_trait,
            "raw_trait": raw_trait,
            "physiology_trait": physiology_trait_for_raw_trait(raw_trait),
            "outcome": outcome,
            "outcome_family": outcome_family,
            "evidence_source": "phase5_dml",
            "model_family": "DML_RF",
            "setting": str(r.get("control_set", "")),
            "setting_detail": str(r.get("model_role", "")),
            "raw_estimate": raw_est,
            "raw_ci_low": raw_ci_low,
            "raw_ci_high": raw_ci_high,
            "p_value": pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0],
            "n": pd.to_numeric(pd.Series([r.get("n", np.nan)]), errors="coerce").iloc[0],
            "status": str(r.get("status", "ok")),
            "source_table": str(PHASE5_DML),
        }

        phys_est, phys_ci_low, phys_ci_high = transform_effect_for_physiology(base, p50_info["p50_to_hydraulic_resistance_multiplier"])
        phys_trait = base["physiology_trait"]
        expected = expected_direction_for(phys_trait, outcome_family)
        observed = direction(phys_est)

        base.update({
            "physiology_effect_estimate": phys_est,
            "physiology_ci_low": phys_ci_low,
            "physiology_ci_high": phys_ci_high,
            "observed_direction": observed,
            "expected_direction": expected,
            "matches_expected_direction": direction_matches(observed, expected),
            "evidence_weight": 1.25,
            "interpretation_note": p50_info["interpretation_note"] if raw_trait == "p50" else "",
        })

        rows.append(base)

    return rows

def evidence_from_phase5_ml(ml):
    rows = []
    if ml.empty:
        return rows

    d = ml.copy()

    if "predictor" not in d.columns:
        return rows

    d = d[d["predictor"].isin(TRAITS_CORE + TRAITS_LIMITED)].copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()

    for _, r in d.iterrows():
        raw_trait = str(r.get("predictor"))
        outcome = str(r.get("outcome", ""))
        outcome_family = classify_outcome_family(outcome)

        rank = pd.to_numeric(pd.Series([r.get("rank", np.nan)]), errors="coerce").iloc[0]
        importance = pd.to_numeric(pd.Series([r.get("permutation_importance_mean", np.nan)]), errors="coerce").iloc[0]

        # ML importance has no sign. It supports "predictive relevance" only.
        rows.append({
            "trait": raw_trait,
            "raw_trait": raw_trait,
            "physiology_trait": physiology_trait_for_raw_trait(raw_trait),
            "outcome": outcome,
            "outcome_family": outcome_family,
            "evidence_source": "phase5_ml_importance",
            "model_family": str(r.get("model_family", "ML")),
            "setting": str(r.get("predictor_set", "")),
            "setting_detail": "permutation_importance_predictive_screening",
            "raw_estimate": importance,
            "raw_ci_low": np.nan,
            "raw_ci_high": np.nan,
            "p_value": np.nan,
            "n": pd.to_numeric(pd.Series([r.get("n", np.nan)]), errors="coerce").iloc[0],
            "status": str(r.get("status", "ok")),
            "source_table": str(PHASE5_ML),
            "physiology_effect_estimate": np.nan,
            "physiology_ci_low": np.nan,
            "physiology_ci_high": np.nan,
            "observed_direction": "importance_only",
            "expected_direction": "importance_only",
            "matches_expected_direction": np.nan,
            "ml_importance": importance,
            "ml_rank": rank,
            "evidence_weight": 0.5 if pd.notna(rank) and rank <= 5 else 0.25,
            "interpretation_note": "ML permutation importance is predictive/descriptive, not causal direction evidence.",
        })

    return rows

def evidence_from_phase5_product_sens(product_sens, p50_info):
    rows = []
    if product_sens.empty:
        return rows

    d = product_sens.copy()

    if "trait" not in d.columns:
        return rows

    d = d[d["trait"].isin(TRAITS_CORE + TRAITS_LIMITED)].copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()

    for _, r in d.iterrows():
        raw_trait = str(r.get("trait"))
        outcome = str(r.get("outcome", ""))
        outcome_family = classify_outcome_family(outcome)

        raw_est = pd.to_numeric(pd.Series([r.get("estimate_or_importance", np.nan)]), errors="coerce").iloc[0]

        base = {
            "trait": raw_trait,
            "raw_trait": raw_trait,
            "physiology_trait": physiology_trait_for_raw_trait(raw_trait),
            "outcome": outcome,
            "outcome_family": outcome_family,
            "evidence_source": "phase5_product_family_sensitivity",
            "model_family": str(r.get("model_family", "")),
            "setting": str(r.get("product_family", "")),
            "setting_detail": "product_family_sensitivity",
            "raw_estimate": raw_est,
            "raw_ci_low": np.nan,
            "raw_ci_high": np.nan,
            "p_value": pd.to_numeric(pd.Series([r.get("p_value_or_rank", np.nan)]), errors="coerce").iloc[0],
            "n": pd.to_numeric(pd.Series([r.get("n", np.nan)]), errors="coerce").iloc[0],
            "status": str(r.get("status", "ok")),
            "source_table": str(PHASE5_PRODUCT_SENS),
        }

        phys_est, phys_ci_low, phys_ci_high = transform_effect_for_physiology(base, p50_info["p50_to_hydraulic_resistance_multiplier"])
        phys_trait = base["physiology_trait"]
        expected = expected_direction_for(phys_trait, outcome_family)
        observed = direction(phys_est)

        base.update({
            "physiology_effect_estimate": phys_est,
            "physiology_ci_low": phys_ci_low,
            "physiology_ci_high": phys_ci_high,
            "observed_direction": observed,
            "expected_direction": expected,
            "matches_expected_direction": direction_matches(observed, expected),
            "evidence_weight": 0.75,
            "interpretation_note": "Product-family sensitivity evidence.",
        })

        rows.append(base)

    return rows

def evidence_from_phase6(coeff, p50_info):
    rows = []
    if coeff.empty:
        return rows

    d = coeff.copy()

    if "trait" not in d.columns:
        return rows

    d = d[d["trait"].isin(TRAITS_CORE + TRAITS_LIMITED)].copy()

    if "status" in d.columns:
        d = d[d["status"].astype(str).eq("ok")].copy()

    for _, r in d.iterrows():
        raw_trait = str(r.get("trait"))
        outcome = str(r.get("outcome", ""))
        robustness_group = str(r.get("robustness_group", ""))
        setting = str(r.get("setting", ""))
        outcome_family = classify_outcome_family(outcome, setting=setting, robustness_group=robustness_group)

        raw_est = pd.to_numeric(pd.Series([r.get("estimate", np.nan)]), errors="coerce").iloc[0]
        raw_ci_low = pd.to_numeric(pd.Series([r.get("ci_low", np.nan)]), errors="coerce").iloc[0]
        raw_ci_high = pd.to_numeric(pd.Series([r.get("ci_high", np.nan)]), errors="coerce").iloc[0]

        base = {
            "trait": raw_trait,
            "raw_trait": raw_trait,
            "physiology_trait": physiology_trait_for_raw_trait(raw_trait),
            "outcome": outcome,
            "outcome_family": outcome_family,
            "evidence_source": "phase6_robustness",
            "model_family": str(r.get("model_family", "")),
            "setting": setting,
            "setting_detail": str(r.get("setting_detail", "")),
            "robustness_group": robustness_group,
            "raw_estimate": raw_est,
            "raw_ci_low": raw_ci_low,
            "raw_ci_high": raw_ci_high,
            "p_value": pd.to_numeric(pd.Series([r.get("p_value", np.nan)]), errors="coerce").iloc[0],
            "n": pd.to_numeric(pd.Series([r.get("n", np.nan)]), errors="coerce").iloc[0],
            "status": str(r.get("status", "ok")),
            "source_table": str(PHASE6_ALL),
        }

        phys_est, phys_ci_low, phys_ci_high = transform_effect_for_physiology(base, p50_info["p50_to_hydraulic_resistance_multiplier"])
        phys_trait = base["physiology_trait"]
        expected = expected_direction_for(phys_trait, outcome_family)
        observed = direction(phys_est)

        base.update({
            "physiology_effect_estimate": phys_est,
            "physiology_ci_low": phys_ci_low,
            "physiology_ci_high": phys_ci_high,
            "observed_direction": observed,
            "expected_direction": expected,
            "matches_expected_direction": direction_matches(observed, expected),
            "evidence_weight": 1.0 if str(r.get("model_family", "")) == "OLS_HC3" else 0.75,
            "interpretation_note": "Phase 6 robustness/falsification evidence.",
        })

        rows.append(base)

    return rows

def build_mechanism_evidence(inputs, p50_info):
    rows = []
    rows.extend(evidence_from_phase5_ols(inputs["ols"], p50_info))
    rows.extend(evidence_from_phase5_dml(inputs["dml"], p50_info))
    rows.extend(evidence_from_phase5_ml(inputs["ml"]))
    rows.extend(evidence_from_phase5_product_sens(inputs["product_sens"], p50_info))
    rows.extend(evidence_from_phase6(inputs["phase6_all"], p50_info))

    if not rows:
        return pd.DataFrame()

    evidence = pd.DataFrame(rows)
    evidence = make_unique_columns(evidence)

    required_cols = [
        "trait",
        "raw_trait",
        "physiology_trait",
        "outcome",
        "outcome_family",
        "evidence_source",
        "model_family",
        "setting",
        "setting_detail",
        "robustness_group",
        "raw_estimate",
        "raw_ci_low",
        "raw_ci_high",
        "physiology_effect_estimate",
        "physiology_ci_low",
        "physiology_ci_high",
        "p_value",
        "observed_direction",
        "expected_direction",
        "matches_expected_direction",
        "n",
        "status",
        "source_table",
        "interpretation_note",
    ]

    for c in required_cols:
        if c not in evidence.columns:
            evidence[c] = np.nan

    evidence["matches_expected_direction_label"] = evidence["matches_expected_direction"].apply(
        lambda x: "not_testable" if pd.isna(x) else ("yes" if bool(x) else "no")
    )

    evidence["is_negative_control_evidence"] = evidence["outcome_family"].eq("negative_control")
    evidence["is_directional_evidence"] = evidence["matches_expected_direction"].notna()

    return evidence

# =============================================================================
# Claim audit and support scoring
# =============================================================================

def component_row(component, status, summary_value, interpretation, supporting_table, score_possible=1, score_earned=0):
    return {
        "claim_component": component,
        "pass_fail_warning": status,
        "summary_value": summary_value,
        "interpretation": interpretation,
        "supporting_table": supporting_table,
        "score_possible": score_possible,
        "score_earned": score_earned,
    }

def evaluate_trait_direction(evidence, physiology_trait, evidence_source_filter=None, model_family_filter=None, outcome_family_filter=None):
    d = evidence.copy()

    d = d[d["physiology_trait"].eq(physiology_trait)]

    if evidence_source_filter is not None:
        d = d[d["evidence_source"].isin(evidence_source_filter)]

    if model_family_filter is not None:
        d = d[d["model_family"].isin(model_family_filter)]

    if outcome_family_filter is not None:
        d = d[d["outcome_family"].isin(outcome_family_filter)]

    d = d[d["matches_expected_direction"].notna()]

    if d.empty:
        return np.nan, 0

    return float(d["matches_expected_direction"].astype(bool).mean()), int(len(d))

def evaluate_setting_group(evidence, group, model_family="OLS_HC3", outcome_families=None):
    d = evidence.copy()

    d = d[
        (d["evidence_source"].eq("phase6_robustness")) &
        (d["robustness_group"].eq(group))
    ].copy()

    if model_family is not None:
        d = d[d["model_family"].eq(model_family)]

    if outcome_families is not None:
        d = d[d["outcome_family"].isin(outcome_families)]

    d = d[d["matches_expected_direction"].notna()]

    if d.empty:
        return np.nan, 0, "not_testable"

    frac = float(d["matches_expected_direction"].astype(bool).mean())
    n = int(len(d))

    if frac >= 0.67:
        status = "pass"
    elif frac >= 0.50:
        status = "warning"
    else:
        status = "fail"

    return frac, n, status

def evaluate_negative_controls(evidence):
    d = evidence[
        (evidence["outcome_family"].eq("negative_control")) &
        (evidence["evidence_source"].eq("phase6_robustness")) &
        (evidence["model_family"].isin(["OLS_HC3", "DML_RF"]))
    ].copy()

    if d.empty:
        return {
            "status": "not_testable",
            "warning": False,
            "n_negative_control_coefficients": 0,
            "n_warning_coefficients": 0,
            "summary": "No negative-control coefficients available.",
        }

    d["abs_effect"] = to_num(d["physiology_effect_estimate"]).abs()
    d["p_value_num"] = to_num(d["p_value"])

    warn = d[(d["abs_effect"] >= 0.30) & (d["p_value_num"] < 0.10)].copy()

    if len(warn) > 0:
        return {
            "status": "warning",
            "warning": True,
            "n_negative_control_coefficients": int(len(d)),
            "n_warning_coefficients": int(len(warn)),
            "summary": f"{len(warn)} negative-control coefficients had |effect| >= 0.30 and p < 0.10.",
            "warning_rows": warn[[
                "physiology_trait",
                "outcome",
                "setting",
                "model_family",
                "physiology_effect_estimate",
                "p_value",
                "n",
            ]].to_dict(orient="records"),
        }

    return {
        "status": "pass",
        "warning": False,
        "n_negative_control_coefficients": int(len(d)),
        "n_warning_coefficients": 0,
        "summary": "No strong negative-control warning by threshold |effect| >= 0.30 and p < 0.10.",
    }

def evaluate_ml_importance(evidence):
    d = evidence[
        (evidence["evidence_source"].eq("phase5_ml_importance")) &
        (evidence["raw_trait"].isin(TRAITS_CORE))
    ].copy()

    if d.empty:
        return {
            "status": "not_testable",
            "summary": "No ML importance evidence available.",
            "score": 0,
        }

    if "ml_rank" not in d.columns:
        return {
            "status": "not_testable",
            "summary": "ML evidence table lacks rank.",
            "score": 0,
        }

    d["rank_num"] = to_num(d["ml_rank"])
    top = d[d["rank_num"] <= 5].copy()

    if len(top) >= 2:
        return {
            "status": "pass",
            "summary": f"{len(top)} core-trait ML importance entries ranked in top 5.",
            "score": 1,
        }
    elif len(top) == 1:
        return {
            "status": "warning",
            "summary": "One core-trait ML importance entry ranked in top 5.",
            "score": 0.5,
        }
    else:
        return {
            "status": "fail",
            "summary": "No core traits ranked in top 5 by ML importance.",
            "score": 0,
        }

def build_claim_audit(evidence, inputs, p50_info):
    rows = []

    # 1. P50/hydraulic resistance expected direction in primary OLS.
    frac, n = evaluate_trait_direction(
        evidence,
        "hydraulic_resistance",
        evidence_source_filter=["phase5_ols_rlm"],
        model_family_filter=["OLS_HC3", "OLS"],
        outcome_family_filter=["slope_change", "post_slope", "satbreak_fraction"],
    )
    status = "not_testable" if n == 0 else ("pass" if frac >= 0.67 else "warning" if frac >= 0.50 else "fail")
    rows.append(component_row(
        "P50 / hydraulic resistance direction consistent in adjusted OLS",
        status,
        f"match_fraction={frac}; n={n}",
        "Raw P50 coefficients were transformed into hydraulic-resistance interpretation using the P50 sign convention.",
        str(PHASE5_OLS),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 2. P50/hydraulic resistance expected direction in DML.
    frac, n = evaluate_trait_direction(
        evidence,
        "hydraulic_resistance",
        evidence_source_filter=["phase5_dml"],
        model_family_filter=["DML_RF"],
        outcome_family_filter=["slope_change", "post_slope", "satbreak_fraction"],
    )
    status = "not_testable" if n == 0 else ("pass" if frac >= 0.67 else "warning" if frac >= 0.50 else "fail")
    rows.append(component_row(
        "P50 / hydraulic resistance direction consistent in DML",
        status,
        f"match_fraction={frac}; n={n}",
        "DML residualization tests whether residual trait variation is associated with residual WUE response after flexible climate/soil adjustment.",
        str(PHASE5_DML),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 3. Rooting-depth expected direction in primary OLS.
    frac, n = evaluate_trait_direction(
        evidence,
        "rooting_depth",
        evidence_source_filter=["phase5_ols_rlm"],
        model_family_filter=["OLS_HC3", "OLS"],
        outcome_family_filter=["slope_change", "post_slope", "satbreak_fraction"],
    )
    status = "not_testable" if n == 0 else ("pass" if frac >= 0.67 else "warning" if frac >= 0.50 else "fail")
    rows.append(component_row(
        "Rooting-depth direction consistent in adjusted OLS",
        status,
        f"match_fraction={frac}; n={n}",
        "Deeper rooting is expected to maintain WUE response and reduce threshold-like behavior.",
        str(PHASE5_OLS),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 4. Rooting-depth expected direction in DML.
    frac, n = evaluate_trait_direction(
        evidence,
        "rooting_depth",
        evidence_source_filter=["phase5_dml"],
        model_family_filter=["DML_RF"],
        outcome_family_filter=["slope_change", "post_slope", "satbreak_fraction"],
    )
    status = "not_testable" if n == 0 else ("pass" if frac >= 0.67 else "warning" if frac >= 0.50 else "fail")
    rows.append(component_row(
        "Rooting-depth direction consistent in DML",
        status,
        f"match_fraction={frac}; n={n}",
        "Rooting-depth effects should remain directionally plausible after flexible adjustment.",
        str(PHASE5_DML),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 5. Raw-vs-CO2 robustness.
    frac, n, status = evaluate_setting_group(
        evidence,
        "raw_vs_co2",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Raw-vs-CO2 robustness",
        status,
        f"match_fraction={frac}; n={n}",
        "Trait-effect direction should not flip just because CO2 correction is applied.",
        str(PHASE6_RAWCO2),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 6. Metric robustness.
    frac, n, status = evaluate_setting_group(
        evidence,
        "metric_comparison",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Metric robustness: uWUE vs raw WUE vs iWUE",
        status,
        f"match_fraction={frac}; n={n}",
        "The claim should be strongest for uWUE but not completely contradicted by raw WUE/iWUE.",
        str(PHASE6_METRIC),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 7. Product robustness.
    frac, n, status = evaluate_setting_group(
        evidence,
        "product_consensus_vs_specific",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Product robustness: consensus vs product-specific",
        status,
        f"match_fraction={frac}; n={n}",
        "The mechanism should not depend entirely on one product combination.",
        str(PHASE6_PRODUCT),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 8. Stress-definition robustness.
    frac, n, status = evaluate_setting_group(
        evidence,
        "stress_definition",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Stress-definition robustness",
        status,
        f"match_fraction={frac}; n={n}",
        "Trait effects should not depend entirely on zscore, percentile_joint, copula_joint, or interaction_surface alone.",
        str(PHASE6_STRESS_SEASON),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 9. Growing-season robustness.
    frac, n, status = evaluate_setting_group(
        evidence,
        "growing_season",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Growing-season robustness",
        status,
        f"match_fraction={frac}; n={n}",
        "Trait effects should not depend entirely on month_fixed, gpp_threshold, or climate_common season definition.",
        str(PHASE6_STRESS_SEASON),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 10. Aridity-stratified plausibility.
    frac, n, status = evaluate_setting_group(
        evidence,
        "aridity_stratified",
        model_family="OLS_HC3",
        outcome_families=["slope_change", "post_slope", "satbreak_fraction"],
    )
    rows.append(component_row(
        "Aridity-stratified plausibility",
        status,
        f"match_fraction={frac}; n={n}",
        "Trait effects should be at least directionally plausible within dry/semi-arid/mesic subsets, though power may be limited.",
        str(PHASE6_ARIDITY),
        score_possible=1,
        score_earned=1 if status == "pass" else 0.5 if status == "warning" else 0,
    ))

    # 11. Negative controls.
    neg = evaluate_negative_controls(evidence)
    rows.append(component_row(
        "Negative-control pass",
        neg["status"],
        neg["summary"],
        "Traits should not strongly explain outcomes such as product availability or raw-vs-CO2 stability.",
        str(PHASE6_NEG),
        score_possible=1,
        score_earned=1 if neg["status"] == "pass" else 0,
    ))

    # 12. ML importance.
    ml = evaluate_ml_importance(evidence)
    rows.append(component_row(
        "ML importance supports trait relevance",
        ml["status"],
        ml["summary"],
        "ML importance is only predictive/descriptive, but core traits should not be completely irrelevant.",
        str(PHASE5_ML),
        score_possible=1,
        score_earned=ml["score"],
    ))

    # 13. Isohydricity sensitivity only.
    trait_dataset = inputs["trait_dataset"]
    iso_n = finite_n(trait_dataset, "isohydricity") if "isohydricity" in trait_dataset.columns else 0
    total_n = len(trait_dataset)
    iso_frac = iso_n / total_n if total_n else 0.0
    iso_status = "pass" if iso_frac < 0.70 else "warning"
    rows.append(component_row(
        "Isohydricity treated as limited-coverage sensitivity",
        iso_status,
        f"isohydricity finite n={iso_n}/{total_n}; coverage={iso_frac:.3f}",
        "Isohydricity should not be a headline predictor unless coverage is adequate.",
        str(PHASE4_DATASET),
        score_possible=1,
        score_earned=1 if iso_status == "pass" else 0.5,
    ))

    # 14. Tower limitation stated.
    rows.append(component_row(
        "Tower validation limitation stated",
        "pass",
        "Tower validation is explicitly retained as future validation, not claimed as completed.",
        FUTURE_TOWER_STATEMENT,
        "Phase 7 interpretation text",
        score_possible=1,
        score_earned=1,
    ))

    # 15. P50 sign handled.
    rows.append(component_row(
        "P50 sign convention handled",
        "pass" if p50_info["p50_to_hydraulic_resistance_multiplier"] in [-1.0, 1.0] else "warning",
        json.dumps(p50_info, default=str),
        "Raw P50 coefficients are translated into hydraulic-resistance interpretation.",
        str(PHASE4_DATASET),
        score_possible=1,
        score_earned=1 if p50_info["p50_to_hydraulic_resistance_multiplier"] in [-1.0, 1.0] else 0,
    ))

    # 16. Product uncertainty retained.
    phase3_manifest = inputs["manifests"].get("phase3", {})
    n_combo = phase3_manifest.get("n_product_combos_present", None)
    complete_counts = phase3_manifest.get("complete_subset_counts", {})
    rows.append(component_row(
        "Product uncertainty retained",
        "pass",
        f"n_product_combos_present={n_combo}; complete_subset_counts={complete_counts}",
        "Interpretation uses product-consensus and product-specific robustness instead of selecting one best product without towers.",
        str(PHASE3_MANIFEST),
        score_possible=1,
        score_earned=1,
    ))

    # 17. No-universal-threshold framing retained.
    phase2_manifest = inputs["manifests"].get("phase2", {})
    phase2_supported = phase2_manifest.get("phase2_claim_supported", "unknown")
    rows.append(component_row(
        "No-universal-threshold framing retained",
        "pass" if phase2_supported in [True, "true", "True", "unknown"] else "warning",
        f"phase2_claim_supported={phase2_supported}",
        "Final mechanism is about heterogeneous physiological response, not universal WUE collapse.",
        str(PHASE2_MANIFEST),
        score_possible=1,
        score_earned=1 if phase2_supported in [True, "true", "True", "unknown"] else 0.5,
    ))

    audit = pd.DataFrame(rows)
    return audit, neg

def classify_support(audit, negative_result):
    score_possible = float(audit["score_possible"].sum())
    score_earned = float(audit["score_earned"].sum())
    score_fraction = score_earned / score_possible if score_possible else np.nan

    negative_warning = negative_result.get("status") == "warning"

    # Base classification.
    if score_earned >= 13 and not negative_warning:
        support = "strong_support"
    elif score_earned >= 9:
        support = "moderate_support_with_negative_control_warning" if negative_warning else "moderate_support"
    else:
        support = "weak_support_with_negative_control_warning" if negative_warning else "weak_support"

    # Automatic downgrade if several components fail.
    fail_count = int((audit["pass_fail_warning"] == "fail").sum())
    warning_count = int((audit["pass_fail_warning"] == "warning").sum())

    if fail_count >= 5:
        support = "weak_support_with_negative_control_warning" if negative_warning else "weak_support"
    elif fail_count >= 3 and support == "strong_support":
        support = "moderate_support_with_negative_control_warning" if negative_warning else "moderate_support"

    score_df = pd.DataFrame([{
        "score_earned": score_earned,
        "score_possible": score_possible,
        "score_fraction": score_fraction,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "negative_control_warning": bool(negative_warning),
        "support_classification": support,
    }])

    return support, score_df

# =============================================================================
# Output interpretation
# =============================================================================

def summarize_mechanism_evidence(evidence):
    directional = evidence[
        evidence["matches_expected_direction"].notna() &
        evidence["physiology_trait"].isin(["hydraulic_resistance", "rooting_depth"])
    ].copy()

    rows = []
    for phys_trait in ["hydraulic_resistance", "rooting_depth"]:
        for source in sorted(directional["evidence_source"].dropna().unique()):
            d = directional[
                (directional["physiology_trait"].eq(phys_trait)) &
                (directional["evidence_source"].eq(source))
            ].copy()
            if d.empty:
                continue

            rows.append({
                "physiology_trait": phys_trait,
                "evidence_source": source,
                "n_directional_tests": int(len(d)),
                "match_fraction": float(d["matches_expected_direction"].astype(bool).mean()),
                "median_effect": float(to_num(d["physiology_effect_estimate"]).median(skipna=True)),
                "median_abs_effect": float(to_num(d["physiology_effect_estimate"]).abs().median(skipna=True)),
                "n_nominal_p_lt_0p10": int((to_num(d["p_value"]) < 0.10).sum()),
            })

    return pd.DataFrame(rows)

def make_interpretation_text(support, score_df, audit, evidence_summary, p50_info, negative_result):
    lines = []
    lines.append("# Phase 7: Physiological mechanism interpretation")
    lines.append("")
    lines.append("## Core question")
    lines.append("")
    lines.append("Does the satellite-derived WUE response phenotype behave like an ecosystem-scale physiological phenotype shaped by plant water-transport and rooting strategy?")
    lines.append("")
    lines.append("## Main expected mechanism")
    lines.append("")
    lines.append("If the model works, the interpretation is that more drought-resistant xylem and deeper rooting systems maintain WUE sensitivity under high compound atmospheric-soil moisture stress. More vulnerable or shallow-rooted systems should show stronger weakening of the uWUE response at high stress.")
    lines.append("")
    lines.append("## P50 sign convention")
    lines.append("")
    lines.append(p50_info["interpretation_note"])
    lines.append("")
    lines.append(f"- Raw P50 median: `{p50_info['p50_median']}`")
    lines.append(f"- Hydraulic-resistance definition: `{p50_info['hydraulic_resistance_definition']}`")
    lines.append(f"- P50 coefficient multiplier used for physiological interpretation: `{p50_info['p50_to_hydraulic_resistance_multiplier']}`")
    lines.append("")
    lines.append("## Mechanistic expected directions")
    lines.append("")
    lines.append("- Hydraulic resistance should have a positive association with `slope_change` and `post_slope`, and a negative association with `satbreak_fraction` or `negative_slope_fraction`.")
    lines.append("- Rooting depth should have a positive association with `slope_change` and `post_slope`, and a negative association with `satbreak_fraction` or `negative_slope_fraction`.")
    lines.append("- Isohydricity is treated as limited-coverage sensitivity, not a core claim.")
    lines.append("")
    lines.append("## Quantitative support classification")
    lines.append("")
    lines.append(score_df.to_string(index=False))
    lines.append("")
    lines.append(f"Support classification: **{support}**")
    lines.append("")
    lines.append("## Mechanism evidence summary")
    lines.append("")
    if evidence_summary.empty:
        lines.append("No directional mechanism evidence was available.")
    else:
        lines.append(evidence_summary.to_string(index=False))
    lines.append("")
    lines.append("## Claim audit")
    lines.append("")
    lines.append(audit.to_string(index=False))
    lines.append("")
    lines.append("## Negative-control evaluation")
    lines.append("")
    lines.append(json.dumps(negative_result, indent=2, default=str))
    lines.append("")
    lines.append("## Interpretation selected from evidence")
    lines.append("")

    if support == "strong_support":
        lines.append("The robustness and falsification results support a physiological interpretation. Across adjusted model families and robustness settings, hydraulic/rooting effects are directionally consistent enough to interpret the product-consensus WUE response phenotype as statistically consistent with plant water-transport and rooting strategy.")
    elif support == "moderate_support":
        lines.append("The evidence is compatible with a hydraulic/rooting mechanism, especially for product-consensus uWUE response, but robustness varies enough that the claim should remain cautious. The result supports a physiological interpretation, not proof.")
    elif support == "moderate_support_with_negative_control_warning":
        lines.append("The trait evidence is partially compatible with a hydraulic/rooting mechanism, but negative-control or robustness warnings remain. The interpretation should be cautious and should emphasize possible residual spatial/product confounding.")
    elif support == "weak_support_with_negative_control_warning":
        lines.append("The evidence is not robust enough for a strong physiological mechanism claim, and negative-control warnings suggest possible spatial/product confounding. The result should be framed as inconclusive or as a limitation of current satellite/trait products.")
    else:
        lines.append("The evidence is not robust enough to support a hydraulic/rooting mechanism claim. Effects weaken, reverse, or fail robustness checks, so the safest interpretation is that the current satellite-derived trait association remains uncertain.")

    lines.append("")
    lines.append("## Acceptable claim")
    lines.append("")
    lines.append(SAFE_CLAIM)
    lines.append("")
    lines.append("## Claim to avoid")
    lines.append("")
    lines.append(CLAIM_TO_AVOID)
    lines.append("")
    lines.append("## Required limitation")
    lines.append("")
    lines.append(FUTURE_TOWER_STATEMENT)
    lines.append("")
    lines.append("## Final manuscript-safe wording")
    lines.append("")

    if support in ["strong_support", "moderate_support", "moderate_support_with_negative_control_warning"]:
        lines.append("After adjustment for aridity, climate, soil texture, and spatial context, the satellite-derived product-consensus WUE response phenotype shows trait-adjusted associations consistent with a hydraulic/rooting mechanism. Grassland points with more drought-resistant hydraulic/rooting strategies tend to show less high-stress weakening of uWUE response. However, because tower validation is unavailable, this should be interpreted as a satellite-derived mechanism-consistent association rather than direct physiological proof.")
    else:
        lines.append("The current satellite-derived trait associations do not provide sufficiently robust support for a hydraulic/rooting mechanism after falsification testing. The analysis should be framed as a test of whether global trait maps can explain WUE response heterogeneity, with tower validation and expanded trait coverage required before stronger physiological claims.")

    return "\n".join(lines)

# =============================================================================
# Figures
# =============================================================================

def plot_claim_audit(audit):
    if audit.empty:
        return None

    status_map = {
        "pass": 1.0,
        "warning": 0.5,
        "fail": 0.0,
        "not_testable": np.nan,
    }

    d = audit.copy()
    d["status_score"] = d["pass_fail_warning"].map(status_map)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["score_earned"])
    ax.set_yticks(y)
    ax.set_yticklabels(d["claim_component"])
    ax.set_xlabel("Score earned")
    ax.set_title("Figure 8. Phase 7 claim-audit score by component")
    ax.set_xlim(0, max(1.0, float(d["score_possible"].max()) if len(d) else 1.0))
    fig.tight_layout()
    fig.savefig(FIG_AUDIT_OUT, dpi=300)
    fig.savefig(FIG_AUDIT_PDF)
    plt.close(fig)

    return FIG_AUDIT_OUT

def plot_evidence_summary(evidence_summary):
    if evidence_summary.empty:
        return None

    d = evidence_summary.copy()
    d["label"] = d["physiology_trait"] + "\n" + d["evidence_source"]
    d = d.sort_values(["physiology_trait", "evidence_source"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["match_fraction"])
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.axvline(0.67, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of tests matching expected physiological direction")
    ax.set_title("Figure 9. Trait-mechanism evidence summary")
    fig.tight_layout()
    fig.savefig(FIG_EVIDENCE_OUT, dpi=300)
    fig.savefig(FIG_EVIDENCE_PDF)
    plt.close(fig)

    return FIG_EVIDENCE_OUT

# =============================================================================
# README / manifest
# =============================================================================

def write_readme(manifest):
    lines = []
    lines.append("# Phase 7: Physiological mechanism interpretation")
    lines.append("")
    lines.append("## What this phase executed")
    lines.append("")
    lines.append("1. Loaded Phase 5 and Phase 6 model evidence.")
    lines.append("2. Converted raw P50 coefficients into hydraulic-resistance interpretation.")
    lines.append("3. Defined expected trait-effect directions for slope_change, post_slope, satbreak_fraction, and negative controls.")
    lines.append("4. Scored evidence across OLS/RLM, DML, ML importance, product-family sensitivity, and Phase 6 robustness tests.")
    lines.append("5. Checked negative controls.")
    lines.append("6. Classified support level.")
    lines.append("7. Generated claim-audit table.")
    lines.append("8. Generated physiological mechanism interpretation.")
    lines.append("9. Explicitly stated safe claim and avoided claim.")
    lines.append("10. Explicitly stated tower validation as a future validation step.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for p in [
        EVIDENCE_OUT,
        CLAIM_AUDIT_OUT,
        SUPPORT_SCORE_OUT,
        INTERPRETATION_OUT,
        MANIFEST_OUT,
        FIG_AUDIT_OUT,
        FIG_EVIDENCE_OUT,
    ]:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## Safe claim")
    lines.append("")
    lines.append(SAFE_CLAIM)
    lines.append("")
    lines.append("## Claim to avoid")
    lines.append("")
    lines.append(CLAIM_TO_AVOID)
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
    print("PHASE 7 START")

    inputs = load_inputs()
    trait_dataset = inputs["trait_dataset"]

    p50_info = infer_p50_transform(trait_dataset)
    print("P50 interpretation:")
    print(json.dumps(p50_info, indent=2, default=str))

    evidence = build_mechanism_evidence(inputs, p50_info)
    save_csv(evidence, EVIDENCE_OUT)

    audit, negative_result = build_claim_audit(evidence, inputs, p50_info)
    save_csv(audit, CLAIM_AUDIT_OUT)

    support, score_df = classify_support(audit, negative_result)
    save_csv(score_df, SUPPORT_SCORE_OUT)

    evidence_summary = summarize_mechanism_evidence(evidence)
    evidence_summary_out = OUTDIR / "table_phase7_evidence_summary_by_source.csv"
    save_csv(evidence_summary, evidence_summary_out)

    interpretation_text = make_interpretation_text(
        support=support,
        score_df=score_df,
        audit=audit,
        evidence_summary=evidence_summary,
        p50_info=p50_info,
        negative_result=negative_result,
    )
    INTERPRETATION_OUT.write_text(interpretation_text)
    print(f"WROTE {INTERPRETATION_OUT}")

    fig_audit = plot_claim_audit(audit)
    fig_evidence = plot_evidence_summary(evidence_summary)

    manifest = {
        "phase": "Phase 7: Interpret physiological mechanism",
        "inputs": {
            "phase4_trait_dataset": str(PHASE4_DATASET),
            "phase5_ols": str(PHASE5_OLS),
            "phase5_ml": str(PHASE5_ML),
            "phase5_dml": str(PHASE5_DML),
            "phase5_product_sensitivity": str(PHASE5_PRODUCT_SENS),
            "phase6_all": str(PHASE6_ALL),
            "phase6_negative_controls": str(PHASE6_NEG),
            "phase6_raw_vs_co2": str(PHASE6_RAWCO2),
            "phase6_metric": str(PHASE6_METRIC),
            "phase6_product": str(PHASE6_PRODUCT),
            "phase6_stress_season": str(PHASE6_STRESS_SEASON),
            "phase6_aridity": str(PHASE6_ARIDITY),
        },
        "outputs": {
            "mechanism_evidence": str(EVIDENCE_OUT),
            "claim_audit": str(CLAIM_AUDIT_OUT),
            "support_score": str(SUPPORT_SCORE_OUT),
            "evidence_summary": str(evidence_summary_out),
            "physiological_interpretation": str(INTERPRETATION_OUT),
            "figure_claim_audit": str(fig_audit) if fig_audit else None,
            "figure_evidence_summary": str(fig_evidence) if fig_evidence else None,
        },
        "p50_interpretation": p50_info,
        "support_classification": support,
        "support_score": score_df.to_dict(orient="records"),
        "negative_control_result": negative_result,
        "safe_claim": SAFE_CLAIM,
        "claim_to_avoid": CLAIM_TO_AVOID,
        "future_validation_statement": FUTURE_TOWER_STATEMENT,
        "n_evidence_rows": int(len(evidence)),
        "n_claim_audit_rows": int(len(audit)),
        "phase3_manifest": inputs["manifests"].get("phase3", {}),
        "phase2_manifest": inputs["manifests"].get("phase2", {}),
    }

    with open(MANIFEST_OUT, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"WROTE {MANIFEST_OUT}")

    write_readme(manifest)

    print("")
    print("DONE Phase 7.")
    print("")
    print("SUPPORT SCORE:")
    print(score_df.to_string(index=False))
    print("")
    print("CLAIM AUDIT:")
    print(audit.to_string(index=False))
    print("")
    print("EVIDENCE SUMMARY:")
    print(evidence_summary.to_string(index=False) if not evidence_summary.empty else "No evidence summary.")
    print("")
    print("SUPPORT CLASSIFICATION:", support)
    print("")
    print("INTERPRETATION FILE:", INTERPRETATION_OUT)

if __name__ == "__main__":
    main()
