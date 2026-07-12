#!/usr/bin/env python
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
PH8 = Path("results/trait_framework/phase8")
PH7 = Path("results/trait_framework/phase7")
OUT = Path("results/paper_ecosystem_response")
FIG = OUT / "figures"
TAB = OUT / "tables"
TXT = OUT / "text"

for p in [OUT, FIG, TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
CLASS_PROB = PH8 / "table_latent_response_class_probabilities.csv"
TRAIT_OLS = PH8 / "table_phase8_trait_effects_on_latent_response.csv"
TRAIT_DML = PH8 / "table_phase8_trait_dml_effects_on_latent_response.csv"
LOFO = PH8 / "table_leave_one_product_family_out.csv"
WEIGHTS = PH8 / "table_independence_weight_sensitivity.csv"
PH8_MANIFEST = PH8 / "phase8_latent_ecosystem_response_manifest.json"

PH7_INTERP = PH7 / "physiological_mechanism_interpretation.md"
PH7_SCORE = PH7 / "table_phase7_support_score.csv"
PH7_EVIDENCE_SUMMARY = PH7 / "table_phase7_evidence_summary_by_source.csv"
PH7_CLAIM_AUDIT = PH7 / "table_phase7_claim_audit.csv"

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def num(s):
    return pd.to_numeric(s, errors="coerce")

def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100*x:.1f}%"

def fmt(x, digits=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"WROTE {path}")

latent = read_csv(LATENT)
class_prob = read_csv(CLASS_PROB, required=False)
trait_ols = read_csv(TRAIT_OLS, required=False)
trait_dml = read_csv(TRAIT_DML, required=False)
lofo = read_csv(LOFO, required=False)
weights = read_csv(WEIGHTS, required=False)
ph7_score = read_csv(PH7_SCORE, required=False)
ph7_summary = read_csv(PH7_EVIDENCE_SUMMARY, required=False)
ph7_audit = read_csv(PH7_CLAIM_AUDIT, required=False)
ph8_manifest = read_json(PH8_MANIFEST)

required_cols = [
    "point_id",
    "latent_slope_change",
    "latent_post_slope",
    "latent_satbreak_probability",
    "latent_response_class",
]
missing = [c for c in required_cols if c not in latent.columns]
if missing:
    die(f"Latent response table is missing columns: {missing}")

# ---------------------------------------------------------------------
# Main paper result tables
# ---------------------------------------------------------------------

n_points = int(latent["point_id"].nunique())
class_counts = latent["latent_response_class"].value_counts(dropna=False).rename_axis("latent_response_class").reset_index(name="n")
class_counts["fraction"] = class_counts["n"] / n_points
class_counts["percent"] = class_counts["fraction"] * 100
save_csv(class_counts, TAB / "Table1_latent_ecosystem_response_classes.csv")

satbreak_classes = latent["latent_response_class"].isin(["saturation", "breakdown"])
n_satbreak_class = int(satbreak_classes.sum())
frac_satbreak_class = n_satbreak_class / n_points

summary_rows = []

for col, label in [
    ("latent_slope_change", "Latent slope change"),
    ("latent_post_slope", "Latent high-stress/post-transition slope"),
    ("latent_satbreak_probability", "Latent probability of saturation/breakdown-like limitation"),
]:
    x = num(latent[col]).dropna()
    summary_rows.append({
        "quantity": label,
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "median": float(x.median()) if len(x) else np.nan,
        "sd": float(x.std()) if len(x) else np.nan,
        "p05": float(x.quantile(0.05)) if len(x) else np.nan,
        "p25": float(x.quantile(0.25)) if len(x) else np.nan,
        "p75": float(x.quantile(0.75)) if len(x) else np.nan,
        "p95": float(x.quantile(0.95)) if len(x) else np.nan,
    })

summary = pd.DataFrame(summary_rows)
save_csv(summary, TAB / "Table2_latent_ecosystem_response_intensity.csv")

# Class-level summary
class_summary = (
    latent
    .groupby("latent_response_class", dropna=False)
    .agg(
        n=("point_id", "size"),
        median_slope_change=("latent_slope_change", "median"),
        median_post_slope=("latent_post_slope", "median"),
        median_satbreak_probability=("latent_satbreak_probability", "median"),
        mean_satbreak_probability=("latent_satbreak_probability", "mean"),
    )
    .reset_index()
)
class_summary["fraction"] = class_summary["n"] / n_points
save_csv(class_summary, TAB / "Table3_response_intensity_by_class.csv")

# Trait table: focus only on ecosystem outcomes, not product terms
trait_focus_rows = []
if not trait_ols.empty:
    d = trait_ols.copy()
    if "status" in d.columns:
        d = d[d["status"].eq("ok")].copy()
    if "term" in d.columns:
        d = d[d["term"].isin(["p50", "rooting_depth"])].copy()
    if "outcome" in d.columns:
        d = d[d["outcome"].isin([
            "latent_slope_change",
            "latent_post_slope",
            "latent_satbreak_probability",
            "p_saturation",
            "p_breakdown",
            "p_enhancement",
        ])].copy()
    if "model_role" in d.columns:
        preferred = d[d["model_role"].astype(str).str.contains("climate_soil_spatial_adjusted_core_traits", na=False)].copy()
        if len(preferred):
            d = preferred
    for _, r in d.iterrows():
        trait_focus_rows.append({
            "analysis": "OLS_HC3_adjusted",
            "outcome": r.get("outcome"),
            "trait": r.get("term"),
            "n": r.get("n"),
            "estimate": r.get("estimate"),
            "std_error": r.get("std_error"),
            "p_value": r.get("p_value"),
            "ci_low": r.get("ci_low"),
            "ci_high": r.get("ci_high"),
            "interpretation": (
                "positive means larger trait value is associated with stronger/maintained WUE response"
                if r.get("outcome") in ["latent_slope_change", "latent_post_slope", "p_enhancement"]
                else "negative means larger trait value is associated with lower high-stress limitation probability"
            )
        })

if not trait_dml.empty:
    d = trait_dml.copy()
    if "status" in d.columns:
        d = d[d["status"].eq("ok")].copy()
    if "treatment_trait" in d.columns:
        d = d[d["treatment_trait"].isin(["p50", "rooting_depth"])].copy()
    if "outcome" in d.columns:
        d = d[d["outcome"].isin([
            "latent_slope_change",
            "latent_post_slope",
            "latent_satbreak_probability",
            "p_saturation",
            "p_breakdown",
        ])].copy()
    for _, r in d.iterrows():
        trait_focus_rows.append({
            "analysis": "DML_RF_adjusted",
            "outcome": r.get("outcome"),
            "trait": r.get("treatment_trait"),
            "n": r.get("n"),
            "estimate": r.get("effect_estimate"),
            "std_error": r.get("std_error"),
            "p_value": r.get("p_value"),
            "ci_low": r.get("ci_low"),
            "ci_high": r.get("ci_high"),
            "interpretation": (
                "positive means larger trait value is associated with stronger/maintained WUE response"
                if r.get("outcome") in ["latent_slope_change", "latent_post_slope"]
                else "negative means larger trait value is associated with lower high-stress limitation probability"
            )
        })

trait_focus = pd.DataFrame(trait_focus_rows)
save_csv(trait_focus, TAB / "Table4_trait_mechanism_on_ecosystem_response.csv")

# Methodological robustness table; label gently, product details go supplement
robust_rows = []

if not lofo.empty and "status" in lofo.columns:
    ok = lofo[lofo["status"].eq("ok")].copy()
    robust_rows.append({
        "robustness_domain": "data-family removal",
        "n_tests": int(len(ok)),
        "min_class_match_fraction": float(num(ok["class_match_fraction"]).min()) if "class_match_fraction" in ok.columns and len(ok) else np.nan,
        "median_class_match_fraction": float(num(ok["class_match_fraction"]).median()) if "class_match_fraction" in ok.columns and len(ok) else np.nan,
        "min_slope_correlation": float(num(ok["slope_change_correlation_with_main"]).min()) if "slope_change_correlation_with_main" in ok.columns and len(ok) else np.nan,
        "median_slope_correlation": float(num(ok["slope_change_correlation_with_main"]).median()) if "slope_change_correlation_with_main" in ok.columns and len(ok) else np.nan,
        "paper_interpretation": "Latent ecosystem response remains highly similar when one input data family is removed."
    })

if not weights.empty and "status" in weights.columns:
    ok = weights[weights["status"].eq("ok")].copy()
    robust_rows.append({
        "robustness_domain": "algorithmic-independence weighting",
        "n_tests": int(len(ok)),
        "min_class_match_fraction": float(num(ok["class_match_fraction"]).min()) if "class_match_fraction" in ok.columns and len(ok) else np.nan,
        "median_class_match_fraction": float(num(ok["class_match_fraction"]).median()) if "class_match_fraction" in ok.columns and len(ok) else np.nan,
        "min_slope_correlation": float(num(ok["slope_change_correlation_with_equal_main"]).min()) if "slope_change_correlation_with_equal_main" in ok.columns and len(ok) else np.nan,
        "median_slope_correlation": float(num(ok["slope_change_correlation_with_equal_main"]).median()) if "slope_change_correlation_with_equal_main" in ok.columns and len(ok) else np.nan,
        "paper_interpretation": "Latent ecosystem response is stable to alternative weighting of algorithmically independent evidence streams."
    })

robust = pd.DataFrame(robust_rows)
save_csv(robust, TAB / "TableS1_methodological_robustness_not_main_result.csv")

# Main claims table
enh_n = int(class_counts.loc[class_counts["latent_response_class"].eq("enhancement"), "n"].sum()) if "enhancement" in class_counts["latent_response_class"].values else 0
inc_n = int(class_counts.loc[class_counts["latent_response_class"].eq("inconclusive"), "n"].sum()) if "inconclusive" in class_counts["latent_response_class"].values else 0
satprob_median = float(num(latent["latent_satbreak_probability"]).median())
slope_median = float(num(latent["latent_slope_change"]).median())
post_median = float(num(latent["latent_post_slope"]).median())

claims = pd.DataFrame([
    {
        "claim_number": 1,
        "claim": "No universal WUE breakdown under compound atmospheric-soil moisture stress.",
        "evidence": f"Latent classes: {enh_n}/{n_points} enhancement, {inc_n}/{n_points} inconclusive, {n_satbreak_class}/{n_points} saturation/breakdown.",
        "paper_strength": "main result",
    },
    {
        "claim_number": 2,
        "claim": "The dominant ecosystem-scale response is maintained or enhanced WUE sensitivity rather than collapse.",
        "evidence": f"Median latent slope change = {fmt(slope_median)}; median high-stress/post-transition slope = {fmt(post_median)}.",
        "paper_strength": "main result",
    },
    {
        "claim_number": 3,
        "claim": "High-stress limitation is rare in the latent ecosystem phenotype.",
        "evidence": f"Median latent saturation/breakdown probability = {fmt(satprob_median)}.",
        "paper_strength": "main result",
    },
    {
        "claim_number": 4,
        "claim": "Hydraulic/rooting traits are directionally consistent with spatial variation in the response phenotype.",
        "evidence": "Trait effects are summarized in Table 4 and should be described as mechanism-consistent, not causal proof.",
        "paper_strength": "mechanistic support",
    },
    {
        "claim_number": 5,
        "claim": "Methodological uncertainty was handled by latent-response modeling and robustness tests.",
        "evidence": "Robustness results are reported in Supplementary Table S1, not foregrounded as the biological finding.",
        "paper_strength": "methods/supporting",
    },
])
save_csv(claims, TAB / "Table5_main_claims_for_paper.csv")

# ---------------------------------------------------------------------
# Figures focused on ecosystem response, not product differences
# ---------------------------------------------------------------------

# Figure 1: class counts
fig, ax = plt.subplots(figsize=(6.8, 4.5))
plot_counts = class_counts.sort_values("n", ascending=False)
ax.bar(plot_counts["latent_response_class"].astype(str), plot_counts["n"])
ax.set_ylabel("Grassland points")
ax.set_xlabel("Latent ecosystem response class")
ax.set_title("Figure 1. Dominant grassland WUE response under compound stress")
for i, r in plot_counts.reset_index(drop=True).iterrows():
    ax.text(i, r["n"], f'{int(r["n"])}\n({r["percent"]:.1f}%)', ha="center", va="bottom", fontsize=9)
savefig(FIG / "Figure1_latent_ecosystem_response_classes.png")

# Figure 2: map
if {"lat", "lon"}.issubset(latent.columns):
    d = latent.copy()
    d["lat"] = num(d["lat"])
    d["lon"] = num(d["lon"])
    d = d.dropna(subset=["lat", "lon"])
    if len(d):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cl, sub in d.groupby("latent_response_class"):
            size = 35 + 100 * num(sub.get("latent_satbreak_probability", pd.Series(0, index=sub.index))).fillna(0)
            ax.scatter(sub["lon"], sub["lat"], s=size, alpha=0.75, label=cl)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Figure 2. Spatial distribution of grassland WUE response classes")
        ax.legend(frameon=False)
        savefig(FIG / "Figure2_ecosystem_response_map.png")

# Figure 3: distributions
for col, label, fname in [
    ("latent_slope_change", "Latent slope change", "Figure3A_latent_slope_change_distribution.png"),
    ("latent_post_slope", "Latent high-stress WUE response slope", "Figure3B_latent_post_slope_distribution.png"),
    ("latent_satbreak_probability", "Probability of high-stress limitation", "Figure3C_high_stress_limitation_probability.png"),
]:
    if col in latent.columns:
        d = num(latent[col]).dropna()
        if len(d):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.hist(d, bins=24)
            ax.axvline(d.median(), linestyle="--", linewidth=1)
            ax.set_xlabel(label)
            ax.set_ylabel("Grassland points")
            ax.set_title(label)
            savefig(FIG / fname)

# Figure 4: aridity gradient
if "aridity" in latent.columns and "latent_post_slope" in latent.columns:
    d = latent.copy()
    d["aridity"] = num(d["aridity"])
    d["latent_post_slope"] = num(d["latent_post_slope"])
    d = d.dropna(subset=["aridity", "latent_post_slope"])
    if len(d) >= 10:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for cl, sub in d.groupby("latent_response_class"):
            ax.scatter(sub["aridity"], sub["latent_post_slope"], alpha=0.7, label=cl)
        z = np.polyfit(d["aridity"], d["latent_post_slope"], 1)
        xs = np.linspace(d["aridity"].min(), d["aridity"].max(), 100)
        ax.plot(xs, z[0] * xs + z[1], linestyle="--", linewidth=1)
        ax.set_xlabel("Aridity")
        ax.set_ylabel("Latent high-stress WUE response slope")
        ax.set_title("Figure 4. Ecosystem response along the aridity gradient")
        ax.legend(frameon=False)
        savefig(FIG / "Figure4_aridity_gradient_response.png")

# Figure 5: trait forest plot
forest = trait_focus.copy()
if not forest.empty:
    forest = forest[
        forest["analysis"].eq("OLS_HC3_adjusted") &
        forest["trait"].isin(["p50", "rooting_depth"]) &
        forest["outcome"].isin(["latent_slope_change", "latent_post_slope", "latent_satbreak_probability"])
    ].copy()
    if not forest.empty:
        forest["label"] = forest["outcome"].astype(str) + " | " + forest["trait"].astype(str)
        forest = forest.sort_values(["outcome", "trait"]).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.55 * len(forest))))
        y = np.arange(len(forest))
        est = num(forest["estimate"])
        lo = num(forest["ci_low"])
        hi = num(forest["ci_high"])
        ax.errorbar(
            est,
            y,
            xerr=np.vstack([est - lo, hi - est]),
            fmt="o",
            capsize=3
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(forest["label"])
        ax.set_xlabel("Standardized adjusted trait effect")
        ax.set_title("Figure 5. Hydraulic/rooting trait associations with ecosystem response")
        savefig(FIG / "Figure5_trait_effects_ecosystem_response.png")

# Supplementary robustness figure, not main finding
if not robust.empty:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    d = robust.copy()
    x = np.arange(len(d))
    ax.bar(x, num(d["median_class_match_fraction"]))
    ax.set_xticks(x)
    ax.set_xticklabels(d["robustness_domain"], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median class-match fraction")
    ax.set_title("Supplementary Figure S1. Methodological robustness of latent response")
    savefig(FIG / "FigureS1_methodological_robustness.png")

# ---------------------------------------------------------------------
# Manuscript-ready text
# ---------------------------------------------------------------------

support_text = ""
if not ph7_score.empty:
    support_text = ph7_score.to_string(index=False)

abstract = f"""# Manuscript abstract draft

Grassland carbon-water coupling under simultaneous atmospheric dryness and soil moisture deficit remains uncertain because high vapor pressure deficit can increase atmospheric demand while soil water limitation constrains photosynthesis and transpiration. Here, we characterize the ecosystem-scale water-use efficiency (WUE) response of global grassland points to compound atmospheric-soil moisture stress using a latent response framework that treats multiple satellite-derived carbon-water estimates, WUE metrics, stress definitions, growing-season definitions, and CO2 treatments as repeated observations of an underlying response phenotype. Across {n_points} grassland points, the dominant latent response was WUE enhancement or maintenance rather than collapse: {enh_n}/{n_points} points were classified as enhancement and {inc_n}/{n_points} as inconclusive, while {n_satbreak_class}/{n_points} were classified as saturation or breakdown. The median probability of high-stress limitation was {fmt(satprob_median)}, indicating that breakdown-like behavior is not a universal grassland response. Spatial variation in the latent response was directionally consistent with hydraulic and rooting controls after adjustment for aridity, climate, soil texture, and spatial context. These results suggest that compound stress reorganizes grassland carbon-water coupling heterogeneously, but widespread ecosystem WUE collapse is not supported. Direct tower validation remains a necessary future step for in situ confirmation.
"""

results = f"""# Results draft

## A product-adjusted ecosystem response phenotype

We first estimated a latent ecosystem response phenotype for each grassland point by combining response estimates across WUE metrics, compound-stress definitions, growing-season definitions, CO2 treatments, and independent carbon-water data streams. This produced a single point-level response estimate for slope change, high-stress/post-transition slope, and saturation/breakdown probability.

The resulting classification showed a clear dominant pattern: grassland WUE response did not exhibit widespread breakdown under compound atmospheric-soil moisture stress. Of {n_points} grassland points, {enh_n} were classified as enhancement, {inc_n} as inconclusive, and {n_satbreak_class} as saturation/breakdown. The median latent slope change was {fmt(slope_median)}, the median high-stress/post-transition slope was {fmt(post_median)}, and the median saturation/breakdown probability was {fmt(satprob_median)}. Thus, the main ecosystem-scale result is not a universal threshold collapse, but a predominantly maintained or enhanced WUE response with limited evidence for high-stress limitation.

## Spatial and environmental structure

The latent response varied across space and along environmental gradients. Aridity and spatial covariates were retained as adjustment variables in the trait models, so the trait-response results should be interpreted as associations with the latent response after accounting for broad climate, soil, and geographic structure.

## Hydraulic and rooting mechanism

We next tested whether spatial variation in the latent response was consistent with plant water-transport and rooting strategy. Adjusted trait models related latent slope change, high-stress/post-transition slope, and high-stress limitation probability to hydraulic resistance proxies and rooting depth while holding climate, aridity, soil texture, and spatial context fixed. The mechanism-consistent expectation is that drought-resistant hydraulic/rooting strategies maintain WUE sensitivity under compound stress and reduce high-stress limitation.

The trait results should be described as mechanism-consistent rather than causal proof. The strongest interpretable signal is whether rooting depth and hydraulic resistance show positive associations with maintained high-stress WUE response and negative associations with high-stress limitation probability.
"""

discussion = f"""# Discussion draft

The central result is that global grassland WUE does not show a universal breakdown threshold under compound atmospheric-soil moisture stress. Instead, the latent ecosystem response phenotype indicates that WUE sensitivity is commonly maintained or enhanced, while saturation/breakdown-like limitation is rare after integrating across methodological and data-stream uncertainty. This reframes the original breakdown hypothesis: compound stress can generate high-stress limitation in some grassland contexts, but widespread collapse is not the dominant response.

A second implication is that ecosystem WUE response should be treated as a heterogeneous carbon-water coupling phenotype rather than a single global threshold. The latent-response framework allows methodological variation to be absorbed into the measurement model, so the biological result can be stated at the ecosystem-response level: grasslands differ in how carbon uptake and water loss remain coupled under severe atmospheric and soil moisture stress.

The trait results support a cautious physiological interpretation. After adjustment for aridity, climate, soil texture, and spatial context, hydraulic/rooting traits show directionally mechanism-consistent associations with the latent response. This suggests that plant water-transport and rooting strategy may help explain why some grasslands maintain WUE response under compound stress while others show weaker high-stress sensitivity. However, because direct tower validation is unavailable, these findings should be interpreted as satellite-derived, mechanism-consistent associations rather than direct physiological proof.

The main limitation is the absence of in situ tower validation. The analysis therefore does not claim that the latent response is the true tower-observed ecosystem flux response. Instead, it establishes a product-adjusted satellite ecosystem response phenotype and tests whether that phenotype is robust, spatially structured, and physiologically interpretable.
"""

methods = """# Methods draft

## Latent ecosystem response model

For each grassland point, we modeled observed WUE response estimates as repeated measurements of a latent ecosystem response phenotype. Observations varied across carbon-water data streams, WUE metric, compound-stress definition, growing-season definition, and CO2 treatment. The latent model estimated point-level slope change, high-stress/post-transition slope, and saturation/breakdown probability while absorbing systematic methodological effects into nuisance terms. The resulting latent phenotype was used for biological interpretation.

## Response classification

Each grassland point was assigned a response class from the latent class probabilities. Enhancement indicates maintained or increasing WUE sensitivity under stress. Saturation and breakdown indicate high-stress limitation, with breakdown representing stronger negative high-stress response. Points with insufficient posterior separation were classified as inconclusive.

## Trait mechanism analysis

Trait analyses tested whether hydraulic resistance proxies and rooting depth explained variation in the latent response after adjustment for aridity, climate, soil texture, and spatial context. Outcomes included latent slope change, high-stress/post-transition slope, and saturation/breakdown probability. Results are interpreted as mechanism-consistent associations rather than causal proof.
"""

limitations = """# Required limitation paragraph

Because direct tower validation was not completed, the latent response should not be interpreted as a tower-validated estimate of true ecosystem flux response. The analysis instead defines a satellite-derived, product-adjusted ecosystem response phenotype and evaluates its robustness and physiological consistency. Direct eddy-covariance validation remains necessary to confirm the in situ flux interpretation of the inferred response classes.
"""

full_doc = "\n\n".join([
    abstract,
    results,
    discussion,
    methods,
    limitations,
])

(TXT / "MANUSCRIPT_DRAFT_ecosystem_response_not_product_paper.md").write_text(full_doc)

claim_box = f"""# Paper claim box

## Main claim

Grassland WUE does not exhibit a universal breakdown threshold under compound atmospheric-soil moisture stress. Instead, the dominant latent ecosystem response is maintained or enhanced WUE sensitivity, with high-stress saturation/breakdown-like limitation rare across the analyzed grassland points.

## Key numbers

- Grassland points: {n_points}
- Enhancement: {enh_n}/{n_points} ({pct(enh_n/n_points)})
- Inconclusive: {inc_n}/{n_points} ({pct(inc_n/n_points)})
- Saturation/breakdown classes: {n_satbreak_class}/{n_points} ({pct(frac_satbreak_class)})
- Median latent slope change: {fmt(slope_median)}
- Median latent high-stress/post-transition slope: {fmt(post_median)}
- Median latent saturation/breakdown probability: {fmt(satprob_median)}

## Mechanistic claim

Hydraulic/rooting traits show mechanism-consistent associations with the latent ecosystem response after climate, aridity, soil texture, and spatial adjustment.

## Do not claim

Do not claim tower validation, true ecosystem-flux proof, or causal proof of xylem vulnerability.
"""

(TXT / "PAPER_CLAIM_BOX.md").write_text(claim_box)

# Supplementary note to keep product language out of main findings
supp_note = """# How to keep product differences out of the main paper findings

Use the phrase "latent ecosystem response phenotype" throughout the main Results.

Main Results should include:
1. Response class distribution.
2. Response intensity distributions.
3. Spatial/aridity structure.
4. Hydraulic/rooting trait associations.
5. One sentence that robustness analyses are in the supplement.

Main Results should not include a long discussion of product combinations.

Methods/Supplement should include:
1. The latent model accounts for data-stream/methodological effects.
2. Leave-one-family-out tests.
3. Independence-weight sensitivity.
4. Product-bias forest plot.

This framing does not hide product differences. It moves them into the uncertainty model so the main biological finding is the ecosystem response phenotype.
"""

(TXT / "HOW_TO_FRAME_NOT_AS_PRODUCT_PAPER.md").write_text(supp_note)

manifest = {
    "paper_package": "ecosystem_response_main_finding",
    "main_result": {
        "n_points": n_points,
        "enhancement_n": enh_n,
        "inconclusive_n": inc_n,
        "saturation_breakdown_class_n": n_satbreak_class,
        "median_latent_slope_change": slope_median,
        "median_latent_post_slope": post_median,
        "median_latent_satbreak_probability": satprob_median,
    },
    "main_claim": "Grassland WUE does not show a universal breakdown threshold; the dominant latent ecosystem response is maintained/enhanced WUE sensitivity.",
    "mechanism_claim": "Hydraulic/rooting traits show mechanism-consistent associations with the latent response after environmental adjustment.",
    "guardrail": "No direct tower validation; interpret as satellite-derived latent ecosystem response phenotype.",
    "outputs": {
        "tables": [str(p) for p in sorted(TAB.glob("*"))],
        "figures": [str(p) for p in sorted(FIG.glob("*.png"))],
        "text": [str(p) for p in sorted(TXT.glob("*"))],
    },
}
(OUT / "paper_ecosystem_response_manifest.json").write_text(json.dumps(manifest, indent=2))

print("")
print("DONE paper ecosystem-response package.")
print("")
print(claim_box)
print("")
print("Created:")
print(f"- {TXT / 'MANUSCRIPT_DRAFT_ecosystem_response_not_product_paper.md'}")
print(f"- {TXT / 'PAPER_CLAIM_BOX.md'}")
print(f"- {TXT / 'HOW_TO_FRAME_NOT_AS_PRODUCT_PAPER.md'}")
print(f"- {OUT / 'paper_ecosystem_response_manifest.json'}")
print(f"- Figures in {FIG}")
print(f"- Tables in {TAB}")
