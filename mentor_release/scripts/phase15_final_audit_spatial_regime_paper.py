#!/usr/bin/env python
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

try:
    import statsmodels.api as sm
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

PH8 = Path("results/trait_framework/phase8")
PH14 = Path("results/paper_spatial_regime_validation")

OUT = Path("results/paper_final_audit_spatial_regime")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
OBS = PH8 / "table_latent_model_observations.csv"
TRAIT = Path("results/trait_framework/trait_model_dataset.csv")

def die(msg):
    raise SystemExit(f"ERROR: {msg}")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing required file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df.loc[:, ~df.columns.duplicated()].copy()

def num(s):
    return pd.to_numeric(s, errors="coerce")

def fmt(x, d=3):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{d}f}"

def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100*float(x):.1f}%"

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"WROTE {path}")

def fisher_rr(local_event, rest_event):
    local_event = pd.Series(local_event).astype(bool)
    rest_event = pd.Series(rest_event).astype(bool)
    a = int(local_event.sum())
    b = int((~local_event).sum())
    c = int(rest_event.sum())
    d = int((~rest_event).sum())
    gf = a / (a + b) if (a + b) else np.nan
    rf = c / (c + d) if (c + d) else np.nan
    rr = gf / rf if rf and rf > 0 else np.nan
    try:
        odds, p = stats.fisher_exact([[a, b], [c, d]])
    except Exception:
        odds, p = np.nan, np.nan
    return a, b, c, d, gf, rf, rr, odds, p

latent = read_csv(LATENT)
obs = read_csv(OBS)
trait = read_csv(TRAIT, required=False)

latent["point_id"] = latent["point_id"].astype(str)
obs["point_id"] = obs["point_id"].astype(str)

if not trait.empty and "point_id" in trait.columns:
    trait["point_id"] = trait["point_id"].astype(str)
    keep = [
        "point_id", "mean_vpd", "aridity", "mean_soil_moisture",
        "mean_annual_temperature", "mean_annual_precipitation",
        "mean_lai", "soil_sand", "soil_clay", "soil_silt",
        "rooting_depth", "p50"
    ]
    keep = [c for c in keep if c in trait.columns]
    latent = latent.merge(trait[keep].drop_duplicates("point_id"), on="point_id", how="left", suffixes=("", "_trait"))

for c in latent.columns:
    if c not in ["point_id", "latent_response_class"]:
        try:
            latent[c] = num(latent[c])
        except Exception:
            pass

for c in ["lat", "lon", "latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]:
    if c not in latent.columns:
        die(f"Missing latent column: {c}")

df = latent.dropna(subset=["lat", "lon", "latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]).copy()
df = df.reset_index(drop=True)

# Define final regimes
df["regime_sahel"] = df["lat"].between(10, 20) & df["lon"].between(-20, 40)
df["regime_western_sahel"] = df["lat"].between(10, 17) & df["lon"].between(-10, 5)
df["regime_russian_steppe_west"] = df["lat"].between(45, 56) & df["lon"].between(30, 60)

if "mean_vpd" in df.columns:
    df["regime_high_vpd"] = df["mean_vpd"] > 2.26
else:
    df["regime_high_vpd"] = False

df["regime_sahel_high_vpd"] = df["regime_sahel"] & df["regime_high_vpd"]

regimes = [
    ("Sahel", "regime_sahel"),
    ("Western_Sahel_core", "regime_western_sahel"),
    ("High_VPD_gt_2p26", "regime_high_vpd"),
    ("Sahel_and_High_VPD", "regime_sahel_high_vpd"),
    ("Russian_steppe_west_secondary", "regime_russian_steppe_west"),
]

# Define event sensitivities
sat = num(df["latent_satbreak_probability"])
post = num(df["latent_post_slope"])
slope = num(df["latent_slope_change"])

event_defs = {}
for q in [0.75, 0.80, 0.85, 0.90]:
    suffix = int(q * 100)
    event_defs[f"satprob_top{100-suffix}"] = sat >= sat.quantile(q)

for q in [0.10, 0.15, 0.20, 0.25]:
    suffix = int(q * 100)
    event_defs[f"post_slope_bottom{suffix}"] = post <= post.quantile(q)
    event_defs[f"slope_change_bottom{suffix}"] = slope <= slope.quantile(q)

event_defs["combined_main_top20_or_weak20"] = (
    event_defs["satprob_top20"] |
    event_defs["post_slope_bottom20"] |
    event_defs["slope_change_bottom20"]
)

# Table 60: regime x event threshold sensitivity
rows = []
for reg_name, reg_col in regimes:
    mask = df[reg_col].fillna(False).astype(bool)
    for event_name, event in event_defs.items():
        local = event[mask]
        rest = event[~mask]
        a,b,c,d,gf,rf,rr,odds,p = fisher_rr(local, rest)
        rows.append({
            "regime": reg_name,
            "n_points": int(mask.sum()),
            "event_definition": event_name,
            "inside_event_n": a,
            "inside_fraction": gf,
            "outside_fraction": rf,
            "risk_ratio": rr,
            "odds_ratio": odds,
            "fisher_p": p,
        })

event_sens = pd.DataFrame(rows)
save_csv(event_sens, TAB / "Table60_regime_event_threshold_sensitivity.csv")

# Table 61: continuous outcome effect sizes
rows = []
for reg_name, reg_col in regimes:
    mask = df[reg_col].fillna(False).astype(bool)
    for outcome in ["latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]:
        inside = num(df.loc[mask, outcome]).dropna()
        outside = num(df.loc[~mask, outcome]).dropna()
        if len(inside) < 3 or len(outside) < 3:
            continue
        try:
            p = stats.mannwhitneyu(inside, outside).pvalue
        except Exception:
            p = np.nan
        rows.append({
            "regime": reg_name,
            "n_points": int(mask.sum()),
            "outcome": outcome,
            "inside_median": float(inside.median()),
            "outside_median": float(outside.median()),
            "median_diff": float(inside.median() - outside.median()),
            "inside_mean": float(inside.mean()),
            "outside_mean": float(outside.mean()),
            "mannwhitney_p": p,
        })
cont = pd.DataFrame(rows)
save_csv(cont, TAB / "Table61_continuous_outcome_effects_by_regime.csv")

# Raw fit threshold-like classification
if "product_combo" not in obs.columns:
    if {"gpp_product", "et_product"}.issubset(obs.columns):
        obs["product_combo"] = obs["gpp_product"].astype(str) + "/" + obs["et_product"].astype(str)
    else:
        obs["product_combo"] = "unknown"

if "response_class_4way" in obs.columns:
    obs["threshold_like_fit"] = obs["response_class_4way"].astype(str).str.lower().isin(["saturation", "breakdown"])
elif "response_class_original" in obs.columns:
    obs["threshold_like_fit"] = obs["response_class_original"].astype(str).str.lower().isin(["saturation", "breakdown"])
elif "latent_response_class" in obs.columns:
    obs["threshold_like_fit"] = obs["latent_response_class"].astype(str).str.lower().isin(["saturation", "breakdown"])
else:
    # fallback: any response class column containing saturation/breakdown
    response_cols = [c for c in obs.columns if "class" in c.lower()]
    if response_cols:
        c = response_cols[0]
        obs["threshold_like_fit"] = obs[c].astype(str).str.lower().isin(["saturation", "breakdown"])
    else:
        obs["threshold_like_fit"] = False

# Attach regime membership to obs
regime_membership = df[["point_id"] + [r[1] for r in regimes]].copy()
obs2 = obs.merge(regime_membership, on="point_id", how="left")

# Table 62: by regime x metric x product_combo
rows = []
strata_cols = []
for c in ["metric", "product_combo", "stress_definition", "growing_season", "co2_version"]:
    if c in obs2.columns:
        strata_cols.append(c)

for reg_name, reg_col in regimes:
    if reg_col not in obs2.columns:
        continue
    for strata in [
        ["metric"],
        ["product_combo"],
        ["metric", "product_combo"],
        ["stress_definition"],
        ["co2_version"],
        ["metric", "stress_definition"],
    ]:
        strata = [c for c in strata if c in obs2.columns]
        if not strata:
            continue

        for keys, sub in obs2.groupby(strata, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            local = sub[sub[reg_col].fillna(False).astype(bool)]
            rest = sub[~sub[reg_col].fillna(False).astype(bool)]
            if len(local) < 20 or len(rest) < 20:
                continue
            a,b,c,d,gf,rf,rr,odds,p = fisher_rr(local["threshold_like_fit"], rest["threshold_like_fit"])
            row = {
                "regime": reg_name,
                "strata": "+".join(strata),
                "n_local_fits": int(len(local)),
                "n_rest_fits": int(len(rest)),
                "threshold_fraction_local": gf,
                "threshold_fraction_rest": rf,
                "risk_ratio": rr,
                "odds_ratio": odds,
                "fisher_p": p,
            }
            for col, val in zip(strata, keys):
                row[col] = val
            rows.append(row)

prod_sens = pd.DataFrame(rows)
save_csv(prod_sens, TAB / "Table62_raw_threshold_sensitivity_by_metric_product_stress.csv")

# Table 63: simple adjusted models
model_rows = []
if STATSMODELS_OK:
    mdf = df.copy()
    predictors = []
    for c in ["mean_vpd", "lat", "aridity", "mean_soil_moisture", "mean_annual_temperature", "mean_annual_precipitation", "mean_lai", "soil_sand", "soil_clay", "soil_silt"]:
        if c in mdf.columns and num(mdf[c]).notna().sum() >= 40:
            mdf[c] = num(mdf[c])
            mdf[c + "_z"] = (mdf[c] - mdf[c].mean()) / (mdf[c].std() if mdf[c].std() else 1)
            predictors.append(c + "_z")

    mdf["sahel"] = mdf["regime_sahel"].astype(int)
    mdf["high_vpd"] = mdf["regime_high_vpd"].astype(int)
    mdf["sahel_x_high_vpd"] = mdf["sahel"] * mdf["high_vpd"]

    base_predictors = ["high_vpd", "sahel", "sahel_x_high_vpd"] + predictors

    for outcome in ["latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]:
        d = mdf[[outcome] + base_predictors].dropna()
        if len(d) < 40:
            continue
        y = num(d[outcome])
        X = sm.add_constant(d[base_predictors].astype(float))
        try:
            fit = sm.OLS(y, X).fit(cov_type="HC3")
            for term in ["high_vpd", "sahel", "sahel_x_high_vpd", "mean_vpd_z", "lat_z", "aridity_z", "mean_soil_moisture_z"]:
                if term in fit.params.index:
                    model_rows.append({
                        "model": "OLS_HC3",
                        "outcome": outcome,
                        "term": term,
                        "n": int(len(d)),
                        "estimate": float(fit.params[term]),
                        "std_error": float(fit.bse[term]),
                        "p_value": float(fit.pvalues[term]),
                        "ci_low": float(fit.conf_int().loc[term, 0]),
                        "ci_high": float(fit.conf_int().loc[term, 1]),
                        "r2": float(fit.rsquared),
                    })
        except Exception as e:
            model_rows.append({
                "model": "OLS_HC3",
                "outcome": outcome,
                "term": "MODEL_FAILED",
                "n": int(len(d)),
                "estimate": np.nan,
                "std_error": np.nan,
                "p_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "r2": np.nan,
                "error": str(e),
            })

models = pd.DataFrame(model_rows)
save_csv(models, TAB / "Table63_adjusted_spatial_hydroclimatic_models.csv")

# Evidence scorecard
score_rows = []

def add_score(name, status, evidence, interpretation):
    score_rows.append({
        "criterion": name,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    })

# Main criteria
high_vpd_main = event_sens[
    (event_sens["regime"].eq("High_VPD_gt_2p26")) &
    (event_sens["event_definition"].eq("combined_main_top20_or_weak20"))
]
sahel_main = event_sens[
    (event_sens["regime"].eq("Sahel")) &
    (event_sens["event_definition"].eq("combined_main_top20_or_weak20"))
]

if len(high_vpd_main):
    r = high_vpd_main.iloc[0]
    add_score(
        "High-VPD regime enrichment",
        "PASS" if r["risk_ratio"] > 2 and r["fisher_p"] < 0.01 else "WEAK",
        f"RR={r['risk_ratio']:.3f}, inside={r['inside_fraction']:.3f}, outside={r['outside_fraction']:.3f}, p={r['fisher_p']:.3g}",
        "This is the core mechanistic-regime result."
    )

if len(sahel_main):
    r = sahel_main.iloc[0]
    add_score(
        "Sahel named-region enrichment",
        "PASS" if r["risk_ratio"] > 2 and r["fisher_p"] < 0.01 else "WEAK",
        f"RR={r['risk_ratio']:.3f}, inside={r['inside_fraction']:.3f}, outside={r['outside_fraction']:.3f}, p={r['fisher_p']:.3g}",
        "Sahel is the clearest named geographic expression."
    )

# Product/metric robustness criterion
if not prod_sens.empty:
    d = prod_sens[
        prod_sens["regime"].eq("High_VPD_gt_2p26") &
        prod_sens["strata"].eq("metric")
    ].copy()
    if len(d):
        min_rr = d["risk_ratio"].min()
        add_score(
            "High-VPD across WUE metrics",
            "PASS" if min_rr > 1 else "WEAK",
            f"metric-level min RR={min_rr:.3f}",
            "Shows whether regime result survives uWUE/raw/iWUE sensitivity."
        )

    d = prod_sens[
        prod_sens["regime"].eq("High_VPD_gt_2p26") &
        prod_sens["strata"].eq("product_combo")
    ].copy()
    if len(d):
        frac_rr_gt_1 = float((d["risk_ratio"] > 1).mean())
        add_score(
            "High-VPD across product combinations",
            "PASS" if frac_rr_gt_1 >= 0.67 else "WEAK",
            f"fraction product combos RR>1 = {frac_rr_gt_1:.3f}",
            "This keeps product differences in supplement rather than main finding."
        )

# Model criterion
if not models.empty:
    d = models[
        models["term"].isin(["high_vpd", "mean_vpd_z"]) &
        models["outcome"].isin(["latent_post_slope", "latent_slope_change"])
    ].copy()
    if len(d):
        add_score(
            "Adjusted high-VPD model evidence",
            "CHECK",
            d[["outcome", "term", "estimate", "p_value"]].to_string(index=False),
            "Use this to decide whether the final wording should emphasize binary high-VPD or continuous VPD."
        )

score = pd.DataFrame(score_rows)
save_csv(score, TAB / "Table64_final_evidence_scorecard.csv")

# Figures
plot = event_sens[event_sens["event_definition"].eq("combined_main_top20_or_weak20")].copy()
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(plot))
ax.bar(x, plot["risk_ratio"])
ax.axhline(1, linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(plot["regime"], rotation=25, ha="right")
ax.set_ylabel("Hotspot risk ratio")
ax.set_title("Figure 1. Final regime enrichment audit")
savefig(FIG / "Figure1_final_regime_hotspot_risk_ratios.png")

plot = cont[cont["outcome"].isin(["latent_post_slope", "latent_slope_change"])].copy()
plot["label"] = plot["regime"] + " | " + plot["outcome"]
fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot))))
y = np.arange(len(plot))
ax.barh(y, plot["median_diff"])
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(plot["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Inside minus outside median difference")
ax.set_title("Figure 2. Regime effect on WUE response slopes")
savefig(FIG / "Figure2_regime_slope_effects.png")

if not prod_sens.empty:
    d = prod_sens[
        prod_sens["regime"].eq("High_VPD_gt_2p26") &
        prod_sens["strata"].eq("metric")
    ].copy()
    if len(d):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(d["metric"].astype(str), d["risk_ratio"])
        ax.axhline(1, linestyle="--", linewidth=1)
        ax.set_ylabel("Raw threshold-like fit risk ratio")
        ax.set_title("Figure 3. High-VPD threshold enrichment across WUE metrics")
        savefig(FIG / "Figure3_high_vpd_metric_robustness.png")

# Final recommendation text
recommendation = """# Phase 15 final audit: what to do with the paper

## Recommended final title

Hydroclimatic Geography Controls Where Grassland WUE Thresholds Emerge Under Compound Stress

## Main claim

High-stress WUE limitation is not a universal grassland response. It is concentrated in a high-VPD, low-latitude dryland regime. The Sahel is the clearest named geographic expression of this regime, while the high-VPD rule is the stronger mechanistic summary.

## How to write the Results

1. Start with the global null: no universal grassland WUE breakdown.
2. Show the spatial-regime scan: Sahel and high-VPD regimes rise to the top.
3. Make high VPD the mechanistic regime: report the High_VPD_gt_2p26 row.
4. Use Sahel as the named ecosystem case: report Sahel and Sahel+High-VPD.
5. Use climate matching to say the Sahel is not uniquely high in limitation probability beyond climate, but still shows weaker high-stress slopes and slope-change response.
6. Put Russian_steppe_west in Supplement as a secondary elevated-limitation-probability contrast, not a clean slope-collapse example.
7. Put product/metric/stress details in Supplement.

## Most important checks from this phase

Read:
- Table60_regime_event_threshold_sensitivity.csv
- Table61_continuous_outcome_effects_by_regime.csv
- Table62_raw_threshold_sensitivity_by_metric_product_stress.csv
- Table63_adjusted_spatial_hydroclimatic_models.csv
- Table64_final_evidence_scorecard.csv

## If the scorecard passes

Start writing the manuscript. Do not run more exploratory phases unless a table fails.

## If the scorecard is weak

Use the safer title:
Spatially Localized High-Stress Limitation of Satellite-Derived Grassland WUE Under Compound Stress

## Non-negotiable limitation

This remains a satellite-derived, product-adjusted latent response phenotype. It is not tower-validated ecosystem flux truth.
"""

(TXT / "PHASE15_FINAL_AUDIT_RECOMMENDATION.md").write_text(recommendation)

manifest = {
    "phase": "Phase 15 final audit for spatial-regime WUE paper",
    "outputs": {
        "event_sensitivity": str(TAB / "Table60_regime_event_threshold_sensitivity.csv"),
        "continuous_effects": str(TAB / "Table61_continuous_outcome_effects_by_regime.csv"),
        "product_metric_sensitivity": str(TAB / "Table62_raw_threshold_sensitivity_by_metric_product_stress.csv"),
        "adjusted_models": str(TAB / "Table63_adjusted_spatial_hydroclimatic_models.csv"),
        "scorecard": str(TAB / "Table64_final_evidence_scorecard.csv"),
        "recommendation": str(TXT / "PHASE15_FINAL_AUDIT_RECOMMENDATION.md"),
    },
    "statsmodels_available": STATSMODELS_OK,
}
(OUT / "phase15_final_audit_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

print("")
print("DONE Phase 15 final audit.")
print("")
print(recommendation)
