#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

PH8 = Path("results/trait_framework/phase8")
PH11 = Path("results/paper_geographic_hotspots")
OUT = Path("results/paper_geographic_hotspots_final")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
OBS = PH8 / "table_latent_model_observations.csv"

PREDEF = PH11 / "tables/Table20_predefined_geographic_region_scan.csv"
KNN = PH11 / "tables/Table22_top_nonoverlapping_knn_geographic_hotspots.csv"
RAW = PH11 / "tables/Table25_local_threshold_evidence_from_all_fits.csv"
BESTPTS = PH11 / "tables/Table26_points_in_best_geographic_hotspot.csv"
EURASIA = PH11 / "tables/Table27_eurasian_russian_steppe_region_scan.csv"

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

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

def fisher_event(group_flag, event_flag):
    g = pd.Series(group_flag).astype(bool)
    e = pd.Series(event_flag).astype(bool)
    a = int((g & e).sum())
    b = int((g & ~e).sum())
    c = int((~g & e).sum())
    d = int((~g & ~e).sum())
    try:
        odds, p = stats.fisher_exact([[a,b],[c,d]], alternative="two-sided")
    except Exception:
        odds, p = np.nan, np.nan
    gf = a / (a+b) if (a+b) else np.nan
    rf = c / (c+d) if (c+d) else np.nan
    rr = gf / rf if rf and rf > 0 else np.nan
    return a,b,c,d,gf,rf,rr,odds,p

def mann_p(a,b):
    a = num(pd.Series(a)).dropna()
    b = num(pd.Series(b)).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a,b,alternative="two-sided").pvalue)
    except Exception:
        return np.nan

latent = read_csv(LATENT)
obs = read_csv(OBS)
predef = read_csv(PREDEF)
knn = read_csv(KNN)
raw = read_csv(RAW, required=False)
bestpts = read_csv(BESTPTS, required=False)
eurasia = read_csv(EURASIA, required=False)

for c in ["lat","lon","latent_satbreak_probability","latent_post_slope","latent_slope_change"]:
    if c in latent.columns:
        latent[c] = num(latent[c])
latent["point_id"] = latent["point_id"].astype(str)

# Global background
n_points = int(latent["point_id"].nunique())
enh_n = int(latent["latent_response_class"].astype(str).eq("enhancement").sum())
inc_n = int(latent["latent_response_class"].astype(str).eq("inconclusive").sum())
hard_satbreak_n = int(latent["latent_response_class"].astype(str).isin(["saturation","breakdown"]).sum())

sat = num(latent["latent_satbreak_probability"])
post = num(latent["latent_post_slope"])
slope = num(latent["latent_slope_change"])

latent["event_satprob_top20"] = sat >= sat.quantile(0.80)
latent["event_post_slope_bottom20"] = post <= post.quantile(0.20)
latent["event_slope_change_bottom20"] = slope <= slope.quantile(0.20)
latent["event_geographic_limitation_hotspot"] = (
    latent["event_satprob_top20"].fillna(False)
    | latent["event_post_slope_bottom20"].fillna(False)
    | latent["event_slope_change_bottom20"].fillna(False)
)

# Region definitions copied from phase 11
REGIONS = {
    "Sahel": (10, 20, -20, 40),
    "Russian_steppe_west": (45, 56, 30, 60),
    "Pontic_Caspian_steppe": (42, 52, 25, 55),
    "Kazakh_steppe_core": (43, 54, 45, 80),
    "Russian_Kazakh_steppe_broad": (42, 58, 35, 95),
    "Mongolian_Manchurian_steppe": (40, 55, 90, 125),
    "East_African_savanna": (-10, 12, 25, 45),
    "Cerrado_savanna_grassland": (-25, -5, -60, -40),
}

def region_mask(name):
    latmin, latmax, lonmin, lonmax = REGIONS[name]
    return (
        (latent["lat"] >= latmin) &
        (latent["lat"] <= latmax) &
        (latent["lon"] >= lonmin) &
        (latent["lon"] <= lonmax)
    )

def summarize_mask(name, mask, role):
    g = pd.Series(mask, index=latent.index).fillna(False).astype(bool)
    row = {
        "region_role": role,
        "region": name,
        "n_points": int(g.sum()),
        "n_rest": int((~g).sum()),
        "center_lat": float(latent.loc[g, "lat"].mean()) if g.sum() else np.nan,
        "center_lon": float(latent.loc[g, "lon"].mean()) if g.sum() else np.nan,
        "min_lat": float(latent.loc[g, "lat"].min()) if g.sum() else np.nan,
        "max_lat": float(latent.loc[g, "lat"].max()) if g.sum() else np.nan,
        "min_lon": float(latent.loc[g, "lon"].min()) if g.sum() else np.nan,
        "max_lon": float(latent.loc[g, "lon"].max()) if g.sum() else np.nan,
    }
    for outcome in ["latent_satbreak_probability","latent_post_slope","latent_slope_change"]:
        row[f"{outcome}_inside_median"] = float(num(latent.loc[g, outcome]).median())
        row[f"{outcome}_outside_median"] = float(num(latent.loc[~g, outcome]).median())
        row[f"{outcome}_median_diff"] = row[f"{outcome}_inside_median"] - row[f"{outcome}_outside_median"]
        row[f"{outcome}_mann_p"] = mann_p(latent.loc[g, outcome], latent.loc[~g, outcome])
    a,b,c,d,gf,rf,rr,odds,p = fisher_event(g, latent["event_geographic_limitation_hotspot"])
    row.update({
        "hotspot_inside_n": a,
        "hotspot_inside_fraction": gf,
        "hotspot_outside_fraction": rf,
        "hotspot_fraction_diff": gf-rf if pd.notna(gf) and pd.notna(rf) else np.nan,
        "hotspot_risk_ratio": rr,
        "hotspot_odds_ratio": odds,
        "hotspot_fisher_p": p,
        "point_ids": ";".join(latent.loc[g, "point_id"].tolist()),
    })

    # raw fit-level threshold evidence
    if not obs.empty and "point_id" in obs.columns:
        pts = set(latent.loc[g, "point_id"].astype(str))
        o = obs.copy()
        o["point_id"] = o["point_id"].astype(str)
        if "response_class_4way" in o.columns:
            o["threshold_like_fit"] = o["response_class_4way"].astype(str).isin(["saturation","breakdown"])
        elif "response_class_original" in o.columns:
            o["threshold_like_fit"] = o["response_class_original"].astype(str).str.lower().isin(["saturation","breakdown"])
        else:
            o["threshold_like_fit"] = False
        loc = o[o["point_id"].isin(pts)]
        rest = o[~o["point_id"].isin(pts)]
        row["n_local_fits"] = int(len(loc))
        row["threshold_like_fit_fraction_inside"] = float(loc["threshold_like_fit"].mean()) if len(loc) else np.nan
        row["threshold_like_fit_fraction_outside"] = float(rest["threshold_like_fit"].mean()) if len(rest) else np.nan
        row["threshold_like_fit_risk_ratio"] = (
            row["threshold_like_fit_fraction_inside"] / row["threshold_like_fit_fraction_outside"]
            if row["threshold_like_fit_fraction_outside"] and row["threshold_like_fit_fraction_outside"] > 0
            else np.nan
        )
        if "metric" in o.columns:
            loc_u = loc[loc["metric"].astype(str).str.lower().eq("uwue")]
            rest_u = rest[rest["metric"].astype(str).str.lower().eq("uwue")]
            row["uwue_threshold_like_fit_fraction_inside"] = float(loc_u["threshold_like_fit"].mean()) if len(loc_u) else np.nan
            row["uwue_threshold_like_fit_fraction_outside"] = float(rest_u["threshold_like_fit"].mean()) if len(rest_u) else np.nan
    return row

# Main region summaries
rows = []
rows.append(summarize_mask("Sahel", region_mask("Sahel"), "primary_predefined_region"))
rows.append(summarize_mask("Russian_steppe_west", region_mask("Russian_steppe_west"), "secondary_eurasian_region"))
rows.append(summarize_mask("Pontic_Caspian_steppe", region_mask("Pontic_Caspian_steppe"), "secondary_eurasian_region"))
rows.append(summarize_mask("Kazakh_steppe_core", region_mask("Kazakh_steppe_core"), "secondary_eurasian_region"))
rows.append(summarize_mask("Russian_Kazakh_steppe_broad", region_mask("Russian_Kazakh_steppe_broad"), "negative_broad_eurasian_check"))

# Best KNN core from Phase 11
if not bestpts.empty:
    bestpts["point_id"] = bestpts["point_id"].astype(str)
    core_pts = set(bestpts["point_id"])
    rows.append(summarize_mask("Western_Sahel_core_KNN6", latent["point_id"].isin(core_pts), "local_core_hotspot"))

summary = pd.DataFrame(rows)
save_csv(summary, TAB / "Table30_final_geographic_regions_for_main_text.csv")

# Jackknife robustness for each region
jack_rows = []
for _, r in summary.iterrows():
    pts = [p for p in str(r["point_ids"]).split(";") if p]
    region = r["region"]
    role = r["region_role"]
    if len(pts) < 5:
        continue

    # leave-one-out
    for leave_k in [1, 2]:
        if len(pts) <= leave_k + 3:
            continue
        combs = list(combinations(pts, leave_k))
        if len(combs) > 250:
            # deterministic sample to keep fast
            np.random.seed(42)
            idx = np.random.choice(len(combs), size=250, replace=False)
            combs = [combs[i] for i in idx]

        vals = []
        for leave in combs:
            keep = set(pts) - set(leave)
            mask = latent["point_id"].isin(keep)
            tmp = summarize_mask(region, mask, f"{role}_leave{leave_k}")
            vals.append(tmp)

        v = pd.DataFrame(vals)
        jack_rows.append({
            "region": region,
            "region_role": role,
            "leave_k": leave_k,
            "n_replicates": int(len(v)),
            "n_points_original": int(len(pts)),
            "satprob_inside_median_min": float(v["latent_satbreak_probability_inside_median"].min()),
            "satprob_inside_median_median": float(v["latent_satbreak_probability_inside_median"].median()),
            "satprob_inside_median_max": float(v["latent_satbreak_probability_inside_median"].max()),
            "post_slope_inside_median_min": float(v["latent_post_slope_inside_median"].min()),
            "post_slope_inside_median_median": float(v["latent_post_slope_inside_median"].median()),
            "post_slope_inside_median_max": float(v["latent_post_slope_inside_median"].max()),
            "hotspot_risk_ratio_min": float(v["hotspot_risk_ratio"].min()),
            "hotspot_risk_ratio_median": float(v["hotspot_risk_ratio"].median()),
            "hotspot_risk_ratio_max": float(v["hotspot_risk_ratio"].max()),
            "fraction_replicates_rr_gt_1": float((v["hotspot_risk_ratio"] > 1).mean()),
            "fraction_replicates_satprob_diff_gt_0": float((v["latent_satbreak_probability_median_diff"] > 0).mean()),
            "fraction_replicates_post_slope_diff_lt_0": float((v["latent_post_slope_median_diff"] < 0).mean()),
        })
jack = pd.DataFrame(jack_rows)
save_csv(jack, TAB / "Table31_jackknife_region_robustness.csv")

# Evidence hierarchy
hierarchy = pd.DataFrame([
    {
        "tier": "Tier 1",
        "claim": "Global grasslands do not show universal breakdown.",
        "evidence": f"{enh_n}/{n_points} enhancement, {inc_n}/{n_points} inconclusive, {hard_satbreak_n}/{n_points} hard saturation/breakdown.",
        "use_in_paper": "Opening result",
    },
    {
        "tier": "Tier 2",
        "claim": "The Sahel is the primary geographic hotspot of high-stress WUE limitation.",
        "evidence": "Predefined Sahel region has n=26 and strong enrichment; use this as main geographic claim.",
        "use_in_paper": "Main geographic ecosystem result",
    },
    {
        "tier": "Tier 3",
        "claim": "A western Sahel core shows the strongest local threshold-like expression.",
        "evidence": "KNN6 core has negative median high-stress slope and 100% hotspot fraction, but n=6.",
        "use_in_paper": "Local intensification / case-study inset",
    },
    {
        "tier": "Tier 4",
        "claim": "Western Russian steppe shows elevated limitation probability, but not clean WUE reversal.",
        "evidence": "n=10, high limitation probability and hotspot enrichment, but high-stress slope is not lower than outside.",
        "use_in_paper": "Supplementary geographic comparison, not main claim",
    },
])
save_csv(hierarchy, TAB / "Table32_final_claim_evidence_hierarchy.csv")

# Figures
# Figure 1 global map
fig, ax = plt.subplots(figsize=(10, 5.8))
sc = ax.scatter(
    latent["lon"], latent["lat"],
    c=latent["latent_satbreak_probability"],
    s=35,
    alpha=0.65,
)
# Sahel box
latmin, latmax, lonmin, lonmax = REGIONS["Sahel"]
ax.plot([lonmin, lonmax, lonmax, lonmin, lonmin], [latmin, latmin, latmax, latmax, latmin], linewidth=2, label="Sahel primary region")
# Russian west box
latmin, latmax, lonmin, lonmax = REGIONS["Russian_steppe_west"]
ax.plot([lonmin, lonmax, lonmax, lonmin, lonmin], [latmin, latmin, latmax, latmax, latmin], linewidth=1.5, linestyle="--", label="Russian steppe west")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Figure 1. Geographic concentration of high-stress WUE limitation")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Latent high-stress limitation probability")
ax.legend(frameon=False)
savefig(FIG / "Figure1_global_map_sahel_and_russian_hotspots.png")

# Figure 2 region comparison
plot = summary.copy()
plot["label"] = plot["region"]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(plot))
ax.bar(x, plot["hotspot_risk_ratio"])
ax.axhline(1, linestyle="--", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(plot["label"], rotation=30, ha="right")
ax.set_ylabel("High-stress hotspot risk ratio")
ax.set_title("Figure 2. Geographic-region enrichment of high-stress WUE limitation")
savefig(FIG / "Figure2_region_hotspot_risk_ratios.png")

# Figure 3 Sahel distributions
sahel_mask = region_mask("Sahel")
for outcome, ylabel, fname in [
    ("latent_satbreak_probability", "Latent high-stress limitation probability", "Figure3A_sahel_limitation_probability.png"),
    ("latent_post_slope", "Latent high-stress/post-transition slope", "Figure3B_sahel_post_slope.png"),
    ("latent_slope_change", "Latent slope change", "Figure3C_sahel_slope_change.png"),
]:
    vals = [
        num(latent.loc[sahel_mask, outcome]).dropna(),
        num(latent.loc[~sahel_mask, outcome]).dropna(),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.boxplot(vals, labels=["Sahel", "all other grasslands"], showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " in the Sahel hotspot")
    savefig(FIG / fname)

# Figure 4 jackknife robustness
if not jack.empty:
    plot = jack[jack["leave_k"].eq(1)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(plot))
    ax.errorbar(
        x,
        plot["hotspot_risk_ratio_median"],
        yerr=[
            plot["hotspot_risk_ratio_median"] - plot["hotspot_risk_ratio_min"],
            plot["hotspot_risk_ratio_max"] - plot["hotspot_risk_ratio_median"],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axhline(1, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["region"], rotation=30, ha="right")
    ax.set_ylabel("Leave-one-out hotspot risk ratio")
    ax.set_title("Figure 4. Jackknife robustness of geographic hotspot enrichment")
    savefig(FIG / "Figure4_jackknife_hotspot_robustness.png")

# Text
sahel = summary[summary["region"].eq("Sahel")].iloc[0]
core = summary[summary["region"].eq("Western_Sahel_core_KNN6")].iloc[0] if "Western_Sahel_core_KNN6" in summary["region"].values else None
rus = summary[summary["region"].eq("Russian_steppe_west")].iloc[0]

decision = f"""# Final geographic ecosystem-response paper decision

## Final recommended paper claim

**Use the Sahel as the primary geographic ecosystem-response finding.**

Main claim:

> Grassland WUE breakdown is not universal globally, but high-stress WUE limitation is geographically concentrated in the Sahel grassland belt, with a smaller western Sahel core showing the strongest local threshold-like behavior.

## Why Sahel is the main claim

The predefined Sahel region is large enough to defend as a geographic ecosystem result:

- Sahel points: {int(sahel['n_points'])}/{n_points}
- Median limitation probability inside Sahel: {fmt(sahel['latent_satbreak_probability_inside_median'])}
- Median limitation probability outside Sahel: {fmt(sahel['latent_satbreak_probability_outside_median'])}
- Median high-stress slope inside Sahel: {fmt(sahel['latent_post_slope_inside_median'])}
- Median high-stress slope outside Sahel: {fmt(sahel['latent_post_slope_outside_median'])}
- Hotspot fraction inside Sahel: {pct(sahel['hotspot_inside_fraction'])}
- Hotspot fraction outside Sahel: {pct(sahel['hotspot_outside_fraction'])}
- Hotspot risk ratio: {fmt(sahel['hotspot_risk_ratio'])}
- Raw threshold-like fit fraction inside Sahel: {pct(sahel['threshold_like_fit_fraction_inside'])}
- Raw threshold-like fit fraction outside Sahel: {pct(sahel['threshold_like_fit_fraction_outside'])}
- Raw threshold-like fit risk ratio: {fmt(sahel['threshold_like_fit_risk_ratio'])}

## How to use the 6-point KNN hotspot

Use the 6-point western Sahel KNN region as a **local core** or **inset**, not the main paper claim.

{'' if core is None else f"- Core points: {int(core['n_points'])}\n- Core median limitation probability: {fmt(core['latent_satbreak_probability_inside_median'])}\n- Core median high-stress slope: {fmt(core['latent_post_slope_inside_median'])}\n- Core hotspot fraction: {pct(core['hotspot_inside_fraction'])}\n- Core raw threshold-like fit risk ratio: {fmt(core['threshold_like_fit_risk_ratio'])}"}

## Russian steppe interpretation

Russian_steppe_west is a real secondary signal, but it is **not** the cleanest breakdown story.

- Russian_steppe_west points: {int(rus['n_points'])}
- Median limitation probability inside: {fmt(rus['latent_satbreak_probability_inside_median'])}
- Median limitation probability outside: {fmt(rus['latent_satbreak_probability_outside_median'])}
- Hotspot risk ratio: {fmt(rus['hotspot_risk_ratio'])}
- Median high-stress slope inside: {fmt(rus['latent_post_slope_inside_median'])}
- Median high-stress slope outside: {fmt(rus['latent_post_slope_outside_median'])}

Interpretation: the western Russian steppe shows elevated high-stress limitation probability, but because its median high-stress slope is not lower than outside, it should be framed as a supplementary geographic contrast, not the main threshold-breakdown claim.

## Final title

**Localized High-Stress Limitation of Grassland Water-Use Efficiency in the Sahel Under Compound Atmospheric–Soil Moisture Stress**

## Final abstract-style result sentence

Across global grassland points, WUE breakdown was not universal; however, the Sahel grassland belt showed a geographically concentrated high-stress limitation phenotype, with higher latent limitation probability, weaker high-stress WUE slope, and roughly doubled threshold-like fit frequency relative to other grasslands.

## Claims to avoid

- Do not claim global grassland WUE breakdown.
- Do not make the n=6 KNN hotspot the only headline.
- Do not claim Russian grasslands are the main threshold region unless you specifically frame it as secondary elevated limitation probability.
- Do not claim tower validation.
"""

(TXT / "FINAL_GEOGRAPHIC_PAPER_DECISION.md").write_text(decision)

abstract = f"""# Final abstract draft

Grassland water-use efficiency (WUE) responses to compound atmospheric and soil-moisture stress are often framed as a search for a universal breakdown threshold. Here we instead show that high-stress WUE limitation is geographically localized. Using a product-adjusted latent ecosystem response phenotype across {n_points} grassland points, we found no evidence for universal WUE collapse: {enh_n}/{n_points} points were classified as enhancement and {hard_satbreak_n}/{n_points} as hard saturation/breakdown. However, the Sahel grassland belt showed a strong localized limitation phenotype. Compared with all other grasslands, Sahel points had higher median high-stress limitation probability ({fmt(sahel['latent_satbreak_probability_inside_median'])} vs {fmt(sahel['latent_satbreak_probability_outside_median'])}), weaker high-stress WUE response slope ({fmt(sahel['latent_post_slope_inside_median'])} vs {fmt(sahel['latent_post_slope_outside_median'])}), and enriched high-stress hotspot occurrence ({pct(sahel['hotspot_inside_fraction'])} vs {pct(sahel['hotspot_outside_fraction'])}; risk ratio {fmt(sahel['hotspot_risk_ratio'])}). Local fit-level threshold-like classifications were also more frequent inside the Sahel than outside ({pct(sahel['threshold_like_fit_fraction_inside'])} vs {pct(sahel['threshold_like_fit_fraction_outside'])}). These results suggest that compound stress does not induce a universal grassland WUE breakdown, but can produce geographically concentrated high-stress limitation in vulnerable dryland grassland systems.
"""

(TXT / "FINAL_ABSTRACT_DRAFT_SAHEL_GEOGRAPHIC_HOTSPOT.md").write_text(abstract)

results = f"""# Final Results structure

## Result 1: No universal grassland WUE breakdown

Start with the global result: {enh_n}/{n_points} enhancement, {inc_n}/{n_points} inconclusive, and {hard_satbreak_n}/{n_points} hard saturation/breakdown. This sets up the important contrast: the global signal is not collapse.

## Result 2: The Sahel is the primary geographic hotspot

Report the predefined Sahel region, not the tiny KNN core, as the main geographic result. Use the following numbers:

- n = {int(sahel['n_points'])}
- median limitation probability: {fmt(sahel['latent_satbreak_probability_inside_median'])} vs {fmt(sahel['latent_satbreak_probability_outside_median'])}
- median high-stress slope: {fmt(sahel['latent_post_slope_inside_median'])} vs {fmt(sahel['latent_post_slope_outside_median'])}
- hotspot occurrence: {pct(sahel['hotspot_inside_fraction'])} vs {pct(sahel['hotspot_outside_fraction'])}
- hotspot risk ratio: {fmt(sahel['hotspot_risk_ratio'])}
- threshold-like fit fraction: {pct(sahel['threshold_like_fit_fraction_inside'])} vs {pct(sahel['threshold_like_fit_fraction_outside'])}

## Result 3: The western Sahel core is the local maximum

Use the KNN6 region as an inset or sensitivity result. It shows the strongest local expression but is too small for the main claim.

## Result 4: Eurasian/Russian steppe is a secondary comparison

The western Russian steppe has elevated limitation probability but does not show the cleanest high-stress slope weakening. Use this as a supplementary comparison, not the main title.

## Result 5: Mechanism

Connect Sahel limitation to high VPD/low moisture context and trait/rooting/hydraulic results, while keeping tower validation as a limitation.
"""

(TXT / "FINAL_RESULTS_STRUCTURE.md").write_text(results)

manifest = {
    "phase": "Phase 12 final geographic ecosystem paper package",
    "main_claim": "High-stress WUE limitation is geographically concentrated in the Sahel rather than universal across grasslands.",
    "n_points": n_points,
    "global": {
        "enhancement": enh_n,
        "inconclusive": inc_n,
        "hard_satbreak": hard_satbreak_n,
    },
    "sahel": sahel.to_dict(),
    "russian_steppe_west": rus.to_dict(),
    "outputs": {
        "summary": str(TAB / "Table30_final_geographic_regions_for_main_text.csv"),
        "jackknife": str(TAB / "Table31_jackknife_region_robustness.csv"),
        "hierarchy": str(TAB / "Table32_final_claim_evidence_hierarchy.csv"),
        "decision": str(TXT / "FINAL_GEOGRAPHIC_PAPER_DECISION.md"),
        "abstract": str(TXT / "FINAL_ABSTRACT_DRAFT_SAHEL_GEOGRAPHIC_HOTSPOT.md"),
        "results_structure": str(TXT / "FINAL_RESULTS_STRUCTURE.md"),
    }
}
(OUT / "phase12_final_geographic_paper_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

print("")
print("DONE Phase 12 final geographic ecosystem-response paper package.")
print("")
print(decision)
