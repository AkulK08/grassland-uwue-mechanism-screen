#!/usr/bin/env python
from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

PH8 = Path("results/trait_framework/phase8")
PH7 = Path("results/trait_framework/phase7")
OUT = Path("results/paper_biome_subset_response")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
TRAIT_DATASET = Path("results/trait_framework/trait_model_dataset.csv")
PH8_OBS = PH8 / "table_latent_model_observations.csv"
PH9_CLAIMS = Path("results/paper_ecosystem_response/tables/Table5_main_claims_for_paper.csv")

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing required file: {path}")
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

def bh_qvalues(pvals):
    p = pd.to_numeric(pd.Series(pvals), errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p[ok].values
    if len(vals) == 0:
        return q
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    qvals = ranked * m / (np.arange(m) + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0, 1)
    out = np.empty_like(qvals)
    out[order] = qvals
    q.loc[p[ok].index] = out
    return q

def cohen_d(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))

def fisher_test(group_flag, event_flag):
    g = pd.Series(group_flag).astype(bool)
    e = pd.Series(event_flag).astype(bool)
    a = int((g & e).sum())
    b = int((g & ~e).sum())
    c = int((~g & e).sum())
    d = int((~g & ~e).sum())
    try:
        odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    except Exception:
        odds, p = np.nan, np.nan
    return a, b, c, d, odds, p

def mannwhitney(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def infer_lat_lon(df):
    lat = None
    lon = None
    for c in ["lat", "latitude", "LAT", "Latitude"]:
        if c in df.columns:
            lat = c
            break
    for c in ["lon", "longitude", "LON", "Longitude"]:
        if c in df.columns:
            lon = c
            break
    return lat, lon

latent = read_csv(LATENT)

needed = [
    "point_id",
    "latent_slope_change",
    "latent_post_slope",
    "latent_satbreak_probability",
    "latent_response_class",
]
missing = [c for c in needed if c not in latent.columns]
if missing:
    die(f"Missing latent columns: {missing}")

latent["point_id"] = latent["point_id"].astype(str)
for c in ["latent_slope_change", "latent_post_slope", "latent_satbreak_probability"]:
    latent[c] = num(latent[c])

trait = read_csv(TRAIT_DATASET, required=False)
if not trait.empty and "point_id" in trait.columns:
    trait["point_id"] = trait["point_id"].astype(str)
    trait = trait.drop_duplicates("point_id")
    df = latent.merge(trait, on="point_id", how="left", suffixes=("", "_trait"))
else:
    df = latent.copy()

df = df.loc[:, ~df.columns.duplicated()].copy()

# Normalize lat/lon if needed
lat_col, lon_col = infer_lat_lon(df)
if lat_col and lat_col != "lat":
    df["lat"] = num(df[lat_col])
if lon_col and lon_col != "lon":
    df["lon"] = num(df[lon_col])
if "lat" in df.columns:
    df["lat"] = num(df["lat"])
if "lon" in df.columns:
    df["lon"] = num(df["lon"])

# ---------------------------------------------------------------------
# Create biome/ecoclimatic grouping columns
# ---------------------------------------------------------------------

group_info = []

# Existing categorical biome/landcover/ecoregion columns
exclude_words = ["point_id", "file", "path", "url", "geometry", "date"]
biome_keywords = [
    "biome",
    "ecoregion",
    "ecozone",
    "realm",
    "igbp",
    "landcover",
    "land_cover",
    "land_class",
    "vegetation",
    "veg",
    "pft",
    "koppen",
    "climate_class",
    "plant_functional",
]

for c in df.columns:
    lc = c.lower()
    if any(w in lc for w in exclude_words):
        continue
    if not any(k in lc for k in biome_keywords):
        continue
    s = df[c]
    nunique = s.dropna().astype(str).nunique()
    nfinite = s.notna().sum()
    if 2 <= nunique <= 30 and nfinite >= 20:
        outcol = f"group_existing_{c}"
        df[outcol] = s.astype(str).replace({"nan": np.nan})
        group_info.append({
            "group_col": outcol,
            "source": c,
            "group_type": "existing_biome_landcover_climate_label",
            "n_levels": int(nunique),
            "n_nonmissing": int(nfinite),
        })

# Numeric ecoclimatic bins
numeric_candidates = [
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
    "rooting_depth",
    "p50",
]

def add_quantile_group(col, q=4):
    if col not in df.columns:
        return
    x = num(df[col])
    if x.notna().sum() < 30 or x.nunique(dropna=True) < q:
        return
    try:
        labels = [f"{col}_Q{i+1}" for i in range(q)]
        out = pd.qcut(x, q=q, labels=labels, duplicates="drop")
        outcol = f"group_{col}_q{q}"
        df[outcol] = out.astype(str).replace({"nan": np.nan})
        group_info.append({
            "group_col": outcol,
            "source": col,
            "group_type": f"empirical_{q}_quantile_ecoclimatic_stratum",
            "n_levels": int(pd.Series(out).dropna().nunique()),
            "n_nonmissing": int(x.notna().sum()),
        })
    except Exception:
        pass

for c in numeric_candidates:
    add_quantile_group(c, q=3)
    add_quantile_group(c, q=4)

# Latitudinal biome proxy
if "lat" in df.columns:
    lat = num(df["lat"])
    bins = [-90, -45, -23.5, 23.5, 45, 90]
    labels = ["south_temperate_highlat", "south_subtropical", "tropical", "north_subtropical", "north_temperate_highlat"]
    df["group_latitudinal_zone"] = pd.cut(lat, bins=bins, labels=labels, include_lowest=True).astype(str).replace({"nan": np.nan})
    group_info.append({
        "group_col": "group_latitudinal_zone",
        "source": "lat",
        "group_type": "derived_latitudinal_zone",
        "n_levels": int(df["group_latitudinal_zone"].dropna().nunique()),
        "n_nonmissing": int(df["group_latitudinal_zone"].notna().sum()),
    })

# Aridity-latitude proxy for biome-like subsets
if "lat" in df.columns and "aridity" in df.columns:
    lat = num(df["lat"])
    arid = num(df["aridity"])
    try:
        arid_bin = pd.qcut(arid, q=3, labels=["aridity_low", "aridity_mid", "aridity_high"], duplicates="drop").astype(str)
        lat_zone = pd.cut(
            lat,
            bins=[-90, -23.5, 23.5, 90],
            labels=["southern_extratropical", "tropical", "northern_extratropical"],
            include_lowest=True
        ).astype(str)
        df["group_biome_proxy_aridity_latitude"] = (lat_zone + "__" + arid_bin).replace({"nan__nan": np.nan})
        group_info.append({
            "group_col": "group_biome_proxy_aridity_latitude",
            "source": "lat + aridity",
            "group_type": "derived_biome_proxy_aridity_by_latitude",
            "n_levels": int(df["group_biome_proxy_aridity_latitude"].dropna().nunique()),
            "n_nonmissing": int(df["group_biome_proxy_aridity_latitude"].notna().sum()),
        })
    except Exception:
        pass

group_info = pd.DataFrame(group_info).drop_duplicates("group_col")
save_csv(group_info, TAB / "TableS10_grouping_variables_used.csv")

if group_info.empty:
    die("No biome/ecoclimatic grouping variables found or derived.")

# ---------------------------------------------------------------------
# Define subset signals
# ---------------------------------------------------------------------

satprob = num(df["latent_satbreak_probability"])
post = num(df["latent_post_slope"])
slope = num(df["latent_slope_change"])

df["event_high_limitation_top10"] = satprob >= satprob.quantile(0.90)
df["event_high_limitation_top20"] = satprob >= satprob.quantile(0.80)
df["event_high_limitation_top25"] = satprob >= satprob.quantile(0.75)

df["event_weak_post_slope_bottom10"] = post <= post.quantile(0.10)
df["event_weak_post_slope_bottom20"] = post <= post.quantile(0.20)
df["event_weak_post_slope_bottom25"] = post <= post.quantile(0.25)

df["event_weak_slope_change_bottom20"] = slope <= slope.quantile(0.20)
df["event_weak_slope_change_bottom25"] = slope <= slope.quantile(0.25)

# combined high-stress limitation hotspot:
# high limitation probability OR weak high-stress slope
df["event_limitation_hotspot_top20_or_weak20"] = (
    df["event_high_limitation_top20"].fillna(False) |
    df["event_weak_post_slope_bottom20"].fillna(False)
)

event_cols = [
    "event_high_limitation_top10",
    "event_high_limitation_top20",
    "event_high_limitation_top25",
    "event_weak_post_slope_bottom10",
    "event_weak_post_slope_bottom20",
    "event_weak_post_slope_bottom25",
    "event_weak_slope_change_bottom20",
    "event_weak_slope_change_bottom25",
    "event_limitation_hotspot_top20_or_weak20",
]

# ---------------------------------------------------------------------
# Scan groups
# ---------------------------------------------------------------------

rows = []
min_n = 8

for group_col in group_info["group_col"]:
    if group_col not in df.columns:
        continue

    levels = sorted([x for x in df[group_col].dropna().astype(str).unique() if x.lower() != "nan"])

    for level in levels:
        g = df[group_col].astype(str).eq(level)
        n_group = int(g.sum())
        n_rest = int((~g).sum())

        if n_group < min_n or n_rest < min_n:
            continue

        row = {
            "group_col": group_col,
            "level": level,
            "n_group": n_group,
            "n_rest": n_rest,
        }

        # continuous outcomes
        for outcome in ["latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]:
            xg = num(df.loc[g, outcome])
            xr = num(df.loc[~g, outcome])
            row[f"{outcome}_group_median"] = float(xg.median(skipna=True))
            row[f"{outcome}_rest_median"] = float(xr.median(skipna=True))
            row[f"{outcome}_median_diff"] = row[f"{outcome}_group_median"] - row[f"{outcome}_rest_median"]
            row[f"{outcome}_group_mean"] = float(xg.mean(skipna=True))
            row[f"{outcome}_rest_mean"] = float(xr.mean(skipna=True))
            row[f"{outcome}_cohen_d"] = cohen_d(xg, xr)
            row[f"{outcome}_mannwhitney_p"] = mannwhitney(xg, xr)

        # event enrichment
        for event in event_cols:
            a, b, c, d, odds, p = fisher_test(g, df[event])
            group_frac = a / (a + b) if (a + b) else np.nan
            rest_frac = c / (c + d) if (c + d) else np.nan
            rr = (group_frac / rest_frac) if rest_frac and rest_frac > 0 else np.nan
            row[f"{event}_group_n"] = a
            row[f"{event}_group_fraction"] = group_frac
            row[f"{event}_rest_fraction"] = rest_frac
            row[f"{event}_fraction_diff"] = group_frac - rest_frac
            row[f"{event}_risk_ratio"] = rr
            row[f"{event}_odds_ratio"] = odds
            row[f"{event}_fisher_p"] = p

        rows.append(row)

scan = pd.DataFrame(rows)

if scan.empty:
    die("No valid group scans were produced. Try lowering min_n or check metadata.")

# q-values for main tests
p_cols = [c for c in scan.columns if c.endswith("_p")]
for c in p_cols:
    scan[c.replace("_p", "_q")] = bh_qvalues(scan[c])

# limitation score:
# higher satbreak probability, lower post slope, lower slope change, and hotspot enrichment
global_sat_sd = num(df["latent_satbreak_probability"]).std(skipna=True)
global_post_sd = num(df["latent_post_slope"]).std(skipna=True)
global_slope_sd = num(df["latent_slope_change"]).std(skipna=True)

if not global_sat_sd or not np.isfinite(global_sat_sd):
    global_sat_sd = 1.0
if not global_post_sd or not np.isfinite(global_post_sd):
    global_post_sd = 1.0
if not global_slope_sd or not np.isfinite(global_slope_sd):
    global_slope_sd = 1.0

scan["limitation_score"] = (
    scan["latent_satbreak_probability_median_diff"].fillna(0) / global_sat_sd
    - scan["latent_post_slope_median_diff"].fillna(0) / global_post_sd
    - scan["latent_slope_change_median_diff"].fillna(0) / global_slope_sd
    + 2.0 * scan["event_limitation_hotspot_top20_or_weak20_fraction_diff"].fillna(0)
)

scan["maintenance_score"] = -scan["limitation_score"]

# classify candidate strength
main_q_candidates = [
    "latent_satbreak_probability_mannwhitney_q",
    "latent_post_slope_mannwhitney_q",
    "latent_slope_change_mannwhitney_q",
    "event_limitation_hotspot_top20_or_weak20_fisher_q",
    "event_high_limitation_top20_fisher_q",
    "event_weak_post_slope_bottom20_fisher_q",
]
existing_q = [c for c in main_q_candidates if c in scan.columns]
scan["best_q"] = scan[existing_q].min(axis=1, skipna=True)

scan["candidate_type"] = "not_candidate"
scan.loc[
    (scan["n_group"] >= 10) &
    (scan["limitation_score"] > 0) &
    (
        (scan["best_q"] <= 0.10) |
        (scan["event_limitation_hotspot_top20_or_weak20_risk_ratio"] >= 1.5)
    ),
    "candidate_type"
] = "high_stress_limitation_subset"

scan.loc[
    (scan["n_group"] >= 10) &
    (scan["maintenance_score"] > 0) &
    (
        (scan["best_q"] <= 0.10) |
        (scan["event_limitation_hotspot_top20_or_weak20_risk_ratio"] <= 0.67)
    ),
    "candidate_type"
] = "maintained_response_subset"

scan = scan.sort_values(["candidate_type", "limitation_score"], ascending=[True, False])
save_csv(scan, TAB / "Table10_all_biome_ecoclimatic_subset_scan.csv")

high = scan[scan["candidate_type"].eq("high_stress_limitation_subset")].copy()
high = high.sort_values("limitation_score", ascending=False)
save_csv(high, TAB / "Table11_candidate_high_stress_limitation_biomes.csv")

maint = scan[scan["candidate_type"].eq("maintained_response_subset")].copy()
maint = maint.sort_values("maintenance_score", ascending=False)
save_csv(maint, TAB / "Table12_candidate_maintained_response_biomes.csv")

# concise top tables
top_cols = [
    "group_col",
    "level",
    "n_group",
    "limitation_score",
    "maintenance_score",
    "best_q",
    "latent_satbreak_probability_group_median",
    "latent_satbreak_probability_rest_median",
    "latent_satbreak_probability_median_diff",
    "latent_post_slope_group_median",
    "latent_post_slope_rest_median",
    "latent_post_slope_median_diff",
    "latent_slope_change_group_median",
    "latent_slope_change_rest_median",
    "latent_slope_change_median_diff",
    "event_limitation_hotspot_top20_or_weak20_group_fraction",
    "event_limitation_hotspot_top20_or_weak20_rest_fraction",
    "event_limitation_hotspot_top20_or_weak20_risk_ratio",
    "candidate_type",
]
save_csv(scan[[c for c in top_cols if c in scan.columns]].head(80), TAB / "Table13_ranked_subset_summary_for_paper.csv")

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

# Figure 1: ranked high-limitation subsets
plot_high = high.head(15).copy()
if plot_high.empty:
    plot_high = scan.sort_values("limitation_score", ascending=False).head(15).copy()
plot_high["label"] = plot_high["group_col"].str.replace("group_", "", regex=False) + " = " + plot_high["level"].astype(str)

fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(plot_high))))
y = np.arange(len(plot_high))
ax.barh(y, plot_high["limitation_score"])
ax.set_yticks(y)
ax.set_yticklabels(plot_high["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("High-stress limitation enrichment score")
ax.set_title("Figure 1. Candidate biome/ecoclimatic subsets with stronger high-stress limitation")
savefig(FIG / "Figure1_candidate_limitation_subsets.png")

# Figure 2: maintained-response subsets
plot_maint = maint.head(15).copy()
if plot_maint.empty:
    plot_maint = scan.sort_values("maintenance_score", ascending=False).head(15).copy()
plot_maint["label"] = plot_maint["group_col"].str.replace("group_", "", regex=False) + " = " + plot_maint["level"].astype(str)

fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(plot_maint))))
y = np.arange(len(plot_maint))
ax.barh(y, plot_maint["maintenance_score"])
ax.set_yticks(y)
ax.set_yticklabels(plot_maint["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Maintained-response enrichment score")
ax.set_title("Figure 2. Candidate subsets with maintained/enhanced WUE response")
savefig(FIG / "Figure2_candidate_maintained_response_subsets.png")

# Figure 3: boxplots for strongest subset
best = high.iloc[0].to_dict() if len(high) else scan.sort_values("limitation_score", ascending=False).iloc[0].to_dict()
best_group_col = best["group_col"]
best_level = best["level"]
df["best_subset_flag"] = df[best_group_col].astype(str).eq(str(best_level))
df["best_subset_label"] = np.where(df["best_subset_flag"], str(best_level), "all_other_points")

for outcome, title, fname in [
    ("latent_satbreak_probability", "High-stress limitation probability", "Figure3A_best_subset_limitation_probability.png"),
    ("latent_post_slope", "High-stress WUE response slope", "Figure3B_best_subset_post_slope.png"),
    ("latent_slope_change", "Slope-change response", "Figure3C_best_subset_slope_change.png"),
]:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vals = [
        num(df.loc[df["best_subset_label"].eq(str(best_level)), outcome]).dropna(),
        num(df.loc[df["best_subset_label"].eq("all_other_points"), outcome]).dropna(),
    ]
    ax.boxplot(vals, labels=[str(best_level), "all other points"], showfliers=False)
    ax.set_ylabel(outcome)
    ax.set_title(title + "\nStrongest candidate subset: " + str(best_level))
    savefig(FIG / fname)

# Figure 4: map strongest subset
if {"lat", "lon"}.issubset(df.columns):
    m = df.dropna(subset=["lat", "lon"]).copy()
    if len(m):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        rest = m[~m["best_subset_flag"]]
        sub = m[m["best_subset_flag"]]
        ax.scatter(rest["lon"], rest["lat"], s=25, alpha=0.35, label="all other points")
        ax.scatter(sub["lon"], sub["lat"], s=70, alpha=0.9, label=f"{best_group_col} = {best_level}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Figure 4. Spatial location of candidate high-stress subset")
        ax.legend(frameon=False)
        savefig(FIG / "Figure4_candidate_subset_map.png")

# Figure 5: top grouping variable distribution
if best_group_col in df.columns:
    d = df.dropna(subset=[best_group_col, "latent_satbreak_probability"]).copy()
    level_order = (
        d.groupby(best_group_col)["latent_satbreak_probability"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    level_order = level_order[:12]
    d = d[d[best_group_col].isin(level_order)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [num(d.loc[d[best_group_col].eq(level), "latent_satbreak_probability"]).dropna() for level in level_order]
    ax.boxplot(vals, labels=[str(x) for x in level_order], showfliers=False)
    ax.set_xticklabels([str(x) for x in level_order], rotation=35, ha="right")
    ax.set_ylabel("Latent high-stress limitation probability")
    ax.set_title("Figure 5. High-stress limitation probability across candidate subset groups")
    savefig(FIG / "Figure5_limitation_probability_by_candidate_group.png")

# ---------------------------------------------------------------------
# Decision text
# ---------------------------------------------------------------------

n_points = int(df["point_id"].nunique())
n_high_candidates = int(len(high))
n_maint_candidates = int(len(maint))

global_enh = int(df["latent_response_class"].astype(str).eq("enhancement").sum())
global_inc = int(df["latent_response_class"].astype(str).eq("inconclusive").sum())
global_satbreak_class = int(df["latent_response_class"].astype(str).isin(["saturation", "breakdown"]).sum())

best_group_n = int(best["n_group"])
best_group_frac = best_group_n / n_points
best_hotspot_frac = best.get("event_limitation_hotspot_top20_or_weak20_group_fraction", np.nan)
best_rest_hotspot_frac = best.get("event_limitation_hotspot_top20_or_weak20_rest_fraction", np.nan)
best_rr = best.get("event_limitation_hotspot_top20_or_weak20_risk_ratio", np.nan)
best_q = best.get("best_q", np.nan)

if n_high_candidates > 0:
    recommendation = "YES: a subset/biome paper is viable. Frame it as heterogeneous ecosystem response with high-stress limitation concentrated in specific ecoclimatic/biome subsets."
else:
    recommendation = "MAYBE/NO: no strong high-limitation biome subset passed the current candidate thresholds. The safer paper remains global no-universal-breakdown plus maintained-response heterogeneity."

claim_box = f"""# Biome/subset paper decision box

## Recommendation

{recommendation}

## Global background result

Across {n_points} grassland points, the latent response is not a universal breakdown pattern:
- Enhancement: {global_enh}/{n_points} ({pct(global_enh/n_points)})
- Inconclusive: {global_inc}/{n_points} ({pct(global_inc/n_points)})
- Hard saturation/breakdown classes: {global_satbreak_class}/{n_points} ({pct(global_satbreak_class/n_points)})

## Best candidate high-stress subset

- Grouping variable: `{best_group_col}`
- Subset level: `{best_level}`
- Points in subset: {best_group_n}/{n_points} ({pct(best_group_frac)})
- Limitation enrichment score: {fmt(best.get("limitation_score"))}
- Best q-value across tests: {fmt(best_q)}
- Hotspot fraction in subset: {pct(best_hotspot_frac)}
- Hotspot fraction outside subset: {pct(best_rest_hotspot_frac)}
- Hotspot risk ratio: {fmt(best_rr)}

## Number of candidates

- High-stress limitation candidate subsets: {n_high_candidates}
- Maintained-response candidate subsets: {n_maint_candidates}

## Best paper framing if candidate is real

Compound atmospheric-soil moisture stress does not produce universal WUE breakdown across grasslands. Instead, high-stress limitation is concentrated in identifiable biome/ecoclimatic subsets, while most grassland points maintain or enhance WUE sensitivity.

## Claim to avoid

Do not say all grasslands break down. Do not say the subset is a true biome mechanism unless the grouping variable is an actual biome/landcover/ecoregion label. If the subset is based on aridity/latitude quantiles, call it an ecoclimatic subset or biome proxy.
"""

(TXT / "BIOME_SUBSET_DECISION_BOX.md").write_text(claim_box)

manuscript_pivot = f"""# Manuscript pivot plan: subset/biome ecosystem response paper

## Main thesis

Grassland WUE response to compound atmospheric-soil moisture stress is heterogeneous. The global latent response does not support universal breakdown, but high-stress limitation is enriched in specific biome/ecoclimatic subsets.

## Results structure

1. Global result: no universal breakdown.
2. Subset result: identify the strongest high-stress limitation subset.
3. Spatial result: map the subset.
4. Gradient result: show that limitation probability differs across the grouping variable.
5. Mechanism result: test whether rooting/hydraulic traits explain response intensity within or across subsets.
6. Supplement: methodological robustness and latent-response uncertainty.

## Strongest current candidate

- `{best_group_col} = {best_level}`
- n = {best_group_n}
- hotspot risk ratio = {fmt(best_rr)}
- best q-value = {fmt(best_q)}

## Main wording

The analysis shows that high-stress WUE limitation is not a universal grassland response. Instead, limitation is concentrated in a subset defined by `{best_group_col} = {best_level}`, suggesting that compound-stress response depends on biome/ecoclimatic context.

## If this grouping variable is not an official biome

Use "ecoclimatic subset" or "biome proxy" rather than "biome."
"""

(TXT / "MANUSCRIPT_PIVOT_PLAN_BIOME_SUBSET.md").write_text(manuscript_pivot)

manifest = {
    "phase": "Phase 10 biome/ecoclimatic subset ecosystem-response scan",
    "n_points": n_points,
    "n_high_limitation_candidates": n_high_candidates,
    "n_maintained_response_candidates": n_maint_candidates,
    "best_candidate": {
        "group_col": best_group_col,
        "level": str(best_level),
        "n_group": best_group_n,
        "limitation_score": best.get("limitation_score"),
        "best_q": best_q,
        "hotspot_fraction_group": best_hotspot_frac,
        "hotspot_fraction_rest": best_rest_hotspot_frac,
        "hotspot_risk_ratio": best_rr,
    },
    "recommendation": recommendation,
    "outputs": {
        "all_scan": str(TAB / "Table10_all_biome_ecoclimatic_subset_scan.csv"),
        "high_limitation_candidates": str(TAB / "Table11_candidate_high_stress_limitation_biomes.csv"),
        "maintained_candidates": str(TAB / "Table12_candidate_maintained_response_biomes.csv"),
        "ranked_summary": str(TAB / "Table13_ranked_subset_summary_for_paper.csv"),
        "decision_box": str(TXT / "BIOME_SUBSET_DECISION_BOX.md"),
        "pivot_plan": str(TXT / "MANUSCRIPT_PIVOT_PLAN_BIOME_SUBSET.md"),
    }
}
(OUT / "phase10_biome_subset_response_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

print("")
print("DONE Phase 10 biome/subset scan.")
print("")
print(claim_box)
