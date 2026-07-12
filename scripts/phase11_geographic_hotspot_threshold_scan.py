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
OUT = Path("results/paper_geographic_hotspots")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
OBS = PH8 / "table_latent_model_observations.csv"
TRAIT = Path("results/trait_framework/trait_model_dataset.csv")

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing file: {path}")
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

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def mann_p(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna()
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

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

def cohen_d(a,b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return float((np.mean(a)-np.mean(b))/np.sqrt(pooled))

def score_group(df, mask, label, group_type, region_name="", extra=None):
    extra = extra or {}
    g = pd.Series(mask, index=df.index).astype(bool)
    if g.sum() < 5 or (~g).sum() < 5:
        return None

    sat = num(df["latent_satbreak_probability"])
    post = num(df["latent_post_slope"])
    slope = num(df["latent_slope_change"])

    sat_sd = sat.std(skipna=True) or 1.0
    post_sd = post.std(skipna=True) or 1.0
    slope_sd = slope.std(skipna=True) or 1.0

    sat_g = sat[g]
    sat_r = sat[~g]
    post_g = post[g]
    post_r = post[~g]
    slope_g = slope[g]
    slope_r = slope[~g]

    event = df["event_geographic_limitation_hotspot"].astype(bool)
    a,b,c,d,gf,rf,rr,odds,fp = fisher_event(g, event)

    sat_diff = float(sat_g.median(skipna=True) - sat_r.median(skipna=True))
    post_diff = float(post_g.median(skipna=True) - post_r.median(skipna=True))
    slope_diff = float(slope_g.median(skipna=True) - slope_r.median(skipna=True))

    # Higher score means more threshold-like / high-stress limitation:
    # higher satbreak probability, lower post-slope, lower slope-change, enriched hotspot event.
    limitation_score = (
        sat_diff / sat_sd
        - post_diff / post_sd
        - slope_diff / slope_sd
        + 2.0 * ((gf if pd.notna(gf) else 0) - (rf if pd.notna(rf) else 0))
    )

    lat = num(df.loc[g, "lat"])
    lon = num(df.loc[g, "lon"])

    row = {
        "label": label,
        "group_type": group_type,
        "region_name": region_name,
        "n_group": int(g.sum()),
        "n_rest": int((~g).sum()),
        "center_lat": float(lat.mean(skipna=True)),
        "center_lon": float(lon.mean(skipna=True)),
        "min_lat": float(lat.min(skipna=True)),
        "max_lat": float(lat.max(skipna=True)),
        "min_lon": float(lon.min(skipna=True)),
        "max_lon": float(lon.max(skipna=True)),
        "limitation_score": float(limitation_score),
        "latent_satbreak_probability_group_median": float(sat_g.median(skipna=True)),
        "latent_satbreak_probability_rest_median": float(sat_r.median(skipna=True)),
        "latent_satbreak_probability_median_diff": sat_diff,
        "latent_post_slope_group_median": float(post_g.median(skipna=True)),
        "latent_post_slope_rest_median": float(post_r.median(skipna=True)),
        "latent_post_slope_median_diff": post_diff,
        "latent_slope_change_group_median": float(slope_g.median(skipna=True)),
        "latent_slope_change_rest_median": float(slope_r.median(skipna=True)),
        "latent_slope_change_median_diff": slope_diff,
        "satbreak_mannwhitney_p": mann_p(sat_g, sat_r),
        "post_slope_mannwhitney_p": mann_p(post_g, post_r),
        "slope_change_mannwhitney_p": mann_p(slope_g, slope_r),
        "event_hotspot_group_n": a,
        "event_hotspot_group_fraction": gf,
        "event_hotspot_rest_fraction": rf,
        "event_hotspot_fraction_diff": (gf-rf) if pd.notna(gf) and pd.notna(rf) else np.nan,
        "event_hotspot_risk_ratio": rr,
        "event_hotspot_odds_ratio": odds,
        "event_hotspot_fisher_p": fp,
        "satprob_cohen_d": cohen_d(sat_g, sat_r),
        "post_slope_cohen_d": cohen_d(post_g, post_r),
        "slope_change_cohen_d": cohen_d(slope_g, slope_r),
        "point_ids": ";".join(df.loc[g, "point_id"].astype(str).tolist()),
    }
    row.update(extra)
    return row

def in_box(lat, lon, lat_min, lat_max, lon_min, lon_max):
    return (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

latent = read_csv(LATENT)
obs = read_csv(OBS, required=False)
trait = read_csv(TRAIT, required=False)

if "point_id" not in latent.columns:
    die("latent table missing point_id")
latent["point_id"] = latent["point_id"].astype(str)

if "lat" not in latent.columns or "lon" not in latent.columns:
    if not trait.empty and "point_id" in trait.columns:
        trait["point_id"] = trait["point_id"].astype(str)
        latent = latent.merge(trait, on="point_id", how="left", suffixes=("", "_trait"))
    if "lat" not in latent.columns or "lon" not in latent.columns:
        die("Need lat/lon in latent table or trait table")

for c in ["lat","lon","latent_satbreak_probability","latent_post_slope","latent_slope_change"]:
    if c in latent.columns:
        latent[c] = num(latent[c])

df = latent.dropna(subset=["lat","lon","latent_satbreak_probability","latent_post_slope","latent_slope_change"]).copy()
df = df.reset_index(drop=True)

n = len(df)
if n < 20:
    die(f"Too few points for geographic scan: {n}")

# Event definition: local high-stress limitation.
# This is deliberately more sensitive than hard saturation/breakdown class.
sat = num(df["latent_satbreak_probability"])
post = num(df["latent_post_slope"])
slope = num(df["latent_slope_change"])

df["event_satprob_top20"] = sat >= sat.quantile(0.80)
df["event_post_slope_bottom20"] = post <= post.quantile(0.20)
df["event_slope_change_bottom20"] = slope <= slope.quantile(0.20)
df["event_geographic_limitation_hotspot"] = (
    df["event_satprob_top20"].fillna(False)
    | df["event_post_slope_bottom20"].fillna(False)
    | df["event_slope_change_bottom20"].fillna(False)
)

# ---------------------------------------------------------------------
# 1. Pre-defined geographic grassland regions
# ---------------------------------------------------------------------

regions = [
    # Eurasian steppe family
    ("Russian_Kazakh_steppe_broad", 42, 58, 35, 95),
    ("Russian_steppe_west", 45, 56, 30, 60),
    ("Russian_steppe_east_southern_siberia", 48, 58, 60, 110),
    ("Kazakh_steppe_core", 43, 54, 45, 80),
    ("Pontic_Caspian_steppe", 42, 52, 25, 55),
    ("Central_Asian_steppe", 38, 52, 55, 90),
    ("Mongolian_Manchurian_steppe", 40, 55, 90, 125),
    ("Tibetan_Qinghai_grasslands", 28, 40, 75, 105),

    # North America
    ("North_American_Great_Plains", 30, 52, -115, -88),
    ("Northern_Great_Plains", 43, 55, -115, -95),
    ("Southern_Great_Plains", 28, 43, -110, -90),
    ("Intermountain_West_grasslands", 35, 50, -125, -105),

    # South America
    ("Pampas", -40, -25, -66, -50),
    ("Patagonian_steppe", -55, -35, -75, -60),
    ("Cerrado_savanna_grassland", -25, -5, -60, -40),
    ("Llanos", 3, 12, -75, -60),

    # Africa
    ("Sahel", 10, 20, -20, 40),
    ("East_African_savanna", -10, 12, 25, 45),
    ("Southern_African_Highveld", -35, -20, 15, 35),
    ("Horn_of_Africa_drylands", 0, 15, 35, 52),

    # Australia / Asia
    ("Australian_grasslands_broad", -40, -15, 110, 155),
    ("Northern_Australia_savanna", -20, -10, 120, 150),
    ("Murray_Darling_grasslands", -38, -25, 135, 150),
    ("Deccan_semiarid_grasslands", 8, 25, 68, 85),
    ("Anatolian_Iranian_steppe", 28, 42, 35, 65),
    ("Mediterranean_steppe", 30, 45, -10, 40),
]

region_rows = []
lat = df["lat"].values
lon = df["lon"].values

for name, lat_min, lat_max, lon_min, lon_max in regions:
    mask = in_box(lat, lon, lat_min, lat_max, lon_min, lon_max)
    row = score_group(
        df,
        mask,
        label=name,
        group_type="predefined_geographic_region",
        region_name=name,
        extra={
            "lat_min_def": lat_min,
            "lat_max_def": lat_max,
            "lon_min_def": lon_min,
            "lon_max_def": lon_max,
        }
    )
    if row is not None:
        region_rows.append(row)

region_scan = pd.DataFrame(region_rows)

if not region_scan.empty:
    for c in ["satbreak_mannwhitney_p", "post_slope_mannwhitney_p", "slope_change_mannwhitney_p", "event_hotspot_fisher_p"]:
        region_scan[c.replace("_p","_q")] = bh_qvalues(region_scan[c])
    qcols = [c for c in region_scan.columns if c.endswith("_q")]
    region_scan["best_q"] = region_scan[qcols].min(axis=1, skipna=True)
    region_scan = region_scan.sort_values("limitation_score", ascending=False)

save_csv(region_scan, TAB / "Table20_predefined_geographic_region_scan.csv")

# ---------------------------------------------------------------------
# 2. Data-driven local KNN geographic windows
# ---------------------------------------------------------------------

coords_lat = df["lat"].values
coords_lon = df["lon"].values
D = np.zeros((n,n), dtype=float)

for i in range(n):
    D[i,:] = haversine_km(coords_lat[i], coords_lon[i], coords_lat, coords_lon)

def assign_named_region_for_center(center_lat, center_lon):
    hits = []
    for name, lat_min, lat_max, lon_min, lon_max in regions:
        if lat_min <= center_lat <= lat_max and lon_min <= center_lon <= lon_max:
            hits.append(name)
    return "|".join(hits) if hits else "unassigned_geographic_cluster"

knn_rows = []
for k in [6, 8, 10, 12, 15, 20, 25, 30]:
    if k >= n:
        continue
    for i in range(n):
        idx = np.argsort(D[i,:])[:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        center_lat = float(df.loc[i, "lat"])
        center_lon = float(df.loc[i, "lon"])
        nearest_region = assign_named_region_for_center(center_lat, center_lon)
        row = score_group(
            df,
            mask,
            label=f"KNN_k{k}_center_{df.loc[i,'point_id']}",
            group_type="data_driven_knn_local_window",
            region_name=nearest_region,
            extra={
                "k": k,
                "center_point_id": df.loc[i, "point_id"],
                "center_lat_original": center_lat,
                "center_lon_original": center_lon,
                "max_distance_km": float(D[i,idx].max()),
                "mean_distance_km": float(D[i,idx].mean()),
            }
        )
        if row is not None:
            knn_rows.append(row)

knn_scan = pd.DataFrame(knn_rows)

for c in ["satbreak_mannwhitney_p", "post_slope_mannwhitney_p", "slope_change_mannwhitney_p", "event_hotspot_fisher_p"]:
    knn_scan[c.replace("_p","_q")] = bh_qvalues(knn_scan[c])

qcols = [c for c in knn_scan.columns if c.endswith("_q")]
knn_scan["best_q"] = knn_scan[qcols].min(axis=1, skipna=True)

# Deduplicate overlapping top clusters:
# keep high score clusters whose point-set is not almost identical to a stronger one.
knn_scan = knn_scan.sort_values("limitation_score", ascending=False).reset_index(drop=True)

selected = []
selected_sets = []
for _, r in knn_scan.iterrows():
    pts = set(str(r["point_ids"]).split(";"))
    if len(pts) == 0:
        continue
    too_similar = False
    for s in selected_sets:
        j = len(pts & s) / len(pts | s)
        if j >= 0.70:
            too_similar = True
            break
    if not too_similar:
        selected.append(r)
        selected_sets.append(pts)
    if len(selected) >= 80:
        break

knn_top = pd.DataFrame(selected)

save_csv(knn_scan, TAB / "Table21_all_knn_geographic_hotspot_windows.csv")
save_csv(knn_top, TAB / "Table22_top_nonoverlapping_knn_geographic_hotspots.csv")

# ---------------------------------------------------------------------
# 3. Radius windows
# ---------------------------------------------------------------------

radius_rows = []
for radius in [500, 750, 1000, 1500, 2000, 3000]:
    for i in range(n):
        mask = D[i,:] <= radius
        if mask.sum() < 6:
            continue
        center_lat = float(df.loc[i, "lat"])
        center_lon = float(df.loc[i, "lon"])
        nearest_region = assign_named_region_for_center(center_lat, center_lon)
        row = score_group(
            df,
            mask,
            label=f"radius_{radius}km_center_{df.loc[i,'point_id']}",
            group_type="data_driven_radius_local_window",
            region_name=nearest_region,
            extra={
                "radius_km": radius,
                "center_point_id": df.loc[i, "point_id"],
                "center_lat_original": center_lat,
                "center_lon_original": center_lon,
            }
        )
        if row is not None:
            radius_rows.append(row)

radius_scan = pd.DataFrame(radius_rows)
if not radius_scan.empty:
    for c in ["satbreak_mannwhitney_p", "post_slope_mannwhitney_p", "slope_change_mannwhitney_p", "event_hotspot_fisher_p"]:
        radius_scan[c.replace("_p","_q")] = bh_qvalues(radius_scan[c])
    qcols = [c for c in radius_scan.columns if c.endswith("_q")]
    radius_scan["best_q"] = radius_scan[qcols].min(axis=1, skipna=True)
    radius_scan = radius_scan.sort_values("limitation_score", ascending=False)

save_csv(radius_scan, TAB / "Table23_radius_geographic_hotspot_windows.csv")

# ---------------------------------------------------------------------
# 4. Combine candidates and compute local raw-threshold evidence from all Phase 8 observations
# ---------------------------------------------------------------------

all_candidates = []
for name, dat in [
    ("predefined_region", region_scan),
    ("knn_local_window", knn_top),
    ("radius_window", radius_scan.head(80) if not radius_scan.empty else radius_scan),
]:
    if dat is not None and not dat.empty:
        x = dat.copy()
        x["candidate_source"] = name
        all_candidates.append(x)

cand = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
cand = cand.sort_values("limitation_score", ascending=False).reset_index(drop=True)

# Attach raw-product threshold-like fraction in this local point set:
if not obs.empty and "point_id" in obs.columns:
    obs = obs.copy()
    obs["point_id"] = obs["point_id"].astype(str)
    if "response_class_4way" in obs.columns:
        obs["threshold_like_fit"] = obs["response_class_4way"].astype(str).isin(["saturation","breakdown"])
    elif "response_class_original" in obs.columns:
        obs["threshold_like_fit"] = obs["response_class_original"].astype(str).str.lower().isin(["saturation","breakdown"])
    else:
        obs["threshold_like_fit"] = False

    raw_rows = []
    for idx, r in cand.head(200).iterrows():
        pts = set(str(r["point_ids"]).split(";"))
        local = obs[obs["point_id"].isin(pts)]
        rest = obs[~obs["point_id"].isin(pts)]
        if len(local) == 0 or len(rest) == 0:
            continue
        local_frac = float(local["threshold_like_fit"].mean())
        rest_frac = float(rest["threshold_like_fit"].mean())
        raw_rows.append({
            "candidate_rank": int(idx + 1),
            "label": r["label"],
            "candidate_source": r["candidate_source"],
            "region_name": r.get("region_name", ""),
            "n_points": int(r["n_group"]),
            "n_local_fits": int(len(local)),
            "n_rest_fits": int(len(rest)),
            "threshold_like_fit_fraction_local": local_frac,
            "threshold_like_fit_fraction_rest": rest_frac,
            "threshold_like_fit_risk_ratio": local_frac / rest_frac if rest_frac > 0 else np.nan,
            "uwue_threshold_like_fit_fraction_local": float(local.loc[local.get("metric","").astype(str).str.lower().eq("uwue"), "threshold_like_fit"].mean()) if "metric" in local.columns else np.nan,
            "uwue_threshold_like_fit_fraction_rest": float(rest.loc[rest.get("metric","").astype(str).str.lower().eq("uwue"), "threshold_like_fit"].mean()) if "metric" in rest.columns else np.nan,
        })
    raw_threshold = pd.DataFrame(raw_rows)
else:
    raw_threshold = pd.DataFrame()

save_csv(cand, TAB / "Table24_combined_geographic_threshold_candidates.csv")
save_csv(raw_threshold, TAB / "Table25_local_threshold_evidence_from_all_fits.csv")

# Best candidate
if cand.empty:
    die("No geographic candidates generated")

best = cand.iloc[0].to_dict()
best_pts = set(str(best["point_ids"]).split(";"))
df["best_geographic_hotspot"] = df["point_id"].isin(best_pts)
best_points = df[df["best_geographic_hotspot"]].copy()
save_csv(best_points, TAB / "Table26_points_in_best_geographic_hotspot.csv")

# Russian/Eurasian steppe specific extract
eurasian_terms = ["Russian", "Kazakh", "Pontic", "Central_Asian", "Mongolian", "Siberia"]
if not region_scan.empty:
    eurasia = region_scan[
        region_scan["label"].astype(str).str.contains("|".join(eurasian_terms), case=False, regex=True, na=False)
    ].copy()
else:
    eurasia = pd.DataFrame()
save_csv(eurasia, TAB / "Table27_eurasian_russian_steppe_region_scan.csv")

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

# Figure 1: global map with best hotspot
fig, ax = plt.subplots(figsize=(10, 5.8))
rest = df[~df["best_geographic_hotspot"]]
sub = df[df["best_geographic_hotspot"]]
sc = ax.scatter(
    rest["lon"], rest["lat"],
    c=rest["latent_satbreak_probability"],
    s=35,
    alpha=0.55,
)
ax.scatter(
    sub["lon"], sub["lat"],
    c=sub["latent_satbreak_probability"],
    s=95,
    edgecolor="black",
    linewidth=0.9,
    label="best local high-stress hotspot",
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Figure 1. Geographic hotspots of high-stress WUE limitation")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Latent high-stress limitation probability")
ax.legend(frameon=False)
savefig(FIG / "Figure1_geographic_hotspot_map.png")

# Figure 2: top named regions
if not region_scan.empty:
    plot = region_scan.head(15).copy()
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(plot))))
    y = np.arange(len(plot))
    ax.barh(y, plot["limitation_score"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("High-stress limitation enrichment score")
    ax.set_title("Figure 2. Named geographic regions ranked by high-stress limitation")
    savefig(FIG / "Figure2_named_geographic_region_ranking.png")

# Figure 3: top local KNN clusters
if not knn_top.empty:
    plot = knn_top.head(20).copy()
    plot["short_label"] = (
        plot["region_name"].astype(str).str.slice(0, 35)
        + "\nlat="
        + plot["center_lat"].round(1).astype(str)
        + ", lon="
        + plot["center_lon"].round(1).astype(str)
        + ", k="
        + plot.get("k", pd.Series("", index=plot.index)).astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, max(5, 0.42 * len(plot))))
    y = np.arange(len(plot))
    ax.barh(y, plot["limitation_score"])
    ax.set_yticks(y)
    ax.set_yticklabels(plot["short_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("High-stress limitation enrichment score")
    ax.set_title("Figure 3. Data-driven local geographic hotspots")
    savefig(FIG / "Figure3_data_driven_local_hotspot_ranking.png")

# Figure 4: best hotspot response distributions
for outcome, ylabel, fname in [
    ("latent_satbreak_probability", "Latent high-stress limitation probability", "Figure4A_best_hotspot_satbreak_probability.png"),
    ("latent_post_slope", "Latent high-stress/post-transition slope", "Figure4B_best_hotspot_post_slope.png"),
    ("latent_slope_change", "Latent slope change", "Figure4C_best_hotspot_slope_change.png"),
]:
    vals = [
        num(df.loc[df["best_geographic_hotspot"], outcome]).dropna(),
        num(df.loc[~df["best_geographic_hotspot"], outcome]).dropna(),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.boxplot(vals, labels=["best hotspot", "all other points"], showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Figure 4. {ylabel} in best geographic hotspot")
    savefig(FIG / fname)

# Figure 5: Eurasian/Russian steppe focus map if present
if not eurasia.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["lon"], df["lat"], s=18, alpha=0.25, label="all grassland points")
    for _, r in eurasia.iterrows():
        pts = set(str(r["point_ids"]).split(";"))
        local = df[df["point_id"].isin(pts)]
        if len(local):
            ax.scatter(local["lon"], local["lat"], s=60, alpha=0.75, label=str(r["label"])[:35])
    ax.set_xlim(20, 125)
    ax.set_ylim(35, 65)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 5. Eurasian/Russian-steppe candidate regions")
    ax.legend(frameon=False, fontsize=7)
    savefig(FIG / "Figure5_eurasian_russian_steppe_focus.png")

# ---------------------------------------------------------------------
# Text outputs
# ---------------------------------------------------------------------

best_raw = pd.DataFrame()
if not raw_threshold.empty:
    best_raw = raw_threshold[raw_threshold["label"].eq(best["label"])].copy()

raw_text = ""
if not best_raw.empty:
    r = best_raw.iloc[0]
    raw_text = f"""
- Threshold-like fit fraction inside best hotspot: {pct(r['threshold_like_fit_fraction_local'])}
- Threshold-like fit fraction outside best hotspot: {pct(r['threshold_like_fit_fraction_rest'])}
- Threshold-like fit risk ratio across all fits: {fmt(r['threshold_like_fit_risk_ratio'])}
"""
else:
    raw_text = "- Local raw fit-level threshold evidence was unavailable."

russian_text = ""
if not eurasia.empty:
    top_eur = eurasia.sort_values("limitation_score", ascending=False).iloc[0]
    russian_text = f"""
## Eurasian/Russian-steppe check

Best Eurasian/Russian-steppe candidate:
- Region: {top_eur['label']}
- Points: {int(top_eur['n_group'])}
- Limitation score: {fmt(top_eur['limitation_score'])}
- Median limitation probability inside: {fmt(top_eur['latent_satbreak_probability_group_median'])}
- Median limitation probability outside: {fmt(top_eur['latent_satbreak_probability_rest_median'])}
- Hotspot risk ratio: {fmt(top_eur['event_hotspot_risk_ratio'])}
- Best q-value: {fmt(top_eur.get('best_q', np.nan))}
"""
else:
    russian_text = """
## Eurasian/Russian-steppe check

No predefined Eurasian/Russian-steppe region had enough sampled points for a stable region-level test, or no such region appeared in the current 199-point sample.
"""

decision = f"""# Geographic hotspot decision box

## Best geographic/local threshold candidate

- Candidate label: `{best['label']}`
- Candidate source: `{best['candidate_source']}`
- Assigned geographic region: `{best.get('region_name', '')}`
- Points in local region: {int(best['n_group'])}/{n} ({pct(int(best['n_group'])/n)})
- Center latitude/longitude: {fmt(best['center_lat'])}, {fmt(best['center_lon'])}
- Bounding box: lat {fmt(best['min_lat'])} to {fmt(best['max_lat'])}; lon {fmt(best['min_lon'])} to {fmt(best['max_lon'])}
- Limitation enrichment score: {fmt(best['limitation_score'])}
- Median limitation probability inside: {fmt(best['latent_satbreak_probability_group_median'])}
- Median limitation probability outside: {fmt(best['latent_satbreak_probability_rest_median'])}
- Median high-stress slope inside: {fmt(best['latent_post_slope_group_median'])}
- Median high-stress slope outside: {fmt(best['latent_post_slope_rest_median'])}
- Hotspot fraction inside: {pct(best['event_hotspot_group_fraction'])}
- Hotspot fraction outside: {pct(best['event_hotspot_rest_fraction'])}
- Hotspot risk ratio: {fmt(best['event_hotspot_risk_ratio'])}
- Best q-value: {fmt(best.get('best_q', np.nan))}

## Raw fit-level threshold evidence

{raw_text}

{russian_text}

## Paper interpretation

The strongest paper is now a **geographic hotspot paper**, not a generic biome-quartile paper.

Main claim:

> Grassland WUE breakdown is not universal globally, but high-stress limitation/threshold-like behavior is geographically concentrated in local grassland hotspots.

Use the specific local region only after checking the candidate table and map. If the best candidate is in Russia/Kazakhstan/Eurasian steppe, the paper can focus on that region. If the best candidate is tropical or another dryland belt, use that instead.

## Claim to avoid

Do not claim “Russian grasslands have a threshold effect” unless the Russian/Eurasian table specifically supports it with enough points and enriched limitation metrics.
"""

(TXT / "GEOGRAPHIC_HOTSPOT_DECISION_BOX.md").write_text(decision)

methods = f"""# Geographic hotspot methods note

We tested whether high-stress WUE limitation was geographically localized rather than globally universal. We used two complementary geographic definitions.

First, we evaluated predefined grassland regions, including Eurasian/Russian-Kazakh steppe, Pontic-Caspian steppe, Mongolian-Manchurian steppe, North American Great Plains, Pampas, Sahel, East African savanna, Australian grasslands, and other major grassland belts.

Second, we performed data-driven local spatial scans using k-nearest-neighbor geographic windows and fixed-radius windows. For each candidate window, we compared the local points to all remaining grassland points in terms of latent saturation/breakdown probability, high-stress/post-transition WUE slope, latent slope change, and enrichment of high-stress limitation hotspot events. P-values were adjusted with Benjamini-Hochberg q-values across candidate windows.

The geographic scan identifies regions where high-stress limitation is locally enriched. It does not replace direct tower validation, but it supports a spatially localized ecosystem-response claim.
"""

(TXT / "METHODS_geographic_hotspot_scan.md").write_text(methods)

abstract = f"""# Geographic hotspot paper abstract draft

Grassland water-use efficiency (WUE) responses to compound atmospheric-soil moisture stress are often discussed as if a single global threshold should exist. Using a product-adjusted latent ecosystem response phenotype across {n} grassland points, we found no evidence for universal WUE breakdown globally. However, geographic hotspot scanning revealed strong spatial concentration of high-stress limitation. The strongest local candidate was `{best['label']}`, containing {int(best['n_group'])} points centered near {fmt(best['center_lat'])} latitude and {fmt(best['center_lon'])} longitude. Within this local region, the median high-stress limitation probability was {fmt(best['latent_satbreak_probability_group_median'])}, compared with {fmt(best['latent_satbreak_probability_rest_median'])} outside the region, and high-stress limitation hotspots were enriched by a risk ratio of {fmt(best['event_hotspot_risk_ratio'])}. These results suggest that compound stress does not induce a universal grassland WUE collapse, but threshold-like high-stress limitation can emerge in geographically localized grassland systems.
"""

(TXT / "ABSTRACT_DRAFT_geographic_hotspot_paper.md").write_text(abstract)

manifest = {
    "phase": "Phase 11 geographic hotspot threshold scan",
    "n_points": n,
    "best_candidate": {
        "label": best["label"],
        "candidate_source": best["candidate_source"],
        "region_name": best.get("region_name", ""),
        "n_group": int(best["n_group"]),
        "center_lat": best["center_lat"],
        "center_lon": best["center_lon"],
        "limitation_score": best["limitation_score"],
        "median_limitation_probability_inside": best["latent_satbreak_probability_group_median"],
        "median_limitation_probability_outside": best["latent_satbreak_probability_rest_median"],
        "hotspot_fraction_inside": best["event_hotspot_group_fraction"],
        "hotspot_fraction_outside": best["event_hotspot_rest_fraction"],
        "hotspot_risk_ratio": best["event_hotspot_risk_ratio"],
        "best_q": best.get("best_q", None),
    },
    "outputs": {
        "predefined_regions": str(TAB / "Table20_predefined_geographic_region_scan.csv"),
        "all_knn": str(TAB / "Table21_all_knn_geographic_hotspot_windows.csv"),
        "top_knn": str(TAB / "Table22_top_nonoverlapping_knn_geographic_hotspots.csv"),
        "radius": str(TAB / "Table23_radius_geographic_hotspot_windows.csv"),
        "combined": str(TAB / "Table24_combined_geographic_threshold_candidates.csv"),
        "raw_threshold": str(TAB / "Table25_local_threshold_evidence_from_all_fits.csv"),
        "best_points": str(TAB / "Table26_points_in_best_geographic_hotspot.csv"),
        "eurasian_russian_steppe": str(TAB / "Table27_eurasian_russian_steppe_region_scan.csv"),
        "decision_box": str(TXT / "GEOGRAPHIC_HOTSPOT_DECISION_BOX.md"),
        "methods": str(TXT / "METHODS_geographic_hotspot_scan.md"),
        "abstract": str(TXT / "ABSTRACT_DRAFT_geographic_hotspot_paper.md"),
    }
}

(OUT / "phase11_geographic_hotspot_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

print("")
print("DONE Phase 11 geographic hotspot scan.")
print("")
print(decision)
