#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 16: Point-level geography annotation and thesis locking.

Purpose:
1. Annotate every grassland point with objective geography:
   country, continent, subregion, WWF/RESOLVE ecoregion, biome, realm,
   latitude band, longitude sector, named geographic region, dryland/aridity class,
   high-VPD regime, Sahel regime, Russian-steppe regime, and response phenotype.

2. Convert the current spatial-regime result into a defensible manuscript thesis:
   - Is the result really Sahelian?
   - Is it a broader high-VPD dryland regime?
   - Is it concentrated in one or two ecoregions?
   - Is the Russian steppe a real second phenotype?
   - Are product/metric sensitivities too weak for a strong ecosystem claim?

3. Produce paper-ready tables:
   Table70_point_level_geography_response_annotation.csv
   Table71_hotspot_point_geography.csv
   Table72_high_vpd_point_geography.csv
   Table73_grouped_geography_response_summary.csv
   Table74_candidate_thesis_ranking.csv
   Table75_top_candidate_permutation_tests.csv
   Table76_top_candidate_raw_fit_sensitivity.csv
   Table77_spatial_autocorrelation_tests.csv
   Table78_thesis_lock_evidence_scorecard.csv

4. Produce text:
   FINAL_THESIS_LOCK.md
   METHODS_point_geography_annotation.md
   RESULTS_point_geography_summary.md
   LITERATURE_POSITIONING_FOR_THESIS.md
"""

from pathlib import Path
import os
import sys
import json
import math
import zipfile
import shutil
import warnings
import subprocess
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Package setup
# ---------------------------------------------------------------------

def ensure_import(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        return True
    except Exception:
        print(f"Installing missing package: {pip_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            __import__(import_name)
            return True
        except Exception as e:
            print(f"WARNING: Could not install/import {pip_name}: {e}")
            return False

for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("requests", "requests"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
]:
    ensure_import(import_name, pip_name)

GEOPANDAS_OK = True
for import_name, pip_name in [
    ("geopandas", "geopandas"),
    ("shapely", "shapely"),
    ("pyproj", "pyproj"),
    ("pyogrio", "pyogrio"),
]:
    ok = ensure_import(import_name, pip_name)
    GEOPANDAS_OK = GEOPANDAS_OK and ok

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception as e:
    GEOPANDAS_OK = False
    print("WARNING: geopandas stack unavailable. Will run non-vector fallback only.")
    print(e)

try:
    import statsmodels.api as sm
    STATSMODELS_OK = True
except Exception:
    STATSMODELS_OK = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(".")
PH8 = Path("results/trait_framework/phase8")
PH14 = Path("results/paper_spatial_regime_validation")
PH15 = Path("results/paper_final_audit_spatial_regime")

OUT = Path("results/paper_point_geography_thesis_lock")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"
GEO = Path("data/external/geography")

for p in [OUT, TAB, FIG, TXT, GEO]:
    p.mkdir(parents=True, exist_ok=True)

LATENT = PH8 / "table_latent_response_by_point.csv"
OBS = PH8 / "table_latent_model_observations.csv"
TRAIT = Path("results/trait_framework/trait_model_dataset.csv")

PH15_SCORECARD = PH15 / "tables/Table64_final_evidence_scorecard.csv"
PH15_EVENT = PH15 / "tables/Table60_regime_event_threshold_sensitivity.csv"
PH15_CONT = PH15 / "tables/Table61_continuous_outcome_effects_by_regime.csv"
PH15_RAW = PH15 / "tables/Table62_raw_threshold_sensitivity_by_metric_product_stress.csv"
PH15_MODELS = PH15 / "tables/Table63_adjusted_spatial_hydroclimatic_models.csv"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")

def read_csv(path, required=True):
    if not path.exists():
        if required:
            die(f"Missing required file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df.loc[:, ~df.columns.duplicated()].copy()

def num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def str_missing(series):
    """
    Convert any Series, including pandas Categorical, to string while safely
    replacing missing values with 'missing'. This avoids:
    TypeError: Cannot setitem on a Categorical with a new category.
    """
    s = pd.Series(series).astype("object")
    return s.where(pd.notna(s), "missing").astype(str)

def fmt(x, d=3):
    x = safe_float(x)
    if pd.isna(x):
        return "NA"
    return f"{x:.{d}f}"

def pct(x):
    x = safe_float(x)
    if pd.isna(x):
        return "NA"
    return f"{100*x:.1f}%"

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def save_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"WROTE {path}")

def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()
    print(f"WROTE {path}")

def download_file(urls, out_path, timeout=120):
    import requests
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"FOUND existing: {out_path}")
        return True

    for url in urls:
        try:
            print(f"Downloading {url}")
            r = requests.get(url, timeout=timeout, stream=True)
            if r.status_code != 200:
                print(f"  status {r.status_code}; trying next")
                continue
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(out_path)
            print(f"DOWNLOADED {out_path} ({out_path.stat().st_size:,} bytes)")
            return True
        except Exception as e:
            print(f"  failed: {e}")

    print(f"WARNING: Could not download any URL for {out_path}")
    return False

def select_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None

def bh_qvalues(pvals):
    p = num(pd.Series(pvals))
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

def fisher_event(local_event, rest_event):
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
        odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    except Exception:
        odds, p = np.nan, np.nan

    return a, b, c, d, gf, rf, rr, odds, p

def mann_p(a, b):
    a = num(pd.Series(a)).dropna()
    b = num(pd.Series(b)).dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

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


# ---------------------------------------------------------------------
# Load core data
# ---------------------------------------------------------------------

if not LATENT.exists():
    die(f"Missing latent response table: {LATENT}")

latent = read_csv(LATENT)
obs = read_csv(OBS, required=False)
trait = read_csv(TRAIT, required=False)

latent["point_id"] = latent["point_id"].astype(str)

if not trait.empty and "point_id" in trait.columns:
    trait["point_id"] = trait["point_id"].astype(str)

    keep = [
        "point_id",
        "lat", "lon",
        "mean_vpd", "aridity", "mean_soil_moisture",
        "mean_annual_temperature", "mean_temperature",
        "mean_annual_precipitation", "mean_precipitation",
        "mean_lai", "growing_season_mean_lai",
        "soil_sand", "soil_clay", "soil_silt",
        "rooting_depth", "p50", "isohydricity",
    ]
    keep = [c for c in keep if c in trait.columns]
    trait_small = trait[keep].drop_duplicates("point_id")

    add_cols = [c for c in trait_small.columns if c == "point_id" or c not in latent.columns]
    latent = latent.merge(trait_small[add_cols], on="point_id", how="left")

for c in latent.columns:
    if c not in ["point_id", "latent_response_class"]:
        try:
            latent[c] = num(latent[c])
        except Exception:
            pass

for c in ["lat", "lon", "latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]:
    if c not in latent.columns:
        die(f"Missing required latent column: {c}")

points = latent.dropna(subset=["lat", "lon", "latent_satbreak_probability", "latent_post_slope", "latent_slope_change"]).copy()
points = points.reset_index(drop=True)

print(f"Loaded point table: {points.shape}")
print(f"Unique points: {points['point_id'].nunique()}")


# ---------------------------------------------------------------------
# Define response events and point-level response score
# ---------------------------------------------------------------------

sat = num(points["latent_satbreak_probability"])
post = num(points["latent_post_slope"])
slope = num(points["latent_slope_change"])

points["event_satprob_top20"] = sat >= sat.quantile(0.80)
points["event_post_slope_bottom20"] = post <= post.quantile(0.20)
points["event_slope_change_bottom20"] = slope <= slope.quantile(0.20)
points["event_limitation_hotspot"] = (
    points["event_satprob_top20"].fillna(False)
    | points["event_post_slope_bottom20"].fillna(False)
    | points["event_slope_change_bottom20"].fillna(False)
)

def zscore(series, invert=False):
    s = num(series)
    sd = s.std(skipna=True)
    if not sd or not np.isfinite(sd):
        z = s * np.nan
    else:
        z = (s - s.mean(skipna=True)) / sd
    if invert:
        z = -z
    return z

points["point_limitation_rank_score"] = (
    zscore(points["latent_satbreak_probability"], invert=False).fillna(0)
    + zscore(points["latent_post_slope"], invert=True).fillna(0)
    + zscore(points["latent_slope_change"], invert=True).fillna(0)
)

points["point_limitation_rank"] = points["point_limitation_rank_score"].rank(ascending=False, method="min").astype(int)


# ---------------------------------------------------------------------
# Product/metric support per point
# ---------------------------------------------------------------------

if not obs.empty:
    obs["point_id"] = obs["point_id"].astype(str)

    if "product_combo" not in obs.columns:
        if {"gpp_product", "et_product"}.issubset(obs.columns):
            obs["product_combo"] = obs["gpp_product"].astype(str) + "/" + obs["et_product"].astype(str)
        elif "combo" in obs.columns:
            obs["product_combo"] = obs["combo"].astype(str)
        else:
            obs["product_combo"] = "unknown"

    if "response_class_4way" in obs.columns:
        class_col = "response_class_4way"
    elif "response_class_strict" in obs.columns:
        class_col = "response_class_strict"
    elif "response_class_original" in obs.columns:
        class_col = "response_class_original"
    elif "latent_response_class" in obs.columns:
        class_col = "latent_response_class"
    else:
        class_candidates = [c for c in obs.columns if "class" in c.lower()]
        class_col = class_candidates[0] if class_candidates else None

    if class_col:
        obs["threshold_like_fit"] = obs[class_col].astype(str).str.lower().isin(["saturation", "breakdown"])
        obs["enhancement_like_fit"] = obs[class_col].astype(str).str.lower().str.contains("enhancement", na=False)
        obs["inconclusive_fit"] = obs[class_col].astype(str).str.lower().eq("inconclusive")
    else:
        obs["threshold_like_fit"] = False
        obs["enhancement_like_fit"] = False
        obs["inconclusive_fit"] = False

    point_support = (
        obs.groupby("point_id")
        .agg(
            n_response_fits=("threshold_like_fit", "size"),
            threshold_like_fit_fraction=("threshold_like_fit", "mean"),
            enhancement_like_fit_fraction=("enhancement_like_fit", "mean"),
            inconclusive_fit_fraction=("inconclusive_fit", "mean"),
            n_product_combos=("product_combo", "nunique"),
        )
        .reset_index()
    )

    if "metric" in obs.columns:
        metric_pivot = (
            obs.groupby(["point_id", "metric"])["threshold_like_fit"]
            .mean()
            .reset_index()
            .pivot(index="point_id", columns="metric", values="threshold_like_fit")
            .add_prefix("threshold_fraction_metric_")
            .reset_index()
        )
        point_support = point_support.merge(metric_pivot, on="point_id", how="left")

    combo_point = (
        obs.groupby(["point_id", "product_combo"])["threshold_like_fit"]
        .mean()
        .reset_index(name="threshold_fraction_combo")
    )
    combo_summary = (
        combo_point.groupby("point_id")
        .agg(
            n_product_combos_with_any_threshold=("threshold_fraction_combo", lambda x: int((x > 0).sum())),
            n_product_combos_threshold_gt_0p05=("threshold_fraction_combo", lambda x: int((x > 0.05).sum())),
            max_combo_threshold_fraction=("threshold_fraction_combo", "max"),
            median_combo_threshold_fraction=("threshold_fraction_combo", "median"),
        )
        .reset_index()
    )
    point_support = point_support.merge(combo_summary, on="point_id", how="left")

    points = points.merge(point_support, on="point_id", how="left")
else:
    for c in [
        "n_response_fits",
        "threshold_like_fit_fraction",
        "enhancement_like_fit_fraction",
        "inconclusive_fit_fraction",
        "n_product_combos",
        "n_product_combos_with_any_threshold",
        "n_product_combos_threshold_gt_0p05",
        "max_combo_threshold_fraction",
        "median_combo_threshold_fraction",
    ]:
        points[c] = np.nan

points["product_support_score"] = (
    num(points.get("threshold_like_fit_fraction", np.nan)).fillna(0)
    + 0.05 * num(points.get("n_product_combos_threshold_gt_0p05", np.nan)).fillna(0)
)

points["point_limitation_rank_score_with_product"] = (
    points["point_limitation_rank_score"] + zscore(points["product_support_score"]).fillna(0)
)
points["point_limitation_rank_with_product"] = points["point_limitation_rank_score_with_product"].rank(ascending=False, method="min").astype(int)


# ---------------------------------------------------------------------
# Geography layers
# ---------------------------------------------------------------------

countries_zip = GEO / "ne_50m_admin_0_countries.zip"
ecoregions_zip = GEO / "Ecoregions2017.zip"

country_urls = [
    "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip",
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip",
]

ecoregion_urls = [
    "https://storage.googleapis.com/teow2016/Ecoregions2017.zip",
    "https://storage.googleapis.com/teow2016/ecoregions2017.zip",
]

if GEOPANDAS_OK:
    download_file(country_urls, countries_zip)
    download_file(ecoregion_urls, ecoregions_zip)

countries_gdf = None
ecoregions_gdf = None
geo_notes = []

if GEOPANDAS_OK:
    try:
        countries_gdf = gpd.read_file(f"zip://{countries_zip}")
        countries_gdf = countries_gdf.to_crs("EPSG:4326")
        geo_notes.append(f"Loaded Natural Earth countries: {countries_zip}")
        print("Loaded countries:", countries_gdf.shape)
    except Exception as e:
        countries_gdf = None
        geo_notes.append(f"FAILED Natural Earth countries: {e}")
        print("WARNING: country layer failed:", e)

    try:
        ecoregions_gdf = gpd.read_file(f"zip://{ecoregions_zip}")
        ecoregions_gdf = ecoregions_gdf.to_crs("EPSG:4326")
        geo_notes.append(f"Loaded RESOLVE/WWF Ecoregions 2017: {ecoregions_zip}")
        print("Loaded ecoregions:", ecoregions_gdf.shape)
    except Exception as e:
        ecoregions_gdf = None
        geo_notes.append(f"FAILED RESOLVE ecoregions: {e}")
        print("WARNING: ecoregion layer failed:", e)


# ---------------------------------------------------------------------
# Spatial annotation
# ---------------------------------------------------------------------

def annotate_with_polygon_layer(points_df, layer_gdf, field_candidates_map, prefix):
    out = pd.DataFrame(index=points_df.index)

    for new_col in field_candidates_map:
        out[prefix + new_col] = np.nan

    if layer_gdf is None or layer_gdf.empty or not GEOPANDAS_OK:
        return out

    gdf = gpd.GeoDataFrame(
        points_df[["point_id", "lat", "lon"]].copy(),
        geometry=[Point(xy) for xy in zip(points_df["lon"], points_df["lat"])],
        crs="EPSG:4326",
    )

    field_map = {}
    for new_col, candidates in field_candidates_map.items():
        col = select_col(layer_gdf, candidates)
        field_map[new_col] = col

    selected_cols = [c for c in field_map.values() if c is not None]
    selected_cols = list(dict.fromkeys(selected_cols))
    if not selected_cols:
        return out

    layer_small = layer_gdf[selected_cols + ["geometry"]].copy()

    try:
        joined = gpd.sjoin(gdf, layer_small, how="left", predicate="within")
        joined = joined[~joined.index.duplicated(keep="first")]
    except Exception as e:
        print(f"WARNING: spatial join failed for {prefix}: {e}")
        joined = pd.DataFrame(index=gdf.index)

    for new_col, old_col in field_map.items():
        if old_col is not None and old_col in joined.columns:
            out[prefix + new_col] = joined.reindex(points_df.index)[old_col].values

    # Fill missing by nearest polygon centroid if possible.
    try:
        key_col = prefix + list(field_candidates_map.keys())[0]
        missing = out[key_col].isna()
        if missing.any() and hasattr(gpd, "sjoin_nearest"):
            gdf_m = gdf.loc[missing].to_crs("EPSG:3857")
            layer_m = layer_small.to_crs("EPSG:3857")
            nn = gpd.sjoin_nearest(gdf_m, layer_m, how="left", max_distance=250000)
            nn = nn[~nn.index.duplicated(keep="first")]
            for new_col, old_col in field_map.items():
                if old_col is not None and old_col in nn.columns:
                    vals = nn[old_col]
                    out.loc[vals.index, prefix + new_col] = vals
    except Exception as e:
        print(f"WARNING: nearest fill failed for {prefix}: {e}")

    return out

country_fields = {
    "country": ["ADMIN", "NAME", "NAME_LONG", "SOVEREIGNT"],
    "country_iso3": ["ISO_A3", "ADM0_A3", "GU_A3"],
    "continent": ["CONTINENT"],
    "subregion": ["SUBREGION", "REGION_UN"],
    "region_un": ["REGION_UN"],
    "sovereignt": ["SOVEREIGNT"],
}

ecoregion_fields = {
    "ecoregion": ["ECO_NAME", "ECOREGION", "ECO_NAME_1", "name"],
    "biome": ["BIOME_NAME", "BIOME", "MHT_NAME"],
    "realm": ["REALM", "REALM_NAME"],
    "eco_id": ["ECO_ID", "ECO_ID_U"],
    "biome_num": ["BIOME_NUM"],
    "nnh_name": ["NNH_NAME"],
}

country_ann = annotate_with_polygon_layer(points, countries_gdf, country_fields, "geo_")
eco_ann = annotate_with_polygon_layer(points, ecoregions_gdf, ecoregion_fields, "eco_")

points = pd.concat([points, country_ann, eco_ann], axis=1)


# ---------------------------------------------------------------------
# Derived geography labels
# ---------------------------------------------------------------------

REGIONS = {
    "Sahel": (10, 20, -20, 40),
    "Western_Sahel": (10, 17, -10, 5),
    "Central_Sahel": (10, 17, 5, 25),
    "Eastern_Sahel": (10, 17, 25, 40),
    "Sudanian_Savanna_Broad": (5, 13, -20, 40),
    "Horn_of_Africa_Drylands": (0, 15, 35, 52),
    "East_African_Savanna": (-10, 12, 25, 45),
    "Russian_steppe_west": (45, 56, 30, 60),
    "Pontic_Caspian_steppe": (42, 52, 25, 55),
    "Kazakh_steppe_core": (43, 54, 45, 80),
    "Russian_Kazakh_steppe_broad": (42, 58, 35, 95),
    "Mongolian_Manchurian_steppe": (40, 55, 90, 125),
    "North_American_Great_Plains": (30, 52, -115, -88),
    "Pampas": (-40, -25, -66, -50),
    "Cerrado": (-25, -5, -60, -40),
    "Australian_grasslands": (-40, -15, 110, 155),
    "Deccan_semiarid_grasslands": (8, 25, 68, 85),
}

def in_box(lat, lon, box):
    latmin, latmax, lonmin, lonmax = box
    return (lat >= latmin) & (lat <= latmax) & (lon >= lonmin) & (lon <= lonmax)

for name, box in REGIONS.items():
    points[f"region_{name}"] = in_box(points["lat"], points["lon"], box)

def named_regions_for_row(row):
    hits = []
    for name in REGIONS:
        if bool(row.get(f"region_{name}", False)):
            hits.append(name)
    return ";".join(hits) if hits else "unassigned"

points["named_geographic_regions"] = points.apply(named_regions_for_row, axis=1)

lat_bins = [-90, -60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60, 90]
lat_labels = [
    "far_south_polar",
    "south_highlatitude",
    "south_midlatitude",
    "south_subtropical",
    "south_tropical",
    "equatorial_south",
    "equatorial_north",
    "sahelian_10N_20N",
    "north_subtropical_20N_30N",
    "north_midlatitude_30N_45N",
    "north_steppe_45N_60N",
    "north_highlatitude",
]
points["latitude_band"] = pd.cut(points["lat"], bins=lat_bins, labels=lat_labels, include_lowest=True).astype("object").where(pd.notna(pd.cut(points["lat"], bins=lat_bins, labels=lat_labels, include_lowest=True)), "missing").astype(str)

lon_bins = [-180, -120, -60, -20, 20, 60, 100, 140, 180]
lon_labels = [
    "americas_west",
    "americas_east",
    "atlantic_west_africa",
    "africa_europe_west_asia",
    "west_central_asia",
    "east_central_asia",
    "australia_east_asia",
    "pacific",
]
points["longitude_sector"] = pd.cut(points["lon"], bins=lon_bins, labels=lon_labels, include_lowest=True).astype("object").where(pd.notna(pd.cut(points["lon"], bins=lon_bins, labels=lon_labels, include_lowest=True)), "missing").astype(str)

points["abs_lat"] = points["lat"].abs()
points["low_latitude_30deg"] = points["abs_lat"] <= 30
points["sahelian_latitude_10N_20N"] = points["lat"].between(10, 20)

if "mean_vpd" in points.columns:
    points["high_vpd_gt_2p26"] = num(points["mean_vpd"]) > 2.26
    try:
        points["mean_vpd_quartile"] = pd.qcut(num(points["mean_vpd"]), 4, labels=["vpd_Q1_low", "vpd_Q2", "vpd_Q3", "vpd_Q4_high"])
    except Exception:
        points["mean_vpd_quartile"] = "unknown"
else:
    points["high_vpd_gt_2p26"] = False
    points["mean_vpd_quartile"] = "unknown"

points["sahel_and_high_vpd"] = points["region_Sahel"] & points["high_vpd_gt_2p26"]
points["low_latitude_high_vpd"] = points["low_latitude_30deg"] & points["high_vpd_gt_2p26"]

if "aridity" in points.columns:
    ar = num(points["aridity"])

    # UNEP-style aridity-index class only if the data appear to be P/PET-ish.
    # If not, still keep quantile-based aridity classes.
    if ar.notna().sum() and ar.quantile(0.99) <= 2.0 and ar.quantile(0.01) >= 0:
        bins = [-np.inf, 0.05, 0.20, 0.50, 0.65, np.inf]
        labels = ["hyper_arid_AI_lt_0p05", "arid_AI_0p05_0p20", "semi_arid_AI_0p20_0p50", "dry_subhumid_AI_0p50_0p65", "humid_AI_gt_0p65"]
        points["dryland_class_unep_if_ai"] = pd.cut(ar, bins=bins, labels=labels, include_lowest=True).astype("object").where(pd.notna(pd.cut(ar, bins=bins, labels=labels, include_lowest=True)), "missing").astype(str)
        points["is_unep_dryland_ai_lt_0p65"] = ar < 0.65
    else:
        points["dryland_class_unep_if_ai"] = "aridity_not_in_AI_scale"
        points["is_unep_dryland_ai_lt_0p65"] = np.nan

    try:
        points["aridity_quantile"] = pd.qcut(ar, 4, labels=["aridity_Q1", "aridity_Q2", "aridity_Q3", "aridity_Q4"])
    except Exception:
        points["aridity_quantile"] = "unknown"
else:
    points["dryland_class_unep_if_ai"] = "missing_aridity"
    points["is_unep_dryland_ai_lt_0p65"] = np.nan
    points["aridity_quantile"] = "unknown"

points["hydroclimatic_regime"] = "other"
points.loc[points["high_vpd_gt_2p26"] & points["low_latitude_30deg"], "hydroclimatic_regime"] = "low_latitude_high_vpd"
points.loc[points["region_Sahel"] & points["high_vpd_gt_2p26"], "hydroclimatic_regime"] = "sahel_high_vpd"
points.loc[points["region_Sahel"] & ~points["high_vpd_gt_2p26"], "hydroclimatic_regime"] = "sahel_not_high_vpd"
points.loc[points["region_Russian_steppe_west"], "hydroclimatic_regime"] = "russian_steppe_west_secondary"


# ---------------------------------------------------------------------
# Group summaries
# ---------------------------------------------------------------------

def group_summary(points_df, mask, label, group_type, min_n=3):
    g = pd.Series(mask, index=points_df.index).fillna(False).astype(bool)
    if g.sum() < min_n or (~g).sum() < min_n:
        return None

    row = {
        "label": label,
        "group_type": group_type,
        "n_group": int(g.sum()),
        "n_rest": int((~g).sum()),
        "center_lat": float(num(points_df.loc[g, "lat"]).mean()),
        "center_lon": float(num(points_df.loc[g, "lon"]).mean()),
        "min_lat": float(num(points_df.loc[g, "lat"]).min()),
        "max_lat": float(num(points_df.loc[g, "lat"]).max()),
        "min_lon": float(num(points_df.loc[g, "lon"]).min()),
        "max_lon": float(num(points_df.loc[g, "lon"]).max()),
    }

    for outcome in ["latent_satbreak_probability", "latent_post_slope", "latent_slope_change", "point_limitation_rank_score"]:
        inside = num(points_df.loc[g, outcome])
        outside = num(points_df.loc[~g, outcome])
        row[f"{outcome}_inside_median"] = float(inside.median())
        row[f"{outcome}_outside_median"] = float(outside.median())
        row[f"{outcome}_median_diff"] = row[f"{outcome}_inside_median"] - row[f"{outcome}_outside_median"]
        row[f"{outcome}_mann_p"] = mann_p(inside, outside)

    a, b, c, d, gf, rf, rr, odds, p = fisher_event(
        points_df.loc[g, "event_limitation_hotspot"],
        points_df.loc[~g, "event_limitation_hotspot"],
    )
    row.update({
        "hotspot_inside_n": a,
        "hotspot_inside_fraction": gf,
        "hotspot_outside_fraction": rf,
        "hotspot_risk_ratio": rr,
        "hotspot_odds_ratio": odds,
        "hotspot_fisher_p": p,
    })

    if "threshold_like_fit_fraction" in points_df.columns:
        row["raw_threshold_fit_fraction_inside_median"] = float(num(points_df.loc[g, "threshold_like_fit_fraction"]).median())
        row["raw_threshold_fit_fraction_outside_median"] = float(num(points_df.loc[~g, "threshold_like_fit_fraction"]).median())
        row["raw_threshold_fit_fraction_median_diff"] = row["raw_threshold_fit_fraction_inside_median"] - row["raw_threshold_fit_fraction_outside_median"]

    sat_sd = num(points_df["latent_satbreak_probability"]).std(skipna=True) or 1
    post_sd = num(points_df["latent_post_slope"]).std(skipna=True) or 1
    slope_sd = num(points_df["latent_slope_change"]).std(skipna=True) or 1

    row["limitation_score"] = (
        row["latent_satbreak_probability_median_diff"] / sat_sd
        - row["latent_post_slope_median_diff"] / post_sd
        - row["latent_slope_change_median_diff"] / slope_sd
        + 2.0 * ((gf if pd.notna(gf) else 0) - (rf if pd.notna(rf) else 0))
    )

    row["point_ids"] = ";".join(points_df.loc[g, "point_id"].astype(str).tolist())
    return row

group_rows = []

# Core a priori regimes
core_groups = [
    ("regime::High_VPD_gt_2p26", "hydroclimatic_regime", points["high_vpd_gt_2p26"]),
    ("regime::Sahel", "named_region", points["region_Sahel"]),
    ("regime::Western_Sahel", "named_region", points["region_Western_Sahel"]),
    ("regime::Sahel_and_High_VPD", "spatial_hydroclimatic_intersection", points["sahel_and_high_vpd"]),
    ("regime::Low_Latitude_High_VPD", "spatial_hydroclimatic_intersection", points["low_latitude_high_vpd"]),
    ("regime::Russian_steppe_west", "secondary_named_region", points["region_Russian_steppe_west"]),
]

for label, typ, mask in core_groups:
    r = group_summary(points, mask, label, typ, min_n=3)
    if r:
        group_rows.append(r)

# Named region boxes
for name in REGIONS:
    r = group_summary(points, points[f"region_{name}"], f"named_region::{name}", "named_region", min_n=3)
    if r:
        group_rows.append(r)

# Country, ecoregion, biome, realm, dryland, lat band, lon sector
grouping_specs = [
    ("country", "geo_country", 3),
    ("continent", "geo_continent", 5),
    ("subregion", "geo_subregion", 5),
    ("ecoregion", "eco_ecoregion", 3),
    ("biome", "eco_biome", 5),
    ("realm", "eco_realm", 5),
    ("nnh_name", "eco_nnh_name", 5),
    ("latitude_band", "latitude_band", 5),
    ("longitude_sector", "longitude_sector", 5),
    ("aridity_quantile", "aridity_quantile", 5),
    ("dryland_class_unep_if_ai", "dryland_class", 5),
    ("hydroclimatic_regime", "hydroclimatic_regime", 3),
]

for group_name, col, min_n in grouping_specs:
    if col not in points.columns:
        continue
    vals = str_missing(points[col])
    for level in sorted(vals.unique()):
        if level.lower() in ["nan", "missing", "none", ""]:
            continue
        mask = vals.eq(level)
        r = group_summary(points, mask, f"{group_name}::{level}", group_name, min_n=min_n)
        if r:
            group_rows.append(r)

# Ecoregion/country/biome intersected with high VPD
for base_name, col, min_n in [
    ("high_vpd_ecoregion", "eco_ecoregion", 3),
    ("high_vpd_biome", "eco_biome", 3),
    ("high_vpd_country", "geo_country", 3),
    ("high_vpd_latitude_band", "latitude_band", 3),
]:
    if col not in points.columns:
        continue
    vals = str_missing(points[col])
    for level in sorted(vals.unique()):
        if level.lower() in ["nan", "missing", "none", ""]:
            continue
        mask = vals.eq(level) & points["high_vpd_gt_2p26"]
        r = group_summary(points, mask, f"{base_name}::{level}", base_name, min_n=min_n)
        if r:
            group_rows.append(r)

group_summary_df = pd.DataFrame(group_rows)
if not group_summary_df.empty:
    for c in [
        "latent_satbreak_probability_mann_p",
        "latent_post_slope_mann_p",
        "latent_slope_change_mann_p",
        "point_limitation_rank_score_mann_p",
        "hotspot_fisher_p",
    ]:
        if c in group_summary_df.columns:
            group_summary_df[c.replace("_p", "_q")] = bh_qvalues(group_summary_df[c])
    qcols = [c for c in group_summary_df.columns if c.endswith("_q")]
    group_summary_df["best_q"] = group_summary_df[qcols].min(axis=1, skipna=True)
else:
    group_summary_df["best_q"] = np.nan

# Candidate thesis ranking
cand = group_summary_df.copy()

if not cand.empty:
    cand["n_score"] = np.where(cand["n_group"] >= 20, 1.0, np.where(cand["n_group"] >= 10, 0.75, 0.45))
    cand["interpretability_score"] = 0.5
    cand.loc[cand["group_type"].isin([
        "hydroclimatic_regime",
        "spatial_hydroclimatic_intersection",
        "named_region",
        "secondary_named_region",
        "ecoregion",
        "biome",
        "country",
        "latitude_band",
        "dryland_class",
        "high_vpd_ecoregion",
        "high_vpd_biome",
        "high_vpd_country",
    ]), "interpretability_score"] = 1.0

    cand["q_score"] = np.where(cand["best_q"] <= 0.01, 1.0, np.where(cand["best_q"] <= 0.10, 0.6, 0.25))
    cand["event_score"] = np.where(cand["hotspot_risk_ratio"] >= 2.0, 1.0, np.where(cand["hotspot_risk_ratio"] >= 1.25, 0.6, 0.2))
    cand["slope_weakening_score"] = np.where(cand["latent_post_slope_median_diff"] < 0, 1.0, 0.25)
    cand["slope_change_score"] = np.where(cand["latent_slope_change_median_diff"] < 0, 1.0, 0.25)

    # Avoid letting tiny groups win too easily.
    cand["tiny_group_penalty"] = np.where(cand["n_group"] < 6, -1.0, 0.0)

    cand["thesis_strength_score"] = (
        cand["limitation_score"].rank(pct=True).fillna(0)
        + cand["n_score"]
        + cand["interpretability_score"]
        + cand["q_score"]
        + cand["event_score"]
        + cand["slope_weakening_score"]
        + cand["slope_change_score"]
        + cand["tiny_group_penalty"]
    )

    cand = cand.sort_values("thesis_strength_score", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------
# Permutation tests for top thesis candidates
# ---------------------------------------------------------------------

def permutation_test_for_group(points_df, mask, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    mask = pd.Series(mask, index=points_df.index).fillna(False).astype(bool)
    n = int(mask.sum())
    if n < 3 or n >= len(points_df) - 3:
        return None

    obs_row = group_summary(points_df, mask, "observed", "observed", min_n=3)
    obs_score = obs_row["limitation_score"]

    scores = []
    idx = np.arange(len(points_df))
    for _ in range(n_perm):
        choice = rng.choice(idx, size=n, replace=False)
        m = np.zeros(len(points_df), dtype=bool)
        m[choice] = True
        r = group_summary(points_df, m, "random", "random", min_n=3)
        scores.append(r["limitation_score"])

    scores = np.array(scores)
    emp_p = float((np.sum(scores >= obs_score) + 1) / (len(scores) + 1))
    return {
        "observed_score": obs_score,
        "random_mean": float(np.mean(scores)),
        "random_sd": float(np.std(scores)),
        "random_p95": float(np.quantile(scores, 0.95)),
        "random_p99": float(np.quantile(scores, 0.99)),
        "empirical_p_score_ge_observed": emp_p,
    }

perm_rows = []
if not cand.empty:
    # Use top 25 plus fixed core hypotheses.
    top_labels = cand.head(25)["label"].tolist()
    for fixed in [
        "regime::High_VPD_gt_2p26",
        "regime::Sahel",
        "regime::Sahel_and_High_VPD",
        "regime::Low_Latitude_High_VPD",
        "regime::Russian_steppe_west",
    ]:
        if fixed not in top_labels:
            top_labels.append(fixed)

    for i, label in enumerate(top_labels):
        rr = cand[cand["label"].eq(label)]
        if rr.empty:
            continue
        pts = set(str(rr.iloc[0]["point_ids"]).split(";"))
        mask = points["point_id"].isin(pts)
        res = permutation_test_for_group(points, mask, n_perm=2000, seed=1000 + i)
        if res:
            out = rr.iloc[0][["label", "group_type", "n_group", "limitation_score", "hotspot_risk_ratio", "best_q"]].to_dict()
            out.update(res)
            perm_rows.append(out)

perm_df = pd.DataFrame(perm_rows)
if not perm_df.empty:
    perm_df = perm_df.sort_values("empirical_p_score_ge_observed")

# ---------------------------------------------------------------------
# Raw fit sensitivity for top candidates
# ---------------------------------------------------------------------

raw_sens_rows = []
if not obs.empty and not cand.empty:
    obs2 = obs.copy()
    regime_membership = points[["point_id"]].copy()

    for label in cand.head(20)["label"].tolist():
        pts = set(str(cand.loc[cand["label"].eq(label), "point_ids"].iloc[0]).split(";"))
        regime_membership[label] = regime_membership["point_id"].isin(pts)

    obs2 = obs2.merge(regime_membership, on="point_id", how="left")

    strata_list = [
        ["metric"],
        ["product_combo"],
        ["stress_definition"],
        ["co2_version"],
        ["metric", "product_combo"],
        ["metric", "stress_definition"],
    ]

    for label in cand.head(20)["label"].tolist():
        if label not in obs2.columns:
            continue

        for strata in strata_list:
            strata = [c for c in strata if c in obs2.columns]
            if not strata:
                continue

            grouped = obs2.groupby(strata, dropna=False)
            for keys, sub in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)

                local = sub[sub[label].fillna(False).astype(bool)]
                rest = sub[~sub[label].fillna(False).astype(bool)]
                if len(local) < 20 or len(rest) < 20:
                    continue

                a, b, c, d, gf, rf, rr, odds, p = fisher_event(local["threshold_like_fit"], rest["threshold_like_fit"])

                row = {
                    "candidate_label": label,
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

                raw_sens_rows.append(row)

raw_sens = pd.DataFrame(raw_sens_rows)

# ---------------------------------------------------------------------
# Spatial autocorrelation tests
# ---------------------------------------------------------------------

def morans_i_knn(points_df, value_col, k=8, n_perm=999, seed=42):
    d = points_df.dropna(subset=["lat", "lon", value_col]).copy()
    if len(d) < k + 3:
        return None

    vals = num(d[value_col]).values
    lat = num(d["lat"]).values
    lon = num(d["lon"]).values
    n = len(d)

    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = haversine_km(lat[i], lon[i], lat, lon)

    W = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(D[i, :])[1:k+1]
        W[i, idx] = 1

    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0)

    x = vals - np.mean(vals)
    S0 = W.sum()
    if S0 == 0 or np.sum(x**2) == 0:
        return None

    I = (n / S0) * ((W * np.outer(x, x)).sum() / np.sum(x**2))

    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_perm):
        xp = rng.permutation(x)
        Ip = (n / S0) * ((W * np.outer(xp, xp)).sum() / np.sum(xp**2))
        perms.append(Ip)
    perms = np.array(perms)

    p_upper = float((np.sum(perms >= I) + 1) / (len(perms) + 1))
    p_two = float((np.sum(np.abs(perms) >= abs(I)) + 1) / (len(perms) + 1))

    return {
        "value_col": value_col,
        "k_neighbors": k,
        "n_points": n,
        "morans_i": float(I),
        "perm_mean": float(np.mean(perms)),
        "perm_sd": float(np.std(perms)),
        "p_upper": p_upper,
        "p_two_sided": p_two,
    }

moran_rows = []
for value_col in [
    "latent_satbreak_probability",
    "latent_post_slope",
    "latent_slope_change",
    "point_limitation_rank_score",
    "threshold_like_fit_fraction",
]:
    if value_col in points.columns:
        for k in [5, 8, 12, 20]:
            res = morans_i_knn(points, value_col, k=k, n_perm=999, seed=77 + k)
            if res:
                moran_rows.append(res)

moran_df = pd.DataFrame(moran_rows)

# ---------------------------------------------------------------------
# Hotspot/high-VPD geography concentration
# ---------------------------------------------------------------------

hotspot_points = points[points["event_limitation_hotspot"]].copy()
high_vpd_points = points[points["high_vpd_gt_2p26"]].copy()
high_vpd_hotspot_points = points[points["high_vpd_gt_2p26"] & points["event_limitation_hotspot"]].copy()

def concentration_summary(df, label):
    rows = []
    for col in [
        "geo_country",
        "geo_subregion",
        "geo_continent",
        "eco_ecoregion",
        "eco_biome",
        "eco_realm",
        "eco_nnh_name",
        "latitude_band",
        "longitude_sector",
        "named_geographic_regions",
        "hydroclimatic_regime",
    ]:
        if col not in df.columns:
            continue
        vc = str_missing(df[col]).value_counts()
        total = int(vc.sum())
        for level, n in vc.head(20).items():
            rows.append({
                "set": label,
                "field": col,
                "level": level,
                "n": int(n),
                "fraction": float(n / total) if total else np.nan,
            })
    return rows

concentration_rows = []
concentration_rows += concentration_summary(hotspot_points, "all_limitation_hotspot_points")
concentration_rows += concentration_summary(high_vpd_points, "all_high_vpd_points")
concentration_rows += concentration_summary(high_vpd_hotspot_points, "high_vpd_limitation_hotspot_points")
concentration = pd.DataFrame(concentration_rows)

# ---------------------------------------------------------------------
# Thesis lock scorecard
# ---------------------------------------------------------------------

def get_candidate(label):
    if cand.empty:
        return None
    d = cand[cand["label"].eq(label)]
    if d.empty:
        return None
    return d.iloc[0]

high_vpd_row = get_candidate("regime::High_VPD_gt_2p26")
sahel_row = get_candidate("regime::Sahel")
sahel_high_vpd_row = get_candidate("regime::Sahel_and_High_VPD")
lowlat_high_vpd_row = get_candidate("regime::Low_Latitude_High_VPD")
russia_row = get_candidate("regime::Russian_steppe_west")

score_rows = []

def add_score(criterion, status, evidence, implication):
    score_rows.append({
        "criterion": criterion,
        "status": status,
        "evidence": evidence,
        "implication": implication,
    })

if high_vpd_row is not None:
    add_score(
        "High-VPD regime is enriched",
        "PASS" if high_vpd_row["hotspot_risk_ratio"] >= 2 and high_vpd_row["hotspot_fisher_p"] < 0.01 else "WEAK",
        f"n={int(high_vpd_row['n_group'])}; RR={fmt(high_vpd_row['hotspot_risk_ratio'])}; p={fmt(high_vpd_row['hotspot_fisher_p'], 4)}; score={fmt(high_vpd_row['limitation_score'])}",
        "Supports a hydroclimatic-regime thesis."
    )

if sahel_row is not None:
    add_score(
        "Sahel is a named geographic expression",
        "PASS" if sahel_row["hotspot_risk_ratio"] >= 2 and sahel_row["hotspot_fisher_p"] < 0.01 else "WEAK",
        f"n={int(sahel_row['n_group'])}; RR={fmt(sahel_row['hotspot_risk_ratio'])}; p={fmt(sahel_row['hotspot_fisher_p'], 4)}; post-slope diff={fmt(sahel_row['latent_post_slope_median_diff'])}",
        "Sahel can be used as the clearest named case."
    )

if sahel_high_vpd_row is not None:
    add_score(
        "Sahel + high-VPD intersection is coherent",
        "PASS" if sahel_high_vpd_row["hotspot_risk_ratio"] >= 2 and sahel_high_vpd_row["latent_post_slope_median_diff"] < 0 else "WEAK",
        f"n={int(sahel_high_vpd_row['n_group'])}; RR={fmt(sahel_high_vpd_row['hotspot_risk_ratio'])}; post-slope diff={fmt(sahel_high_vpd_row['latent_post_slope_median_diff'])}",
        "This is the cleanest case-study regime if n is acceptable."
    )

if russia_row is not None:
    russia_status = "CONTRAST"
    if russia_row["latent_post_slope_median_diff"] < 0:
        russia_status = "SECONDARY_SUPPORT"
    add_score(
        "Russian steppe phenotype",
        russia_status,
        f"n={int(russia_row['n_group'])}; RR={fmt(russia_row['hotspot_risk_ratio'])}; satprob diff={fmt(russia_row['latent_satbreak_probability_median_diff'])}; post-slope diff={fmt(russia_row['latent_post_slope_median_diff'])}",
        "Use as supplementary contrast, not the main slope-weakening claim, unless post-slope is negative."
    )

# Ecoregion concentration decision
eco_conc = concentration[
    (concentration["set"].eq("high_vpd_limitation_hotspot_points"))
    & (concentration["field"].eq("eco_ecoregion"))
].copy()

top_eco_level = None
top_eco_fraction = np.nan
top_eco_n = 0
if not eco_conc.empty:
    top_eco = eco_conc.sort_values("fraction", ascending=False).iloc[0]
    top_eco_level = top_eco["level"]
    top_eco_fraction = top_eco["fraction"]
    top_eco_n = int(top_eco["n"])
    add_score(
        "Ecoregion concentration of high-VPD hotspots",
        "ECOSYSTEM_SPECIFIC" if top_eco_fraction >= 0.45 and top_eco_n >= 8 else "DISTRIBUTED",
        f"Top ecoregion={top_eco_level}; n={top_eco_n}; fraction={pct(top_eco_fraction)}",
        "Determines whether final title should be ecoregion-specific or broader hydroclimatic geography."
    )
else:
    add_score(
        "Ecoregion concentration of high-VPD hotspots",
        "UNKNOWN",
        "Ecoregion layer unavailable or no ecoregion matches.",
        "Cannot lock an ecoregion-specific thesis without ecoregion labels."
    )

# Product/metric caveat from Phase 15 if available
scorecard15 = read_csv(PH15_SCORECARD, required=False)
if not scorecard15.empty:
    weak_rows = scorecard15[scorecard15["status"].astype(str).str.upper().eq("WEAK")]
    if len(weak_rows):
        add_score(
            "Product/metric robustness caveat",
            "LIMITATION",
            "; ".join((weak_rows["criterion"] + ": " + weak_rows["evidence"]).astype(str).tolist()),
            "Keep product/metric sensitivity in Supplement; do not claim fully product-independent truth."
        )

scorecard = pd.DataFrame(score_rows)

# ---------------------------------------------------------------------
# Decide final thesis
# ---------------------------------------------------------------------

if top_eco_level and top_eco_fraction >= 0.45 and top_eco_n >= 8:
    thesis_type = "ecoregion_specific_hydroclimatic_hotspot"
    thesis_title = f"High-Stress Limitation of Grassland Water-Use Efficiency in {top_eco_level}"
    thesis_claim = (
        f"High-stress WUE limitation is concentrated in a high-VPD dryland regime, "
        f"with the strongest ecoregion-level concentration in {top_eco_level}. "
        f"The result should be framed as an ecoregion-specific expression of a broader hydroclimatic threshold regime."
    )
elif high_vpd_row is not None and high_vpd_row["hotspot_risk_ratio"] >= 2 and high_vpd_row["latent_post_slope_median_diff"] < 0:
    thesis_type = "spatial_hydroclimatic_regime"
    thesis_title = "Hydroclimatic Geography Controls Where Grassland WUE Thresholds Emerge Under Compound Stress"
    thesis_claim = (
        "High-stress WUE limitation is not a universal grassland response. "
        "It is concentrated in a high-VPD, low-latitude dryland regime. "
        "The Sahel should be presented as the clearest named geographic expression, "
        "while high VPD is the stronger mechanistic summary."
    )
elif sahel_row is not None and sahel_row["hotspot_risk_ratio"] >= 2:
    thesis_type = "sahel_named_hotspot"
    thesis_title = "Localized High-Stress Limitation of Grassland Water-Use Efficiency in Sahelian Drylands"
    thesis_claim = (
        "High-stress WUE limitation is not universal, but is concentrated in the Sahel named region. "
        "This should be framed as a geographic hotspot, not a global grassland threshold."
    )
else:
    thesis_type = "safe_spatial_localization"
    thesis_title = "Spatially Localized High-Stress Limitation of Satellite-Derived Grassland WUE Under Compound Stress"
    thesis_claim = (
        "The evidence supports spatial localization of a satellite-derived WUE response phenotype, "
        "but not a strong ecoregion- or mechanism-specific claim."
    )

# ---------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------

point_cols_front = [
    "point_id", "lat", "lon",
    "geo_country", "geo_country_iso3", "geo_continent", "geo_subregion",
    "eco_ecoregion", "eco_biome", "eco_realm", "eco_nnh_name",
    "named_geographic_regions", "latitude_band", "longitude_sector",
    "dryland_class_unep_if_ai", "aridity_quantile", "mean_vpd_quartile",
    "hydroclimatic_regime",
    "high_vpd_gt_2p26", "region_Sahel", "sahel_and_high_vpd", "low_latitude_high_vpd", "region_Russian_steppe_west",
    "latent_response_class",
    "latent_satbreak_probability", "latent_post_slope", "latent_slope_change",
    "event_limitation_hotspot", "event_satprob_top20", "event_post_slope_bottom20", "event_slope_change_bottom20",
    "point_limitation_rank_score", "point_limitation_rank",
    "threshold_like_fit_fraction", "n_product_combos_with_any_threshold", "n_product_combos_threshold_gt_0p05",
    "threshold_fraction_metric_uwue", "threshold_fraction_metric_iwue", "threshold_fraction_metric_raw_wue",
    "mean_vpd", "aridity", "mean_soil_moisture", "mean_annual_temperature", "mean_annual_precipitation", "mean_lai",
    "rooting_depth", "p50",
]

point_cols = [c for c in point_cols_front if c in points.columns] + [c for c in points.columns if c not in point_cols_front]

points_out = points.sort_values("point_limitation_rank_with_product").copy()
save_csv(points_out[point_cols], TAB / "Table70_point_level_geography_response_annotation.csv")

save_csv(hotspot_points.sort_values("point_limitation_rank_score"), TAB / "Table71_hotspot_point_geography.csv")
save_csv(high_vpd_points.sort_values("point_limitation_rank_score"), TAB / "Table72_high_vpd_point_geography.csv")
save_csv(group_summary_df.sort_values("limitation_score", ascending=False), TAB / "Table73_grouped_geography_response_summary.csv")
save_csv(cand, TAB / "Table74_candidate_thesis_ranking.csv")
save_csv(perm_df, TAB / "Table75_top_candidate_permutation_tests.csv")
save_csv(raw_sens, TAB / "Table76_top_candidate_raw_fit_sensitivity.csv")
save_csv(moran_df, TAB / "Table77_spatial_autocorrelation_tests.csv")
save_csv(scorecard, TAB / "Table78_thesis_lock_evidence_scorecard.csv")
save_csv(concentration, TAB / "Table79_hotspot_and_high_vpd_geography_concentration.csv")

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

try:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    sc = ax.scatter(
        points["lon"], points["lat"],
        c=points["point_limitation_rank_score"],
        s=45,
        alpha=0.75
    )
    ax.scatter(
        points.loc[points["high_vpd_gt_2p26"], "lon"],
        points.loc[points["high_vpd_gt_2p26"], "lat"],
        s=95,
        facecolors="none",
        edgecolors="black",
        linewidths=1.0,
        label="High VPD > 2.26"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Figure 1. Point-level WUE limitation score and high-VPD regime")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Point limitation rank score")
    ax.legend(frameon=False)
    savefig(FIG / "Figure1_point_level_limitation_score_map.png")
except Exception as e:
    print("WARNING: Figure1 failed:", e)

try:
    if not cand.empty:
        plot = cand.head(20).copy()
        plot["short_label"] = plot["label"].astype(str).str.replace("regime::", "", regex=False).str.replace("named_region::", "", regex=False).str.slice(0, 55)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(plot))))
        y = np.arange(len(plot))
        ax.barh(y, plot["thesis_strength_score"])
        ax.set_yticks(y)
        ax.set_yticklabels(plot["short_label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Thesis strength score")
        ax.set_title("Figure 2. Ranked geography/ecosystem thesis candidates")
        savefig(FIG / "Figure2_ranked_thesis_candidates.png")
except Exception as e:
    print("WARNING: Figure2 failed:", e)

try:
    if not concentration.empty:
        plot = concentration[
            concentration["set"].eq("high_vpd_limitation_hotspot_points")
            & concentration["field"].isin(["eco_ecoregion", "geo_country", "eco_biome", "named_geographic_regions"])
        ].copy()
        plot = plot.sort_values("fraction", ascending=False).head(20)
        plot["label"] = plot["field"].str.replace("eco_", "", regex=False).str.replace("geo_", "", regex=False) + "::" + plot["level"].astype(str).str.slice(0, 45)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(plot))))
        y = np.arange(len(plot))
        ax.barh(y, plot["fraction"])
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction of high-VPD hotspot points")
        ax.set_title("Figure 3. Geographic concentration of high-VPD hotspot points")
        savefig(FIG / "Figure3_high_vpd_hotspot_geographic_concentration.png")
except Exception as e:
    print("WARNING: Figure3 failed:", e)

try:
    if "mean_vpd" in points.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            points["lat"], points["mean_vpd"],
            c=points["latent_slope_change"],
            s=50,
            alpha=0.75
        )
        ax.axhline(2.26, linestyle="--", linewidth=1, label="High VPD rule")
        ax.axvspan(10, 20, alpha=0.12, label="Sahelian latitude")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Mean VPD")
        ax.set_title("Figure 4. Latitude-VPD space and slope-change phenotype")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Latent slope change")
        ax.legend(frameon=False)
        savefig(FIG / "Figure4_latitude_vpd_slope_change_space.png")
except Exception as e:
    print("WARNING: Figure4 failed:", e)

# ---------------------------------------------------------------------
# Text outputs
# ---------------------------------------------------------------------

geo_source_note = f"""# Geography source note

## Loaded sources

{chr(10).join("- " + x for x in geo_notes) if geo_notes else "- No external geography polygons loaded."}

## External geography files

- Natural Earth countries ZIP expected at: `{countries_zip}`
- RESOLVE/WWF Ecoregions 2017 ZIP expected at: `{ecoregions_zip}`

If the ecoregion download failed, manually download `Ecoregions2017.zip` and place it at:

`data/external/geography/Ecoregions2017.zip`

Then rerun Phase 16.

## Interpretation guardrail

Country and ecoregion annotation is used to name the geography of the response phenotype. It does not create tower validation and does not prove causal mechanism by itself.
"""
save_text(geo_source_note, TXT / "GEOGRAPHY_SOURCE_NOTE.md")

methods_text = """# Methods: point-level geography annotation and thesis locking

Each grassland point was annotated with administrative and ecological geography. Country and continent labels were assigned by spatial intersection with Natural Earth country polygons. Ecoregion, biome, and realm labels were assigned by spatial intersection with the RESOLVE/WWF Terrestrial Ecoregions 2017 polygon layer when available. Points were also classified into predefined geographic boxes, latitude bands, longitude sectors, dryland/aridity classes, high-VPD regimes, and combined spatial–hydroclimatic regimes.

For each annotated group, we compared points inside the group to all remaining grassland points using latent high-stress limitation probability, latent high-stress/post-transition WUE slope, latent slope-change response, and a combined high-stress limitation hotspot event. We computed median differences, Mann–Whitney tests, Fisher exact tests, risk ratios, and a composite limitation enrichment score. Candidate thesis groups were ranked by sample size, interpretability, response-strength, event enrichment, slope weakening, slope-change weakening, and q-value support.

To guard against geographic cherry-picking, top candidate groups were tested against random same-size point selections. Spatial autocorrelation was assessed using k-nearest-neighbor Moran's I permutation tests. Product and metric sensitivity were summarized from all raw response fits by candidate group.
"""
save_text(methods_text, TXT / "METHODS_point_geography_annotation.md")

# Build a readable summary of top points.
top_points_summary = points_out[point_cols].head(25).copy()
top_points_text = top_points_summary.to_string(index=False)

top_candidates_text = cand.head(20)[[
    c for c in [
        "label", "group_type", "n_group", "thesis_strength_score", "limitation_score", "best_q",
        "hotspot_inside_fraction", "hotspot_outside_fraction", "hotspot_risk_ratio",
        "latent_satbreak_probability_inside_median", "latent_satbreak_probability_outside_median",
        "latent_post_slope_inside_median", "latent_post_slope_outside_median",
        "latent_slope_change_inside_median", "latent_slope_change_outside_median"
    ] if c in cand.columns
]].to_string(index=False) if not cand.empty else "No candidates."

concentration_text = concentration.head(80).to_string(index=False) if not concentration.empty else "No concentration table."

final_thesis_text = f"""# FINAL THESIS LOCK

## Final thesis type

`{thesis_type}`

## Recommended title

**{thesis_title}**

## Locked claim

{thesis_claim}

## Why this is the best claim

The point-level annotation step is designed to prevent map cherry-picking. The question is not simply whether a few points in the Sahel look interesting. The question is whether the high-stress limitation phenotype is coherently associated with objective geography: high VPD, low latitude, named dryland regions, countries, ecoregions, biomes, and product-support patterns.

## Evidence scorecard

{scorecard.to_string(index=False) if not scorecard.empty else "No scorecard generated."}

## Top thesis candidates

{top_candidates_text}

## Geographic concentration of hotspot points

{concentration_text}

## Top 25 high-limitation points

{top_points_text}

## Manuscript wording

Use this as the main result:

> High-stress WUE limitation is not a universal grassland response. Instead, threshold-like response weakening is concentrated in a spatial–hydroclimatic regime characterized by high atmospheric water demand and dryland geography. The Sahel is the clearest named geographic expression, while ecoregion-level annotations determine whether the final paper should be framed as Sahelian, Sudanian/Sahelian-savanna, or broader high-VPD dryland grassland response.

## What not to claim

- Do not claim all grasslands break down.
- Do not claim VPD alone causally proves the threshold.
- Do not claim the Sahel is uniquely special if climate-matched controls explain the limitation-probability difference.
- Do not claim tower-validated truth.
- Do not make product differences the headline.
- Do not make a tiny KNN hotspot the entire paper.

## Next action

If the top hotspot/high-VPD points concentrate in one or two named ecoregions, write the paper as an ecoregion-specific dryland threshold paper. If they are spread across multiple dryland ecoregions but share high VPD and low latitude, write it as a spatial–hydroclimatic regime paper.
"""
save_text(final_thesis_text, TXT / "FINAL_THESIS_LOCK.md")

lit_positioning = """# Literature positioning for the locked thesis

The thesis should be positioned against four literatures.

## 1. VPD and atmospheric demand

Vapor pressure deficit is atmospheric water demand. Rising VPD increases evaporative demand and can drive stomatal closure, changing the balance between carbon uptake and water loss. This supports framing the result as a high-atmospheric-demand regime rather than a simple location effect.

## 2. Compound hot-dry stress

High VPD and low soil moisture are not independent stressors. Dry soils reduce evaporative cooling, raise sensible heating, and can intensify atmospheric demand. The paper should frame the predictor as compound atmospheric–soil moisture stress.

## 3. Dryland thresholds and spatial regimes

Drylands are already studied as systems where nonlinear responses, spatial heterogeneity, and threshold-like transitions can emerge. This supports your move away from a global grassland average toward a spatial-regime claim.

## 4. Ecoregions and biogeography

WWF/RESOLVE ecoregions provide an objective way to translate point coordinates into named ecosystems. The final paper becomes stronger if the high-VPD limitation points concentrate in named ecoregions rather than just an arbitrary latitude-longitude box.

## Final literature gap

Most WUE work asks whether ecosystem WUE increases or decreases under stress globally. Your paper should instead ask where WUE response-shape weakening emerges. The contribution is spatially explicit: threshold-like WUE limitation is not a universal grassland response, but a property of identifiable hydroclimatic/geographic regimes.
"""
save_text(lit_positioning, TXT / "LITERATURE_POSITIONING_FOR_THESIS.md")

results_summary = f"""# Results summary for point-level thesis locking

## Main thesis

{thesis_claim}

## Key outputs

- `Table70_point_level_geography_response_annotation.csv`: every grassland point with country, ecoregion, biome, response phenotype, and product-support metrics.
- `Table71_hotspot_point_geography.csv`: all points in the high-stress limitation hotspot event.
- `Table72_high_vpd_point_geography.csv`: all high-VPD points.
- `Table73_grouped_geography_response_summary.csv`: all geography/ecosystem group comparisons.
- `Table74_candidate_thesis_ranking.csv`: ranked candidate claims.
- `Table75_top_candidate_permutation_tests.csv`: anti-cherry-picking same-n permutation tests.
- `Table76_top_candidate_raw_fit_sensitivity.csv`: product/metric/stress sensitivity.
- `Table77_spatial_autocorrelation_tests.csv`: Moran's I spatial-dependence tests.
- `Table78_thesis_lock_evidence_scorecard.csv`: final decision evidence.
- `Table79_hotspot_and_high_vpd_geography_concentration.csv`: concentration of high-VPD/hotspot points by country, ecoregion, biome, etc.

## Paste-back instructions

Paste:
1. FINAL_THESIS_LOCK.md
2. Table74 top 30 rows
3. Table79 top 80 rows
4. Table78 scorecard
5. If ecoregions failed, paste GEOGRAPHY_SOURCE_NOTE.md
"""
save_text(results_summary, TXT / "RESULTS_point_geography_summary.md")

manifest = {
    "phase": "Phase 16 point-level geography thesis lock",
    "thesis_type": thesis_type,
    "recommended_title": thesis_title,
    "locked_claim": thesis_claim,
    "n_points": int(len(points)),
    "n_hotspot_points": int(points["event_limitation_hotspot"].sum()),
    "n_high_vpd_points": int(points["high_vpd_gt_2p26"].sum()),
    "geopandas_ok": GEOPANDAS_OK,
    "country_layer_loaded": countries_gdf is not None,
    "ecoregion_layer_loaded": ecoregions_gdf is not None,
    "top_ecoregion_level": top_eco_level,
    "top_ecoregion_fraction": None if pd.isna(top_eco_fraction) else float(top_eco_fraction),
    "outputs": {
        "point_annotation": str(TAB / "Table70_point_level_geography_response_annotation.csv"),
        "hotspot_points": str(TAB / "Table71_hotspot_point_geography.csv"),
        "high_vpd_points": str(TAB / "Table72_high_vpd_point_geography.csv"),
        "grouped_summary": str(TAB / "Table73_grouped_geography_response_summary.csv"),
        "candidate_ranking": str(TAB / "Table74_candidate_thesis_ranking.csv"),
        "permutation_tests": str(TAB / "Table75_top_candidate_permutation_tests.csv"),
        "raw_fit_sensitivity": str(TAB / "Table76_top_candidate_raw_fit_sensitivity.csv"),
        "spatial_autocorrelation": str(TAB / "Table77_spatial_autocorrelation_tests.csv"),
        "scorecard": str(TAB / "Table78_thesis_lock_evidence_scorecard.csv"),
        "concentration": str(TAB / "Table79_hotspot_and_high_vpd_geography_concentration.csv"),
        "final_thesis_lock": str(TXT / "FINAL_THESIS_LOCK.md"),
        "methods": str(TXT / "METHODS_point_geography_annotation.md"),
        "literature_positioning": str(TXT / "LITERATURE_POSITIONING_FOR_THESIS.md"),
    }
}
(OUT / "phase16_point_geography_thesis_lock_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

print("")
print("DONE Phase 16 point-level geography thesis lock.")
print("")
print(final_thesis_text)
