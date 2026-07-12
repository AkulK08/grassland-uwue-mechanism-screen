#!/usr/bin/env python
from pathlib import Path
import re
import json
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(".")
OUT = Path("results/tower_validation_broad_inventory")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

DAILY_PATH = TAB / "Table84_tower_daily_wue_fluxes.csv"
META_PATH = TAB / "Table82_tower_site_metadata_extracted.csv"

PH8_LATENT = Path("results/trait_framework/phase8/table_latent_response_by_point.csv")
PH16_POINTS = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")


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


def num(x):
    return pd.to_numeric(x, errors="coerce")


def infer_site_from_text(text):
    text = str(text)
    for pat in [
        r"AMF_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"FLX_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"([A-Z]{2}-[A-Za-z0-9]{3})",
    ]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def infer_site_row(row):
    text = str(row.get("source_zip", "")) + " " + str(row.get("source_member", ""))
    return infer_site_from_text(text)


def zscore_site(s):
    s = num(s)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean(skipna=True)) / sd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fit_piecewise_grid(x, y, min_n=18, min_side=6, n_boot=60, seed=42):
    x = num(pd.Series(x)).values
    y = num(pd.Series(y)).values
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < min_n:
        return None
    if np.nanmax(x) - np.nanmin(x) < 1.0:
        return None

    qs = np.linspace(0.35, 0.75, 13)
    knots = np.unique(np.nanquantile(x, qs))
    best = None

    for k in knots:
        left = x <= k
        right = x > k
        if left.sum() < min_side or right.sum() < min_side:
            continue

        X = np.column_stack([np.ones(len(x)), x, np.maximum(0, x - k)])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            rss = float(np.sum((y - pred) ** 2))
            tss = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - rss / tss if tss > 0 else np.nan
        except Exception:
            continue

        row = {
            "breakpoint": float(k),
            "pre_slope": float(beta[1]),
            "slope_change": float(beta[2]),
            "post_slope": float(beta[1] + beta[2]),
            "rss": rss,
            "r2": r2,
            "n_obs": int(len(x)),
            "n_left": int(left.sum()),
            "n_right": int(right.sum()),
        }

        if best is None or row["rss"] < best["rss"]:
            best = row

    if best is None:
        return None

    rng = np.random.default_rng(seed)
    boots = []
    n = len(x)

    for _ in range(n_boot):
        ii = rng.choice(np.arange(n), size=n, replace=True)
        xb = x[ii]
        yb = y[ii]
        try:
            knots_b = np.unique(np.nanquantile(xb, qs))
        except Exception:
            continue

        best_b = None
        for k in knots_b:
            left = xb <= k
            right = xb > k
            if left.sum() < min_side or right.sum() < min_side:
                continue
            Xb = np.column_stack([np.ones(len(xb)), xb, np.maximum(0, xb - k)])
            try:
                beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
                pred = Xb @ beta
                rss = float(np.sum((yb - pred) ** 2))
            except Exception:
                continue

            rb = {
                "breakpoint": float(k),
                "pre_slope": float(beta[1]),
                "slope_change": float(beta[2]),
                "post_slope": float(beta[1] + beta[2]),
                "rss": rss,
            }
            if best_b is None or rb["rss"] < best_b["rss"]:
                best_b = rb

        if best_b is not None:
            boots.append(best_b)

    boot = pd.DataFrame(boots)

    for col in ["breakpoint", "pre_slope", "slope_change", "post_slope"]:
        if len(boot) >= 10:
            best[col + "_ci_low"] = float(boot[col].quantile(0.025))
            best[col + "_ci_high"] = float(boot[col].quantile(0.975))
        else:
            best[col + "_ci_low"] = np.nan
            best[col + "_ci_high"] = np.nan

    post = best["post_slope"]
    change = best["slope_change"]

    if post < 0 and change < 0:
        response_class = "breakdown"
    elif change < 0 and post < 0.25:
        response_class = "saturation"
    elif post > 0:
        response_class = "enhancement"
    else:
        response_class = "inconclusive"

    best["response_class"] = response_class

    if len(boot):
        satbreak = ((boot["post_slope"] < 0) & (boot["slope_change"] < 0)) | ((boot["slope_change"] < 0) & (boot["post_slope"] < 0.25))
        enhance = boot["post_slope"] > 0
        best["p_tower_saturation_breakdown"] = float(satbreak.mean())
        best["p_tower_enhancement"] = float(enhance.mean())
    else:
        best["p_tower_saturation_breakdown"] = np.nan
        best["p_tower_enhancement"] = np.nan

    return best


def classify_tower_sites(eight):
    rows = []

    stress_methods = [
        ("compound_vpd_swc", "stress_compound_vpd_swc"),
        ("vpd_precip_proxy", "stress_vpd_precip_proxy"),
        ("vpd_only", "stress_vpd_only"),
    ]

    metrics = [
        ("uwue", "log_uwue_8day"),
        ("wue", "log_wue_8day"),
    ]

    for site, g0 in eight.groupby("site"):
        g0 = g0.copy()

        g_gs = g0[g0["growing_season_proxy"]].copy()
        if len(g_gs) >= 18:
            g_use = g_gs
            growing_season_used = "gpp_top75_percent"
        else:
            g_use = g0
            growing_season_used = "all_valid_periods"

        for metric_name, metric_col in metrics:
            for stress_name, stress_col in stress_methods:
                if stress_col not in g_use.columns:
                    continue

                usable = g_use[[stress_col, metric_col]].dropna()
                if len(usable) < 18:
                    continue

                fit = fit_piecewise_grid(
                    usable[stress_col],
                    usable[metric_col],
                    min_n=18,
                    min_side=6,
                    n_boot=60,
                    seed=abs(hash(site + metric_name + stress_name)) % (2**32 - 1),
                )

                if fit is None:
                    continue

                row = {
                    "site": site,
                    "tower_metric": metric_name,
                    "stress_method": stress_name,
                    "stress_col": stress_col,
                    "metric_col": metric_col,
                    "growing_season_used": growing_season_used,
                    "n_site_8day_total": int(len(g0)),
                    "n_fit_8day": int(len(usable)),
                    "n_years": int(g0["year"].nunique()),
                    "has_swc": bool(g0["swc_8day_mean"].notna().sum() >= 12),
                    "has_precip": bool(g0["precip_8day_mm"].notna().sum() >= 12),
                    "vpd_mean_kpa": float(g0["vpd_8day_kpa_mean"].mean(skipna=True)),
                    "vpd_p90_kpa": float(g0["vpd_8day_kpa_mean"].quantile(0.90)),
                    "gpp_sum": float(g0["gpp_8day_gC_m2"].sum(skipna=True)),
                    "et_sum": float(g0["et_8day_mm"].sum(skipna=True)),
                }
                row.update(fit)
                rows.append(row)

    fits = pd.DataFrame(rows)
    if fits.empty:
        return fits

    metric_rank = {"uwue": 0, "wue": 1}
    stress_rank = {"compound_vpd_swc": 0, "vpd_precip_proxy": 1, "vpd_only": 2}

    fits["metric_rank"] = fits["tower_metric"].map(metric_rank).fillna(99)
    fits["stress_rank"] = fits["stress_method"].map(stress_rank).fillna(99)
    fits["primary_rank"] = fits["metric_rank"] * 10 + fits["stress_rank"]
    fits["is_primary_tower_fit"] = False

    primary_idx = (
        fits.sort_values(["site", "primary_rank", "n_fit_8day"], ascending=[True, True, False])
        .groupby("site")
        .head(1)
        .index
    )
    fits.loc[primary_idx, "is_primary_tower_fit"] = True
    return fits


def load_satellite_points():
    if PH16_POINTS.exists():
        sat = pd.read_csv(PH16_POINTS, low_memory=False)
        source = str(PH16_POINTS)
    elif PH8_LATENT.exists():
        sat = pd.read_csv(PH8_LATENT, low_memory=False)
        source = str(PH8_LATENT)
    else:
        return pd.DataFrame(), ""

    if "point_id" not in sat.columns:
        sat["point_id"] = sat.index.astype(str)

    if "lat" not in sat.columns or "lon" not in sat.columns:
        return pd.DataFrame(), source

    sat["lat"] = num(sat["lat"])
    sat["lon"] = num(sat["lon"])
    sat = sat.dropna(subset=["lat", "lon"]).copy()
    return sat, source


def nearest_satellite_matches(site_summary, sat):
    rows = []
    if site_summary.empty or sat.empty:
        return pd.DataFrame()

    if "tower_lat" not in site_summary.columns or "tower_lon" not in site_summary.columns:
        return pd.DataFrame()

    sites = site_summary.dropna(subset=["tower_lat", "tower_lon"]).copy()
    if sites.empty:
        return pd.DataFrame()

    sat_lat = num(sat["lat"]).values
    sat_lon = num(sat["lon"]).values

    for _, r in sites.iterrows():
        dist = haversine_km(float(r["tower_lat"]), float(r["tower_lon"]), sat_lat, sat_lon)
        if len(dist) == 0:
            continue
        j = int(np.nanargmin(dist))
        sr = sat.iloc[j]

        row = {
            "site": r["site"],
            "tower_lat": r["tower_lat"],
            "tower_lon": r["tower_lon"],
            "nearest_point_id": sr.get("point_id", ""),
            "nearest_point_lat": sr.get("lat", np.nan),
            "nearest_point_lon": sr.get("lon", np.nan),
            "distance_km": float(dist[j]),
            "match_within_25km": bool(dist[j] <= 25),
            "match_within_50km": bool(dist[j] <= 50),
            "match_within_100km": bool(dist[j] <= 100),
        }

        for c in [
            "latent_response_class",
            "latent_satbreak_probability",
            "latent_post_slope",
            "latent_slope_change",
            "event_limitation_hotspot",
            "eco_ecoregion",
            "eco_biome",
            "hydroclimatic_regime",
        ]:
            if c in sr.index:
                row["satellite_" + c] = sr[c]

        rows.append(row)

    return pd.DataFrame(rows)


def compare_tower_satellite(tower_primary, matches):
    if tower_primary.empty or matches.empty:
        return pd.DataFrame()

    comp = tower_primary.merge(matches, on="site", how="left")

    if "satellite_latent_post_slope" in comp.columns:
        comp["tower_post_slope_sign"] = np.sign(num(comp["post_slope"]))
        comp["sat_post_slope_sign"] = np.sign(num(comp["satellite_latent_post_slope"]))
        comp["post_slope_sign_agreement"] = comp["tower_post_slope_sign"].eq(comp["sat_post_slope_sign"])

    if "satellite_latent_slope_change" in comp.columns:
        comp["tower_slope_change_sign"] = np.sign(num(comp["slope_change"]))
        comp["sat_slope_change_sign"] = np.sign(num(comp["satellite_latent_slope_change"]))
        comp["slope_change_sign_agreement"] = comp["tower_slope_change_sign"].eq(comp["sat_slope_change_sign"])

    if "match_within_50km" in comp.columns:
        comp["usable_for_rigorous_validation_now"] = comp["match_within_50km"].fillna(False)
    else:
        comp["usable_for_rigorous_validation_now"] = False

    return comp


if not DAILY_PATH.exists():
    raise SystemExit("Missing Table84_tower_daily_wue_fluxes.csv")

print("READING", DAILY_PATH)
d = pd.read_csv(DAILY_PATH, low_memory=False)

if "site" not in d.columns:
    if "site_x" in d.columns:
        d["site"] = d["site_x"]
    elif "site_y" in d.columns:
        d["site"] = d["site_y"]
    else:
        d["site"] = ""

d["site"] = d["site"].astype("object")
bad_site = d["site"].isna() | d["site"].astype(str).str.strip().isin(["", "nan", "None"])
if bad_site.any():
    inferred_sites = d.loc[bad_site].apply(infer_site_row, axis=1).astype(str).values
    d.loc[bad_site, "site"] = inferred_sites

d["site"] = d["site"].astype(str).str.strip()

if "date" not in d.columns:
    raise SystemExit("Table84 has no date column.")

d["date"] = pd.to_datetime(d["date"], errors="coerce")
d["year"] = d["date"].dt.year
d["doy"] = d["date"].dt.dayofyear

needed_numeric = [
    "gpp_gC_m2_day",
    "et_mm_day",
    "vpd_kpa",
    "wue_gC_per_mm",
    "uwue_gC_kPa05_per_mm",
    "swc",
    "precip_mm_day",
]

for c in needed_numeric:
    if c not in d.columns:
        d[c] = np.nan
    d[c] = num(d[c])

d = d.dropna(
    subset=[
        "site",
        "date",
        "year",
        "doy",
        "gpp_gC_m2_day",
        "et_mm_day",
        "vpd_kpa",
        "wue_gC_per_mm",
        "uwue_gC_kPa05_per_mm",
    ]
).copy()

d = d[d["site"].str.len() > 0].copy()
d["year"] = d["year"].astype(int)
d["doy"] = d["doy"].astype(int)

print("===== FIXED DAILY TABLE =====")
print("rows:", len(d))
print("unique sites:", d["site"].nunique())
print("years:", int(d["year"].min()), "to", int(d["year"].max()))

save_csv(d, TAB / "Table84_tower_daily_wue_fluxes_FIXED_SITE_YEAR.csv")

if META_PATH.exists():
    site_meta = pd.read_csv(META_PATH, low_memory=False)
else:
    site_meta = pd.DataFrame()

if not site_meta.empty:
    if "site" not in site_meta.columns:
        site_meta["site"] = ""
    site_meta["site"] = site_meta["site"].astype(str).str.strip()
    site_meta = site_meta.drop_duplicates("site")

site_year = (
    d.groupby(["site", "year"])
    .agg(
        n_daily=("date", "size"),
        gpp_mean=("gpp_gC_m2_day", "mean"),
        et_sum=("et_mm_day", "sum"),
        vpd_mean=("vpd_kpa", "mean"),
        swc_coverage=("swc", lambda x: float(x.notna().mean())),
        precip_coverage=("precip_mm_day", lambda x: float(x.notna().mean())),
    )
    .reset_index()
)

site_year["passes_min_daily_rows"] = site_year["n_daily"] >= 90

site_summary = (
    site_year.groupby("site")
    .agg(
        years_total=("year", "nunique"),
        years_with_90_daily_rows=("passes_min_daily_rows", "sum"),
        n_daily_total=("n_daily", "sum"),
        mean_daily_rows_per_year=("n_daily", "mean"),
        mean_vpd_kpa=("vpd_mean", "mean"),
        mean_swc_coverage=("swc_coverage", "mean"),
        mean_precip_coverage=("precip_coverage", "mean"),
    )
    .reset_index()
)

if not site_meta.empty:
    keep = [
        c for c in [
            "site",
            "tower_lat",
            "tower_lon",
            "igbp",
            "is_strict_grassland_gra",
            "is_grassland_savanna_extension",
            "igbp_missing",
            "passes_landcover_screen_lenient",
            "metadata_source",
        ]
        if c in site_meta.columns
    ]
    site_summary = site_summary.merge(site_meta[keep], on="site", how="left")

for c in ["tower_lat", "tower_lon"]:
    if c in site_summary.columns:
        site_summary[c] = num(site_summary[c])
    else:
        site_summary[c] = np.nan

if "igbp" not in site_summary.columns:
    site_summary["igbp"] = ""
site_summary["igbp"] = site_summary["igbp"].fillna("").astype(str).str.upper().str.slice(0, 3)

grassland_like = {"GRA", "SAV", "WSA", "OSH", "CSH"}
if "is_strict_grassland_gra" not in site_summary.columns:
    site_summary["is_strict_grassland_gra"] = site_summary["igbp"].eq("GRA")
if "is_grassland_savanna_extension" not in site_summary.columns:
    site_summary["is_grassland_savanna_extension"] = site_summary["igbp"].isin(grassland_like)
if "igbp_missing" not in site_summary.columns:
    site_summary["igbp_missing"] = site_summary["igbp"].isin(["", "NAN", "NONE"])
if "passes_landcover_screen_lenient" not in site_summary.columns:
    site_summary["passes_landcover_screen_lenient"] = site_summary["is_grassland_savanna_extension"] | site_summary["igbp_missing"]

site_summary["passes_record_length_3yr_lenient"] = site_summary["years_with_90_daily_rows"] >= 3
site_summary["usable_for_tower_response_lenient"] = (
    site_summary["passes_record_length_3yr_lenient"]
    & site_summary["passes_landcover_screen_lenient"].fillna(True)
)

save_csv(site_year, TAB / "Table85_tower_site_year_quality.csv")
save_csv(site_summary, TAB / "Table86_tower_usable_site_summary.csv")

d["modis_8day_index"] = ((d["doy"] - 1) // 8).astype(int)
d["year_start"] = pd.to_datetime(d["year"].astype(str) + "-01-01")
d["period_start"] = d["year_start"] + pd.to_timedelta(d["modis_8day_index"] * 8, unit="D")
d["period_end"] = d["period_start"] + pd.to_timedelta(7, unit="D")

tower_8day = (
    d.groupby(["site", "period_start", "period_end", "year", "modis_8day_index"])
    .agg(
        n_valid_days=("date", "size"),
        gpp_8day_gC_m2=("gpp_gC_m2_day", "sum"),
        et_8day_mm=("et_mm_day", "sum"),
        vpd_8day_kpa_mean=("vpd_kpa", "mean"),
        vpd_8day_kpa_max=("vpd_kpa", "max"),
        swc_8day_mean=("swc", "mean"),
        precip_8day_mm=("precip_mm_day", "sum"),
    )
    .reset_index()
)

tower_8day = tower_8day[
    (tower_8day["n_valid_days"] >= 4)
    & (tower_8day["gpp_8day_gC_m2"] > 0)
    & (tower_8day["et_8day_mm"] > 0)
    & tower_8day["vpd_8day_kpa_mean"].notna()
].copy()

tower_8day["wue_8day"] = tower_8day["gpp_8day_gC_m2"] / tower_8day["et_8day_mm"]
tower_8day["uwue_8day"] = (
    tower_8day["gpp_8day_gC_m2"]
    * np.sqrt(tower_8day["vpd_8day_kpa_mean"].clip(lower=0))
    / tower_8day["et_8day_mm"]
)
tower_8day["log_wue_8day"] = np.log(tower_8day["wue_8day"].where(tower_8day["wue_8day"] > 0))
tower_8day["log_uwue_8day"] = np.log(tower_8day["uwue_8day"].where(tower_8day["uwue_8day"] > 0))

if tower_8day.empty:
    raise SystemExit("Still no 8-day tower records.")

tower_8day["vpd_z_site"] = tower_8day.groupby("site")["vpd_8day_kpa_mean"].transform(zscore_site)
tower_8day["swc_z_site"] = tower_8day.groupby("site")["swc_8day_mean"].transform(zscore_site)
tower_8day["dry_swc_z_site"] = -tower_8day["swc_z_site"]
tower_8day["precip_z_site"] = tower_8day.groupby("site")["precip_8day_mm"].transform(zscore_site)
tower_8day["dry_precip_z_site"] = -tower_8day["precip_z_site"]

tower_8day["stress_vpd_only"] = tower_8day["vpd_z_site"]
tower_8day["stress_compound_vpd_swc"] = (tower_8day["vpd_z_site"] + tower_8day["dry_swc_z_site"]) / 2.0
tower_8day["stress_vpd_precip_proxy"] = (tower_8day["vpd_z_site"] + tower_8day["dry_precip_z_site"]) / 2.0

tower_8day["growing_season_proxy"] = False
for site, idx in tower_8day.groupby("site").groups.items():
    g = tower_8day.loc[idx]
    if g["gpp_8day_gC_m2"].notna().sum() >= 8:
        thr = g["gpp_8day_gC_m2"].quantile(0.25)
        tower_8day.loc[idx, "growing_season_proxy"] = g["gpp_8day_gC_m2"] > thr

merge_cols = [
    c for c in [
        "site",
        "tower_lat",
        "tower_lon",
        "igbp",
        "is_strict_grassland_gra",
        "is_grassland_savanna_extension",
        "igbp_missing",
        "passes_landcover_screen_lenient",
        "passes_record_length_3yr_lenient",
        "usable_for_tower_response_lenient",
        "metadata_source",
    ]
    if c in site_summary.columns
]
tower_8day = tower_8day.merge(site_summary[merge_cols], on="site", how="left")

save_csv(tower_8day, TAB / "Table87_tower_8day_wue_stress.csv")

print("===== FITTING TOWER RESPONSE SHAPES =====")
tower_fits_all = classify_tower_sites(tower_8day)

if tower_fits_all.empty:
    save_csv(tower_fits_all, TAB / "Table88_tower_response_phenotypes_all_fits.csv")
    raise SystemExit("8-day records exist, but no response fits were produced.")

tower_fits_all = tower_fits_all.merge(site_summary[merge_cols], on="site", how="left")
save_csv(tower_fits_all, TAB / "Table88_tower_response_phenotypes_all_fits.csv")

tower_primary = tower_fits_all[tower_fits_all["is_primary_tower_fit"]].copy()
save_csv(tower_primary, TAB / "Table89_tower_response_phenotypes_primary_by_site.csv")

by_class = (
    tower_primary.groupby("response_class")
    .agg(
        n_sites=("site", "nunique"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
        median_p_satbreak=("p_tower_saturation_breakdown", "median"),
    )
    .reset_index()
    .sort_values("n_sites", ascending=False)
)
save_csv(by_class, TAB / "Table90_tower_response_class_summary.csv")

by_igbp = (
    tower_primary.groupby("igbp", dropna=False)
    .agg(
        n_sites=("site", "nunique"),
        n_grassland_extension=("is_grassland_savanna_extension", "sum"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
        median_p_satbreak=("p_tower_saturation_breakdown", "median"),
    )
    .reset_index()
    .sort_values("n_sites", ascending=False)
)
save_csv(by_igbp, TAB / "Table91_tower_response_summary_by_igbp.csv")

sat, sat_source = load_satellite_points()
matches = nearest_satellite_matches(site_summary, sat)
save_csv(matches, TAB / "Table92_existing_satellite_point_nearest_tower_matches.csv")

comp = compare_tower_satellite(tower_primary, matches)
save_csv(comp, TAB / "Table93_existing_satellite_tower_provisional_comparison.csv")

if not comp.empty and "usable_for_rigorous_validation_now" in comp.columns:
    close_comp = comp[comp["usable_for_rigorous_validation_now"].fillna(False)].copy()
else:
    close_comp = pd.DataFrame()

if not close_comp.empty:
    rows = []
    for c in ["post_slope_sign_agreement", "slope_change_sign_agreement"]:
        if c in close_comp.columns:
            rows.append(
                {
                    "comparison_metric": c,
                    "n_close_matches": int(close_comp[c].notna().sum()),
                    "agreement_fraction": float(close_comp[c].mean()) if close_comp[c].notna().sum() else np.nan,
                }
            )
    arbitration = pd.DataFrame(rows)
else:
    arbitration = pd.DataFrame(
        [
            {
                "comparison_metric": "none",
                "n_close_matches": 0,
                "agreement_fraction": np.nan,
                "interpretation": "No rigorous tower-satellite arbitration from existing sampled satellite points; tower-centered extraction required.",
            }
        ]
    )
save_csv(arbitration, TAB / "Table94_product_arbitration_status_from_existing_matches.csv")

extraction_targets = site_summary[site_summary["usable_for_tower_response_lenient"].fillna(False)].copy()
if extraction_targets.empty:
    extraction_targets = site_summary.copy()

extraction_targets = extraction_targets.dropna(subset=["tower_lat", "tower_lon"]).copy()
extraction_targets["target_id"] = extraction_targets["site"]
extraction_targets["extract_lat"] = extraction_targets["tower_lat"]
extraction_targets["extract_lon"] = extraction_targets["tower_lon"]
extraction_targets["recommended_satellite_window"] = "tower_pixel_plus_3x3_sensitivity"
extraction_targets["target_reason"] = np.where(
    extraction_targets["usable_for_tower_response_lenient"].fillna(False),
    "usable_tower_response_site",
    "metadata_site_needs_quality_review",
)

target_cols = [
    c for c in [
        "target_id",
        "site",
        "extract_lat",
        "extract_lon",
        "igbp",
        "is_strict_grassland_gra",
        "is_grassland_savanna_extension",
        "years_total",
        "years_with_90_daily_rows",
        "n_daily_total",
        "mean_vpd_kpa",
        "mean_swc_coverage",
        "mean_precip_coverage",
        "recommended_satellite_window",
        "target_reason",
    ]
    if c in extraction_targets.columns
]
save_csv(extraction_targets[target_cols], TAB / "Table95_tower_centered_satellite_extraction_targets.csv")

coord = extraction_targets[["site", "extract_lat", "extract_lon"]].rename(
    columns={"site": "id", "extract_lat": "latitude", "extract_lon": "longitude"}
)
save_csv(coord, TAB / "tower_centered_coordinates_for_satellite_extraction.csv")

try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = tower_primary["response_class"].value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_ylabel("Tower sites")
    ax.set_title("Tower response classes")
    ax.tick_params(axis="x", rotation=25)
    savefig(FIG / "Figure1_tower_response_class_counts.png")
except Exception as e:
    print("Figure1 skipped:", e)

try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(tower_primary["post_slope"], tower_primary["slope_change"], s=55, alpha=0.8)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Tower high-stress/post-transition slope")
    ax.set_ylabel("Tower slope change")
    ax.set_title("Tower WUE response phenotype")
    savefig(FIG / "Figure2_tower_post_slope_vs_slope_change.png")
except Exception as e:
    print("Figure2 skipped:", e)

try:
    if not matches.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(matches["tower_lon"], matches["tower_lat"], label="Tower sites", s=55)
        if not sat.empty:
            ax.scatter(sat["lon"], sat["lat"], label="Existing satellite sample points", s=20, alpha=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Tower sites vs existing satellite sample points")
        ax.legend(frameon=False)
        savefig(FIG / "Figure3_tower_sites_vs_existing_satellite_points.png")
except Exception as e:
    print("Figure3 skipped:", e)

try:
    if not comp.empty and "satellite_latent_post_slope" in comp.columns:
        plot = comp.dropna(subset=["post_slope", "satellite_latent_post_slope"]).copy()
        if len(plot):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(plot["satellite_latent_post_slope"], plot["post_slope"], s=55)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_xlabel("Nearest satellite latent post-slope")
            ax.set_ylabel("Tower post-slope")
            ax.set_title("Provisional tower-satellite comparison")
            savefig(FIG / "Figure4_provisional_tower_satellite_post_slope.png")
except Exception as e:
    print("Figure4 skipped:", e)

n_candidate_sites = int(site_summary["site"].nunique())
n_sites_daily = int(d["site"].nunique())
n_sites_8day = int(tower_8day["site"].nunique())
n_fit_sites = int(tower_primary["site"].nunique())
n_close_50 = int(matches["match_within_50km"].sum()) if not matches.empty and "match_within_50km" in matches.columns else 0
n_extraction_targets = int(len(extraction_targets))
class_counts = {str(k): int(v) for k, v in tower_primary["response_class"].value_counts().to_dict().items()}

verdict = {
    "candidate_sites_from_fixed_daily_table": n_candidate_sites,
    "sites_with_daily_wue": n_sites_daily,
    "sites_with_8day_wue": n_sites_8day,
    "sites_with_primary_tower_response_fit": n_fit_sites,
    "primary_tower_response_class_counts": class_counts,
    "existing_satellite_source": sat_source,
    "existing_satellite_points": int(len(sat)) if not sat.empty else 0,
    "existing_satellite_matches_within_50km": n_close_50,
    "tower_centered_satellite_extraction_targets": n_extraction_targets,
    "can_claim_tower_flux_phenotype_now": bool(n_fit_sites >= 10),
    "requires_tower_centered_satellite_extraction": bool(n_close_50 < 5),
}
(OUT / "phase17_tower_validation_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

class_counts_text = pd.Series(class_counts).to_string() if class_counts else "No tower classes."
validation_status = "The broad tower inventory now supports construction of an independent tower-observed WUE/uWUE flux phenotype. Direct satellite-vs-tower validation still requires tower-centered satellite extraction if existing sampled satellite points are not colocated with enough towers."

text_lines = [
    "# Phase 17 broad tower validation verdict",
    "",
    "## Summary numbers",
    "",
    f"- Sites with parsed daily tower WUE: `{n_sites_daily}`",
    f"- Sites with 8-day tower WUE: `{n_sites_8day}`",
    f"- Sites with primary tower response fits: `{n_fit_sites}`",
    f"- Existing satellite sample points: `{verdict.get('existing_satellite_points')}`",
    f"- Existing tower-satellite matches within 50 km: `{n_close_50}`",
    f"- Tower-centered satellite extraction targets: `{n_extraction_targets}`",
    "",
    "## Tower response class counts",
    "",
    "```text",
    class_counts_text,
    "```",
    "",
    "## Interpretation",
    "",
    validation_status,
    "",
    "## Manuscript-safe wording",
    "",
    "Tower observations provide an independent ecosystem-flux check on the satellite-derived WUE response phenotype. We computed tower WUE and uWUE from eddy-covariance GPP and latent heat flux, constructed tower-level high-stress indices from VPD and soil moisture or precipitation proxies, and classified tower response shape using the same slope-change logic as the satellite analysis. These tower results test whether the product-adjusted satellite phenotype corresponds to observed ecosystem carbon-water flux behavior.",
    "",
    "## Next step",
    "",
    "Use Table95_tower_centered_satellite_extraction_targets.csv and tower_centered_coordinates_for_satellite_extraction.csv to extract satellite GPP, ET, VPD, soil moisture, and WUE products at the tower coordinates. Then rerun satellite response classification at the tower sites and compare tower-vs-satellite slopes, slope changes, and classes.",
]
text = "\n".join(text_lines) + "\n"

save_text(text, TXT / "PHASE17_TOWER_VALIDATION_VERDICT.md")
save_text(text, OUT / "README_phase17_broad_tower_validation.md")

methods = "Broad tower validation used daily AmeriFlux/FLUXNET files with GPP, LE, and VPD columns. Daily ET was computed from latent heat flux as ET = LE * 86400 / 2.45e6. Daily WUE was computed as GPP / ET and uWUE as GPP * sqrt(VPD) / ET after converting VPD to kPa. Daily observations were aggregated to 8-day periods, and tower response shape was estimated using segmented regressions of log WUE or log uWUE against site-standardized high-stress indices. Tower-satellite comparison against existing sampled satellite points is only diagnostic unless tower-centered satellite extraction is performed."
save_text(methods, TXT / "METHODS_BROAD_TOWER_VALIDATION.md")

summary_rows = [{"metric": k, "value": v} for k, v in verdict.items() if not isinstance(v, dict)]
save_csv(pd.DataFrame(summary_rows), TAB / "Table96_phase17_verdict_summary.csv")

print("")
print("==============================")
print("PHASE 17 TOWER VALIDATION VERDICT")
print("==============================")
print(json.dumps(verdict, indent=2, default=str))
print("")
print(text)
