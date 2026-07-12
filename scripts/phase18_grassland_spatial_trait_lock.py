#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phase 18: Grassland-limited tower validation + spatial/characteristic/trait narrowing.

Run after Phase 17 has created:
  results/tower_validation_broad_inventory/tables/Table89_tower_response_phenotypes_primary_by_site.csv
  results/tower_validation_broad_inventory/tables/Table86_tower_usable_site_summary.csv
  results/tower_validation_broad_inventory/tables/Table87_tower_8day_wue_stress.csv
  results/tower_validation_broad_inventory/tables/Table81_selected_daily_tower_files.csv

Purpose:
  1. Repair landcover labels where possible.
  2. Define three validation scopes:
       all tower response sites
       strict grassland: IGBP == GRA
       expanded open grassland/savanna/shrubland: GRA/SAV/WSA/OSH/CSH and/or ecoregion biome says grassland/savanna/shrubland
  3. Annotate tower sites with country/ecoregion/biome using local Natural Earth + RESOLVE ecoregion layers.
  4. Add provisional nearest satellite-point trait/environment proxies.
  5. Test which features separate tower saturation/breakdown vs enhancement:
       VPD, latitude, aridity proxy, soil moisture availability, ecoregion, biome, nearest satellite trait proxies.
  6. Produce satellite extraction target files for:
       all 49 tower response sites
       strict grassland towers
       expanded grassland/savanna/open towers
       high-priority trait/spatial candidate towers
"""

from pathlib import Path
import re
import json
import math
import zipfile
import warnings
import subprocess
import sys

warnings.filterwarnings("ignore")

def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        return True
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
            __import__(import_name)
            return True
        except Exception as e:
            print(f"WARNING: could not import/install {pip_name}: {e}")
            return False

for import_name, pip_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
]:
    ensure_package(import_name, pip_name)

GEOPANDAS_OK = True
for import_name, pip_name in [
    ("geopandas", "geopandas"),
    ("shapely", "shapely"),
    ("pyproj", "pyproj"),
    ("pyogrio", "pyogrio"),
]:
    ok = ensure_package(import_name, pip_name)
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
    print("WARNING: geopandas unavailable; spatial ecoregion annotation will be skipped.")
    print(e)

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


ROOT = Path(".")
PH17 = Path("results/tower_validation_broad_inventory")
PH17_TAB = PH17 / "tables"

IN_FITS = PH17_TAB / "Table89_tower_response_phenotypes_primary_by_site.csv"
IN_ALL_FITS = PH17_TAB / "Table88_tower_response_phenotypes_all_fits.csv"
IN_SITE_SUMMARY = PH17_TAB / "Table86_tower_usable_site_summary.csv"
IN_8DAY = PH17_TAB / "Table87_tower_8day_wue_stress.csv"
IN_SELECTED = PH17_TAB / "Table81_selected_daily_tower_files.csv"
IN_TABLE82 = PH17_TAB / "Table82_tower_site_metadata_extracted.csv"

PH16_POINTS = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")
PH8_LATENT = Path("results/trait_framework/phase8/table_latent_response_by_point.csv")
TRAIT_DATA = Path("results/trait_framework/trait_model_dataset.csv")

GEO_DIR = Path("data/external/geography")
COUNTRIES_ZIP = GEO_DIR / "ne_50m_admin_0_countries.zip"
ECOREGIONS_ZIP = GEO_DIR / "Ecoregions2017.zip"

OUT = Path("results/tower_grassland_spatial_trait_lock")
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"

for p in [OUT, TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)


VALID_IGBP = {
    "ENF", "EBF", "DNF", "DBF", "MF",
    "CSH", "OSH", "WSA", "SAV", "GRA",
    "WET", "CRO", "URB", "CVM", "SNO", "BSV", "WAT"
}

STRICT_GRASS_IGBP = {"GRA"}
EXPANDED_GRASS_SAVANNA_IGBP = {"GRA", "SAV", "WSA", "OSH", "CSH"}
FOREST_IGBP = {"ENF", "EBF", "DNF", "DBF", "MF"}
NON_TARGET_IGBP = {"WET", "CRO", "URB", "CVM", "SNO", "WAT"}


def die(msg):
    raise SystemExit(f"\nERROR: {msg}\n")


def read_csv(path, required=True):
    if not Path(path).exists():
        if required:
            die(f"Missing required file: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df.loc[:, ~df.columns.duplicated()].copy()


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


def num(s):
    return pd.to_numeric(s, errors="coerce")


def fmt(x, d=3):
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{d}f}"
    except Exception:
        return "NA"


def pct(x):
    try:
        if pd.isna(x):
            return "NA"
        return f"{100 * float(x):.1f}%"
    except Exception:
        return "NA"


def clean_igbp(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)

    # Common broken parser artifact.
    if s in {"", "NAN", "NONE", "NULL", "IGB", "IGBP"}:
        return ""

    # Try exact 3-letter valid code.
    if s in VALID_IGBP:
        return s

    # Sometimes metadata value has extra text; find a valid isolated code.
    for code in sorted(VALID_IGBP, key=len, reverse=True):
        if re.search(rf"\b{code}\b", str(x).upper()):
            return code

    # As a last resort, first three chars only if valid.
    if len(s) >= 3 and s[:3] in VALID_IGBP:
        return s[:3]

    return ""


def infer_site_from_text(*parts):
    text = " ".join([str(x) for x in parts if x is not None])
    pats = [
        r"AMF_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"FLX_([A-Z]{2}-[A-Za-z0-9]{3})_",
        r"\b([A-Z]{2}-[A-Za-z0-9]{3})\b",
    ]
    for pat in pats:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return ""


def fix_site_column(df):
    d = df.copy()
    if "site" not in d.columns:
        d["site"] = ""
    d["site"] = d["site"].astype("object")
    source_cols = [c for c in ["source_zip", "source_member", "zip_path", "zip_name", "member"] if c in d.columns]
    site_as_str = d["site"].astype(str)
    bad = (
        d["site"].isna()
        | site_as_str.str.lower().isin(["", "nan", "none", "null", "<na>"])
        | site_as_str.str.fullmatch(r"\d+(\.0)?", na=False)
    )
    if bad.any():
        def infer_row(row):
            return infer_site_from_text(*[row.get(c, "") for c in source_cols])
        d.loc[bad, "site"] = d.loc[bad].apply(infer_row, axis=1).astype("object").values
    d["site"] = d["site"].astype(str).str.strip()
    return d


def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def get_valid_igbp_from_text(text):
    lines = str(text).splitlines()

    # Prefer line-level extraction to avoid capturing the word IGBP as a value.
    for line in lines:
        upper = line.upper()
        if "IGBP" not in upper and "LAND_COVER" not in upper and "VEG" not in upper:
            continue

        # CSV-ish line: VARIABLE,DATAVALUE
        fields = re.split(r"[,;\t|]", upper)
        fields = [f.strip().strip('"').strip("'") for f in fields if f.strip()]
        for f in fields[::-1]:
            code = clean_igbp(f)
            if code:
                return code

        # Generic token scan after IGBP mention.
        pos = upper.find("IGBP")
        if pos >= 0:
            tail = upper[pos:pos + 250]
            for code in VALID_IGBP:
                if re.search(rf"\b{code}\b", tail):
                    return code

    return ""


def regex_float_from_text(text, keys):
    text = str(text)
    for key in keys:
        pat = rf"{re.escape(key)}[^-\d\.]{{0,80}}(-?\d+\.?\d*)"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan


def extract_metadata_from_zip(zip_path, site):
    meta = {
        "site": site,
        "parsed_tower_lat": np.nan,
        "parsed_tower_lon": np.nan,
        "parsed_igbp": "",
        "parsed_metadata_source": "",
    }

    zip_path = Path(str(zip_path))
    if not zip_path.exists():
        meta["parsed_metadata_source"] = "zip_not_found"
        return meta

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            members = z.namelist()
            candidates = [
                m for m in members
                if m.lower().endswith((".csv", ".txt"))
                and any(k in m.lower() for k in ["badm", "bif", "meta", "site"])
            ]
            candidates = sorted(
                candidates,
                key=lambda x: (0 if ("badm" in x.lower() or "bif" in x.lower()) else 1, x)
            )

            for member in candidates:
                try:
                    raw = z.open(member).read(2_000_000)
                    text = raw.decode("utf-8", errors="ignore")

                    parsed_igbp = get_valid_igbp_from_text(text)
                    parsed_lat = regex_float_from_text(text, ["LOCATION_LAT", "LATITUDE", "SITE_LAT"])
                    parsed_lon = regex_float_from_text(text, ["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON"])

                    # Try structured CSV too.
                    try:
                        dfm = pd.read_csv(z.open(member), low_memory=False)
                        cols = list(dfm.columns)

                        lat_col = pick_col(cols, ["LOCATION_LAT", "LATITUDE", "SITE_LAT", "LAT"])
                        lon_col = pick_col(cols, ["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON", "LON"])
                        igbp_col = pick_col(cols, ["IGBP", "LAND_COVER_IGBP", "IGBP_CLASS", "LOCATION_IGBP"])

                        if lat_col and lon_col:
                            lat_vals = num(dfm[lat_col]).dropna()
                            lon_vals = num(dfm[lon_col]).dropna()
                            if len(lat_vals):
                                parsed_lat = float(lat_vals.iloc[0])
                            if len(lon_vals):
                                parsed_lon = float(lon_vals.iloc[0])

                        if igbp_col:
                            vals = dfm[igbp_col].dropna().astype(str)
                            for v in vals:
                                code = clean_igbp(v)
                                if code:
                                    parsed_igbp = code
                                    break

                        var_col = pick_col(cols, ["VARIABLE", "VAR", "BADM_VARIABLE", "DATAVARIABLE"])
                        val_col = pick_col(cols, ["DATAVALUE", "VALUE", "DATA_VALUE", "VAR_VALUE"])

                        if var_col and val_col:
                            dd = dfm[[var_col, val_col]].dropna().copy()
                            dd[var_col] = dd[var_col].astype(str).str.upper()
                            dd[val_col] = dd[val_col].astype(str)

                            def find_long(keys):
                                keys = [k.upper() for k in keys]
                                exact = dd[dd[var_col].isin(keys)]
                                if len(exact):
                                    return exact[val_col].iloc[0]
                                for key in keys:
                                    sub = dd[dd[var_col].str.contains(key, regex=False, na=False)]
                                    if len(sub):
                                        return sub[val_col].iloc[0]
                                return None

                            lat_v = find_long(["LOCATION_LAT", "LATITUDE", "SITE_LAT"])
                            lon_v = find_long(["LOCATION_LONG", "LOCATION_LON", "LONGITUDE", "SITE_LON"])
                            igbp_v = find_long(["IGBP", "LOCATION_IGBP", "LAND_COVER_IGBP", "IGBP_CLASS"])

                            if lat_v is not None:
                                try:
                                    parsed_lat = float(lat_v)
                                except Exception:
                                    pass
                            if lon_v is not None:
                                try:
                                    parsed_lon = float(lon_v)
                                except Exception:
                                    pass
                            if igbp_v is not None:
                                code = clean_igbp(igbp_v)
                                if code:
                                    parsed_igbp = code

                    except Exception:
                        pass

                    if pd.notna(parsed_lat) and pd.notna(parsed_lon) or parsed_igbp:
                        meta["parsed_tower_lat"] = parsed_lat
                        meta["parsed_tower_lon"] = parsed_lon
                        meta["parsed_igbp"] = parsed_igbp
                        meta["parsed_metadata_source"] = member
                        return meta

                except Exception:
                    continue

    except Exception as e:
        meta["parsed_metadata_source"] = "zip_error:" + str(e)
        return meta

    meta["parsed_metadata_source"] = "no_metadata_found"
    return meta


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


def annotate_spatial(points):
    out = points.copy()

    for c in [
        "geo_country", "geo_continent", "geo_subregion",
        "eco_ecoregion", "eco_biome", "eco_realm", "eco_nnh_name"
    ]:
        if c not in out.columns:
            out[c] = ""

    if not GEOPANDAS_OK:
        return out

    gdf = gpd.GeoDataFrame(
        out.copy(),
        geometry=[Point(xy) for xy in zip(out["tower_lon"], out["tower_lat"])],
        crs="EPSG:4326",
    )

    if COUNTRIES_ZIP.exists():
        try:
            countries = gpd.read_file(f"zip://{COUNTRIES_ZIP}").to_crs("EPSG:4326")
            country_fields = {
                "geo_country": ["ADMIN", "NAME", "NAME_LONG"],
                "geo_continent": ["CONTINENT"],
                "geo_subregion": ["SUBREGION", "REGION_UN"],
            }

            keep = []
            field_map = {}
            for new_col, candidates in country_fields.items():
                col = pick_col(list(countries.columns), candidates)
                field_map[new_col] = col
                if col:
                    keep.append(col)
            keep = list(dict.fromkeys(keep))

            if keep:
                joined = gpd.sjoin(gdf[["site", "geometry"]], countries[keep + ["geometry"]], how="left", predicate="within")
                joined = joined[~joined.index.duplicated(keep="first")]
                for new_col, old_col in field_map.items():
                    if old_col and old_col in joined.columns:
                        out.loc[joined.index, new_col] = joined[old_col].values
        except Exception as e:
            print("WARNING: country annotation failed:", e)

    if ECOREGIONS_ZIP.exists():
        try:
            eco = gpd.read_file(f"zip://{ECOREGIONS_ZIP}").to_crs("EPSG:4326")
            eco_fields = {
                "eco_ecoregion": ["ECO_NAME", "ECOREGION", "ECO_NAME_1", "name"],
                "eco_biome": ["BIOME_NAME", "BIOME", "MHT_NAME"],
                "eco_realm": ["REALM", "REALM_NAME"],
                "eco_nnh_name": ["NNH_NAME"],
            }

            keep = []
            field_map = {}
            for new_col, candidates in eco_fields.items():
                col = pick_col(list(eco.columns), candidates)
                field_map[new_col] = col
                if col:
                    keep.append(col)
            keep = list(dict.fromkeys(keep))

            if keep:
                joined = gpd.sjoin(gdf[["site", "geometry"]], eco[keep + ["geometry"]], how="left", predicate="within")
                joined = joined[~joined.index.duplicated(keep="first")]
                for new_col, old_col in field_map.items():
                    if old_col and old_col in joined.columns:
                        out.loc[joined.index, new_col] = joined[old_col].values
        except Exception as e:
            print("WARNING: ecoregion annotation failed:", e)

    return out


def load_satellite_points():
    if PH16_POINTS.exists():
        sat = read_csv(PH16_POINTS, required=True)
        source = str(PH16_POINTS)
    elif PH8_LATENT.exists():
        sat = read_csv(PH8_LATENT, required=True)
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


def attach_nearest_satellite_trait_proxy(towers):
    sat, source = load_satellite_points()

    if sat.empty:
        towers["nearest_satellite_source"] = ""
        towers["nearest_satellite_distance_km"] = np.nan
        return towers, source

    sat_lat = sat["lat"].values
    sat_lon = sat["lon"].values

    attach_cols = [
        "point_id", "lat", "lon",
        "latent_response_class",
        "latent_satbreak_probability",
        "latent_post_slope",
        "latent_slope_change",
        "event_limitation_hotspot",
        "eco_ecoregion",
        "eco_biome",
        "geo_country",
        "hydroclimatic_regime",
        "high_vpd_gt_2p26",
        "region_Sahel",
        "mean_vpd",
        "aridity",
        "mean_soil_moisture",
        "mean_annual_temperature",
        "mean_annual_precipitation",
        "mean_lai",
        "rooting_depth",
        "p50",
        "isohydricity",
        "soil_sand",
        "soil_clay",
        "soil_silt",
    ]
    attach_cols = [c for c in attach_cols if c in sat.columns]

    rows = []
    for _, r in towers.iterrows():
        if pd.isna(r.get("tower_lat")) or pd.isna(r.get("tower_lon")):
            rows.append({})
            continue

        dist = haversine_km(float(r["tower_lat"]), float(r["tower_lon"]), sat_lat, sat_lon)
        j = int(np.nanargmin(dist))
        sr = sat.iloc[j]
        row = {
            "nearest_satellite_source": source,
            "nearest_satellite_distance_km": float(dist[j]),
        }
        for c in attach_cols:
            row["nearest_sat_" + c] = sr[c]
        rows.append(row)

    add = pd.DataFrame(rows)
    out = pd.concat([towers.reset_index(drop=True), add.reset_index(drop=True)], axis=1)
    return out, source


def mann_auc(a, b):
    a = num(pd.Series(a)).dropna()
    b = num(pd.Series(b)).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan

    try:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = float(u.statistic / (len(a) * len(b)))
        return float(u.pvalue), auc
    except Exception:
        return np.nan, np.nan


def fisher_rr(mask, event):
    mask = pd.Series(mask).fillna(False).astype(bool)
    event = pd.Series(event).fillna(False).astype(bool)

    if mask.sum() == 0 or (~mask).sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, 0, 0

    a = int((mask & event).sum())
    b = int((mask & ~event).sum())
    c = int((~mask & event).sum())
    d = int((~mask & ~event).sum())

    frac_in = a / (a + b) if (a + b) else np.nan
    frac_out = c / (c + d) if (c + d) else np.nan
    rr = frac_in / frac_out if frac_out and frac_out > 0 else np.nan

    try:
        odds, p = stats.fisher_exact([[a, b], [c, d]])
    except Exception:
        odds, p = np.nan, np.nan

    return frac_in, frac_out, rr, p, a, a + b


def bh_q(pvals):
    p = num(pd.Series(pvals))
    out = pd.Series(np.nan, index=p.index, dtype=float)
    ok = p.notna()
    vals = p[ok].values
    if len(vals) == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty_like(q)
    res[order] = q
    out.loc[p[ok].index] = res
    return out


def summarize_subset(df, subset_name, mask):
    d = df[mask].copy()
    if d.empty:
        return {
            "subset": subset_name,
            "n_sites": 0,
        }

    rows = {
        "subset": subset_name,
        "n_sites": int(len(d)),
        "n_saturation": int(d["tower_satbreak_event"].sum() if "tower_satbreak_event" in d.columns else 0),
        "satbreak_fraction": float(d["tower_satbreak_event"].mean() if "tower_satbreak_event" in d.columns else np.nan),
        "n_breakdown_only": int((d["response_class"] == "breakdown").sum()),
        "n_enhancement": int((d["response_class"] == "enhancement").sum()),
        "n_inconclusive": int((d["response_class"] == "inconclusive").sum()),
        "median_post_slope": float(num(d["post_slope"]).median()),
        "median_slope_change": float(num(d["slope_change"]).median()),
        "median_p_tower_satbreak": float(num(d["p_tower_saturation_breakdown"]).median()),
        "median_vpd_mean_kpa": float(num(d.get("vpd_mean_kpa", np.nan)).median()),
        "median_vpd_p90_kpa": float(num(d.get("vpd_p90_kpa", np.nan)).median()),
        "median_abs_lat": float(num(d.get("abs_lat", np.nan)).median()),
    }
    return rows


def characteristic_tests(df, subset_name, mask):
    d = df[mask].copy()
    rows = []

    if len(d) < 4:
        return pd.DataFrame()

    d = d[d["response_class"].isin(["saturation", "breakdown", "enhancement"])].copy()
    if d.empty:
        return pd.DataFrame()

    event = d["response_class"].isin(["saturation", "breakdown"])

    variables = [
        "vpd_mean_kpa",
        "vpd_p90_kpa",
        "post_slope",
        "slope_change",
        "p_tower_saturation_breakdown",
        "n_fit_8day",
        "n_years",
        "abs_lat",
        "tower_lat",
        "tower_lon",
        "nearest_sat_mean_vpd",
        "nearest_sat_aridity",
        "nearest_sat_mean_soil_moisture",
        "nearest_sat_mean_annual_temperature",
        "nearest_sat_mean_annual_precipitation",
        "nearest_sat_mean_lai",
        "nearest_sat_rooting_depth",
        "nearest_sat_p50",
        "nearest_sat_isohydricity",
        "nearest_sat_soil_sand",
        "nearest_sat_soil_clay",
        "nearest_sat_soil_silt",
        "nearest_sat_latent_satbreak_probability",
        "nearest_sat_latent_post_slope",
        "nearest_sat_latent_slope_change",
        "nearest_satellite_distance_km",
    ]

    variables = [v for v in variables if v in d.columns]

    for v in variables:
        a = num(d.loc[event, v]).dropna()
        b = num(d.loc[~event, v]).dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        p, auc = mann_auc(a, b)

        rows.append({
            "subset": subset_name,
            "variable": v,
            "n_satbreak": int(len(a)),
            "n_enhancement": int(len(b)),
            "median_satbreak": float(a.median()),
            "median_enhancement": float(b.median()),
            "median_diff_satbreak_minus_enhancement": float(a.median() - b.median()),
            "mann_p": p,
            "auc_satbreak_gt_enhancement": auc,
            "direction": "higher_in_satbreak" if a.median() > b.median() else "lower_in_satbreak",
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["mann_q"] = bh_q(out["mann_p"])
    return out


def group_enrichment(df, subset_name, mask):
    d = df[mask].copy()
    rows = []

    if len(d) < 4:
        return pd.DataFrame()

    event = d["tower_satbreak_event"]

    group_cols = [
        "igbp_final",
        "geo_country",
        "geo_continent",
        "geo_subregion",
        "eco_ecoregion",
        "eco_biome",
        "eco_realm",
        "nearest_sat_eco_ecoregion",
        "nearest_sat_eco_biome",
        "nearest_sat_hydroclimatic_regime",
    ]

    group_cols = [c for c in group_cols if c in d.columns]

    for c in group_cols:
        vals = d[c].fillna("").astype(str)
        for level in sorted(vals.unique()):
            if level.strip() == "" or level.lower() in ["nan", "none", "null"]:
                continue
            m = vals.eq(level)
            if m.sum() < 2 or (~m).sum() < 2:
                continue

            frac_in, frac_out, rr, p, n_event_in, n_in = fisher_rr(m, event)

            rows.append({
                "subset": subset_name,
                "group_col": c,
                "level": level,
                "n_group": int(n_in),
                "n_satbreak_in_group": int(n_event_in),
                "satbreak_fraction_group": frac_in,
                "satbreak_fraction_rest": frac_out,
                "risk_ratio": rr,
                "fisher_p": p,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fisher_q"] = bh_q(out["fisher_p"])
        out = out.sort_values(["fisher_q", "risk_ratio"], ascending=[True, False])
    return out


def candidate_rules(df, subset_name, mask):
    d = df[mask].copy()
    rows = []

    if len(d) < 8:
        return pd.DataFrame()

    event = d["tower_satbreak_event"]

    variables = [
        "vpd_mean_kpa",
        "vpd_p90_kpa",
        "abs_lat",
        "nearest_sat_mean_vpd",
        "nearest_sat_aridity",
        "nearest_sat_mean_soil_moisture",
        "nearest_sat_mean_annual_temperature",
        "nearest_sat_mean_annual_precipitation",
        "nearest_sat_mean_lai",
        "nearest_sat_rooting_depth",
        "nearest_sat_p50",
        "nearest_sat_soil_sand",
        "nearest_sat_soil_clay",
        "nearest_sat_soil_silt",
    ]
    variables = [v for v in variables if v in d.columns]

    for v in variables:
        x = num(d[v])
        if x.notna().sum() < 8:
            continue

        for q in [0.25, 0.33, 0.50, 0.67, 0.75]:
            thr = x.quantile(q)
            for op in [">=", "<="]:
                if op == ">=":
                    m = x >= thr
                else:
                    m = x <= thr

                if m.sum() < 3 or (~m).sum() < 3:
                    continue

                frac_in, frac_out, rr, p, n_event_in, n_in = fisher_rr(m, event)

                rows.append({
                    "subset": subset_name,
                    "rule": f"{v} {op} {thr:.5g}",
                    "variable": v,
                    "operator": op,
                    "threshold": float(thr),
                    "n_rule": int(n_in),
                    "n_satbreak_rule": int(n_event_in),
                    "satbreak_fraction_rule": frac_in,
                    "satbreak_fraction_else": frac_out,
                    "risk_ratio": rr,
                    "fisher_p": p,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["fisher_q"] = bh_q(out["fisher_p"])
        out = out.sort_values(["fisher_q", "risk_ratio"], ascending=[True, False])
    return out


def decision_tree_rules(df, subset_name, mask):
    if not SKLEARN_OK:
        return pd.DataFrame(), "sklearn unavailable"

    d = df[mask].copy()
    d = d[d["response_class"].isin(["saturation", "breakdown", "enhancement"])].copy()
    if len(d) < 12:
        return pd.DataFrame(), "too few sites for decision tree"

    y = d["tower_satbreak_event"].astype(int)

    candidate_features = [
        "vpd_mean_kpa",
        "vpd_p90_kpa",
        "abs_lat",
        "nearest_sat_mean_vpd",
        "nearest_sat_aridity",
        "nearest_sat_mean_soil_moisture",
        "nearest_sat_mean_annual_temperature",
        "nearest_sat_mean_annual_precipitation",
        "nearest_sat_mean_lai",
        "nearest_sat_rooting_depth",
        "nearest_sat_p50",
        "nearest_sat_soil_sand",
        "nearest_sat_soil_clay",
        "nearest_sat_soil_silt",
    ]
    features = [f for f in candidate_features if f in d.columns and num(d[f]).notna().sum() >= 8]

    if len(features) < 2:
        return pd.DataFrame(), "too few numeric features"

    X = d[features].apply(num)
    X = X.fillna(X.median(numeric_only=True))

    if y.nunique() < 2:
        return pd.DataFrame(), "only one response class"

    clf = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=max(2, int(math.ceil(len(d) * 0.12))),
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, y)

    txt = export_text(clf, feature_names=features)

    imp = pd.DataFrame({
        "subset": subset_name,
        "feature": features,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return imp, txt


def make_coordinate_file(df, mask, filename_base, priority_label):
    d = df[mask].dropna(subset=["tower_lat", "tower_lon"]).copy()
    d["target_id"] = d["site"]
    d["latitude"] = d["tower_lat"]
    d["longitude"] = d["tower_lon"]
    d["satellite_extraction_priority"] = priority_label

    rich_cols = [
        "target_id", "site", "latitude", "longitude",
        "igbp_final", "igbp_final_source",
        "is_strict_grassland_tower",
        "is_expanded_grassland_savanna_tower",
        "is_ecoregion_grassland_savanna_tower",
        "response_class",
        "post_slope",
        "slope_change",
        "p_tower_saturation_breakdown",
        "tower_metric",
        "stress_method",
        "n_fit_8day",
        "n_years",
        "vpd_mean_kpa",
        "vpd_p90_kpa",
        "geo_country",
        "eco_ecoregion",
        "eco_biome",
        "nearest_sat_rooting_depth",
        "nearest_sat_p50",
        "nearest_sat_aridity",
        "nearest_sat_mean_vpd",
        "nearest_satellite_distance_km",
        "satellite_extraction_priority",
    ]
    rich_cols = [c for c in rich_cols if c in d.columns]

    coords = d[["target_id", "latitude", "longitude"]].copy()

    out_rich = TAB / f"{filename_base}.csv"
    out_coords = TAB / f"{filename_base}_coordinates_only.csv"

    save_csv(d[rich_cols], out_rich)
    save_csv(coords, out_coords)

    return out_rich, out_coords, len(d)


def main():
    if not IN_FITS.exists():
        die(f"Missing {IN_FITS}. Run Phase 17 first.")

    fits = read_csv(IN_FITS)
    fits = fix_site_column(fits)

    summary = read_csv(IN_SITE_SUMMARY, required=False)
    if not summary.empty:
        summary = fix_site_column(summary)

    eight = read_csv(IN_8DAY, required=False)
    if not eight.empty:
        eight = fix_site_column(eight)

    selected = read_csv(IN_SELECTED, required=False)
    if not selected.empty:
        selected = fix_site_column(selected)

    table82 = read_csv(IN_TABLE82, required=False)
    if not table82.empty:
        table82 = fix_site_column(table82)

    # Base merge.
    df = fits.copy()

    if not summary.empty:
        keep = [
            "site", "tower_lat", "tower_lon", "igbp",
            "years_total", "years_with_90_daily_rows",
            "n_daily_total", "mean_vpd_kpa", "mean_swc_coverage",
            "mean_precip_coverage", "metadata_source"
        ]
        keep = [c for c in keep if c in summary.columns]
        df = df.merge(summary[keep], on="site", how="left", suffixes=("", "_summary"))

    # Prefer fit values, then summary values.
    for c in ["tower_lat", "tower_lon", "igbp", "metadata_source"]:
        csum = c + "_summary"
        if c not in df.columns:
            df[c] = np.nan if c in ["tower_lat", "tower_lon"] else ""
        if csum in df.columns:
            df[c] = df[c].replace(["", "nan", "NaN", "None"], np.nan).combine_first(df[csum])

    # Reparse metadata from zips.
    parsed_rows = []
    if not selected.empty and "zip_path" in selected.columns:
        selected_site = selected.drop_duplicates("site")
        for _, r in selected_site.iterrows():
            parsed_rows.append(extract_metadata_from_zip(r["zip_path"], r["site"]))
    parsed = pd.DataFrame(parsed_rows) if parsed_rows else pd.DataFrame(columns=["site", "parsed_tower_lat", "parsed_tower_lon", "parsed_igbp", "parsed_metadata_source"])

    save_csv(parsed, TAB / "Table100_reparsed_tower_metadata_from_zip.csv")

    df = df.merge(parsed, on="site", how="left")

    # Final coordinate and IGBP choice.
    df["tower_lat"] = num(df["tower_lat"]).combine_first(num(df.get("parsed_tower_lat", np.nan)))
    df["tower_lon"] = num(df["tower_lon"]).combine_first(num(df.get("parsed_tower_lon", np.nan)))

    existing_igbp = df.get("igbp", "").apply(clean_igbp) if "igbp" in df.columns else pd.Series("", index=df.index)
    parsed_igbp = df.get("parsed_igbp", "").apply(clean_igbp) if "parsed_igbp" in df.columns else pd.Series("", index=df.index)

    df["igbp_existing_clean"] = existing_igbp
    df["igbp_parsed_clean"] = parsed_igbp
    df["igbp_final"] = parsed_igbp.where(parsed_igbp.ne(""), existing_igbp)

    df["igbp_final_source"] = np.where(
        df["igbp_parsed_clean"].ne(""),
        "reparsed_zip_metadata",
        np.where(df["igbp_existing_clean"].ne(""), "existing_phase17_metadata", "missing_or_invalid")
    )

    # Add 8-day climate summary if not present.
    if not eight.empty:
        e = eight.copy()
        e["site"] = e["site"].astype(str)

        clim = (
            e.groupby("site")
            .agg(
                tower_8day_n=("site", "size"),
                tower_8day_years=("year", "nunique"),
                tower_vpd_mean_kpa_8day=("vpd_8day_kpa_mean", "mean"),
                tower_vpd_p90_kpa_8day=("vpd_8day_kpa_mean", lambda x: float(num(x).quantile(0.90))),
                tower_vpd_p95_kpa_8day=("vpd_8day_kpa_mean", lambda x: float(num(x).quantile(0.95))),
                tower_swc_coverage_8day=("swc_8day_mean", lambda x: float(pd.Series(x).notna().mean())),
                tower_precip_coverage_8day=("precip_8day_mm", lambda x: float(pd.Series(x).notna().mean())),
                tower_median_gpp_8day=("gpp_8day_gC_m2", "median"),
                tower_median_et_8day=("et_8day_mm", "median"),
                tower_median_uwue_8day=("uwue_8day", "median"),
            )
            .reset_index()
        )
        df = df.merge(clim, on="site", how="left")
        df["vpd_mean_kpa"] = num(df.get("vpd_mean_kpa", np.nan)).combine_first(num(df.get("tower_vpd_mean_kpa_8day", np.nan)))
        df["vpd_p90_kpa"] = num(df.get("vpd_p90_kpa", np.nan)).combine_first(num(df.get("tower_vpd_p90_kpa_8day", np.nan)))

    # Tower event.
    df["tower_satbreak_event"] = df["response_class"].astype(str).isin(["saturation", "breakdown"])
    df["tower_breakdown_event"] = df["response_class"].astype(str).eq("breakdown")
    df["tower_enhancement_event"] = df["response_class"].astype(str).eq("enhancement")

    df["tower_lat"] = num(df["tower_lat"])
    df["tower_lon"] = num(df["tower_lon"])
    df["abs_lat"] = df["tower_lat"].abs()

    # Spatial annotation.
    df = annotate_spatial(df)

    # Ecoregion-based open grassland/savanna.
    biome_text = df.get("eco_biome", "").fillna("").astype(str).str.lower()
    eco_text = df.get("eco_ecoregion", "").fillna("").astype(str).str.lower()

    df["is_ecoregion_grassland_savanna_tower"] = (
        biome_text.str.contains("grassland", na=False)
        | biome_text.str.contains("savanna", na=False)
        | biome_text.str.contains("shrubland", na=False)
        | eco_text.str.contains("grassland", na=False)
        | eco_text.str.contains("savanna", na=False)
        | eco_text.str.contains("steppe", na=False)
        | eco_text.str.contains("prairie", na=False)
        | eco_text.str.contains("shrub", na=False)
    )

    df["is_strict_grassland_tower"] = df["igbp_final"].isin(STRICT_GRASS_IGBP)

    df["is_expanded_grassland_savanna_tower"] = (
        df["igbp_final"].isin(EXPANDED_GRASS_SAVANNA_IGBP)
        | df["is_ecoregion_grassland_savanna_tower"]
    )

    df["is_forest_tower"] = df["igbp_final"].isin(FOREST_IGBP)

    df["is_open_nonforest_tower"] = (
        df["is_expanded_grassland_savanna_tower"]
        & ~df["is_forest_tower"]
        & ~df["igbp_final"].isin(NON_TARGET_IGBP)
    )

    # Attach nearest satellite/trait proxy.
    df, sat_source = attach_nearest_satellite_trait_proxy(df)

    # If nearest satellite ecoregion suggests grassland/savanna, mark as provisional.
    ns_biome = df.get("nearest_sat_eco_biome", "").fillna("").astype(str).str.lower()
    ns_eco = df.get("nearest_sat_eco_ecoregion", "").fillna("").astype(str).str.lower()
    df["is_nearest_satellite_grassland_savanna_proxy"] = (
        ns_biome.str.contains("grassland", na=False)
        | ns_biome.str.contains("savanna", na=False)
        | ns_biome.str.contains("shrubland", na=False)
        | ns_eco.str.contains("grassland", na=False)
        | ns_eco.str.contains("savanna", na=False)
        | ns_eco.str.contains("steppe", na=False)
        | ns_eco.str.contains("prairie", na=False)
        | ns_eco.str.contains("shrub", na=False)
    )

    df["is_expanded_plus_satellite_proxy_grassland"] = (
        df["is_expanded_grassland_savanna_tower"]
        | (
            df["is_nearest_satellite_grassland_savanna_proxy"]
            & (num(df.get("nearest_satellite_distance_km", np.nan)) <= 100)
        )
    )

    # Scope labels.
    df["validation_scope"] = "all_tower_ecosystems"
    df.loc[df["is_expanded_grassland_savanna_tower"], "validation_scope"] = "expanded_grassland_savanna_open"
    df.loc[df["is_strict_grassland_tower"], "validation_scope"] = "strict_grassland_GRA"

    save_csv(df, TAB / "Table101_tower_landcover_spatial_trait_annotation.csv")

    # Subsets.
    masks = {
        "all_49_tower_response_sites": pd.Series(True, index=df.index),
        "strict_grassland_GRA": df["is_strict_grassland_tower"].fillna(False),
        "expanded_grassland_savanna_open": df["is_expanded_grassland_savanna_tower"].fillna(False),
        "expanded_plus_nearest_satellite_proxy_grassland": df["is_expanded_plus_satellite_proxy_grassland"].fillna(False),
        "nonforest_open_ecosystems": df["is_open_nonforest_tower"].fillna(False),
        "forest_contrast": df["is_forest_tower"].fillna(False),
    }

    # Summary by subset.
    subset_summary = pd.DataFrame([summarize_subset(df, name, mask) for name, mask in masks.items()])
    save_csv(subset_summary, TAB / "Table102_tower_response_summary_by_validation_scope.csv")

    # Class counts by subset.
    rows = []
    for name, mask in masks.items():
        d = df[mask].copy()
        counts = d["response_class"].value_counts()
        for cls, n in counts.items():
            rows.append({
                "subset": name,
                "response_class": cls,
                "n_sites": int(n),
                "fraction": float(n / len(d)) if len(d) else np.nan,
            })
    class_counts = pd.DataFrame(rows)
    save_csv(class_counts, TAB / "Table103_tower_response_class_counts_by_scope.csv")

    # Spatial/ecoregion enrichment.
    group_rows = []
    for name, mask in masks.items():
        gr = group_enrichment(df, name, mask)
        if not gr.empty:
            group_rows.append(gr)
    groups = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    save_csv(groups, TAB / "Table104_spatial_ecoregion_landcover_enrichment.csv")

    # Characteristic tests.
    test_rows = []
    for name, mask in masks.items():
        tt = characteristic_tests(df, name, mask)
        if not tt.empty:
            test_rows.append(tt)
    tests = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    if not tests.empty:
        tests = tests.sort_values(["subset", "mann_q", "mann_p"])
    save_csv(tests, TAB / "Table105_characteristic_trait_association_tests.csv")

    # Candidate rules.
    rule_rows = []
    for name, mask in masks.items():
        rr = candidate_rules(df, name, mask)
        if not rr.empty:
            rule_rows.append(rr)
    rules = pd.concat(rule_rows, ignore_index=True) if rule_rows else pd.DataFrame()
    save_csv(rules, TAB / "Table106_candidate_characteristic_rules_for_tower_satbreak.csv")

    # Decision tree rules.
    tree_text_lines = []
    tree_imps = []
    for name, mask in masks.items():
        imp, tree_txt = decision_tree_rules(df, name, mask)
        tree_text_lines.append(f"\n\n===== {name} =====\n{tree_txt}")
        if isinstance(imp, pd.DataFrame) and not imp.empty:
            tree_imps.append(imp)

    tree_text = "\n".join(tree_text_lines)
    save_text(tree_text, TXT / "DECISION_TREE_RULES_tower_satbreak.txt")

    tree_imp_df = pd.concat(tree_imps, ignore_index=True) if tree_imps else pd.DataFrame()
    save_csv(tree_imp_df, TAB / "Table107_decision_tree_feature_importance.csv")

    # Extraction target files.
    target_info = []
    for subset_name, mask in masks.items():
        if subset_name == "forest_contrast":
            continue
        out_rich, out_coords, n = make_coordinate_file(
            df,
            mask,
            f"Table108_satellite_extraction_targets_{subset_name}",
            subset_name,
        )
        target_info.append({
            "subset": subset_name,
            "n_targets_with_coordinates": int(n),
            "rich_file": str(out_rich),
            "coordinate_file": str(out_coords),
        })

    target_info_df = pd.DataFrame(target_info)
    save_csv(target_info_df, TAB / "Table109_satellite_extraction_target_files_manifest.csv")

    # High-priority candidates: open/grassland-like and satbreak, plus enhancement controls.
    priority = df[
        df["is_expanded_grassland_savanna_tower"].fillna(False)
        & df["tower_lat"].notna()
        & df["tower_lon"].notna()
    ].copy()

    if not priority.empty:
        priority["priority_score"] = (
            2.0 * priority["tower_satbreak_event"].astype(float)
            + num(priority["p_tower_saturation_breakdown"]).fillna(0)
            - 0.15 * num(priority["nearest_satellite_distance_km"]).fillna(1000) / 100.0
            + 0.25 * priority["is_strict_grassland_tower"].astype(float)
        )
        priority = priority.sort_values("priority_score", ascending=False)

    save_csv(priority, TAB / "Table110_high_priority_grassland_open_tower_sites_for_satellite_extraction.csv")

    if not priority.empty:
        save_csv(
            priority[["site", "tower_lat", "tower_lon"]].rename(
                columns={"site": "target_id", "tower_lat": "latitude", "tower_lon": "longitude"}
            ),
            TAB / "high_priority_grassland_open_tower_coordinates.csv"
        )
    else:
        save_csv(pd.DataFrame(columns=["target_id", "latitude", "longitude"]), TAB / "high_priority_grassland_open_tower_coordinates.csv")

    # Figures.
    try:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        plot = class_counts[class_counts["subset"].isin([
            "all_49_tower_response_sites",
            "strict_grassland_GRA",
            "expanded_grassland_savanna_open",
            "nonforest_open_ecosystems",
        ])].copy()
        if not plot.empty:
            pivot = plot.pivot(index="subset", columns="response_class", values="n_sites").fillna(0)
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Tower sites")
            ax.set_title("Tower WUE response classes by validation scope")
            ax.tick_params(axis="x", rotation=30)
            savefig(FIG / "Figure1_tower_response_classes_by_scope.png")
    except Exception as e:
        print("WARNING Figure1 failed:", e)

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {
            "breakdown": "tab:red",
            "saturation": "tab:orange",
            "enhancement": "tab:green",
            "inconclusive": "tab:gray",
        }
        for cls, g in df.groupby("response_class"):
            ax.scatter(g["post_slope"], g["slope_change"], label=cls, s=65, alpha=0.8)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Tower post-stress slope")
        ax.set_ylabel("Tower slope change")
        ax.set_title("Tower response phenotype space")
        ax.legend(frameon=False)
        savefig(FIG / "Figure2_tower_response_phenotype_space.png")
    except Exception as e:
        print("WARNING Figure2 failed:", e)

    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        dplot = df.dropna(subset=["tower_lat", "vpd_mean_kpa"]).copy()
        for cls, g in dplot.groupby("response_class"):
            ax.scatter(g["tower_lat"], g["vpd_mean_kpa"], label=cls, s=65, alpha=0.8)
        ax.set_xlabel("Tower latitude")
        ax.set_ylabel("Tower mean VPD, kPa")
        ax.set_title("Tower response classes in latitude–VPD space")
        ax.legend(frameon=False)
        savefig(FIG / "Figure3_latitude_vpd_tower_response_classes.png")
    except Exception as e:
        print("WARNING Figure3 failed:", e)

    try:
        if not tests.empty:
            top = tests[tests["subset"].eq("expanded_grassland_savanna_open")].head(15).copy()
            if top.empty:
                top = tests.head(15).copy()
            top["label"] = top["variable"].astype(str)
            fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
            y = np.arange(len(top))
            ax.barh(y, top["median_diff_satbreak_minus_enhancement"])
            ax.set_yticks(y)
            ax.set_yticklabels(top["label"], fontsize=8)
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.invert_yaxis()
            ax.set_xlabel("Median difference: saturation/breakdown minus enhancement")
            ax.set_title("Candidate characteristics associated with tower WUE limitation")
            savefig(FIG / "Figure4_trait_characteristic_differences.png")
    except Exception as e:
        print("WARNING Figure4 failed:", e)

    # Recommendation text.
    strict_n = int(df["is_strict_grassland_tower"].sum())
    expanded_n = int(df["is_expanded_grassland_savanna_tower"].sum())
    open_n = int(df["is_open_nonforest_tower"].sum())
    all_n = int(len(df))

    strict_sat = float(df.loc[df["is_strict_grassland_tower"], "tower_satbreak_event"].mean()) if strict_n else np.nan
    expanded_sat = float(df.loc[df["is_expanded_grassland_savanna_tower"], "tower_satbreak_event"].mean()) if expanded_n else np.nan
    all_sat = float(df["tower_satbreak_event"].mean()) if all_n else np.nan

    top_tests_txt = "No characteristic tests available."
    if not tests.empty:
        top_tests_txt = tests.head(30).to_string(index=False)

    top_groups_txt = "No spatial enrichment tests available."
    if not groups.empty:
        top_groups_txt = groups.head(30).to_string(index=False)

    top_rules_txt = "No rules available."
    if not rules.empty:
        top_rules_txt = rules.head(30).to_string(index=False)


    # Recommendation text.
    strict_n = int(df["is_strict_grassland_tower"].sum())
    expanded_n = int(df["is_expanded_grassland_savanna_tower"].sum())
    open_n = int(df["is_open_nonforest_tower"].sum())
    all_n = int(len(df))

    strict_sat = float(df.loc[df["is_strict_grassland_tower"], "tower_satbreak_event"].mean()) if strict_n else np.nan
    expanded_sat = float(df.loc[df["is_expanded_grassland_savanna_tower"], "tower_satbreak_event"].mean()) if expanded_n else np.nan
    open_sat = float(df.loc[df["is_open_nonforest_tower"], "tower_satbreak_event"].mean()) if open_n else np.nan
    all_sat = float(df["tower_satbreak_event"].mean()) if all_n else np.nan

    top_tests_txt = "No characteristic tests available."
    if not tests.empty:
        top_tests_txt = tests.head(30).to_string(index=False)

    top_groups_txt = "No spatial enrichment tests available."
    if not groups.empty:
        top_groups_txt = groups.head(30).to_string(index=False)

    top_rules_txt = "No rules available."
    if not rules.empty:
        top_rules_txt = rules.head(30).to_string(index=False)

    top_tree_txt = tree_text if tree_text.strip() else "No decision-tree rules available."

    verdict = {
        "all_tower_response_sites": all_n,
        "strict_grassland_GRA_sites": strict_n,
        "expanded_grassland_savanna_open_sites": expanded_n,
        "open_nonforest_sites": open_n,
        "all_tower_satbreak_fraction": all_sat,
        "strict_grassland_satbreak_fraction": strict_sat,
        "expanded_grassland_savanna_satbreak_fraction": expanded_sat,
        "open_nonforest_satbreak_fraction": open_sat,
        "satellite_proxy_source": sat_source,
        "geopandas_spatial_annotation_available": bool(GEOPANDAS_OK),
        "strict_grassland_main_analysis_ready": bool(strict_n >= 8),
        "expanded_open_main_analysis_ready": bool(expanded_n >= 10),
        "recommended_tower_validation_scope": (
            "strict_grassland_GRA" if strict_n >= 8
            else "expanded_grassland_savanna_open" if expanded_n >= 10
            else "all_tower_ecosystems_with_grassland_subset_as_sensitivity"
        ),
    }

    (OUT / "phase18_grassland_spatial_trait_lock_verdict.json").write_text(
        json.dumps(verdict, indent=2, default=str),
        encoding="utf-8"
    )

    recommendation_lines = [
        "# Phase 18 grassland + spatial/trait lock",
        "",
        "## Core counts",
        "",
        f"- All tower response sites: `{all_n}`",
        f"- Strict IGBP grassland sites, GRA only: `{strict_n}`",
        f"- Expanded grassland/savanna/open sites: `{expanded_n}`",
        f"- Open nonforest sites: `{open_n}`",
        "",
        "## Tower saturation/breakdown fractions",
        "",
        f"- All tower sites: `{pct(all_sat)}`",
        f"- Strict grassland sites: `{pct(strict_sat)}`",
        f"- Expanded grassland/savanna/open sites: `{pct(expanded_sat)}`",
        f"- Open nonforest sites: `{pct(open_sat)}`",
        "",
        "## Interpretation",
        "",
        "This phase separates the mentor thesis into two levels.",
        "",
        "1. General ecosystem-flux thesis: the all-site tower phenotype can support a broad ecosystem WUE/uWUE response paper because it uses many eddy-covariance sites and captures saturation, breakdown, enhancement, and inconclusive response classes.",
        "",
        "2. Grassland-specific thesis: the strict grassland version depends on how many valid GRA sites remain after metadata repair. If strict GRA has too few sites, use expanded grassland/savanna/open ecosystems as the tower validation scope and present strict GRA as a sensitivity check.",
        "",
        "## Recommended hierarchy for the paper",
        "",
        "- Main tower validation: expanded grassland/savanna/open tower subset if n is large enough.",
        "- Sensitivity: strict GRA-only towers.",
        "- Contrast: forest towers or all ecosystem towers.",
        "- Final validation: extract satellite products at the tower coordinates listed in the Table108 coordinate files.",
        "",
        "## Top characteristic/trait association tests",
        "",
        "```text",
        top_tests_txt,
        "```",
        "",
        "## Top spatial/ecoregion enrichment tests",
        "",
        "```text",
        top_groups_txt,
        "```",
        "",
        "## Top candidate rules",
        "",
        "```text",
        top_rules_txt,
        "```",
        "",
        "## Decision-tree rules",
        "",
        "```text",
        top_tree_txt,
        "```",
        "",
        "## Manuscript-safe conclusion",
        "",
        "The tower results support a tower-observed ecosystem flux phenotype, but the final grassland-specific thesis depends on the size and quality of the repaired grassland/savanna/open-tower subset. The next required step is tower-centered satellite extraction for the selected target files, followed by direct tower-vs-satellite response-class comparison.",
        "",
    ]

    recommendation = "\n".join(recommendation_lines)
    save_text(recommendation, TXT / "PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK_VERDICT.md")
    save_text(recommendation, OUT / "README_phase18_grassland_spatial_trait_lock.md")

    methods_lines = [
        "# Methods: Phase 18 grassland spatial trait lock",
        "",
        "Phase 18 repaired tower land-cover labels, annotated tower sites with spatial/ecoregion information when local GIS layers were available, and separated tower validation into strict grassland, expanded grassland/savanna/open, open nonforest, forest-contrast, and all-site scopes.",
        "",
        "Tower saturation/breakdown was defined as tower response class equal to saturation or breakdown. Enhancement was used as the main contrast class. Characteristic tests compared saturation/breakdown sites against enhancement sites using Mann-Whitney tests for continuous variables and Fisher exact tests for categorical groups.",
        "",
        "Nearest satellite-point traits and environmental variables are treated only as provisional proxies. They do not replace tower-centered satellite extraction.",
        "",
    ]
    save_text("\n".join(methods_lines), TXT / "METHODS_PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK.md")

    print("")
    print("==============================")
    print("PHASE 18 GRASSLAND SPATIAL TRAIT LOCK VERDICT")
    print("==============================")
    print(json.dumps(verdict, indent=2, default=str))
    print("")
    print(recommendation)
    print("")
    print("MAIN OUTPUTS:")
    print(TAB / "Table101_tower_landcover_spatial_trait_annotation.csv")
    print(TAB / "Table102_tower_response_summary_by_validation_scope.csv")
    print(TAB / "Table105_characteristic_trait_association_tests.csv")
    print(TAB / "Table108_satellite_extraction_targets_expanded_grassland_savanna_open_coordinates_only.csv")
    print(TXT / "PHASE18_GRASSLAND_SPATIAL_TRAIT_LOCK_VERDICT.md")


if __name__ == "__main__":
    main()
