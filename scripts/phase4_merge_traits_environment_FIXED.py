#!/usr/bin/env python
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# PHASE 4: Merge plant physiology and environmental covariates
# =============================================================================
#
# Goal:
# Attach hydraulic/rooting/climate/soil information to each response phenotype.
#
# Inputs:
# - results/trait_framework/point_product_consensus_response.csv
# - data/external/liu_2021_psi50_0p1deg.nc
# - data/external/stocker_2023_rooting_depth_0p1deg.nc
# - data/external/konings_gentine_isohydricity_0p1deg.nc
# - data/external/aridity_by_point.csv
# - local point-time matrix for VPD / soil moisture / precipitation / temperature / LAI
# - local soil texture files if available
#
# Main output:
# - results/trait_framework/trait_model_dataset.csv
#
# =============================================================================

# =============================================================================
# Paths
# =============================================================================

RESPONSE_PATH = Path("results/trait_framework/point_product_consensus_response.csv")

TRAIT_PATHS = {
    "p50": Path("data/external/liu_2021_psi50_0p1deg.nc"),
    "rooting_depth": Path("data/external/stocker_2023_rooting_depth_0p1deg.nc"),
    "isohydricity": Path("data/external/konings_gentine_isohydricity_0p1deg.nc"),
}

ARIDITY_PATH = Path("data/external/aridity_by_point.csv")

OUTDIR = Path("results/trait_framework/phase4")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAIN_OUT = Path("results/trait_framework/trait_model_dataset.csv")

POINT_TIME_CANDIDATES = [
    Path("data/raw/agents/merged_full_matrix_co2corrected.csv"),
    Path("data/processed/project_metric_matrix_co2corrected.csv"),
    Path("data/raw/agents/merged_full_matrix_raw.csv"),
    Path("data/processed/project_metric_matrix_raw.csv"),
]

STABLE_POINT_CANDIDATES = [
    Path("data/raw/gee/stable_grassland_points.csv"),
    Path("data/processed/stable_grassland_points.csv"),
]

SOIL_CANDIDATES = [
    Path("data/external/soil_by_point.csv"),
    Path("data/external/soilgrids_by_point.csv"),
    Path("data/external/soil_texture_by_point.csv"),
    Path("data/processed/soil_by_point.csv"),
    Path("data/processed/soilgrids_by_point.csv"),
    Path("results/qc/soil_by_point.csv"),
    Path("results/qc/soilgrids_by_point.csv"),
]

SCAN_DIRS_FOR_SOIL = [
    Path("data/external"),
    Path("data/processed"),
    Path("results"),
]

# =============================================================================
# Utility functions
# =============================================================================

def die(msg):
    raise SystemExit("\nERROR: " + str(msg) + "\n")

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def make_unique_columns(df):
    """
    Fixes duplicate column names so df[c] always returns a Series, not a DataFrame.
    This directly fixes the prior Phase 4 quantile crash.
    """
    df = df.copy()
    seen = {}
    new_cols = []
    for c in df.columns:
        c = str(c)
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__dup{seen[c]}")
    df.columns = new_cols
    return df

def parse_point_id(pid):
    s = str(pid).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        lon = float(s[0])
        lat = float(s[1])
        return lon, lat
    except Exception:
        return np.nan, np.nan

def finite_count(series):
    return int(pd.to_numeric(series, errors="coerce").notna().sum())

def finite_fraction(series):
    if len(series) == 0:
        return np.nan
    return finite_count(series) / len(series)

def to_numeric_series(obj):
    """
    Always return a float numeric Series.

    Why this matters:
    - If duplicate columns exist, df[c] can return a DataFrame instead of a Series.
      We use the first duplicate column instead of crashing.
    - pandas keeps boolean columns as dtype bool after pd.to_numeric(...).
      numpy/pandas quantile cannot interpolate booleans because it tries to subtract
      True/False values. Casting to float makes False=0.0 and True=1.0.
    - If conversion fails, return all-NaN float values on the original index.
    """
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype="float64")
        obj = obj.iloc[:, 0]

    if isinstance(obj, pd.Series):
        idx = obj.index
    else:
        idx = None

    try:
        x = pd.to_numeric(obj, errors="coerce")
        if not isinstance(x, pd.Series):
            x = pd.Series(x, index=idx)
        return x.astype("float64")
    except Exception:
        return pd.Series(np.nan, index=idx, dtype="float64")

def zscore(series):
    x = to_numeric_series(series)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd

def safe_quantile(series, q):
    x = to_numeric_series(series).dropna()
    if len(x) == 0:
        return np.nan
    try:
        return float(x.quantile(q))
    except Exception:
        # Last-resort fallback. Should not be needed after float conversion,
        # but keeps the script from dying on unexpected dtypes.
        arr = np.asarray(x, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        return float(np.nanquantile(arr, q))

def safe_mean(series):
    x = to_numeric_series(series).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.mean())

def safe_median(series):
    x = to_numeric_series(series).dropna()
    if len(x) == 0:
        return np.nan
    return float(x.median())

def safe_std(series):
    x = to_numeric_series(series).dropna()
    if len(x) < 2:
        return np.nan
    return float(x.std())

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_unique_columns(df)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

def normalize_lon_for_dataset(lon, lon_values):
    lon = float(lon)
    arr = np.asarray(lon_values, dtype=float)

    if len(arr) == 0:
        return lon

    mn = np.nanmin(arr)
    mx = np.nanmax(arr)

    if mn >= 0 and lon < 0:
        return lon + 360.0
    if mx > 180 and lon < 0:
        return lon + 360.0
    if mn < 0 and lon > 180:
        return lon - 360.0
    return lon

def json_default(obj):
    """
    JSON helper for numpy/pandas scalar types that sometimes appear in metadata.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def sanitize_lai_values(series):
    """
    Keep only physically plausible LAI-like values.

    The previous run reported mean_lai around -37, which is not physical LAI.
    That usually means the auto-picker grabbed an anomaly/standardized column.
    This sanitizer prevents those values from being treated as real canopy LAI.

    Accepted range here is broad:
    - LAI cannot be negative.
    - Natural/crop LAI above 20 is almost certainly scaled or wrong for this use.
    """
    x = to_numeric_series(series)
    x = x.where((x >= 0) & (x <= 20), np.nan)
    return x

# =============================================================================
# Step 1: Load response phenotype table
# =============================================================================

def load_response():
    if not RESPONSE_PATH.exists():
        die(
            f"Missing Phase 3 response phenotype table:\n{RESPONSE_PATH}\n\n"
            "Run Phase 3 first."
        )

    df = clean_columns(pd.read_csv(RESPONSE_PATH, low_memory=False))
    df = make_unique_columns(df)

    if "point_id" not in df.columns:
        die(f"{RESPONSE_PATH} has no point_id column.")

    df["point_id"] = df["point_id"].astype(str)

    if "lat" not in df.columns or "lon" not in df.columns:
        lonlat = df["point_id"].apply(lambda x: pd.Series(parse_point_id(x)))
        if "lon" not in df.columns:
            df["lon"] = lonlat[0]
        if "lat" not in df.columns:
            df["lat"] = lonlat[1]

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    return df

# =============================================================================
# Step 2: Sample NetCDF trait maps
# =============================================================================

def infer_lat_lon_coord(ds):
    names = list(ds.coords) + list(ds.dims)

    lat = None
    lon = None

    for n in names:
        nl = str(n).lower()
        if nl in ["lat", "latitude", "y"]:
            lat = n
        if nl in ["lon", "longitude", "x"]:
            lon = n

    if lat is None:
        for n in names:
            if "lat" in str(n).lower():
                lat = n
                break

    if lon is None:
        for n in names:
            if "lon" in str(n).lower():
                lon = n
                break

    return lat, lon

def infer_trait_var(ds, trait_name):
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None

    preferred = {
        "p50": ["p50", "psi50", "xylem"],
        "rooting_depth": ["root", "depth"],
        "isohydricity": ["isohyd", "hydric"],
    }.get(trait_name, [trait_name])

    for term in preferred:
        term = term.lower()
        for v in data_vars:
            if term in str(v).lower():
                return v

    # Prefer 2D variable.
    for v in data_vars:
        try:
            if len(ds[v].dims) >= 2:
                return v
        except Exception:
            pass

    return data_vars[0]

def reduce_extra_dims(da, lat_name, lon_name):
    """
    If a trait variable has extra dimensions besides lat/lon,
    take the first index along those dimensions.
    """
    indexers = {}
    for dim in da.dims:
        if dim not in [lat_name, lon_name]:
            indexers[dim] = 0
    if indexers:
        da = da.isel(indexers)
    return da

def sample_trait_nc(points, trait_name, path):
    out = points[["point_id", "lat", "lon"]].copy()
    out[trait_name] = np.nan

    meta = {
        "trait_name": trait_name,
        "path": str(path),
        "exists": path.exists(),
        "status": "not_started",
        "var_name": None,
        "lat_coord": None,
        "lon_coord": None,
        "finite_n": 0,
        "total_n": int(len(points)),
        "coverage_fraction": 0.0,
    }

    if not path.exists():
        meta["status"] = "missing_file"
        return out[["point_id", trait_name]], meta

    try:
        import xarray as xr
    except Exception as e:
        meta["status"] = f"xarray_import_failed: {type(e).__name__}: {e}"
        return out[["point_id", trait_name]], meta

    try:
        ds = xr.open_dataset(path)
        lat_name, lon_name = infer_lat_lon_coord(ds)
        var_name = infer_trait_var(ds, trait_name)

        meta["lat_coord"] = str(lat_name)
        meta["lon_coord"] = str(lon_name)
        meta["var_name"] = str(var_name)

        if lat_name is None or lon_name is None or var_name is None:
            meta["status"] = "could_not_infer_lat_lon_or_var"
            return out[["point_id", trait_name]], meta

        da = reduce_extra_dims(ds[var_name], lat_name, lon_name)
        lon_values = ds[lon_name].values

        vals = []
        for _, r in points.iterrows():
            try:
                lat = float(r["lat"])
                lon = normalize_lon_for_dataset(float(r["lon"]), lon_values)
                val = da.sel({lat_name: lat, lon_name: lon}, method="nearest").values
                val = float(np.asarray(val).squeeze())
                if not np.isfinite(val):
                    val = np.nan
                vals.append(val)
            except Exception:
                vals.append(np.nan)

        out[trait_name] = vals
        meta["finite_n"] = finite_count(out[trait_name])
        meta["coverage_fraction"] = finite_fraction(out[trait_name])
        meta["status"] = "ok"

        try:
            ds.close()
        except Exception:
            pass

        return out[["point_id", trait_name]], meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        return out[["point_id", trait_name]], meta

def add_traits(dataset):
    points = dataset[["point_id", "lat", "lon"]].drop_duplicates("point_id").copy()

    metas = []
    out = dataset.copy()

    for trait_name, path in TRAIT_PATHS.items():
        vals, meta = sample_trait_nc(points, trait_name, path)
        metas.append(meta)
        out = out.merge(vals, on="point_id", how="left")

    # Required aliases for explicit framework wording.
    if "p50" in out.columns:
        out["xylem_vulnerability_p50"] = pd.to_numeric(out["p50"], errors="coerce")
    else:
        out["xylem_vulnerability_p50"] = np.nan

    if "rooting_depth" in out.columns:
        out["rooting_zone_storage_rooting_depth"] = pd.to_numeric(out["rooting_depth"], errors="coerce")
    else:
        out["rooting_zone_storage_rooting_depth"] = np.nan

    if "isohydricity" in out.columns:
        out["stomatal_strategy_isohydricity"] = pd.to_numeric(out["isohydricity"], errors="coerce")
    else:
        out["stomatal_strategy_isohydricity"] = np.nan

    return out, metas

# =============================================================================
# Step 3: Merge aridity
# =============================================================================

def add_aridity(dataset):
    meta = {
        "path": str(ARIDITY_PATH),
        "exists": ARIDITY_PATH.exists(),
        "status": "not_started",
        "columns_added": [],
        "aridity_column_used": None,
    }

    out = dataset.copy()

    if not ARIDITY_PATH.exists():
        meta["status"] = "missing_file"
        out["aridity"] = np.nan
        return out, meta

    try:
        ar = clean_columns(pd.read_csv(ARIDITY_PATH, low_memory=False))
        ar = make_unique_columns(ar)

        if "point_id" not in ar.columns:
            meta["status"] = "no_point_id"
            out["aridity"] = np.nan
            return out, meta

        ar["point_id"] = ar["point_id"].astype(str)

        keep = ["point_id"]
        candidates = []

        for c in ar.columns:
            cl = c.lower()
            if c == "point_id":
                continue
            if cl in ["aridity", "aridity_index", "ai", "arid_index"]:
                candidates.append(c)
            elif "aridity" in cl:
                candidates.append(c)
            elif any(t in cl for t in ["pet", "precip", "map", "mat"]):
                keep.append(c)

        if candidates:
            chosen = candidates[0]
            keep.append(chosen)
            ar = ar[list(dict.fromkeys(keep))].copy()
            if chosen != "aridity":
                ar = ar.rename(columns={chosen: "aridity"})
            meta["aridity_column_used"] = chosen
        else:
            ar = ar[list(dict.fromkeys(keep))].copy()
            ar["aridity"] = np.nan
            meta["aridity_column_used"] = None

        # Numeric conversion.
        for c in ar.columns:
            if c != "point_id":
                ar[c] = pd.to_numeric(ar[c], errors="coerce")

        ar = ar.groupby("point_id", dropna=False).mean(numeric_only=True).reset_index()

        before = set(out.columns)
        out = out.merge(ar, on="point_id", how="left", suffixes=("", "_aridity"))
        out = make_unique_columns(out)
        meta["columns_added"] = [c for c in out.columns if c not in before]
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        out["aridity"] = np.nan
        return out, meta

# =============================================================================
# Step 4: Add climate controls from local point-time data
# =============================================================================

def find_point_time_file():
    for p in POINT_TIME_CANDIDATES:
        if p.exists():
            return p

    patterns = [
        "data/raw/agents/*matrix*co2*.csv",
        "data/raw/agents/*merged*.csv",
        "data/processed/*metric_matrix*co2*.csv",
        "data/processed/*matrix*.csv",
    ]

    for pat in patterns:
        for f in sorted(Path(".").glob(pat)):
            if f.exists() and f.stat().st_size > 0:
                return f

    return None

def column_score_for_variable(col, var):
    cl = col.lower()

    bad_terms = [
        "uwue",
        "iwue",
        "raw_wue",
        "wue",
        "log_",
        "response",
        "slope",
        "class",
        "anomaly",
        "anom",
        "zscore",
        "z_score",
        "standardized",
        "scaled",
        "normalized",
    ]

    if any(b in cl for b in bad_terms):
        return -999

    score = 0

    if var == "vpd":
        if cl == "vpd":
            score += 100
        if "vpd" in cl:
            score += 30

    elif var == "soil_moisture":
        if cl in ["soil_moisture", "sm"]:
            score += 100
        if "soil_moisture" in cl:
            score += 50
        if "swvl" in cl:
            score += 40
        if cl.startswith("sm_") or cl.endswith("_sm") or "_sm_" in cl:
            score += 20
        if "stress" in cl:
            score -= 30

    elif var == "temperature":
        if cl in ["temperature", "temp", "tmean", "t2m"]:
            score += 100
        if "temperature" in cl:
            score += 50
        if "temp" in cl:
            score += 30
        if "t2m" in cl:
            score += 30

    elif var == "precipitation":
        if cl in ["precipitation", "precip", "ppt", "rain"]:
            score += 100
        if "precip" in cl:
            score += 50
        if "ppt" in cl:
            score += 40
        if "rain" in cl:
            score += 30

    elif var == "lai":
        if cl == "lai":
            score += 150
        if cl in ["leaf_area_index", "modis_lai", "mcd15_lai"]:
            score += 120
        if "leaf_area_index" in cl:
            score += 90
        if "lai" in cl:
            score += 50
        if any(t in cl for t in ["anomaly", "anom", "zscore", "z_score", "standardized", "scaled", "normalized"]):
            score -= 200

    return score

def pick_best_column(cols, var):
    scored = []
    for c in cols:
        score = column_score_for_variable(c, var)
        if score > 0:
            scored.append((score, c))
    if not scored:
        return None
    scored = sorted(scored, key=lambda x: (-x[0], x[1]))
    return scored[0][1]

def add_climate_controls(dataset):
    meta = {
        "path": None,
        "exists": False,
        "status": "not_started",
        "column_map": {},
        "columns_added": [],
    }

    out = dataset.copy()

    p = find_point_time_file()
    if p is None:
        meta["status"] = "no_point_time_file_found"
        for c in [
            "mean_vpd",
            "mean_soil_moisture",
            "mean_annual_precipitation",
            "mean_annual_temperature",
            "mean_temperature",
            "mean_precipitation",
            "mean_lai",
            "growing_season_mean_lai",
        ]:
            out[c] = np.nan
        return out, meta

    meta["path"] = str(p)
    meta["exists"] = True

    try:
        header = clean_columns(pd.read_csv(p, nrows=5, low_memory=False))
        header = make_unique_columns(header)

        if "point_id" not in header.columns:
            meta["status"] = "file_has_no_point_id"
            return out, meta

        cols = list(header.columns)

        colmap = {
            "vpd": pick_best_column(cols, "vpd"),
            "soil_moisture": pick_best_column(cols, "soil_moisture"),
            "temperature": pick_best_column(cols, "temperature"),
            "precipitation": pick_best_column(cols, "precipitation"),
            "lai": pick_best_column(cols, "lai"),
        }

        meta["column_map"] = colmap

        keep = ["point_id"]
        for c in ["date", "year", "doy", "month"]:
            if c in cols:
                keep.append(c)

        for c in colmap.values():
            if c is not None and c in cols:
                keep.append(c)

        keep = list(dict.fromkeys(keep))

        pt = clean_columns(pd.read_csv(p, usecols=keep, low_memory=False))
        pt = make_unique_columns(pt)
        pt["point_id"] = pt["point_id"].astype(str)

        for c in pt.columns:
            if c not in ["point_id", "date"]:
                pt[c] = pd.to_numeric(pt[c], errors="coerce")

        # Guard against selecting an LAI anomaly / z-score column as physical LAI.
        # Negative LAI is impossible, and the previous run had mean_lai around -37.
        lai_candidate = colmap.get("lai")
        if lai_candidate is not None and lai_candidate in pt.columns:
            before_lai_n = finite_count(pt[lai_candidate])
            pt[lai_candidate] = sanitize_lai_values(pt[lai_candidate])
            after_lai_n = finite_count(pt[lai_candidate])
            meta["lai_sanity_filter"] = {
                "selected_column": lai_candidate,
                "finite_before": int(before_lai_n),
                "finite_after": int(after_lai_n),
                "rule": "kept only 0 <= LAI <= 20",
            }

        if "date" in pt.columns:
            pt["date"] = pd.to_datetime(pt["date"], errors="coerce")
            pt["year"] = pt["date"].dt.year
            pt["month"] = pt["date"].dt.month
        else:
            if "year" not in pt.columns:
                pt["year"] = np.nan
            if "month" not in pt.columns:
                pt["month"] = np.nan

        agg = {}

        def add_summaries(source_col, prefix):
            if source_col is None or source_col not in pt.columns:
                return
            agg[f"mean_{prefix}"] = (source_col, safe_mean)
            agg[f"median_{prefix}"] = (source_col, safe_median)
            agg[f"sd_{prefix}"] = (source_col, safe_std)
            agg[f"p10_{prefix}"] = (source_col, lambda s: safe_quantile(s, 0.10))
            agg[f"p90_{prefix}"] = (source_col, lambda s: safe_quantile(s, 0.90))
            agg[f"n_{prefix}"] = (source_col, finite_count)

        add_summaries(colmap["vpd"], "vpd")
        add_summaries(colmap["soil_moisture"], "soil_moisture")
        add_summaries(colmap["temperature"], "temperature")
        add_summaries(colmap["precipitation"], "precipitation")
        add_summaries(colmap["lai"], "lai")

        if agg:
            climate = pt.groupby("point_id", dropna=False).agg(**agg).reset_index()
        else:
            climate = pt[["point_id"]].drop_duplicates().copy()

        # Mean annual precipitation: sum precipitation within year, then average annual total.
        precip_col = colmap["precipitation"]
        if precip_col is not None and precip_col in pt.columns and "year" in pt.columns:
            annual_precip = (
                pt.dropna(subset=["year"])
                .groupby(["point_id", "year"], dropna=False)[precip_col]
                .sum(min_count=1)
                .reset_index()
                .groupby("point_id", dropna=False)[precip_col]
                .mean()
                .reset_index()
                .rename(columns={precip_col: "mean_annual_precipitation"})
            )
            climate = climate.merge(annual_precip, on="point_id", how="left")
        else:
            climate["mean_annual_precipitation"] = np.nan

        # Mean annual temperature: annual mean temperature, then average annual mean.
        temp_col = colmap["temperature"]
        if temp_col is not None and temp_col in pt.columns and "year" in pt.columns:
            annual_temp = (
                pt.dropna(subset=["year"])
                .groupby(["point_id", "year"], dropna=False)[temp_col]
                .mean()
                .reset_index()
                .groupby("point_id", dropna=False)[temp_col]
                .mean()
                .reset_index()
                .rename(columns={temp_col: "mean_annual_temperature"})
            )
            climate = climate.merge(annual_temp, on="point_id", how="left")
        else:
            climate["mean_annual_temperature"] = np.nan

        # Convenience aliases.
        if "mean_temperature" not in climate.columns:
            climate["mean_temperature"] = climate["mean_annual_temperature"]
        if "mean_precipitation" not in climate.columns:
            climate["mean_precipitation"] = climate["mean_annual_precipitation"]

        # Growing-season LAI:
        # Northern Hemisphere = Apr-Sep; Southern Hemisphere = Oct-Mar.
        lai_col = colmap["lai"]
        if lai_col is not None and lai_col in pt.columns and "month" in pt.columns:
            coords = out[["point_id", "lat"]].drop_duplicates("point_id")
            tmp = pt[["point_id", "month", lai_col]].merge(coords, on="point_id", how="left")
            tmp["month"] = pd.to_numeric(tmp["month"], errors="coerce")
            tmp["lat"] = pd.to_numeric(tmp["lat"], errors="coerce")
            tmp["is_north"] = tmp["lat"] >= 0
            tmp["is_growing_month"] = np.where(
                tmp["is_north"],
                tmp["month"].isin([4, 5, 6, 7, 8, 9]),
                tmp["month"].isin([10, 11, 12, 1, 2, 3]),
            )
            gs_lai = (
                tmp[tmp["is_growing_month"]]
                .groupby("point_id", dropna=False)[lai_col]
                .mean()
                .reset_index()
                .rename(columns={lai_col: "growing_season_mean_lai"})
            )
            climate = climate.merge(gs_lai, on="point_id", how="left")
        else:
            climate["growing_season_mean_lai"] = np.nan

        before = set(out.columns)
        out = out.merge(climate, on="point_id", how="left")
        out = make_unique_columns(out)
        meta["columns_added"] = [c for c in out.columns if c not in before]
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        return out, meta

# =============================================================================
# Step 5: Add soil controls
# =============================================================================

def scan_soil_files():
    found = []

    for p in SOIL_CANDIDATES:
        if p.exists():
            found.append(p)

    for d in SCAN_DIRS_FOR_SOIL:
        if not d.exists():
            continue
        for f in d.rglob("*.csv"):
            fl = str(f).lower()
            if any(t in fl for t in ["soil", "soilgrids", "sand", "silt", "clay"]):
                found.append(f)

    unique = []
    seen = set()
    for f in found:
        s = str(f)
        if s not in seen:
            unique.append(f)
            seen.add(s)

    return unique

def infer_soil_cols(df):
    sand = None
    silt = None
    clay = None

    for c in df.columns:
        cl = c.lower()
        if sand is None and "sand" in cl:
            sand = c
        if silt is None and "silt" in cl:
            silt = c
        if clay is None and "clay" in cl:
            clay = c

    return sand, silt, clay

def add_soil_controls(dataset):
    meta = {
        "status": "not_started",
        "files_scanned": [],
        "selected_file": None,
        "soil_columns_detected": {},
        "columns_added": [],
    }

    out = dataset.copy()
    files = scan_soil_files()
    meta["files_scanned"] = [str(f) for f in files]

    best_file = None
    best_score = -1
    best_cols = None

    for f in files:
        try:
            head = clean_columns(pd.read_csv(f, nrows=5, low_memory=False))
            head = make_unique_columns(head)
            if "point_id" not in head.columns:
                continue
            sand, silt, clay = infer_soil_cols(head)
            score = sum(x is not None for x in [sand, silt, clay])
            if score > best_score:
                best_score = score
                best_file = f
                best_cols = (sand, silt, clay)
        except Exception:
            continue

    if best_file is None or best_score <= 0:
        meta["status"] = "no_soil_texture_file_found"
        for c in ["soil_sand", "soil_silt", "soil_clay"]:
            if c not in out.columns:
                out[c] = np.nan
        return out, meta

    try:
        soil = clean_columns(pd.read_csv(best_file, low_memory=False))
        soil = make_unique_columns(soil)

        if "point_id" not in soil.columns:
            meta["status"] = "selected_file_has_no_point_id"
            return out, meta

        soil["point_id"] = soil["point_id"].astype(str)

        sand, silt, clay = best_cols

        keep = ["point_id"]
        rename = {}

        if sand is not None:
            keep.append(sand)
            rename[sand] = "soil_sand"
        if silt is not None:
            keep.append(silt)
            rename[silt] = "soil_silt"
        if clay is not None:
            keep.append(clay)
            rename[clay] = "soil_clay"

        for c in soil.columns:
            cl = c.lower()
            if c == "point_id":
                continue
            if any(t in cl for t in ["bulk", "soc", "ph", "texture"]):
                keep.append(c)

        keep = list(dict.fromkeys([c for c in keep if c in soil.columns]))
        soil = soil[keep].rename(columns=rename)

        for c in soil.columns:
            if c != "point_id":
                soil[c] = pd.to_numeric(soil[c], errors="coerce")

        soil = soil.groupby("point_id", dropna=False).mean(numeric_only=True).reset_index()

        before = set(out.columns)
        out = out.merge(soil, on="point_id", how="left")
        out = make_unique_columns(out)

        for c in ["soil_sand", "soil_silt", "soil_clay"]:
            if c not in out.columns:
                out[c] = np.nan

        meta["selected_file"] = str(best_file)
        meta["soil_columns_detected"] = {
            "soil_sand": "soil_sand" in out.columns,
            "soil_silt": "soil_silt" in out.columns,
            "soil_clay": "soil_clay" in out.columns,
        }
        meta["columns_added"] = [c for c in out.columns if c not in before]
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        for c in ["soil_sand", "soil_silt", "soil_clay"]:
            if c not in out.columns:
                out[c] = np.nan
        return out, meta

# =============================================================================
# Step 6: Add region / biome labels
# =============================================================================

def add_stable_point_labels(dataset):
    meta = {
        "status": "not_started",
        "selected_file": None,
        "columns_added": [],
    }

    out = dataset.copy()

    selected = None
    for p in STABLE_POINT_CANDIDATES:
        if p.exists():
            selected = p
            break

    if selected is None:
        meta["status"] = "no_stable_point_file_found"
        if "biome_label" not in out.columns:
            out["biome_label"] = "grassland"
        return out, meta

    try:
        pts = clean_columns(pd.read_csv(selected, low_memory=False))
        pts = make_unique_columns(pts)

        if "point_id" not in pts.columns:
            meta["status"] = "stable_point_file_has_no_point_id"
            if "biome_label" not in out.columns:
                out["biome_label"] = "grassland"
            return out, meta

        pts["point_id"] = pts["point_id"].astype(str)

        keep = ["point_id"]
        for c in pts.columns:
            cl = c.lower()
            if any(t in cl for t in ["igbp", "biome", "landcover", "land_cover", "class", "eco"]):
                keep.append(c)

        keep = list(dict.fromkeys(keep))
        pts = pts[keep].drop_duplicates("point_id")

        before = set(out.columns)
        out = out.merge(pts, on="point_id", how="left", suffixes=("", "_stable"))
        out = make_unique_columns(out)

        # Create biome_label.
        biome_source = None
        for c in out.columns:
            cl = c.lower()
            if c == "point_id":
                continue
            if any(t in cl for t in ["igbp", "biome", "landcover", "land_cover"]):
                biome_source = c
                break

        if biome_source is not None:
            out["biome_label"] = out[biome_source].astype(str)
            out.loc[out["biome_label"].isin(["nan", "None", ""]), "biome_label"] = "grassland"
        else:
            out["biome_label"] = "grassland"

        meta["selected_file"] = str(selected)
        meta["columns_added"] = [c for c in out.columns if c not in before]
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        if "biome_label" not in out.columns:
            out["biome_label"] = "grassland"
        return out, meta

def add_spatial_labels(dataset):
    out = dataset.copy()

    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["abs_lat"] = out["lat"].abs()

    out["hemisphere"] = np.where(out["lat"] >= 0, "north", "south")

    def lat_band(lat):
        if pd.isna(lat):
            return "unknown"
        a = abs(float(lat))
        if a < 15:
            return "tropical"
        if a < 35:
            return "subtropical"
        if a < 55:
            return "temperate"
        return "boreal_arctic"

    def lon_region(lon):
        if pd.isna(lon):
            return "unknown"
        lon = float(lon)
        if -170 <= lon < -30:
            return "americas"
        if -30 <= lon < 60:
            return "africa_europe_west_asia"
        if 60 <= lon < 150:
            return "asia_oceania"
        return "pacific_high_longitude"

    out["lat_band"] = out["lat"].apply(lat_band)
    out["lon_region"] = out["lon"].apply(lon_region)
    out["region_label"] = out["lat_band"].astype(str) + "__" + out["lon_region"].astype(str)

    if "aridity" in out.columns and finite_count(out["aridity"]) >= 4:
        ar = pd.to_numeric(out["aridity"], errors="coerce")
        try:
            out["aridity_quartile"] = pd.qcut(
                ar,
                q=4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            ).astype(str)
        except Exception:
            out["aridity_quartile"] = "unknown"
    else:
        out["aridity_quartile"] = "unknown"

    return out

# =============================================================================
# Step 7: Add z-scores and model-ready flags
# =============================================================================

def add_zscores_and_flags(dataset):
    out = dataset.copy()
    out = make_unique_columns(out)

    # Important response phenotypes from Phase 3.
    response_candidates = [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
        "satbreak_fraction_independent",
        "satbreak_fraction_all",
        "negative_slope_fraction_independent",
        "negative_slope_fraction_all",
        "product_agreement_independent",
        "product_agreement_all",
    ]

    predictor_candidates = [
        "p50",
        "xylem_vulnerability_p50",
        "rooting_depth",
        "rooting_zone_storage_rooting_depth",
        "isohydricity",
        "stomatal_strategy_isohydricity",
        "aridity",
        "mean_vpd",
        "median_vpd",
        "p90_vpd",
        "mean_soil_moisture",
        "median_soil_moisture",
        "p10_soil_moisture",
        "mean_annual_precipitation",
        "mean_precipitation",
        "mean_annual_temperature",
        "mean_temperature",
        "mean_lai",
        "growing_season_mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "lat",
        "lon",
        "abs_lat",
    ]

    z_cols = response_candidates + predictor_candidates

    for c in z_cols:
        if c in out.columns:
            out[f"z_{c}"] = zscore(out[c])

    preferred_response = None
    for c in [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
    ]:
        if c in out.columns and finite_count(out[c]) > 0:
            preferred_response = c
            break

    out["preferred_trait_response"] = preferred_response if preferred_response else "missing"

    def complete_row(row, cols):
        for c in cols:
            if c not in row.index:
                return False
            val = row[c]
            try:
                if pd.isna(float(val)):
                    return False
            except Exception:
                if pd.isna(val):
                    return False
        return True

    # Core physiology predictors: P50 + rooting depth.
    core_cols = []
    if preferred_response:
        core_cols.append(preferred_response)

    if "p50" in out.columns:
        core_cols.append("p50")
    elif "xylem_vulnerability_p50" in out.columns:
        core_cols.append("xylem_vulnerability_p50")

    if "rooting_depth" in out.columns:
        core_cols.append("rooting_depth")
    elif "rooting_zone_storage_rooting_depth" in out.columns:
        core_cols.append("rooting_zone_storage_rooting_depth")

    # Minimal environmental adjustment.
    minimal_control_cols = []
    if "aridity" in out.columns:
        minimal_control_cols.append("aridity")

    climate_control_cols = []
    for c in [
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        "mean_annual_precipitation",
        "mean_annual_temperature",
        "growing_season_mean_lai",
    ]:
        if c in out.columns:
            climate_control_cols.append(c)

    # If annual aliases are missing but generic columns exist, include generic columns.
    for c in ["mean_precipitation", "mean_temperature", "mean_lai"]:
        if c in out.columns and c not in climate_control_cols:
            climate_control_cols.append(c)

    soil_control_cols = []
    for c in ["soil_sand", "soil_silt", "soil_clay"]:
        if c in out.columns:
            soil_control_cols.append(c)

    full_trait_cols = core_cols.copy()
    if "isohydricity" in out.columns:
        full_trait_cols.append("isohydricity")
    elif "stomatal_strategy_isohydricity" in out.columns:
        full_trait_cols.append("stomatal_strategy_isohydricity")

    out["core_trait_model_ready"] = out.apply(lambda r: complete_row(r, core_cols + minimal_control_cols), axis=1)
    out["core_trait_climate_model_ready"] = out.apply(lambda r: complete_row(r, core_cols + climate_control_cols), axis=1)
    out["core_trait_climate_soil_model_ready"] = out.apply(lambda r: complete_row(r, core_cols + climate_control_cols + soil_control_cols), axis=1)
    out["full_trait_with_isohydricity_ready"] = out.apply(lambda r: complete_row(r, full_trait_cols + minimal_control_cols), axis=1)

    # Explicit issue flag.
    if "isohydricity" in out.columns:
        out["isohydricity_available"] = pd.to_numeric(out["isohydricity"], errors="coerce").notna()
    else:
        out["isohydricity_available"] = False

    # Product-support flags should already exist from Phase 3.
    for c in [
        "has_all_9_product_combos",
        "has_independent_subset_complete",
        "has_pml_containing_subset_complete",
        "has_gosif_gpp_subset_complete",
        "has_gleam_et_subset_complete",
    ]:
        if c not in out.columns:
            out[c] = False

    model_meta = {
        "preferred_response": preferred_response,
        "core_trait_cols": core_cols,
        "minimal_control_cols": minimal_control_cols,
        "climate_control_cols": climate_control_cols,
        "soil_control_cols": soil_control_cols,
        "full_trait_cols": full_trait_cols,
    }

    return out, model_meta

# =============================================================================
# Step 8: Build coverage, summary, correlation tables
# =============================================================================

def build_coverage_table(df):
    variables = [
        # Response phenotype variables.
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
        "satbreak_fraction_independent",
        "satbreak_fraction_all",
        "negative_slope_fraction_independent",
        "negative_slope_fraction_all",
        "product_agreement_independent",
        "product_agreement_all",

        # Plant physiology.
        "p50",
        "xylem_vulnerability_p50",
        "rooting_depth",
        "rooting_zone_storage_rooting_depth",
        "isohydricity",
        "stomatal_strategy_isohydricity",

        # Environmental controls.
        "aridity",
        "mean_vpd",
        "median_vpd",
        "p90_vpd",
        "mean_soil_moisture",
        "median_soil_moisture",
        "p10_soil_moisture",
        "mean_annual_precipitation",
        "mean_precipitation",
        "mean_annual_temperature",
        "mean_temperature",
        "mean_lai",
        "growing_season_mean_lai",

        # Soil controls.
        "soil_sand",
        "soil_silt",
        "soil_clay",

        # Spatial controls.
        "lat",
        "lon",
        "abs_lat",
    ]

    rows = []
    for c in variables:
        if c not in df.columns:
            rows.append({
                "variable": c,
                "present_in_dataset": False,
                "finite_n": 0,
                "total_n": len(df),
                "coverage_fraction": 0.0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
            continue

        x = to_numeric_series(df[c])
        n = int(x.notna().sum())
        rows.append({
            "variable": c,
            "present_in_dataset": True,
            "finite_n": n,
            "total_n": len(df),
            "coverage_fraction": n / len(df) if len(df) else np.nan,
            "mean": float(x.mean()) if n else np.nan,
            "sd": float(x.std()) if n > 1 else np.nan,
            "min": float(x.min()) if n else np.nan,
            "max": float(x.max()) if n else np.nan,
        })

    # Model-ready flags.
    for c in [
        "core_trait_model_ready",
        "core_trait_climate_model_ready",
        "core_trait_climate_soil_model_ready",
        "full_trait_with_isohydricity_ready",
        "isohydricity_available",
        "has_all_9_product_combos",
        "has_independent_subset_complete",
    ]:
        if c in df.columns:
            s = df[c].fillna(False).astype(bool)
            rows.append({
                "variable": c,
                "present_in_dataset": True,
                "finite_n": int(s.sum()),
                "total_n": len(df),
                "coverage_fraction": float(s.mean()) if len(df) else np.nan,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
        else:
            rows.append({
                "variable": c,
                "present_in_dataset": False,
                "finite_n": 0,
                "total_n": len(df),
                "coverage_fraction": 0.0,
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
            })

    return pd.DataFrame(rows)

def build_summary_table(df):
    """
    Robust version: never calls quantile on a DataFrame.
    This fixes the crash from the previous Phase 4 run.
    """
    df = make_unique_columns(df)
    rows = []

    for c in df.columns:
        x = to_numeric_series(df[c])
        n = int(x.notna().sum())
        if n == 0:
            continue

        rows.append({
            "variable": c,
            "n": n,
            "mean": float(x.mean()) if n else np.nan,
            "sd": float(x.std()) if n > 1 else np.nan,
            "median": float(x.median()) if n else np.nan,
            "p10": safe_quantile(x, 0.10),
            "p25": safe_quantile(x, 0.25),
            "p75": safe_quantile(x, 0.75),
            "p90": safe_quantile(x, 0.90),
            "min": float(x.min()) if n else np.nan,
            "max": float(x.max()) if n else np.nan,
        })

    return pd.DataFrame(rows)

def build_correlation_table(df):
    preferred = [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
        "satbreak_fraction_all",
        "product_agreement_all",

        "p50",
        "rooting_depth",
        "isohydricity",
        "aridity",

        "mean_vpd",
        "mean_soil_moisture",
        "mean_annual_precipitation",
        "mean_annual_temperature",
        "growing_season_mean_lai",

        "soil_sand",
        "soil_silt",
        "soil_clay",

        "lat",
        "lon",
        "abs_lat",
    ]

    cols = [c for c in preferred if c in df.columns and finite_count(df[c]) >= 3]

    if len(cols) < 2:
        return pd.DataFrame(columns=["var1", "var2", "pearson_r"]), cols

    d = pd.DataFrame({c: to_numeric_series(df[c]) for c in cols})
    corr = d.corr(method="pearson")

    # Pandas-version-safe reshape.
    # Older pandas allowed corr.stack(dropna=False); newer pandas errors because
    # the new stack implementation does not support dropna.
    # melt() preserves the full correlation matrix, including NaN cells.
    corr_long = (
        corr.rename_axis(index="var1")
        .reset_index()
        .melt(id_vars="var1", var_name="var2", value_name="pearson_r")
    )

    return corr_long, cols

def build_model_sample_manifest(df):
    rows = []
    for flag in [
        "core_trait_model_ready",
        "core_trait_climate_model_ready",
        "core_trait_climate_soil_model_ready",
        "full_trait_with_isohydricity_ready",
        "isohydricity_available",
        "has_all_9_product_combos",
        "has_independent_subset_complete",
        "has_pml_containing_subset_complete",
        "has_gosif_gpp_subset_complete",
        "has_gleam_et_subset_complete",
    ]:
        if flag in df.columns:
            s = df[flag].fillna(False).astype(bool)
            rows.append({
                "sample_flag": flag,
                "n_points": int(s.sum()),
                "total_points": int(len(df)),
                "fraction": float(s.mean()) if len(df) else np.nan,
            })
        else:
            rows.append({
                "sample_flag": flag,
                "n_points": 0,
                "total_points": int(len(df)),
                "fraction": 0.0,
            })

    return pd.DataFrame(rows)

# =============================================================================
# Step 9: Figures
# =============================================================================

def plot_coverage_matrix(df):
    cols = [
        c for c in [
            "consensus_slope_change_independent",
            "consensus_slope_change_all",
            "consensus_post_slope_independent",
            "consensus_post_slope_all",
            "p50",
            "rooting_depth",
            "isohydricity",
            "aridity",
            "mean_vpd",
            "mean_soil_moisture",
            "mean_annual_precipitation",
            "mean_annual_temperature",
            "growing_season_mean_lai",
            "soil_sand",
            "soil_silt",
            "soil_clay",
        ]
        if c in df.columns
    ]

    if not cols:
        return None

    mat = pd.DataFrame({c: df[c].notna().astype(int) for c in cols}).T

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(cols))))
    im = ax.imshow(mat.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("Point index")
    ax.set_title("Figure 3A. Covariate coverage matrix")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Present")
    fig.tight_layout()

    out_png = OUTDIR / "Figure3A_covariate_coverage_matrix.png"
    out_pdf = OUTDIR / "Figure3A_covariate_coverage_matrix.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png

def plot_correlation_heatmap(df, cols):
    if len(cols) < 2:
        return None

    d = pd.DataFrame({c: to_numeric_series(df[c]) for c in cols})
    corr = d.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(cols)), max(7, 0.55 * len(cols))))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Figure 3B. Covariate correlation heatmap")

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()

    out_png = OUTDIR / "Figure3B_covariate_correlation_heatmap.png"
    out_pdf = OUTDIR / "Figure3B_covariate_correlation_heatmap.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png

def plot_response_trait_scatters(df):
    response = None
    for c in [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
    ]:
        if c in df.columns and finite_count(df[c]) >= 10:
            response = c
            break

    if response is None:
        return []

    traits = [
        "p50",
        "rooting_depth",
        "isohydricity",
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        "soil_sand",
        "soil_clay",
    ]

    out_paths = []

    for trait in traits:
        if trait not in df.columns or finite_count(df[trait]) < 3:
            continue

        x = to_numeric_series(df[trait])
        y = to_numeric_series(df[response])
        ok = x.notna() & y.notna()

        if ok.sum() < 3:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x[ok], y[ok], alpha=0.75)
        ax.set_xlabel(trait)
        ax.set_ylabel(response)
        ax.set_title(f"{response} vs {trait} (n={int(ok.sum())})")

        try:
            coef = np.polyfit(x[ok], y[ok], 1)
            xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, linewidth=1)
        except Exception:
            pass

        fig.tight_layout()
        out_png = OUTDIR / f"Figure3_scatter_{response}_vs_{trait}.png"
        out_pdf = OUTDIR / f"Figure3_scatter_{response}_vs_{trait}.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)

        out_paths.append(str(out_png))

    return out_paths

# =============================================================================
# Main execution
# =============================================================================

def main():
    print("PHASE 4 START")

    # 1. Load product-consensus response phenotypes.
    dataset = load_response()
    print(f"Loaded response phenotype table: {RESPONSE_PATH} {dataset.shape}")

    # 2. Merge plant physiology traits.
    dataset, trait_metas = add_traits(dataset)
    print("Merged trait rasters.")

    # 3. Merge aridity.
    dataset, aridity_meta = add_aridity(dataset)
    print("Merged aridity.")

    # 4. Add climate controls.
    dataset, climate_meta = add_climate_controls(dataset)
    print("Merged climate controls.")

    # 5. Add soil texture controls.
    dataset, soil_meta = add_soil_controls(dataset)
    print("Merged soil controls.")

    # 6. Add stable point labels / biome labels where available.
    dataset, stable_point_meta = add_stable_point_labels(dataset)
    print("Merged stable point / biome labels.")

    # 7. Add spatial labels and region fixed-effect style labels.
    dataset = add_spatial_labels(dataset)
    print("Added spatial labels.")

    # 8. Add z-scores and model-ready flags.
    dataset, model_meta = add_zscores_and_flags(dataset)
    print("Added z-scores and model-ready flags.")

    # 9. Final duplicate-column safety.
    dataset = make_unique_columns(dataset)

    # 10. Save main dataset.
    save_csv(dataset, MAIN_OUT)

    # 11. Coverage table.
    coverage = build_coverage_table(dataset)
    save_csv(coverage, OUTDIR / "trait_covariate_coverage.csv")

    # 12. Summary table.
    summary = build_summary_table(dataset)
    save_csv(summary, OUTDIR / "trait_covariate_summary.csv")

    # 13. Correlation table.
    corr_long, corr_cols = build_correlation_table(dataset)
    save_csv(corr_long, OUTDIR / "trait_covariate_correlations.csv")

    # 14. Model sample manifest.
    sample_manifest = build_model_sample_manifest(dataset)
    save_csv(sample_manifest, OUTDIR / "model_sample_manifest.csv")

    # 15. Trait raster metadata.
    trait_meta_df = pd.DataFrame(trait_metas)
    save_csv(trait_meta_df, OUTDIR / "trait_raster_sampling_metadata.csv")

    # 16. Environmental merge metadata.
    env_meta = {
        "aridity_merge": aridity_meta,
        "climate_merge": climate_meta,
        "soil_merge": soil_meta,
        "stable_point_merge": stable_point_meta,
    }
    with open(OUTDIR / "environmental_merge_metadata.json", "w") as f:
        json.dump(env_meta, f, indent=2, default=json_default)

    # 17. Figures.
    fig_cov = plot_coverage_matrix(dataset)
    fig_corr = plot_correlation_heatmap(dataset, corr_cols)
    scatter_paths = plot_response_trait_scatters(dataset)

    # 18. Manifest.
    manifest = {
        "phase": "Phase 4: Merge plant physiology and environmental covariates",
        "input_response": str(RESPONSE_PATH),
        "output_dataset": str(MAIN_OUT),
        "n_points": int(dataset["point_id"].nunique()),
        "dataset_shape": list(dataset.shape),
        "trait_predictors": {
            "p50_xylem_vulnerability": "p50 / xylem_vulnerability_p50",
            "rooting_depth_rooting_zone_storage": "rooting_depth / rooting_zone_storage_rooting_depth",
            "isohydricity_stomatal_strategy": "isohydricity / stomatal_strategy_isohydricity",
        },
        "environmental_controls": {
            "aridity": "aridity",
            "mean_vpd": "mean_vpd",
            "mean_soil_moisture": "mean_soil_moisture",
            "mean_annual_precipitation": "mean_annual_precipitation",
            "mean_annual_temperature": "mean_annual_temperature",
            "growing_season_lai": "growing_season_mean_lai",
            "soil_sand_silt_clay": ["soil_sand", "soil_silt", "soil_clay"],
            "spatial_labels": ["lat", "lon", "abs_lat", "hemisphere", "lat_band", "lon_region", "region_label", "aridity_quartile", "biome_label"],
        },
        "model_metadata": model_meta,
        "trait_sampling": trait_metas,
        "merge_metadata": env_meta,
        "coverage_table": str(OUTDIR / "trait_covariate_coverage.csv"),
        "summary_table": str(OUTDIR / "trait_covariate_summary.csv"),
        "correlation_table": str(OUTDIR / "trait_covariate_correlations.csv"),
        "model_sample_manifest": str(OUTDIR / "model_sample_manifest.csv"),
        "figures": {
            "coverage_matrix": str(fig_cov) if fig_cov else None,
            "correlation_heatmap": str(fig_corr) if fig_corr else None,
            "scatterplots": scatter_paths,
        },
    }

    with open(OUTDIR / "phase4_trait_environment_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=json_default)

    # 19. README.
    readme = []
    readme.append("# Phase 4: Trait and environmental covariate merge")
    readme.append("")
    readme.append("## Goal")
    readme.append("")
    readme.append("Attach hydraulic/rooting/climate/soil information to each product-consensus WUE response phenotype.")
    readme.append("")
    readme.append("## Main output")
    readme.append("")
    readme.append(f"- `{MAIN_OUT}`")
    readme.append("")
    readme.append("## Main trait predictors")
    readme.append("")
    readme.append("- `p50` / `xylem_vulnerability_p50`: P50 / xylem vulnerability")
    readme.append("- `rooting_depth` / `rooting_zone_storage_rooting_depth`: rooting-zone storage")
    readme.append("- `isohydricity` / `stomatal_strategy_isohydricity`: stomatal strategy; limited coverage sensitivity predictor")
    readme.append("")
    readme.append("## Environmental controls")
    readme.append("")
    readme.append("- `aridity`")
    readme.append("- `mean_vpd`, `p90_vpd`")
    readme.append("- `mean_soil_moisture`, `p10_soil_moisture`")
    readme.append("- `mean_annual_precipitation`")
    readme.append("- `mean_annual_temperature`")
    readme.append("- `mean_lai`, `growing_season_mean_lai`")
    readme.append("- `soil_sand`, `soil_silt`, `soil_clay` when available")
    readme.append("- `lat`, `lon`, `abs_lat`, `lat_band`, `lon_region`, `region_label`, `aridity_quartile`, `biome_label`")
    readme.append("")
    readme.append("## Output tables")
    readme.append("")
    for p in [
        MAIN_OUT,
        OUTDIR / "trait_covariate_coverage.csv",
        OUTDIR / "trait_covariate_summary.csv",
        OUTDIR / "trait_covariate_correlations.csv",
        OUTDIR / "model_sample_manifest.csv",
        OUTDIR / "trait_raster_sampling_metadata.csv",
        OUTDIR / "environmental_merge_metadata.json",
        OUTDIR / "phase4_trait_environment_manifest.json",
    ]:
        readme.append(f"- `{p}`")
    readme.append("")
    readme.append("## Output figures")
    readme.append("")
    if fig_cov:
        readme.append(f"- `{fig_cov}`")
    if fig_corr:
        readme.append(f"- `{fig_corr}`")
    for p in scatter_paths:
        readme.append(f"- `{p}`")
    readme.append("")
    readme.append("## Model sample manifest")
    readme.append("")
    readme.append(sample_manifest.to_string(index=False))
    readme.append("")
    readme.append("## Trait raster sampling metadata")
    readme.append("")
    readme.append(trait_meta_df.to_string(index=False))
    readme.append("")
    readme.append("## Interpretation")
    readme.append("")
    readme.append("P50/xylem vulnerability and rooting depth/rooting-zone storage are the core physiology predictors. Isohydricity is included but should be treated as a limited-coverage sensitivity analysis unless its sample size is adequate.")
    readme.append("")
    readme.append("Use `core_trait_model_ready` or `core_trait_climate_model_ready` for the first Phase 5 models. Use `full_trait_with_isohydricity_ready` only if the N is scientifically usable.")
    readme.append("")
    readme.append("## Manifest")
    readme.append("")
    readme.append(json.dumps(manifest, indent=2, default=json_default))

    readme_path = OUTDIR / "README_phase4_trait_environment_merge.md"
    readme_path.write_text("\n".join(readme))
    print(f"WROTE {readme_path}")

    print("")
    print("DONE Phase 4.")
    print("")
    print("MODEL SAMPLE MANIFEST:")
    print(sample_manifest.to_string(index=False))
    print("")
    print("COVARIATE COVERAGE:")
    print(coverage.to_string(index=False))
    print("")
    print("MANIFEST:")
    print(json.dumps(manifest, indent=2, default=json_default))

if __name__ == "__main__":
    main()
