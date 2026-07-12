#!/usr/bin/env python
from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# Point-time data candidates, in preferred order.
POINT_TIME_CANDIDATES = [
    Path("data/raw/agents/merged_full_matrix_co2corrected.csv"),
    Path("data/processed/reza_metric_matrix_co2corrected.csv"),
    Path("data/raw/agents/merged_full_matrix_raw.csv"),
    Path("data/processed/reza_metric_matrix_raw.csv"),
]

# Soil/covariate file candidates. The script also scans folders below.
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
# Helpers
# =============================================================================

def die(msg):
    raise SystemExit("\nERROR: " + msg + "\n")

def normalize_cols(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

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

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def pick_cols_contains(cols, must_terms=None, any_terms=None, avoid_terms=None):
    must_terms = [t.lower() for t in (must_terms or [])]
    any_terms = [t.lower() for t in (any_terms or [])]
    avoid_terms = [t.lower() for t in (avoid_terms or [])]
    out = []
    for c in cols:
        cl = c.lower()
        if any(a in cl for a in avoid_terms):
            continue
        if must_terms and not all(t in cl for t in must_terms):
            continue
        if any_terms and not any(t in cl for t in any_terms):
            continue
        out.append(c)
    return out

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(skipna=True)
    mu = x.mean(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def finite_count(s):
    return int(pd.to_numeric(s, errors="coerce").notna().sum())

def finite_frac(s):
    n = len(s)
    if n == 0:
        return np.nan
    return finite_count(s) / n

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"WROTE {path} {df.shape}")

# =============================================================================
# Trait raster sampling
# =============================================================================

def infer_lat_lon_coord(ds):
    coords = list(ds.coords)
    dims = list(ds.dims)

    lat_candidates = []
    lon_candidates = []

    for c in coords + dims:
        cl = str(c).lower()
        if cl in ["lat", "latitude", "y"]:
            lat_candidates.append(c)
        if cl in ["lon", "longitude", "x"]:
            lon_candidates.append(c)

    if not lat_candidates:
        for c in coords + dims:
            cl = str(c).lower()
            if "lat" in cl:
                lat_candidates.append(c)
    if not lon_candidates:
        for c in coords + dims:
            cl = str(c).lower()
            if "lon" in cl:
                lon_candidates.append(c)

    lat_name = lat_candidates[0] if lat_candidates else None
    lon_name = lon_candidates[0] if lon_candidates else None
    return lat_name, lon_name

def infer_data_var(ds, preferred_terms):
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None

    # Prefer variables containing requested terms.
    for term in preferred_terms:
        term = term.lower()
        for v in data_vars:
            if term in str(v).lower():
                return v

    # Otherwise first variable with at least 2 dimensions.
    for v in data_vars:
        try:
            if len(ds[v].dims) >= 2:
                return v
        except Exception:
            pass

    return data_vars[0]

def normalize_lon_for_dataset(lon, lon_values):
    lon = float(lon)
    arr = np.asarray(lon_values, dtype=float)
    if np.nanmin(arr) >= 0 and lon < 0:
        return lon + 360.0
    if np.nanmax(arr) > 180 and lon < 0:
        return lon + 360.0
    if np.nanmin(arr) < 0 and lon > 180:
        return lon - 360.0
    return lon

def sample_trait_nc(points, trait_name, path):
    """
    Returns:
      values_df: point_id + trait column
      meta: dict
    """
    values_col = trait_name
    out = points[["point_id", "lat", "lon"]].copy()
    out[values_col] = np.nan

    meta = {
        "trait_name": trait_name,
        "path": str(path),
        "exists": path.exists(),
        "var_name": None,
        "lat_coord": None,
        "lon_coord": None,
        "finite_count": 0,
        "total_points": int(len(points)),
        "coverage_fraction": 0.0,
        "status": "not_started",
    }

    if not path.exists():
        meta["status"] = "missing_file"
        return out[["point_id", values_col]], meta

    try:
        import xarray as xr
    except Exception as e:
        meta["status"] = f"xarray_import_failed: {type(e).__name__}: {e}"
        return out[["point_id", values_col]], meta

    try:
        ds = xr.open_dataset(path)
        lat_name, lon_name = infer_lat_lon_coord(ds)

        preferred = {
            "p50": ["p50", "psi50", "xylem"],
            "rooting_depth": ["root", "depth"],
            "isohydricity": ["isohyd", "hydric"],
        }.get(trait_name, [trait_name])

        var_name = infer_data_var(ds, preferred)

        meta["var_name"] = str(var_name)
        meta["lat_coord"] = str(lat_name)
        meta["lon_coord"] = str(lon_name)

        if lat_name is None or lon_name is None or var_name is None:
            meta["status"] = "could_not_infer_coords_or_var"
            return out[["point_id", values_col]], meta

        lon_values = ds[lon_name].values

        vals = []
        for _, r in points.iterrows():
            lat = float(r["lat"])
            lon = normalize_lon_for_dataset(float(r["lon"]), lon_values)
            try:
                da = ds[var_name].sel({lat_name: lat, lon_name: lon}, method="nearest")
                val = float(np.asarray(da.values).squeeze())
                if not np.isfinite(val):
                    val = np.nan
                vals.append(val)
            except Exception:
                vals.append(np.nan)

        out[values_col] = vals
        meta["finite_count"] = finite_count(out[values_col])
        meta["coverage_fraction"] = finite_frac(out[values_col])
        meta["status"] = "ok"
        try:
            ds.close()
        except Exception:
            pass
        return out[["point_id", values_col]], meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        return out[["point_id", values_col]], meta

# =============================================================================
# Aridity merge
# =============================================================================

def merge_aridity(base):
    meta = {
        "path": str(ARIDITY_PATH),
        "exists": ARIDITY_PATH.exists(),
        "status": "not_started",
        "columns_added": [],
    }

    if not ARIDITY_PATH.exists():
        meta["status"] = "missing_file"
        return base, meta

    try:
        ar = normalize_cols(pd.read_csv(ARIDITY_PATH, low_memory=False))
    except Exception as e:
        meta["status"] = f"read_failed: {type(e).__name__}: {e}"
        return base, meta

    if "point_id" in ar.columns:
        ar["point_id"] = ar["point_id"].astype(str)
    else:
        meta["status"] = "no_point_id"
        return base, meta

    # Keep likely aridity columns and coordinates if present.
    keep = ["point_id"]
    for c in ar.columns:
        cl = c.lower()
        if c == "point_id":
            continue
        if (
            "aridity" in cl
            or cl in ["ai", "arid_index", "aridity_index"]
            or "pet" in cl
            or "precip" in cl
            or "map" == cl
            or "mat" == cl
        ):
            keep.append(c)

    keep = list(dict.fromkeys(keep))
    ar = ar[keep].drop_duplicates("point_id")

    # Rename only the clearest aridity column to aridity if no aridity exists.
    aridity_col = None
    for cand in ["aridity", "aridity_index", "ai", "arid_index"]:
        if cand in ar.columns:
            aridity_col = cand
            break

    if aridity_col is None:
        possible = [c for c in ar.columns if "aridity" in c.lower()]
        if possible:
            aridity_col = possible[0]

    if aridity_col is not None and aridity_col != "aridity":
        ar = ar.rename(columns={aridity_col: "aridity"})

    before = set(base.columns)
    out = base.merge(ar, on="point_id", how="left", suffixes=("", "_aridity"))
    added = [c for c in out.columns if c not in before]
    meta["columns_added"] = added
    meta["status"] = "ok"
    return out, meta

# =============================================================================
# Climate controls from point-time matrix
# =============================================================================

def find_point_time_file():
    for p in POINT_TIME_CANDIDATES:
        if p.exists():
            return p

    # fallback scan
    patterns = [
        "data/raw/agents/*matrix*co2*.csv",
        "data/raw/agents/*merged*.csv",
        "data/processed/*metric_matrix*co2*.csv",
        "data/processed/*matrix*.csv",
    ]

    for pat in patterns:
        files = sorted(Path(".").glob(pat))
        for f in files:
            if f.exists() and f.stat().st_size > 0:
                return f
    return None

def infer_climate_columns(df):
    cols = list(df.columns)

    vpd_cols = pick_cols_contains(cols, any_terms=["vpd"], avoid_terms=["uwue", "iwue", "wue"])
    sm_cols = pick_cols_contains(cols, any_terms=["soil_moisture", "sm", "swvl"], avoid_terms=["stress", "index"])
    temp_cols = pick_cols_contains(cols, any_terms=["temperature", "temp", "tmean", "t2m"], avoid_terms=[])
    precip_cols = pick_cols_contains(cols, any_terms=["precip", "rain", "ppt", "pr"], avoid_terms=[])
    lai_cols = pick_cols_contains(cols, any_terms=["lai"], avoid_terms=[])
    gpp_cols = pick_cols_contains(cols, any_terms=["gpp"], avoid_terms=["log", "wue", "uwue", "iwue"])
    et_cols = pick_cols_contains(cols, any_terms=["et", "evap"], avoid_terms=["log", "wue", "uwue", "iwue"])

    # Pick one primary column for each broad variable.
    # Use more literal names first.
    def choose(candidates, preferred_exact):
        for pe in preferred_exact:
            for c in candidates:
                if c.lower() == pe.lower():
                    return c
        return candidates[0] if candidates else None

    out = {
        "vpd": choose(vpd_cols, ["vpd", "vpd_mean", "era5_vpd"]),
        "soil_moisture": choose(sm_cols, ["soil_moisture", "sm", "swvl", "era5_soil_moisture"]),
        "temperature": choose(temp_cols, ["temperature", "temp", "tmean", "t2m", "era5_temperature"]),
        "precipitation": choose(precip_cols, ["precipitation", "precip", "ppt", "rain"]),
        "lai": choose(lai_cols, ["lai", "modis_lai"]),
    }

    return out

def add_climate_controls(base):
    meta = {
        "path": None,
        "exists": False,
        "status": "not_started",
        "column_map": {},
        "columns_added": [],
    }

    p = find_point_time_file()
    if p is None:
        meta["status"] = "no_point_time_file_found"
        return base, meta

    meta["path"] = str(p)
    meta["exists"] = True

    try:
        # Read only header first.
        header = pd.read_csv(p, nrows=5, low_memory=False)
        header = normalize_cols(header)
        if "point_id" not in header.columns:
            meta["status"] = "point_time_file_has_no_point_id"
            return base, meta

        cols = list(header.columns)
        keep = ["point_id"]

        # Keep date/year if present.
        for c in cols:
            cl = c.lower()
            if cl in ["date", "year", "doy"]:
                keep.append(c)

        candidates = infer_climate_columns(header)
        meta["column_map"] = candidates

        for c in candidates.values():
            if c is not None and c in cols:
                keep.append(c)

        # Also include maybe climate columns with readable names.
        extra_terms = ["vpd", "soil", "moisture", "swvl", "temp", "precip", "rain", "ppt", "lai"]
        for c in cols:
            cl = c.lower()
            if any(t in cl for t in extra_terms):
                keep.append(c)

        keep = list(dict.fromkeys([c for c in keep if c in cols]))

        pt = pd.read_csv(p, usecols=keep, low_memory=False)
        pt = normalize_cols(pt)
        pt["point_id"] = pt["point_id"].astype(str)

        # Convert all non-id/date fields to numeric when possible.
        for c in pt.columns:
            if c not in ["point_id", "date"]:
                pt[c] = pd.to_numeric(pt[c], errors="coerce")

        if "date" in pt.columns:
            pt["date"] = pd.to_datetime(pt["date"], errors="coerce")
            pt["month"] = pt["date"].dt.month
            pt["year"] = pt["date"].dt.year
        elif "year" in pt.columns:
            pt["year"] = pd.to_numeric(pt["year"], errors="coerce")
            pt["month"] = np.nan

        colmap = infer_climate_columns(pt)

        agg_dict = {}
        rename_dict = {}

        def add_numeric_summary(source_col, prefix):
            if source_col is None or source_col not in pt.columns:
                return
            agg_dict[f"mean_{prefix}"] = (source_col, "mean")
            agg_dict[f"median_{prefix}"] = (source_col, "median")
            agg_dict[f"p10_{prefix}"] = (source_col, lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.10)))
            agg_dict[f"p90_{prefix}"] = (source_col, lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90)))
            agg_dict[f"sd_{prefix}"] = (source_col, "std")
            agg_dict[f"n_{prefix}"] = (source_col, lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))

        add_numeric_summary(colmap.get("vpd"), "vpd")
        add_numeric_summary(colmap.get("soil_moisture"), "soil_moisture")
        add_numeric_summary(colmap.get("temperature"), "temperature")
        add_numeric_summary(colmap.get("precipitation"), "precipitation")
        add_numeric_summary(colmap.get("lai"), "lai")

        # Growing-season LAI: approximate as months Apr-Sep in Northern Hemisphere and Oct-Mar in Southern Hemisphere.
        # Need lat from base merged later. Here compute all-month mean_lai; do growing-season after merge.
        if not agg_dict:
            meta["status"] = "no_climate_columns_detected"
            return base, meta

        climate = pt.groupby("point_id", dropna=False).agg(**agg_dict).reset_index()

        # Growing-season LAI by hemisphere if date/month and lai exist.
        lai_col = colmap.get("lai")
        if lai_col and lai_col in pt.columns and "month" in pt.columns:
            coords = base[["point_id", "lat"]].copy()
            tmp = pt[["point_id", "month", lai_col]].merge(coords, on="point_id", how="left")
            tmp["is_north"] = tmp["lat"] >= 0
            tmp["is_growing_month"] = np.where(
                tmp["is_north"],
                tmp["month"].isin([4,5,6,7,8,9]),
                tmp["month"].isin([10,11,12,1,2,3]),
            )
            gs = (
                tmp[tmp["is_growing_month"]]
                .groupby("point_id")[lai_col]
                .mean()
                .reset_index()
                .rename(columns={lai_col: "growing_season_mean_lai"})
            )
            climate = climate.merge(gs, on="point_id", how="left")

        before = set(base.columns)
        out = base.merge(climate, on="point_id", how="left")
        added = [c for c in out.columns if c not in before]
        meta["columns_added"] = added
        meta["column_map"] = colmap
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        return base, meta

# =============================================================================
# Soil controls
# =============================================================================

def scan_soil_files():
    files = []
    for p in SOIL_CANDIDATES:
        if p.exists():
            files.append(p)

    for d in SCAN_DIRS_FOR_SOIL:
        if not d.exists():
            continue
        for f in d.rglob("*.csv"):
            fl = str(f).lower()
            if any(t in fl for t in ["soil", "sand", "silt", "clay", "soilgrids"]):
                files.append(f)

    # remove duplicates
    seen = set()
    uniq = []
    for f in files:
        s = str(f)
        if s not in seen:
            seen.add(s)
            uniq.append(f)
    return uniq

def infer_soil_cols(df):
    cols = list(df.columns)
    sand = None
    silt = None
    clay = None

    for c in cols:
        cl = c.lower()
        if sand is None and "sand" in cl:
            sand = c
        if silt is None and "silt" in cl:
            silt = c
        if clay is None and "clay" in cl:
            clay = c

    return sand, silt, clay

def add_soil_controls(base):
    meta = {
        "status": "not_started",
        "files_scanned": [],
        "selected_file": None,
        "columns_added": [],
        "soil_columns_detected": {},
    }

    files = scan_soil_files()
    meta["files_scanned"] = [str(f) for f in files]

    best = None
    best_info = None

    for f in files:
        try:
            head = normalize_cols(pd.read_csv(f, nrows=5, low_memory=False))
            if "point_id" not in head.columns:
                continue
            sand, silt, clay = infer_soil_cols(head)
            score = sum(x is not None for x in [sand, silt, clay])
            if score > 0:
                best = f
                best_info = (sand, silt, clay, score)
                if score == 3:
                    break
        except Exception:
            continue

    if best is None:
        meta["status"] = "no_soil_file_found"
        return base, meta

    try:
        soil = normalize_cols(pd.read_csv(best, low_memory=False))
        soil["point_id"] = soil["point_id"].astype(str)
        sand, silt, clay, score = best_info
        keep = ["point_id"]
        rename = {}

        if sand:
            keep.append(sand)
            rename[sand] = "soil_sand"
        if silt:
            keep.append(silt)
            rename[silt] = "soil_silt"
        if clay:
            keep.append(clay)
            rename[clay] = "soil_clay"

        # Keep other useful soil descriptors if present.
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

        # Collapse duplicate point rows.
        soil = soil.groupby("point_id", dropna=False).mean(numeric_only=True).reset_index()

        before = set(base.columns)
        out = base.merge(soil, on="point_id", how="left")
        added = [c for c in out.columns if c not in before]

        meta["selected_file"] = str(best)
        meta["columns_added"] = added
        meta["soil_columns_detected"] = {
            "soil_sand": "soil_sand" in out.columns,
            "soil_silt": "soil_silt" in out.columns,
            "soil_clay": "soil_clay" in out.columns,
        }
        meta["status"] = "ok"
        return out, meta

    except Exception as e:
        meta["status"] = f"failed: {type(e).__name__}: {e}"
        return base, meta

# =============================================================================
# Region/spatial labels
# =============================================================================

def add_spatial_labels(df):
    out = df.copy()
    out["abs_lat"] = pd.to_numeric(out["lat"], errors="coerce").abs()

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

    if "aridity" in out.columns:
        ar = pd.to_numeric(out["aridity"], errors="coerce")
        try:
            out["aridity_quartile"] = pd.qcut(ar, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
            out["aridity_quartile"] = out["aridity_quartile"].astype(str)
        except Exception:
            out["aridity_quartile"] = "unknown"
    else:
        out["aridity_quartile"] = "unknown"

    return out

# =============================================================================
# Model-ready flags and z-scores
# =============================================================================

def add_zscores_and_flags(df):
    out = df.copy()

    # Standardize likely numeric predictors and outcomes.
    z_candidates = [
        "p50",
        "rooting_depth",
        "isohydricity",
        "aridity",
        "mean_vpd",
        "median_vpd",
        "p90_vpd",
        "mean_soil_moisture",
        "median_soil_moisture",
        "p10_soil_moisture",
        "mean_temperature",
        "mean_precipitation",
        "mean_lai",
        "growing_season_mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "lat",
        "lon",
        "abs_lat",
        "consensus_slope_change_all",
        "consensus_slope_change_independent",
        "consensus_post_slope_all",
        "consensus_post_slope_independent",
        "satbreak_fraction_all",
        "satbreak_fraction_independent",
        "product_agreement_all",
        "product_agreement_independent",
    ]

    for c in z_candidates:
        if c in out.columns:
            out[f"z_{c}"] = zscore(out[c])

    # Model predictor columns.
    core_predictors = [
        "p50",
        "rooting_depth",
    ]

    climate_controls = [
        c for c in [
            "aridity",
            "mean_vpd",
            "mean_soil_moisture",
            "mean_precipitation",
            "mean_temperature",
            "growing_season_mean_lai",
            "mean_lai",
        ]
        if c in out.columns
    ]

    soil_controls = [c for c in ["soil_sand", "soil_silt", "soil_clay"] if c in out.columns]

    core_response_options = [
        c for c in [
            "consensus_slope_change_independent",
            "consensus_slope_change_all",
            "consensus_post_slope_independent",
            "consensus_post_slope_all",
        ]
        if c in out.columns
    ]

    # Preferred response: independent if available; otherwise all.
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

    def row_complete(row, cols):
        for c in cols:
            if c not in row.index:
                return False
            try:
                if pd.isna(float(row[c])):
                    return False
            except Exception:
                if pd.isna(row[c]):
                    return False
        return True

    # Core model:
    # response + P50 + rooting depth + at least aridity if available.
    core_cols = []
    if preferred_response:
        core_cols.append(preferred_response)
    core_cols += [c for c in core_predictors if c in out.columns]
    if "aridity" in out.columns:
        core_cols.append("aridity")

    out["core_trait_model_ready"] = out.apply(lambda r: row_complete(r, core_cols), axis=1) if core_cols else False

    # Core + climate controls
    core_climate_cols = core_cols + climate_controls
    out["core_trait_climate_model_ready"] = out.apply(lambda r: row_complete(r, core_climate_cols), axis=1) if core_climate_cols else False

    # Core + climate + soil
    core_soil_cols = core_climate_cols + soil_controls
    out["core_trait_climate_soil_model_ready"] = out.apply(lambda r: row_complete(r, core_soil_cols), axis=1) if core_soil_cols else False

    # Full physiology with isohydricity
    full_cols = core_cols + (["isohydricity"] if "isohydricity" in out.columns else [])
    out["full_trait_with_isohydricity_ready"] = out.apply(lambda r: row_complete(r, full_cols), axis=1) if full_cols else False

    # Product support flags already exist from Phase 3, but ensure.
    for c in [
        "has_all_9_product_combos",
        "has_independent_subset_complete",
        "has_pml_containing_subset_complete",
        "has_gosif_gpp_subset_complete",
        "has_gleam_et_subset_complete",
    ]:
        if c not in out.columns:
            out[c] = False

    return out, {
        "preferred_response": preferred_response,
        "core_cols": core_cols,
        "climate_controls": climate_controls,
        "soil_controls": soil_controls,
        "core_climate_cols": core_climate_cols,
        "core_soil_cols": core_soil_cols,
        "full_cols": full_cols,
    }

# =============================================================================
# Output summaries and figures
# =============================================================================

def build_coverage_table(df):
    important = [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
        "product_agreement_all",
        "satbreak_fraction_all",
        "p50",
        "rooting_depth",
        "isohydricity",
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        "mean_precipitation",
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

    rows = []
    for c in important:
        if c in df.columns:
            rows.append({
                "variable": c,
                "finite_n": finite_count(df[c]),
                "total_n": len(df),
                "coverage_fraction": finite_frac(df[c]),
                "mean": float(pd.to_numeric(df[c], errors="coerce").mean(skipna=True)) if finite_count(df[c]) else np.nan,
                "sd": float(pd.to_numeric(df[c], errors="coerce").std(skipna=True)) if finite_count(df[c]) else np.nan,
                "min": float(pd.to_numeric(df[c], errors="coerce").min(skipna=True)) if finite_count(df[c]) else np.nan,
                "max": float(pd.to_numeric(df[c], errors="coerce").max(skipna=True)) if finite_count(df[c]) else np.nan,
            })

    # Add flags as coverage-ish counts.
    for c in [
        "core_trait_model_ready",
        "core_trait_climate_model_ready",
        "core_trait_climate_soil_model_ready",
        "full_trait_with_isohydricity_ready",
        "has_all_9_product_combos",
        "has_independent_subset_complete",
    ]:
        if c in df.columns:
            s = df[c].fillna(False).astype(bool)
            rows.append({
                "variable": c,
                "finite_n": int(s.sum()),
                "total_n": len(df),
                "coverage_fraction": float(s.mean()),
                "mean": np.nan,
                "sd": np.nan,
                "min": np.nan,
                "max": np.nan,
            })

    return pd.DataFrame(rows)

def build_summary_table(df):
    numeric = []
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= 1:
            numeric.append(c)

    rows = []
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "variable": c,
            "n": int(x.notna().sum()),
            "mean": float(x.mean(skipna=True)) if x.notna().sum() else np.nan,
            "sd": float(x.std(skipna=True)) if x.notna().sum() else np.nan,
            "median": float(x.median(skipna=True)) if x.notna().sum() else np.nan,
            "p10": float(x.quantile(0.10)) if x.notna().sum() else np.nan,
            "p25": float(x.quantile(0.25)) if x.notna().sum() else np.nan,
            "p75": float(x.quantile(0.75)) if x.notna().sum() else np.nan,
            "p90": float(x.quantile(0.90)) if x.notna().sum() else np.nan,
            "min": float(x.min(skipna=True)) if x.notna().sum() else np.nan,
            "max": float(x.max(skipna=True)) if x.notna().sum() else np.nan,
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
        "mean_precipitation",
        "mean_temperature",
        "growing_season_mean_lai",
        "mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
        "lat",
        "lon",
        "abs_lat",
    ]
    cols = [c for c in preferred if c in df.columns and finite_count(df[c]) >= 3]
    if len(cols) < 2:
        return pd.DataFrame(), cols
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method="pearson")
    corr_long = corr.stack(dropna=False).reset_index()
    corr_long.columns = ["var1", "var2", "pearson_r"]
    return corr_long, cols

def plot_missingness(df, out_path):
    cols = [
        c for c in [
            "consensus_slope_change_independent",
            "consensus_slope_change_all",
            "p50",
            "rooting_depth",
            "isohydricity",
            "aridity",
            "mean_vpd",
            "mean_soil_moisture",
            "mean_precipitation",
            "mean_temperature",
            "growing_season_mean_lai",
            "soil_sand",
            "soil_silt",
            "soil_clay",
        ]
        if c in df.columns
    ]
    if not cols:
        return

    mat = df[cols].notna().astype(int).to_numpy().T
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(cols))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("Point index")
    ax.set_title("Phase 4 covariate coverage matrix")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Present")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)

def plot_correlation_heatmap(df, cols, out_path):
    if len(cols) < 2:
        return
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = d.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(8, 0.6*len(cols)), max(7, 0.6*len(cols))))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Phase 4 covariate correlation heatmap")
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(str(out_path).replace(".png", ".pdf"))
    plt.close(fig)

def scatter_response_traits(df):
    out_paths = []
    response_candidates = [
        "consensus_slope_change_independent",
        "consensus_slope_change_all",
        "consensus_post_slope_independent",
        "consensus_post_slope_all",
    ]
    response = None
    for c in response_candidates:
        if c in df.columns and finite_count(df[c]) >= 10:
            response = c
            break
    if response is None:
        return out_paths

    traits = [c for c in ["p50", "rooting_depth", "isohydricity", "aridity"] if c in df.columns and finite_count(df[c]) >= 3]
    for t in traits:
        x = pd.to_numeric(df[t], errors="coerce")
        y = pd.to_numeric(df[response], errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 3:
            continue
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(x[ok], y[ok], alpha=0.75)
        ax.set_xlabel(t)
        ax.set_ylabel(response)
        ax.set_title(f"{response} vs {t} (n={int(ok.sum())})")
        # Add simple linear fit.
        try:
            coef = np.polyfit(x[ok], y[ok], 1)
            xx = np.linspace(x[ok].min(), x[ok].max(), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, linewidth=1)
        except Exception:
            pass
        fig.tight_layout()
        path = OUTDIR / f"scatter_{response}_vs_{t}.png"
        fig.savefig(path, dpi=300)
        fig.savefig(str(path).replace(".png", ".pdf"))
        plt.close(fig)
        out_paths.append(str(path))
    return out_paths

# =============================================================================
# Main
# =============================================================================

def main():
    if not RESPONSE_PATH.exists():
        die(f"Missing Phase 3 response phenotype table: {RESPONSE_PATH}\nRun Phase 3 first.")

    response = normalize_cols(pd.read_csv(RESPONSE_PATH, low_memory=False))

    if "point_id" not in response.columns:
        die(f"{RESPONSE_PATH} does not contain point_id.")

    response["point_id"] = response["point_id"].astype(str)

    # Ensure lat/lon.
    if "lat" not in response.columns or "lon" not in response.columns:
        lonlat = response["point_id"].apply(lambda x: pd.Series(parse_point_id(x)))
        response["lon"] = lonlat[0]
        response["lat"] = lonlat[1]

    response["lat"] = pd.to_numeric(response["lat"], errors="coerce")
    response["lon"] = pd.to_numeric(response["lon"], errors="coerce")

    # Start dataset.
    dataset = response.copy()

    # Sample traits.
    trait_metas = []
    trait_values = []

    points = dataset[["point_id", "lat", "lon"]].drop_duplicates("point_id").copy()

    for trait_name, path in TRAIT_PATHS.items():
        vals, meta = sample_trait_nc(points, trait_name, path)
        trait_values.append(vals)
        trait_metas.append(meta)

    for vals in trait_values:
        dataset = dataset.merge(vals, on="point_id", how="left")

    # Alias columns for desired wording.
    # Keep p50 as P50/xylem vulnerability.
    if "p50" in dataset.columns:
        dataset["xylem_vulnerability_p50"] = dataset["p50"]
    if "rooting_depth" in dataset.columns:
        dataset["rooting_zone_storage_rooting_depth"] = dataset["rooting_depth"]
    if "isohydricity" in dataset.columns:
        dataset["stomatal_strategy_isohydricity"] = dataset["isohydricity"]

    # Merge aridity.
    dataset, aridity_meta = merge_aridity(dataset)

    # Add climate controls.
    dataset, climate_meta = add_climate_controls(dataset)

    # Add soil controls.
    dataset, soil_meta = add_soil_controls(dataset)

    # Spatial/region controls.
    dataset = add_spatial_labels(dataset)

    # Numeric cleanup for key columns.
    numeric_like = [
        "p50",
        "xylem_vulnerability_p50",
        "rooting_depth",
        "rooting_zone_storage_rooting_depth",
        "isohydricity",
        "stomatal_strategy_isohydricity",
        "aridity",
        "mean_vpd",
        "mean_soil_moisture",
        "mean_precipitation",
        "mean_temperature",
        "mean_lai",
        "growing_season_mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
    ]
    dataset = safe_numeric(dataset, numeric_like)

    # Add flags and z-scores.
    dataset, model_meta = add_zscores_and_flags(dataset)

    # Save main dataset.
    save_csv(dataset, MAIN_OUT)

    # Coverage, summary, correlations.
    coverage = build_coverage_table(dataset)
    save_csv(coverage, OUTDIR / "trait_covariate_coverage.csv")

    summary = build_summary_table(dataset)
    save_csv(summary, OUTDIR / "trait_covariate_summary.csv")

    corr_long, corr_cols = build_correlation_table(dataset)
    if len(corr_long):
        save_csv(corr_long, OUTDIR / "trait_covariate_correlations.csv")
    else:
        save_csv(pd.DataFrame(columns=["var1", "var2", "pearson_r"]), OUTDIR / "trait_covariate_correlations.csv")

    # Model sample manifest table.
    manifest_rows = []
    for flag in [
        "core_trait_model_ready",
        "core_trait_climate_model_ready",
        "core_trait_climate_soil_model_ready",
        "full_trait_with_isohydricity_ready",
        "has_all_9_product_combos",
        "has_independent_subset_complete",
    ]:
        if flag in dataset.columns:
            s = dataset[flag].fillna(False).astype(bool)
            manifest_rows.append({
                "sample_flag": flag,
                "n_points": int(s.sum()),
                "total_points": int(len(dataset)),
                "fraction": float(s.mean()),
            })

    sample_manifest = pd.DataFrame(manifest_rows)
    save_csv(sample_manifest, OUTDIR / "model_sample_manifest.csv")

    # Trait metadata.
    trait_meta_df = pd.DataFrame(trait_metas)
    save_csv(trait_meta_df, OUTDIR / "trait_raster_sampling_metadata.csv")

    merge_meta = {
        "phase": "Phase 4: Merge plant physiology and environmental covariates",
        "input_response": str(RESPONSE_PATH),
        "output_dataset": str(MAIN_OUT),
        "n_points": int(dataset["point_id"].nunique()),
        "dataset_shape": list(dataset.shape),
        "trait_sampling": trait_metas,
        "aridity_merge": aridity_meta,
        "climate_merge": climate_meta,
        "soil_merge": soil_meta,
        "model_metadata": model_meta,
        "coverage_table": str(OUTDIR / "trait_covariate_coverage.csv"),
        "summary_table": str(OUTDIR / "trait_covariate_summary.csv"),
        "correlation_table": str(OUTDIR / "trait_covariate_correlations.csv"),
        "model_sample_manifest": str(OUTDIR / "model_sample_manifest.csv"),
    }

    with open(OUTDIR / "phase4_trait_environment_manifest.json", "w") as f:
        json.dump(merge_meta, f, indent=2)

    # Figures.
    plot_missingness(dataset, OUTDIR / "Figure3A_covariate_coverage_matrix.png")
    if corr_cols:
        plot_correlation_heatmap(dataset, corr_cols, OUTDIR / "Figure3B_covariate_correlation_heatmap.png")
    scatter_paths = scatter_response_traits(dataset)

    # README.
    readme = []
    readme.append("# Phase 4: Trait and environmental covariate merge")
    readme.append("")
    readme.append("## Goal")
    readme.append("")
    readme.append("Attach hydraulic, rooting, stomatal, climate, soil, and spatial covariates to the product-consensus WUE response phenotype.")
    readme.append("")
    readme.append("## Main output")
    readme.append("")
    readme.append(f"- `{MAIN_OUT}`")
    readme.append("")
    readme.append("## Key output tables")
    readme.append("")
    for p in [
        OUTDIR / "trait_covariate_coverage.csv",
        OUTDIR / "trait_covariate_summary.csv",
        OUTDIR / "trait_covariate_correlations.csv",
        OUTDIR / "model_sample_manifest.csv",
        OUTDIR / "trait_raster_sampling_metadata.csv",
        OUTDIR / "phase4_trait_environment_manifest.json",
    ]:
        readme.append(f"- `{p}`")
    readme.append("")
    readme.append("## Key figures")
    readme.append("")
    for p in [
        OUTDIR / "Figure3A_covariate_coverage_matrix.png",
        OUTDIR / "Figure3B_covariate_correlation_heatmap.png",
    ]:
        if p.exists():
            readme.append(f"- `{p}`")
    for p in scatter_paths:
        readme.append(f"- `{p}`")
    readme.append("")
    readme.append("## Trait raster sampling")
    readme.append("")
    readme.append(trait_meta_df.to_string(index=False))
    readme.append("")
    readme.append("## Model sample manifest")
    readme.append("")
    readme.append(sample_manifest.to_string(index=False))
    readme.append("")
    readme.append("## Interpretation")
    readme.append("")
    readme.append("P50/xylem vulnerability and rooting depth/rooting-zone storage are the core physiology predictors. Isohydricity is included where available but should only become a main predictor if coverage is adequate. Otherwise it is a limited-coverage sensitivity analysis.")
    readme.append("")
    readme.append("The model-ready flags prevent overclaiming. Use `core_trait_model_ready` or `core_trait_climate_model_ready` for the first trait models. Use `full_trait_with_isohydricity_ready` only if its N is scientifically usable.")
    readme.append("")
    readme.append("## Manifest")
    readme.append("")
    readme.append(json.dumps(merge_meta, indent=2))
    Path(OUTDIR / "README_phase4_trait_environment_merge.md").write_text("\n".join(readme))

    print("")
    print("DONE Phase 4.")
    print("")
    print(f"MAIN DATASET: {MAIN_OUT}")
    print("")
    print("MODEL SAMPLE MANIFEST:")
    print(sample_manifest.to_string(index=False))
    print("")
    print("COVERAGE:")
    print(coverage.to_string(index=False))
    print("")
    print("MANIFEST:")
    print(json.dumps(merge_meta, indent=2))

if __name__ == "__main__":
    main()
