from pathlib import Path
from datetime import datetime, timedelta
import re
import warnings

import earthaccess
import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore")

RAW = Path("data/raw/smap_8day")
RAW.mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results/stress").mkdir(parents=True, exist_ok=True)
Path("docs").mkdir(parents=True, exist_ok=True)

POINTS_FILE = Path("data/raw/gee/stable_grassland_points.csv")
GEE_DIR = Path("data/raw/gee")

START = "2015-04-01"
END = "2024-12-31"

print("Loading points...", flush=True)
pts = pd.read_csv(POINTS_FILE)
lat_col = "lat" if "lat" in pts.columns else "latitude"
lon_col = "lon" if "lon" in pts.columns else "longitude"
pts = pts[["point_id", lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"})
pts["point_id"] = pts["point_id"].astype(str)
print("Points:", len(pts), flush=True)

print("Loading local ERA5/GEE soil moisture...", flush=True)
gee_files = sorted(GEE_DIR.glob("wue_timeseries_*.csv"))
if not gee_files:
    raise SystemExit("No wue_timeseries_*.csv files found in data/raw/gee")

gee = pd.concat([pd.read_csv(p) for p in gee_files], ignore_index=True)
gee["date"] = pd.to_datetime(gee["date"])
gee = gee[(gee["date"] >= START) & (gee["date"] <= END)].copy()

if "soil_moisture" in gee.columns:
    soil_col = "soil_moisture"
else:
    soil_candidates = [c for c in gee.columns if "soil" in c.lower()]
    if not soil_candidates:
        print("Columns:", list(gee.columns))
        raise SystemExit("Could not find soil moisture column.")
    soil_col = soil_candidates[0]

print("Using ERA5/local soil column:", soil_col, flush=True)
gee = gee[["point_id", "date", soil_col]].rename(columns={soil_col: "soil_moisture"})
gee["point_id"] = gee["point_id"].astype(str)

print("Logging into Earthdata...", flush=True)
earthaccess.login(strategy="netrc")

print("Searching SMAP SPL4SMGP.008...", flush=True)
results = earthaccess.search_data(
    short_name="SPL4SMGP",
    version="008",
    temporal=(START, END),
    count=200000
)

print("Granules found:", len(results), flush=True)
if len(results) == 0:
    raise SystemExit("No SMAP SPL4SMGP.008 granules found.")

def granule_datetime(r):
    meta = None
    for getter in [
        lambda x: getattr(x, "umm", None),
        lambda x: getattr(x, "metadata", None),
        lambda x: x["umm"] if isinstance(x, dict) and "umm" in x else None,
    ]:
        try:
            meta = getter(r)
            if meta:
                break
        except Exception:
            pass

    if isinstance(meta, dict):
        try:
            dt = meta["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            return pd.to_datetime(dt).to_pydatetime()
        except Exception:
            pass

    txt = str(r)

    for m in re.finditer(r"(20\d{2})(\d{2})(\d{2})[T_ -]?(\d{2})?(\d{2})?(\d{2})?", txt):
        y, mo, da = m.group(1), m.group(2), m.group(3)
        hh = m.group(4) or "12"
        mm = m.group(5) or "00"
        ss = m.group(6) or "00"
        try:
            return datetime(int(y), int(mo), int(da), int(hh), int(mm), int(ss))
        except Exception:
            continue

    for m in re.finditer(r"(20\d{2})-(\d{2})-(\d{2})[T ]?(\d{2})?:?(\d{2})?:?(\d{2})?", txt):
        y, mo, da = m.group(1), m.group(2), m.group(3)
        hh = m.group(4) or "12"
        mm = m.group(5) or "00"
        ss = m.group(6) or "00"
        try:
            return datetime(int(y), int(mo), int(da), int(hh), int(mm), int(ss))
        except Exception:
            continue

    return None

selected = {}
bad_dates = 0

for r in results:
    dt = granule_datetime(r)
    if dt is None:
        bad_dates += 1
        continue

    year_start = datetime(dt.year, 1, 1)
    doy0 = (datetime(dt.year, dt.month, dt.day) - year_start).days
    bin_start = year_start + timedelta(days=(doy0 // 8) * 8)

    score = abs(dt.hour - 12)

    if bin_start not in selected or score < selected[bin_start][0]:
        selected[bin_start] = (score, r)

selected_results = [v[1] for _, v in sorted(selected.items())]

print("Selected 8-day granules:", len(selected_results), flush=True)
print("Granules skipped because date could not be parsed:", bad_dates, flush=True)

if len(selected_results) == 0:
    print(str(results[0])[:4000])
    raise SystemExit("Could not parse any SMAP granule dates.")

Path("data/raw/smap_8day_selected_count.txt").write_text(str(len(selected_results)))

print("Downloading selected SMAP 8-day granules into data/raw/smap_8day...", flush=True)
paths = earthaccess.download(selected_results, str(RAW))
paths = [Path(p) for p in paths if Path(p).exists()]

# Also include already-downloaded files in the 8-day folder if rerunning.
all_paths = sorted(set(paths + list(RAW.rglob("*.h5")) + list(RAW.rglob("*.hdf5")) + list(RAW.rglob("*.nc"))))
print("Local SMAP files available:", len(all_paths), flush=True)

def find_dataset(h5, wanted):
    found = []
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            lname = name.lower()
            for w in wanted:
                if w in lname:
                    found.append(name)
    h5.visititems(walk)
    return found

def read_smap_file(path):
    try:
        f = h5py.File(path, "r")
    except Exception:
        return None

    with f:
        sm_paths = find_dataset(f, ["sm_rootzone"])
        if not sm_paths:
            sm_paths = find_dataset(f, ["rootzone"])

        lat_paths = find_dataset(f, ["cell_lat", "lat"])
        lon_paths = find_dataset(f, ["cell_lon", "lon"])

        if not sm_paths or not lat_paths or not lon_paths:
            return None

        sm_path = sm_paths[0]

        lat_path = next((p for p in lat_paths if "cell_lat" in p.lower()), lat_paths[0])
        lon_path = next((p for p in lon_paths if "cell_lon" in p.lower()), lon_paths[0])

        sm = np.array(f[sm_path][()], dtype="float64")
        lat = np.array(f[lat_path][()], dtype="float64")
        lon = np.array(f[lon_path][()], dtype="float64")

        fill = f[sm_path].attrs.get("_FillValue", f[sm_path].attrs.get("missing_value", None))
        if fill is not None:
            try:
                sm[sm == float(np.array(fill).ravel()[0])] = np.nan
            except Exception:
                pass

        sm[(sm < 0) | (sm > 1)] = np.nan

        if sm.shape != lat.shape or sm.shape != lon.shape:
            return None

        return lat, lon, sm

def file_date(path):
    m = re.search(r"(20\d{6})", path.name)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return pd.NaT

rows = []
tree_cache = None
grid_shape = None

print("Sampling SMAP at points...", flush=True)

for path in tqdm(all_paths):
    parsed = read_smap_file(path)
    if parsed is None:
        continue

    lat_grid, lon_grid, sm_grid = parsed

    if tree_cache is None or grid_shape != lat_grid.shape:
        valid = np.isfinite(lat_grid) & np.isfinite(lon_grid)
        coords = np.column_stack([lat_grid[valid].ravel(), lon_grid[valid].ravel()])
        tree = cKDTree(coords)
        valid_indices = np.argwhere(valid)
        tree_cache = (tree, valid_indices)
        grid_shape = lat_grid.shape

    tree, valid_indices = tree_cache
    query = pts[["lat", "lon"]].to_numpy()
    dist, idx = tree.query(query, k=1)
    grid_idx = valid_indices[idx]

    vals = sm_grid[grid_idx[:, 0], grid_idx[:, 1]]
    d = file_date(path)

    for point_id, val in zip(pts["point_id"], vals):
        if np.isfinite(val):
            rows.append({
                "point_id": point_id,
                "smap_source_date": d,
                "smap_sm_rootzone": float(val)
            })

smap = pd.DataFrame(rows)
if smap.empty:
    raise SystemExit("No SMAP samples extracted from downloaded files.")

print("SMAP sampled rows:", len(smap), flush=True)

def eight_day_bin(series):
    s = pd.to_datetime(series)
    year_start = pd.to_datetime(s.dt.year.astype(str) + "-01-01")
    doy0 = (s - year_start).dt.days
    return year_start + pd.to_timedelta((doy0 // 8) * 8, unit="D")

smap["date"] = eight_day_bin(smap["smap_source_date"])
gee["date"] = eight_day_bin(gee["date"])

smap8 = smap.groupby(["point_id", "date"], as_index=False)["smap_sm_rootzone"].mean()
gee8 = gee.groupby(["point_id", "date"], as_index=False)["soil_moisture"].mean()

matched = gee8.merge(smap8, on=["point_id", "date"], how="inner")
matched = matched.sort_values(["point_id", "date"])

matched.to_csv("data/processed/smap_era5_matched_points.csv", index=False)

x = matched["soil_moisture"].astype(float)
y = matched["smap_sm_rootzone"].astype(float)
mask = x.notna() & y.notna()

r = float(pearsonr(x[mask], y[mask]).statistic) if mask.sum() >= 3 else np.nan
rmse = float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2))) if mask.sum() else np.nan
bias = float(np.mean(x[mask] - y[mask])) if mask.sum() else np.nan

summary = pd.DataFrame([{
    "n_matched": int(mask.sum()),
    "pearson_r": r,
    "rmse": rmse,
    "bias_era5_minus_smap": bias,
    "date_min": str(matched["date"].min()),
    "date_max": str(matched["date"].max()),
    "smap_product": "SPL4SMGP.008",
    "smap_variable": "sm_rootzone",
    "download_strategy": "one 3-hourly SMAP granule per 8-day bin, closest to 12:00 UTC"
}])

summary.to_csv("results/stress/smap_era5_comparison.csv", index=False)

Path("docs/smap_validation.md").write_text(
f"""# SMAP validation

This validation was run without Google Earth Engine.

Product: SPL4SMGP.008.

Variable: `sm_rootzone`.

Date range: {START} to {END}.

Method:
- Selected one SMAP L4 granule per 8-day bin, preferring closest to 12:00 UTC.
- Downloaded files to `data/raw/smap_8day/`.
- Sampled nearest SMAP grid cell at each stable grassland point.
- Aggregated SMAP and local ERA5/local soil moisture to matching 8-day bins.
- Matched by `point_id` and `date`.
- Computed Pearson correlation, RMSE, and ERA5-minus-SMAP bias.

Outputs:
- `data/processed/smap_era5_matched_points.csv`
- `results/stress/smap_era5_comparison.csv`

Summary:
{summary.to_string(index=False)}
"""
)

print("DONE", flush=True)
print(summary.to_string(index=False), flush=True)
