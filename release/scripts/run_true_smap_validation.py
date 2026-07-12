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

RAW = Path("data/raw/smap")
RAW.mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results/stress").mkdir(parents=True, exist_ok=True)
Path("docs").mkdir(parents=True, exist_ok=True)

POINTS_FILE = Path("data/raw/gee/stable_grassland_points.csv")
GEE_DIR = Path("data/raw/gee")

START = "2015-04-01"
END = "2024-12-31"

print("Loading points...")
pts = pd.read_csv(POINTS_FILE)
lat_col = "lat" if "lat" in pts.columns else "latitude"
lon_col = "lon" if "lon" in pts.columns else "longitude"

pts = pts[["point_id", lat_col, lon_col]].copy()
pts = pts.rename(columns={lat_col: "lat", lon_col: "lon"})

print("Points:", len(pts))

print("Loading local ERA5/GEE soil moisture...")
gee_files = sorted(GEE_DIR.glob("wue_timeseries_*.csv"))
if not gee_files:
    raise SystemExit("No wue_timeseries_*.csv files found in data/raw/gee")

gee = pd.concat([pd.read_csv(p) for p in gee_files], ignore_index=True)
gee["date"] = pd.to_datetime(gee["date"])
gee = gee[(gee["date"] >= START) & (gee["date"] <= END)].copy()

soil_candidates = [
    c for c in gee.columns
    if ("soil" in c.lower() and ("moist" in c.lower() or "sm" == c.lower()))
]
if "soil_moisture" in gee.columns:
    soil_col = "soil_moisture"
elif soil_candidates:
    soil_col = soil_candidates[0]
else:
    print("Available columns:", list(gee.columns))
    raise SystemExit("Could not find soil moisture column in GEE files.")

print("Using ERA5/local soil column:", soil_col)

gee = gee[["point_id", "date", soil_col]].rename(columns={soil_col: "soil_moisture"})
gee["point_id"] = gee["point_id"].astype(str)

print("Logging into Earthdata...")
earthaccess.login(strategy="interactive", persist=True)

print("Searching SMAP SPL4SMGP.008...")
results = earthaccess.search_data(
    short_name="SPL4SMGP",
    version="008",
    temporal=(START, END),
    count=200000
)

print("Granules found:", len(results))
if len(results) == 0:
    print("Trying without explicit version...")
    results = earthaccess.search_data(
        short_name="SPL4SMGP",
        temporal=(START, END),
        count=200000
    )
    print("Granules found:", len(results))

if len(results) == 0:
    raise SystemExit("No SMAP SPL4SMGP granules found.")

# Download one granule per day near midday if possible.
# SPL4SMGP is 3-hourly, so this avoids downloading every 3-hourly file.

def granule_begin_datetime(r):
    # First try real CMR/UMM metadata.
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

    # Fallback: parse all possible dates in the string and keep the first valid one.
    txt = str(r)

    # Pattern like 20150401T013000 or 20150401_013000.
    for m in re.finditer(r"(20\d{2})(\d{2})(\d{2})[T_ -]?(\d{2})?(\d{2})?(\d{2})?", txt):
        y, mo, da = m.group(1), m.group(2), m.group(3)
        hh = m.group(4) or "12"
        mm = m.group(5) or "00"
        ss = m.group(6) or "00"
        try:
            return datetime(int(y), int(mo), int(da), int(hh), int(mm), int(ss))
        except Exception:
            continue

    # Pattern like 2015-04-01T01:30:00.
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
    dt = granule_begin_datetime(r)
    if dt is None:
        bad_dates += 1
        continue

    # Select one granule per 8-day bin, closest to midday.
    # This matches the 8-day WUE/MODIS analysis cadence and avoids downloading thousands of daily files.
    year_start = datetime(dt.year, 1, 1)
    doy0 = (datetime(dt.year, dt.month, dt.day) - year_start).days
    bin_start = year_start + timedelta(days=(doy0 // 8) * 8)

    score = abs(dt.hour - 12)

    if bin_start not in selected or score < selected[bin_start][0]:
        selected[bin_start] = (score, r)

selected_results = [v[1] for _, v in sorted(selected.items())]
print("Selected daily granules:", len(selected_results))
print("Granules skipped because date could not be parsed:", bad_dates)

if len(selected_results) == 0:
    print("Example result object:")
    print(str(results[0])[:4000])
    raise SystemExit("Could not parse dates from any SMAP granules.")


print("Downloading selected SMAP granules to data/raw/smap...")
paths = earthaccess.download(selected_results, str(RAW))
paths = [Path(p) for p in paths if Path(p).exists()]
print("Downloaded/existing files:", len(paths))

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
    with h5py.File(path, "r") as f:
        sm_paths = find_dataset(f, ["sm_rootzone"])
        lat_paths = find_dataset(f, ["cell_lat", "lat"])
        lon_paths = find_dataset(f, ["cell_lon", "lon"])

        if not sm_paths:
            sm_paths = find_dataset(f, ["rootzone"])
        if not sm_paths:
            return None

        sm_path = sm_paths[0]

        # Prefer cell_lat/cell_lon if available.
        lat_path = None
        lon_path = None
        for p in lat_paths:
            if "cell_lat" in p.lower():
                lat_path = p
                break
        for p in lon_paths:
            if "cell_lon" in p.lower():
                lon_path = p
                break

        if lat_path is None and lat_paths:
            lat_path = lat_paths[0]
        if lon_path is None and lon_paths:
            lon_path = lon_paths[0]

        if lat_path is None or lon_path is None:
            return None

        sm = f[sm_path][()]
        lat = f[lat_path][()]
        lon = f[lon_path][()]

        sm = np.array(sm, dtype="float64")
        lat = np.array(lat, dtype="float64")
        lon = np.array(lon, dtype="float64")

        # Replace fill values.
        ds = f[sm_path]
        fill = ds.attrs.get("_FillValue", ds.attrs.get("missing_value", None))
        if fill is not None:
            try:
                sm[sm == float(np.array(fill).ravel()[0])] = np.nan
            except Exception:
                pass

        sm[(sm < 0) | (sm > 1)] = np.nan

        return lat, lon, sm

def file_date(path):
    m = re.search(r"(20\d{6})", path.name)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return pd.NaT

rows = []
tree_cache = None
grid_shape = None

print("Sampling SMAP at points...")
for path in tqdm(paths):
    parsed = read_smap_file(path)
    if parsed is None:
        continue

    lat_grid, lon_grid, sm_grid = parsed

    if lat_grid.shape != sm_grid.shape or lon_grid.shape != sm_grid.shape:
        continue

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

    for point_id, val in zip(pts["point_id"].astype(str), vals):
        if np.isfinite(val):
            rows.append({
                "point_id": point_id,
                "smap_date": d,
                "smap_sm_rootzone": float(val)
            })

smap = pd.DataFrame(rows)
if smap.empty:
    raise SystemExit("No SMAP samples were extracted. Need to inspect HDF5 paths.")

print("SMAP sampled rows:", len(smap))

# Convert both datasets to 8-day bins anchored to Jan 1.
def eight_day_bin(s):
    s = pd.to_datetime(s)
    year_start = pd.to_datetime(s.dt.year.astype(str) + "-01-01")
    doy0 = (s - year_start).dt.days
    bin_start = year_start + pd.to_timedelta((doy0 // 8) * 8, unit="D")
    return bin_start

smap["date"] = eight_day_bin(smap["smap_date"])
gee["date"] = eight_day_bin(gee["date"])

smap8 = smap.groupby(["point_id", "date"], as_index=False)["smap_sm_rootzone"].mean()
gee8 = gee.groupby(["point_id", "date"], as_index=False)["soil_moisture"].mean()

matched = gee8.merge(smap8, on=["point_id", "date"], how="inner")
matched = matched.sort_values(["point_id", "date"])

matched.to_csv("data/processed/smap_era5_matched_points.csv", index=False)

x = matched["soil_moisture"].astype(float)
y = matched["smap_sm_rootzone"].astype(float)
mask = x.notna() & y.notna()

if mask.sum() >= 3:
    r = float(pearsonr(x[mask], y[mask]).statistic)
else:
    r = np.nan

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
    "download_strategy": "one 3-hourly granule per day closest to 12:00 UTC, aggregated to 8-day bins"
}])

summary.to_csv("results/stress/smap_era5_comparison.csv", index=False)

Path("docs/smap_validation.md").write_text(
f"""# SMAP validation

This validation was run without Google Earth Engine.

Product: SPL4SMGP.008, SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data.

Variable: `sm_rootzone`, root-zone soil moisture.

Date range: {START} to {END}.

Sampling method:
- Downloaded one SPL4SMGP granule per day, preferring the granule closest to 12:00 UTC.
- Sampled nearest SMAP grid cell at each stable grassland point.
- Aggregated SMAP daily samples to 8-day bins.
- Aggregated the local ERA5/local soil-moisture column to the same 8-day bins.
- Matched by point_id and 8-day date.
- Computed Pearson correlation, RMSE, and mean ERA5-minus-SMAP bias.

Outputs:
- `data/processed/smap_era5_matched_points.csv`
- `results/stress/smap_era5_comparison.csv`

Summary:
{summary.to_string(index=False)}
"""
)

print("DONE")
print(summary.to_string(index=False))
