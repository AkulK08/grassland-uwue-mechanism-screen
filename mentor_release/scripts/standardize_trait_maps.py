from pathlib import Path
import re
import numpy as np
import xarray as xr

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
DL = ROOT / "data/external/_downloads/traits"
OUT = ROOT / "data/external"
OUT.mkdir(parents=True, exist_ok=True)

def open_any_nc(path):
    return xr.open_dataset(path)

def detect_lat_lon(ds):
    lat_names = ["lat", "latitude", "y", "LAT", "Latitude"]
    lon_names = ["lon", "longitude", "x", "LON", "Longitude"]

    lat = None
    lon = None

    for n in lat_names:
        if n in ds.coords or n in ds.dims or n in ds.variables:
            lat = n
            break

    for n in lon_names:
        if n in ds.coords or n in ds.dims or n in ds.variables:
            lon = n
            break

    if lat is None or lon is None:
        print(ds)
        raise ValueError("Could not detect lat/lon coordinates.")

    return lat, lon

def choose_var(ds, hints):
    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("No data variables found.")

    lower = {v.lower(): v for v in data_vars}

    for h in hints:
        for v in data_vars:
            if h.lower() in v.lower():
                return v

    # fallback: first numeric variable with at least 2 dims
    for v in data_vars:
        da = ds[v]
        if np.issubdtype(da.dtype, np.number) and da.ndim >= 2:
            return v

    return data_vars[0]

def normalize_lon(ds, lon_name):
    lon = ds[lon_name]
    try:
        if float(lon.max()) > 180:
            new_lon = (((lon + 180) % 360) - 180)
            ds = ds.assign_coords({lon_name: new_lon})
            ds = ds.sortby(lon_name)
    except Exception:
        pass
    return ds

def standardize_nc(in_path, out_path, out_var, hints):
    print("\nStandardizing:")
    print(" input:", in_path)
    print(" output:", out_path)

    ds = open_any_nc(in_path)
    lat, lon = detect_lat_lon(ds)
    ds = normalize_lon(ds, lon)

    var = choose_var(ds, hints)
    print(" selected variable:", var)
    print(" lat/lon:", lat, lon)

    da = ds[var]

    # Drop extra singleton dims if possible.
    for dim in list(da.dims):
        if dim not in [lat, lon] and da.sizes.get(dim, 1) == 1:
            da = da.isel({dim: 0}, drop=True)

    # If still has extra dimensions, take the first slice but report it.
    for dim in list(da.dims):
        if dim not in [lat, lon]:
            print(" extra dim found; taking first index:", dim)
            da = da.isel({dim: 0}, drop=True)

    da = da.rename({lat: "lat", lon: "lon"})
    da = da.sortby("lat")
    da = da.sortby("lon")
    da.name = out_var

    da.attrs["standardized_for"] = "grassland_wue_nature_repo_trait_analysis"
    da.attrs["original_file"] = str(in_path)
    da.attrs["original_variable"] = var

    final = da.to_dataset()
    final.to_netcdf(out_path)

    print("Wrote:", out_path)
    print(final)
    print("min/max:", float(final[out_var].min(skipna=True)), float(final[out_var].max(skipna=True)))

def find_liu_psi50_file():
    base = DL / "liu_2021"
    candidates = []
    for p in base.glob("*"):
        name = p.name.lower()
        if not p.is_file():
            continue
        if not (name.endswith(".nc") or name.endswith(".nc4") or name.endswith(".cdf")):
            continue
        score = 0
        if "p50" in name: score += 10
        if "psi" in name: score += 10
        if "hydraulic" in name: score += 2
        if "trait" in name: score += 1
        candidates.append((score, p))

    candidates = sorted(candidates, reverse=True, key=lambda x: x[0])
    if candidates and candidates[0][0] > 0:
        return candidates[0][1]

    # fallback: inspect variables
    for p in base.glob("*"):
        if not p.name.lower().endswith((".nc", ".nc4", ".cdf")):
            continue
        try:
            ds = xr.open_dataset(p)
            names = " ".join(list(ds.data_vars)).lower()
            if any(k in names for k in ["p50", "psi50", "psi_50"]):
                return p
        except Exception:
            pass

    print("\nCould not auto-detect Liu psi50 NetCDF.")
    print("Files downloaded:")
    for p in sorted(base.glob("*")):
        print(" ", p.name)
    raise SystemExit("Paste the file list back to ChatGPT so we can select the correct Liu P50/psi50 file.")

# 1. Konings/Gentine isohydricity
standardize_nc(
    DL / "isohydricityAMSRE_Global.nc",
    OUT / "konings_gentine_isohydricity_0p1deg.nc",
    "isohydricity",
    hints=["iso", "isohydricity", "slope"],
)

# 2. Stocker rooting depth
standardize_nc(
    DL / "zroot_cwd80.nc",
    OUT / "stocker_2023_rooting_depth_0p1deg.nc",
    "rooting_depth",
    hints=["zroot", "root", "depth"],
)

# 3. Liu 2021 psi50
liu_file = find_liu_psi50_file()
standardize_nc(
    liu_file,
    OUT / "liu_2021_psi50_0p1deg.nc",
    "psi50",
    hints=["p50", "psi50", "psi_50", "xylem"],
)

print("\nALL THREE TRAIT FILES CREATED.")
