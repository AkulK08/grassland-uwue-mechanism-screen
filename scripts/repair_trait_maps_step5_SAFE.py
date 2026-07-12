from pathlib import Path
import shutil
import numpy as np
import xarray as xr

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
EXT = ROOT / "data/external"
BACKUP = EXT / "_trait_backups_before_repair"
BACKUP.mkdir(parents=True, exist_ok=True)

FILES = {
    "psi50": EXT / "liu_2021_psi50_0p1deg.nc",
    "isohydricity": EXT / "konings_gentine_isohydricity_0p1deg.nc",
    "rooting_depth": EXT / "stocker_2023_rooting_depth_0p1deg.nc",
}

for name, path in FILES.items():
    if path.exists():
        shutil.copy2(path, BACKUP / path.name)
        print("Backed up:", path)

target_lat = np.round(np.arange(-89.95, 90.0, 0.1), 4).astype("float32")
target_lon = np.round(np.arange(-179.95, 180.0, 0.1), 4).astype("float32")

def normalize_lon(da):
    if "lon" not in da.coords:
        return da
    lon = da["lon"]
    if float(lon.max()) > 180:
        new_lon = (((lon + 180) % 360) - 180)
        da = da.assign_coords(lon=new_lon).sortby("lon")
    return da

def safe_write(da, out_path, varname, note):
    tmp_path = out_path.with_suffix(".tmp.nc")

    da = da.rename(varname)
    da = da.astype("float32")
    da.attrs["standardized_for"] = "grassland_wue_nature_repo_trait_analysis"
    da.attrs["repair_note"] = note

    ds_out = da.to_dataset()
    encoding = {
        varname: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }

    if tmp_path.exists():
        tmp_path.unlink()

    ds_out.to_netcdf(tmp_path, encoding=encoding)
    ds_out.close()

    tmp_path.replace(out_path)

    ds_check = xr.open_dataset(out_path)
    print("\nWrote:", out_path)
    print(ds_check)
    print(varname, "min/max:", float(ds_check[varname].min(skipna=True)), float(ds_check[varname].max(skipna=True)))
    ds_check.close()

# 1. Liu psi50
with xr.open_dataset(FILES["psi50"]) as ds:
    v = list(ds.data_vars)[0]
    da = ds[v].load()

rename = {}
for d in da.dims:
    if d.lower() in ["latitude", "y"]:
        rename[d] = "lat"
    if d.lower() in ["longitude", "x"]:
        rename[d] = "lon"
da = da.rename(rename)
da = normalize_lon(da)
da = da.sortby("lat").sortby("lon").transpose("lat", "lon")
da = da.interp(lat=target_lat, lon=target_lon, method="linear")
safe_write(
    da,
    FILES["psi50"],
    "psi50",
    "Regridded to regular 0.1 degree lat/lon grid; values otherwise unchanged.",
)

# 2. Konings/Gentine isohydricity
with xr.open_dataset(FILES["isohydricity"]) as ds:
    v = list(ds.data_vars)[0]
    da = ds[v].load()

if set(da.dims) == {"lon", "lat"}:
    nlon = da.sizes["lon"]
    nlat = da.sizes["lat"]
    lon = np.linspace(-180 + 180 / nlon, 180 - 180 / nlon, nlon).astype("float32")
    lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat).astype("float32")
    da = da.assign_coords({"lon": lon, "lat": lat}).transpose("lat", "lon")
else:
    raise ValueError(f"Unexpected isohydricity dimensions: {da.dims}")

da = da.sortby("lat").sortby("lon")
da = da.interp(lat=target_lat, lon=target_lon, method="linear")
safe_write(
    da,
    FILES["isohydricity"],
    "isohydricity",
    "Reconstructed missing/broken global lat/lon coordinates and regridded to 0.1 degree.",
)

# 3. Stocker rooting depth
with xr.open_dataset(FILES["rooting_depth"]) as ds:
    v = list(ds.data_vars)[0]
    da = ds[v].load()

rename = {}
for d in da.dims:
    if d.lower() in ["latitude", "y"]:
        rename[d] = "lat"
    if d.lower() in ["longitude", "x"]:
        rename[d] = "lon"
da = da.rename(rename)
da = normalize_lon(da)
da = da.sortby("lat").sortby("lon").transpose("lat", "lon")

# Mask obvious invalid/fill/extreme values.
da = da.where(np.isfinite(da))
da = da.where(da > 0)
da = da.where(da < 10000)

da = da.interp(lat=target_lat, lon=target_lon, method="linear")
safe_write(
    da,
    FILES["rooting_depth"],
    "rooting_depth",
    "Masked nonpositive and extremely large values >10000, then regridded to 0.1 degree.",
)

print("\nSTEP 5 TRAIT REPAIR COMPLETE.")
