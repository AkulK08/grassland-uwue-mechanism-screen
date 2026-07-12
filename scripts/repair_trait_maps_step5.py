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
        print("Backed up:", path, "->", BACKUP / path.name)

# 0.1 degree global grid, cell centers
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

def write_clean(da, out_path, varname, note):
    da = da.rename(varname)
    da = da.astype("float32")
    da.attrs["standardized_for"] = "grassland_wue_nature_repo_trait_analysis"
    da.attrs["repair_note"] = note

    ds = da.to_dataset()
    encoding = {
        varname: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }
    ds.to_netcdf(out_path, encoding=encoding)

    print("\nWrote:", out_path)
    print(ds)
    print(
        varname,
        "min/max:",
        float(ds[varname].min(skipna=True)),
        float(ds[varname].max(skipna=True)),
    )

# 1. Liu psi50
ds = xr.open_dataset(FILES["psi50"])
v = list(ds.data_vars)[0]
da = ds[v]

# Ensure normal names/order
rename = {}
for d in da.dims:
    if d.lower() in ["latitude", "y"]:
        rename[d] = "lat"
    if d.lower() in ["longitude", "x"]:
        rename[d] = "lon"
da = da.rename(rename)
da = normalize_lon(da)
da = da.sortby("lat").sortby("lon")
da = da.transpose("lat", "lon")
da = da.interp(lat=target_lat, lon=target_lon, method="linear")
write_clean(
    da,
    FILES["psi50"],
    "psi50",
    "Regridded to regular 0.1 degree lat/lon grid; values otherwise unchanged.",
)

# 2. Konings/Gentine isohydricity
ds = xr.open_dataset(FILES["isohydricity"])
v = list(ds.data_vars)[0]
da = ds[v]

# Current file has dims (lon, lat), but lat has no real coordinate and lon is broken.
# Reconstruct as global 0.25 degree cell centers, then interpolate to 0.1.
if set(da.dims) == {"lon", "lat"}:
    nlon = da.sizes["lon"]
    nlat = da.sizes["lat"]

    lon = np.linspace(-180 + 180 / nlon, 180 - 180 / nlon, nlon).astype("float32")
    lat = np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat).astype("float32")

    da = da.assign_coords({"lon": lon, "lat": lat})
    da = da.transpose("lat", "lon")
else:
    raise ValueError(f"Unexpected isohydricity dimensions: {da.dims}")

da = da.sortby("lat").sortby("lon")
da = da.interp(lat=target_lat, lon=target_lon, method="linear")
write_clean(
    da,
    FILES["isohydricity"],
    "isohydricity",
    "Reconstructed missing/broken global lat/lon coordinates and regridded to 0.1 degree.",
)

# 3. Stocker rooting depth
ds = xr.open_dataset(FILES["rooting_depth"])
v = list(ds.data_vars)[0]
da = ds[v]

rename = {}
for d in da.dims:
    if d.lower() in ["latitude", "y"]:
        rename[d] = "lat"
    if d.lower() in ["longitude", "x"]:
        rename[d] = "lon"
da = da.rename(rename)
da = normalize_lon(da)
da = da.sortby("lat").sortby("lon")
da = da.transpose("lat", "lon")

# Mask obvious invalid/fill/extreme values.
# The previous max was 765134.5, which is not a plausible rooting-depth value.
da = da.where(np.isfinite(da))
da = da.where(da > 0)
da = da.where(da < 10000)

da = da.interp(lat=target_lat, lon=target_lon, method="linear")
write_clean(
    da,
    FILES["rooting_depth"],
    "rooting_depth",
    "Masked nonpositive and extremely large values >10000, then regridded to 0.1 degree.",
)

print("\nSTEP 5 TRAIT REPAIR COMPLETE.")
