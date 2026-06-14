"""Grid and calendar harmonization utilities."""

from __future__ import annotations
import xarray as xr


def align_like(reference: xr.DataArray | xr.Dataset, *datasets: xr.DataArray | xr.Dataset):
    return xr.align(reference, *datasets, join="inner")


def to_common_time(*datasets):
    return xr.align(*datasets, join="inner")


def standardize_dims(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    rename = {}
    for old, new in [("latitude", "lat"), ("longitude", "lon"), ("y", "lat"), ("x", "lon")]:
        if old in obj.dims or old in obj.coords:
            if new not in obj.dims and new not in obj.coords:
                rename[old] = new
    return obj.rename(rename) if rename else obj
