"""Spatial and quality masks."""

from __future__ import annotations
import numpy as np
import xarray as xr


def stable_grassland_mask(masks: xr.Dataset, grassland_classes: list[int], extension_classes: list[int] | None = None, include_extensions: bool = False) -> xr.DataArray:
    classes = list(grassland_classes)
    if include_extensions and extension_classes:
        classes += list(extension_classes)
    lc = masks["landcover"]
    if "time" in lc.dims or "year" in lc.dims:
        dim = "time" if "time" in lc.dims else "year"
        valid_each = xr.apply_ufunc(np.isin, lc, np.asarray(classes), dask="allowed")
        return valid_each.all(dim)
    return xr.apply_ufunc(np.isin, lc, np.asarray(classes), dask="allowed")


def disturbance_mask(masks: xr.Dataset, max_irrigation: float) -> xr.DataArray:
    spatial_like = masks["landcover"]
    if "time" in spatial_like.dims:
        spatial_like = spatial_like.isel(time=0)
    if "year" in spatial_like.dims:
        spatial_like = spatial_like.isel(year=0)
    ok = xr.ones_like(spatial_like, dtype=bool)
    if "irrigated_fraction" in masks:
        ok = ok & (masks["irrigated_fraction"] <= max_irrigation)
    if "burned" in masks:
        b = masks["burned"]
        if "time" in b.dims:
            b_any = b.any("time")
        elif "year" in b.dims:
            b_any = b.any("year")
        else:
            b_any = b.astype(bool)
        ok = ok & (~b_any)
    return ok


def combined_analysis_mask(masks: xr.Dataset, grassland_classes: list[int], extension_classes: list[int], max_irrigation: float, include_extensions: bool = False) -> xr.DataArray:
    return stable_grassland_mask(masks, grassland_classes, extension_classes, include_extensions) & disturbance_mask(masks, max_irrigation)


def apply_mask(ds: xr.Dataset, mask: xr.DataArray) -> xr.Dataset:
    return ds.where(mask)
