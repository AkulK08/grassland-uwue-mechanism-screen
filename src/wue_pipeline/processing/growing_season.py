"""Growing season definitions."""

from __future__ import annotations
import numpy as np
import xarray as xr


def gpp_threshold_mask(gpp: xr.DataArray, threshold_fraction: float = 0.20) -> xr.DataArray:
    group = gpp.groupby("time.year")
    annual_peak = group.max("time", skipna=True)
    mask = xr.zeros_like(gpp, dtype=bool)
    for year in annual_peak["year"].values:
        peak = annual_peak.sel(year=year)
        year_mask = gpp["time.year"] == year
        mask = xr.where(year_mask, gpp >= threshold_fraction * peak, mask)
    return mask


def climate_threshold_mask(temperature: xr.DataArray, precipitation: xr.DataArray, temp_min: float = 5.0, precip_30d_min: float = 10.0) -> xr.DataArray:
    # Input is already 8-day. Approximate 30-day with 4 composites.
    temp_roll = temperature.rolling(time=4, min_periods=2).mean()
    precip_roll = precipitation.rolling(time=4, min_periods=2).sum()
    return (temp_roll > temp_min) & (precip_roll > precip_30d_min)


def add_month_fixed_effects(df):
    import pandas as pd
    out = df.copy()
    out["month"] = pd.to_datetime(out["time"]).dt.month
    dummies = pd.get_dummies(out["month"], prefix="month", drop_first=True, dtype=float)
    return pd.concat([out, dummies], axis=1)
