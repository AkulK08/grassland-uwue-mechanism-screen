"""Meteorological transformations."""

from __future__ import annotations
import numpy as np
import xarray as xr


def saturation_vapor_pressure_kpa(temp_c: xr.DataArray) -> xr.DataArray:
    return 0.611 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def vpd_from_temperature_dewpoint(temp_k: xr.DataArray, dewpoint_k: xr.DataArray) -> xr.DataArray:
    temp_c = temp_k - 273.15
    dew_c = dewpoint_k - 273.15
    return saturation_vapor_pressure_kpa(temp_c) - saturation_vapor_pressure_kpa(dew_c)


def rootzone_soil_moisture(swvl1: xr.DataArray, swvl2: xr.DataArray) -> xr.DataArray:
    return 0.25 * swvl1 + 0.75 * swvl2
