"""WUE computation and decomposition."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def compute_log_components(gpp: xr.DataArray, et: xr.DataArray, gpp_floor: float, et_floor: float) -> xr.Dataset:
    gpp_f = gpp.where(gpp > gpp_floor)
    et_f = et.where(et > et_floor)
    log_gpp = np.log(gpp_f)
    log_et = np.log(et_f)
    return xr.Dataset({"log_gpp": log_gpp, "log_et": log_et, "log_wue": log_gpp - log_et})


def apply_co2_correction(log_wue: xr.DataArray, co2: pd.DataFrame, ref_ppm: float) -> xr.DataArray:
    c = co2.copy()
    c["time"] = pd.to_datetime(c["time"])
    target = pd.DatetimeIndex(pd.to_datetime(log_wue["time"].values))
    interp = np.interp(target.view("int64"), c["time"].view("int64"), c["co2_ppm"].astype(float))
    corr = np.log(ref_ppm) - np.log(interp)
    corr_da = xr.DataArray(corr, coords={"time": log_wue.time}, dims="time")
    return log_wue + corr_da
