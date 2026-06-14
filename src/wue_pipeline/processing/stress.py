"""Compound stress definitions."""

from __future__ import annotations
import numpy as np
import xarray as xr
from scipy import stats


def zscore(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    return (da - da.mean(dim, skipna=True)) / da.std(dim, skipna=True)


def csi_zscore(vpd: xr.DataArray, soil_moisture: xr.DataArray) -> xr.DataArray:
    return 0.5 * zscore(vpd) - 0.5 * zscore(soil_moisture)


def _empirical_cdf_1d(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    ranks = stats.rankdata(x[ok], method="average") / (ok.sum() + 1.0)
    out[ok] = ranks
    return out


def empirical_cdf(da: xr.DataArray) -> xr.DataArray:
    out = xr.apply_ufunc(
        _empirical_cdf_1d, da,
        input_core_dims=[["time"]], output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[float]
    )
    return out.transpose(*da.dims)


def csi_percentile(vpd: xr.DataArray, soil_moisture: xr.DataArray, vpd_q: float = 0.75, sm_q: float = 0.25) -> xr.DataArray:
    v_rank = empirical_cdf(vpd)
    sm_rank = empirical_cdf(soil_moisture)
    v_ex = xr.where(v_rank > vpd_q, v_rank - vpd_q, 0.0)
    sm_def = xr.where(sm_rank < sm_q, sm_q - sm_rank, 0.0)
    return v_ex + sm_def


def _empirical_cdf_1d_old_unused(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    ranks = stats.rankdata(x[ok], method="average") / (ok.sum() + 1.0)
    out[ok] = ranks
    return out


def csi_copula(vpd: xr.DataArray, soil_moisture: xr.DataArray) -> xr.DataArray:
    """Empirical joint tail stress.

    This nonparametric implementation estimates marginal ranks and multiplies
    upper-tail VPD probability by lower-tail soil-moisture probability. It is a
    robust fallback when parametric copula fitting is unstable at pixel level.
    """
    u_vpd = empirical_cdf(vpd)
    u_sm = empirical_cdf(soil_moisture)
    return u_vpd * (1.0 - u_sm)


def stress_dataset(vpd: xr.DataArray, soil_moisture: xr.DataArray) -> xr.Dataset:
    return xr.Dataset({
        "csi_zscore": csi_zscore(vpd, soil_moisture),
        "csi_percentile": csi_percentile(vpd, soil_moisture),
        "csi_copula": csi_copula(vpd, soil_moisture),
        "vpd_z": zscore(vpd),
        "sm_z": zscore(soil_moisture),
        "interaction": zscore(vpd) * zscore(soil_moisture),
    })
