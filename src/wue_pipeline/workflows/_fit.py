"""Shared fitting utilities for workflow modules."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ..models.segmented import segmented_with_uncertainty, fit_interaction_model
from ..processing.growing_season import gpp_threshold_mask, climate_threshold_mask, add_month_fixed_effects


def _spatial_points(da: xr.DataArray) -> pd.DataFrame:
    lat = da["lat"].values
    lon = da["lon"].values
    rows = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            rows.append((i, j, float(la), float(lo)))
    return pd.DataFrame(rows, columns=["ilat", "ilon", "lat", "lon"])


def sample_points(points: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    valid = points.copy()
    if len(valid) <= n:
        return valid
    return valid.sample(n=n, random_state=seed).reset_index(drop=True)


def growing_mask(name: str, gpp: xr.DataArray, met: xr.Dataset):
    if name == "gpp_threshold":
        return gpp_threshold_mask(gpp)
    if name == "climate_threshold":
        return climate_threshold_mask(met["temperature"], met["precipitation"])
    if name == "month_fixed_effects":
        return xr.ones_like(gpp, dtype=bool)
    raise ValueError(f"Unknown growing season: {name}")


def stress_var_name(stress_definition: str) -> str:
    return {
        "zscore": "csi_zscore",
        "percentile": "csi_percentile",
        "copula": "csi_copula",
        "interaction": "interaction",
    }[stress_definition]


def fit_pixels_for_combo(components: xr.Dataset, stress: xr.Dataset, met: xr.Dataset, gpp: xr.DataArray, points: pd.DataFrame, gpp_product: str, et_product: str, stress_definition: str, growing_season: str, min_obs: int, n_boot: int, seed: int, use_co2: bool = True) -> pd.DataFrame:
    mask = growing_mask(growing_season, gpp, met)
    y_name = "log_wue_co2" if use_co2 and "log_wue_co2" in components else "log_wue"
    rows = []
    stress_name = stress_var_name(stress_definition)
    for _, p in tqdm(points.iterrows(), total=len(points), desc=f"{gpp_product}-{et_product}-{stress_definition}-{growing_season}", leave=False):
        ilat, ilon = int(p.ilat), int(p.ilon)
        y = components[y_name].isel(lat=ilat, lon=ilon).where(mask.isel(lat=ilat, lon=ilon))
        x = stress[stress_name].isel(lat=ilat, lon=ilon).where(mask.isel(lat=ilat, lon=ilon))
        if stress_definition == "interaction":
            df = pd.DataFrame({
                "time": pd.to_datetime(components.time.values),
                "log_wue": y.values,
                "vpd_z": stress["vpd_z"].isel(lat=ilat, lon=ilon).where(mask.isel(lat=ilat, lon=ilon)).values,
                "sm_z": stress["sm_z"].isel(lat=ilat, lon=ilon).where(mask.isel(lat=ilat, lon=ilon)).values,
            }).dropna()
            res = fit_interaction_model(df)
            row = {"n": res.get("n"), "breakpoint": np.nan, "pre_slope": np.nan, "post_slope": np.nan, "slope_change": res.get("interaction_coef"), "response_class": res.get("response_class"), "reason": "interaction_model"}
            row["interaction_p"] = res.get("interaction_p")
        else:
            if growing_season == "month_fixed_effects":
                # Month fixed effects handled by residualizing y and x over month before segmented fit.
                df = pd.DataFrame({"time": pd.to_datetime(components.time.values), "x": x.values, "y": y.values}).dropna()
                if len(df) >= min_obs:
                    d = add_month_fixed_effects(df)
                    month_cols = [c for c in d.columns if c.startswith("month_")]
                    import statsmodels.api as sm
                    yr = sm.OLS(d["y"], sm.add_constant(d[month_cols])).fit().resid if month_cols else d["y"] - d["y"].mean()
                    xr_ = sm.OLS(d["x"], sm.add_constant(d[month_cols])).fit().resid if month_cols else d["x"] - d["x"].mean()
                    fit = segmented_with_uncertainty(xr_, yr, min_obs=min_obs, n_boot=n_boot, seed=seed)
                else:
                    fit = segmented_with_uncertainty([], [], min_obs=min_obs, n_boot=n_boot, seed=seed)
            else:
                fit = segmented_with_uncertainty(x.values, y.values, min_obs=min_obs, n_boot=n_boot, seed=seed)
            row = fit.to_dict()
        row.update({
            "lat": p.lat,
            "lon": p.lon,
            "ilat": ilat,
            "ilon": ilon,
            "gpp_product": gpp_product,
            "et_product": et_product,
            "stress_definition": stress_definition,
            "growing_season": growing_season,
        })
        rows.append(row)
    return pd.DataFrame(rows)
