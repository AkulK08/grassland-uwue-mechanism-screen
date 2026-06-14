"""Synthetic data generator that mimics the structure of the full project.

The demo data are not scientific results. They let the entire analysis pipeline
run end-to-end without large external downloads, which is essential for testing
code paths, figures, memos, and reproducibility.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from ..config import ProjectConfig
from ..constants import GPP_PRODUCTS, ET_PRODUCTS
from ..utils import safe_to_netcdf, safe_to_csv


def _times(start: str, end: str, freq: str = "8D") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)


def make_demo_data(cfg: ProjectConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    start = cfg.analysis.get("start_date", "2015-01-01")
    end = cfg.analysis.get("end_date", "2023-12-31")
    times = _times(start, end, cfg.analysis.get("base_frequency", "8D"))
    # 16 x 20 grid = 320 pixels; enough for quartiles and towers but fast.
    lat = np.linspace(-45, 55, 16)
    lon = np.linspace(-120, 140, 20)
    t = np.arange(len(times))
    seasonal = 1.0 + 0.35 * np.sin(2 * np.pi * (t % 46) / 46.0)
    aridity = xr.DataArray(
        np.clip(rng.beta(2, 3, size=(len(lat), len(lon))), 0.05, 0.95),
        coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="aridity_index"
    )
    lai = xr.DataArray(
        0.2 + 3.2 * (1 - aridity.values) + rng.normal(0, 0.1, size=aridity.shape),
        coords=aridity.coords, dims=aridity.dims, name="lai"
    ).clip(min=0.05)
    # Stress fields. SM is negatively related to VPD but not perfectly.
    stress_base = rng.normal(0, 1, size=(len(times), len(lat), len(lon)))
    vpd = 1.0 + 0.55 * seasonal[:, None, None] + 0.75 * stress_base + 0.25 * rng.normal(size=stress_base.shape)
    sm = 0.25 + 0.18 * (1 - aridity.values)[None, :, :] - 0.08 * stress_base + 0.03 * rng.normal(size=stress_base.shape)
    temp = 15 + 9 * np.sin(2 * np.pi * (t % 46) / 46.0)[:, None, None] + 8 * aridity.values[None, :, :] + rng.normal(0, 1, stress_base.shape)
    precip = np.clip(2.0 + 5.0 * (1 - aridity.values)[None, :, :] - 1.2 * stress_base + rng.gamma(1.2, 1.0, stress_base.shape), 0, None)
    met = xr.Dataset(
        {
            "vpd": (("time", "lat", "lon"), vpd),
            "soil_moisture": (("time", "lat", "lon"), sm),
            "temperature": (("time", "lat", "lon"), temp),
            "precipitation": (("time", "lat", "lon"), precip),
        }, coords={"time": times, "lat": lat, "lon": lon}
    )
    # Underlying response: enhancement then saturation/reversal varies by aridity.
    vpd_z = (met.vpd - met.vpd.mean("time")) / met.vpd.std("time")
    sm_z = (met.soil_moisture - met.soil_moisture.mean("time")) / met.soil_moisture.std("time")
    csi = (0.5 * vpd_z - 0.5 * sm_z).transpose("time", "lat", "lon")
    psi = 1.3 + 0.9 * aridity
    pre = 0.10 + 0.05 * (1 - aridity)
    post = -0.04 + 0.10 * (1 - aridity)  # dry pixels more likely reversal; wet saturation/enhancement
    hinge = xr.where(csi > psi, csi - psi, 0)
    true_log_wue = (0.3 + pre * csi + (post - pre) * hinge).transpose("time", "lat", "lon") + 0.03 * rng.normal(size=csi.shape)
    # Base ET and GPP with product-specific structural offsets.
    base_log_et = (np.log(1.2 + 1.4 * seasonal[:, None, None]) + 0.09 * csi).transpose("time", "lat", "lon") + 0.04 * rng.normal(size=csi.shape)
    base_log_gpp = (true_log_wue + base_log_et).transpose("time", "lat", "lon")
    product_noise = {"MODIS": 0.07, "GOSIF": 0.10, "PML": 0.06, "GLEAM": 0.11}
    demo_dir = cfg.resolve("demo")
    for product in GPP_PRODUCTS:
        bias = {"MODIS": 0.02 * csi, "GOSIF": -0.01 * csi, "PML": 0.00 * csi}[product]
        gpp = np.exp(base_log_gpp + bias + product_noise[product] * rng.normal(size=csi.shape))
        ds = xr.Dataset({"gpp": gpp.rename("gpp")})
        safe_to_netcdf(ds, demo_dir / f"gpp_{product}.nc")
    for product in ET_PRODUCTS:
        bias = {"MODIS": 0.04 * csi, "GLEAM": -0.01 * csi, "PML": 0.02 * csi}[product]
        et = np.exp(base_log_et + bias + product_noise[product] * rng.normal(size=csi.shape))
        ds = xr.Dataset({"et": et.rename("et")})
        safe_to_netcdf(ds, demo_dir / f"et_{product}.nc")
    safe_to_netcdf(met, demo_dir / "met.nc")
    safe_to_netcdf(xr.Dataset({"aridity_index": aridity, "lai": lai}), demo_dir / "ancillary.nc")
    # Land cover/masks: all pixels stable grasslands except a few invalid flags.
    lc = xr.DataArray(np.full((len(lat), len(lon)), 10, dtype=int), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="landcover")
    burned = xr.DataArray(np.zeros((len(times), len(lat), len(lon)), dtype=bool), coords=met.coords, dims=met.vpd.dims, name="burned")
    irrig = xr.DataArray(rng.random((len(lat), len(lon))) * 0.08, coords=aridity.coords, dims=aridity.dims, name="irrigated_fraction")
    safe_to_netcdf(xr.Dataset({"landcover": lc, "burned": burned, "irrigated_fraction": irrig}), demo_dir / "masks.nc")
    # CO2 table.
    year_frac = times.year + (times.dayofyear - 1) / 365.25
    co2 = 400 + 2.35 * (year_frac - year_frac[0]) + 0.5 * np.sin(2 * np.pi * times.dayofyear / 365.25)
    safe_to_csv(pd.DataFrame({"time": times, "co2_ppm": co2}), demo_dir / "co2.csv")
    # Towers: choose random grid cells and generate time series with flux variables.
    tower_rows = []
    sites = []
    n_sites = 24
    choices = rng.choice(len(lat) * len(lon), size=n_sites, replace=False)
    for k, flat in enumerate(choices):
        iy, ix = divmod(flat, len(lon))
        site = f"DEMO_{k:03d}"
        sites.append((site, float(lat[iy]), float(lon[ix])))
        for ti, tm in enumerate(times):
            gpp_val = float(np.exp(base_log_gpp.isel(time=ti, lat=iy, lon=ix).values + rng.normal(0, 0.07)))
            et_val = float(np.exp(base_log_et.isel(time=ti, lat=iy, lon=ix).values + rng.normal(0, 0.06)))
            le = et_val / 8.0 * 2.45e6 / 86400.0  # W m-2 equivalent for 8-day total approximation
            h = 0.8 * le + rng.normal(0, 10)
            rn = h + le + rng.normal(0, 10)
            tower_rows.append({
                "site_id": site,
                "time": tm,
                "latitude": lat[iy],
                "longitude": lon[ix],
                "igbp": "GRA",
                "GPP_NT_VUT_REF": gpp_val,
                "LE_F_MDS": le,
                "LE_F_MDS_QC": 0 if rng.random() > 0.15 else 1,
                "NEE_VUT_REF_QC": 0.9,
                "H_F_MDS": h,
                "NETRAD": rn,
                "G": 0.05 * rn,
                "VPD_F_MDS": float(met.vpd.isel(time=ti, lat=iy, lon=ix).values),
                "SWC_F_MDS": float(met.soil_moisture.isel(time=ti, lat=iy, lon=ix).values),
                "TA_F_MDS": float(met.temperature.isel(time=ti, lat=iy, lon=ix).values),
            })
    safe_to_csv(pd.DataFrame(tower_rows), demo_dir / "towers.csv")
    # Traits.
    psi50 = xr.DataArray(-2.0 - 4.0 * aridity.values + rng.normal(0, 0.25, aridity.shape), coords=aridity.coords, dims=aridity.dims, name="psi50")
    isoh = xr.DataArray(0.2 + 0.7 * aridity.values + rng.normal(0, 0.1, aridity.shape), coords=aridity.coords, dims=aridity.dims, name="isohydricity")
    rooting = xr.DataArray(0.8 + 2.4 * aridity.values + rng.normal(0, 0.15, aridity.shape), coords=aridity.coords, dims=aridity.dims, name="rooting_depth")
    safe_to_netcdf(xr.Dataset({"psi50": psi50, "isohydricity": isoh, "rooting_depth": rooting}), demo_dir / "traits.nc")
