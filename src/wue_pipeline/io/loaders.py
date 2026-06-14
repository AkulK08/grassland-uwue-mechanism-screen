"""High-level loading interface for demo and production data."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import xarray as xr

from ..config import ProjectConfig
from .catalog import open_dataset, open_table, read_co2_table


def _demo_path(cfg: ProjectConfig, name: str) -> Path:
    return cfg.resolve("demo") / name


def load_gpp(cfg: ProjectConfig, product: str) -> xr.DataArray:
    if cfg.mode == "demo":
        ds = xr.open_dataset(_demo_path(cfg, f"gpp_{product}.nc"))
        return ds["gpp"]
    path = cfg.catalog["gpp"][product]
    return open_dataset(path, variable="gpp")["gpp"]


def load_et(cfg: ProjectConfig, product: str) -> xr.DataArray:
    if cfg.mode == "demo":
        ds = xr.open_dataset(_demo_path(cfg, f"et_{product}.nc"))
        return ds["et"]
    path = cfg.catalog["et"][product]
    return open_dataset(path, variable="et")["et"]


def load_met(cfg: ProjectConfig) -> xr.Dataset:
    if cfg.mode == "demo":
        return xr.open_dataset(_demo_path(cfg, "met.nc"))
    met = cfg.catalog["met"]
    parts = []
    for key, var in [("vpd", "vpd"), ("soil_moisture", "soil_moisture"), ("temperature", "temperature"), ("precipitation", "precipitation")]:
        parts.append(open_dataset(met[key], variable=var))
    return xr.merge(parts)


def load_masks(cfg: ProjectConfig) -> xr.Dataset:
    if cfg.mode == "demo":
        return xr.open_dataset(_demo_path(cfg, "masks.nc"))
    masks = cfg.catalog["masks"]
    parts = []
    for key, var in [("landcover", "landcover"), ("burned_area", "burned"), ("irrigated_fraction", "irrigated_fraction")]:
        if key in masks:
            parts.append(open_dataset(masks[key], variable=var))
    return xr.merge(parts)


def load_ancillary(cfg: ProjectConfig) -> xr.Dataset:
    if cfg.mode == "demo":
        return xr.open_dataset(_demo_path(cfg, "ancillary.nc"))
    anc = cfg.catalog["ancillary"]
    parts = []
    for key, var in [("aridity_index", "aridity_index"), ("lai", "lai")]:
        if key in anc:
            parts.append(open_dataset(anc[key], variable=var))
    return xr.merge(parts)


def load_co2(cfg: ProjectConfig) -> pd.DataFrame:
    if cfg.mode == "demo":
        return read_co2_table(_demo_path(cfg, "co2.csv"))
    return read_co2_table(cfg.catalog["ancillary"]["co2"])


def load_towers(cfg: ProjectConfig) -> pd.DataFrame:
    if cfg.mode == "demo":
        return open_table(_demo_path(cfg, "towers.csv"))
    tables = []
    for _, path in cfg.catalog.get("towers", {}).items():
        try:
            tables.append(open_table(path))
        except FileNotFoundError:
            continue
    if not tables:
        raise FileNotFoundError("No tower tables were found in the data catalog.")
    return pd.concat(tables, ignore_index=True)


def load_traits(cfg: ProjectConfig) -> xr.Dataset:
    if cfg.mode == "demo":
        return xr.open_dataset(_demo_path(cfg, "traits.nc"))
    trait_catalog = cfg.catalog["traits"]
    parts = []
    for key, var in [("psi50", "psi50"), ("isohydricity", "isohydricity"), ("rooting_depth", "rooting_depth")]:
        parts.append(open_dataset(trait_catalog[key], variable=var))
    return xr.merge(parts)
