"""Utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import logging
import json
import numpy as np
import pandas as pd
import xarray as xr


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def safe_to_netcdf(ds: xr.Dataset | xr.DataArray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ds.to_netcdf(tmp)
    tmp.replace(path)
    return path


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def robust_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def ensure_datetime_index(values: Iterable) -> pd.DatetimeIndex:
    return pd.to_datetime(list(values))


def infer_lat_lon_names(ds: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    names = set(ds.coords) | set(ds.dims)
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat = next((n for n in lat_candidates if n in names), None)
    lon = next((n for n in lon_candidates if n in names), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not infer lat/lon names from {names}")
    return lat, lon


def flatten_dataset(ds: xr.Dataset, vars: list[str]) -> pd.DataFrame:
    return ds[vars].to_dataframe().reset_index().dropna()
