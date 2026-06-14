"""Local data catalog loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import xarray as xr


def open_dataset(path: str | Path, variable: str | None = None) -> xr.Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected data file not found: {path}")
    if path.suffix == ".zarr" or path.is_dir():
        ds = xr.open_zarr(path)
    elif path.suffix.lower() in [".nc", ".nc4", ".cdf"]:
        ds = xr.open_dataset(path)
    elif path.suffix.lower() in [".tif", ".tiff"]:
        import rioxarray  # noqa: F401
        da = xr.open_dataarray(path, engine="rasterio")
        name = variable or path.stem
        ds = da.to_dataset(name=name)
    else:
        raise ValueError(f"Unsupported gridded data format: {path}")
    if variable is not None and variable not in ds:
        # If only one variable exists, rename it.
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: variable})
        else:
            raise KeyError(f"Variable {variable} not in {path}; variables={data_vars}")
    return ds


def open_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected table not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_co2_table(path: str | Path) -> pd.DataFrame:
    df = open_table(path)
    needed = {"time", "co2_ppm"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CO2 table must contain {needed}; got {df.columns.tolist()}")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")
