#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PAT = re.compile(r"E_(\d{4})_GLEAM_v4\.3a\.nc$")


def read_points(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    lat = next((c for c in ["lat", "latitude", "LAT", "Latitude", "y", "Y"] if c in df.columns), None)
    lon = next((c for c in ["lon", "longitude", "LON", "Longitude", "x", "X"] if c in df.columns), None)
    if lat is None or lon is None:
        raise SystemExit(f"Could not find lat/lon columns in {path}. Columns={list(df.columns)}")
    if "point_id" not in df.columns:
        df["point_id"] = np.arange(len(df))
    out = df[["point_id", lat, lon]].rename(columns={lat: "lat", lon: "lon"}).dropna()
    out["point_id"] = out["point_id"].astype(str)
    return out


def eight_day_starts(year: int):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    for doy in range(1, 362, 8):
        yield doy, jan1 + pd.Timedelta(days=doy - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--local-glob", default="data/raw/gleam/E_20*_GLEAM_v4.3a.nc")
    ap.add_argument("--var", default="E")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pts = read_points(args.points)
    lat_da = xr.DataArray(pts["lat"].astype(float).values, dims="point")
    lon_da = xr.DataArray(pts["lon"].astype(float).values, dims="point")

    files = []
    for f in sorted(glob(args.local_glob)):
        m = PAT.search(Path(f).name)
        if m:
            y = int(m.group(1))
            if args.start_year <= y <= args.end_year:
                files.append((f, y))

    if not files:
        raise SystemExit("No GLEAM yearly E files matched requested years.")

    rows = []
    for i, (f, year) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Sampling {f}")
        ds = xr.open_dataset(f)
        if args.var not in ds:
            raise SystemExit(f"{args.var} not found in {f}; variables={list(ds.data_vars)}")
        da = ds[args.var]
        sampled = da.sel(lat=lat_da, lon=lon_da, method="nearest")

        for doy, start in eight_day_starts(year):
            end = start + pd.Timedelta(days=7)
            win = sampled.sel(time=slice(start, end))
            vals = win.sum(dim="time", skipna=True).values
            d = pts[["point_id", "lat", "lon"]].copy()
            d["date"] = start.strftime("%Y-%m-%d")
            d["year"] = year
            d["doy"] = doy
            d["et_gleam"] = vals
            rows.append(d)

        ds.close()

    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["et_gleam"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows", len(out), "points", out["point_id"].nunique())


if __name__ == "__main__":
    main()
