#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PAT = re.compile(r"GOSIF_GPP_(\d{4})(\d{3})_Mean\.tif(?:\.gz)?$")


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


def parse_file(path: str):
    m = PAT.search(Path(path).name)
    if not m:
        return None
    year = int(m.group(1))
    doy = int(m.group(2))
    date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    return year, doy, date


def open_raster_maybe_gz(path: str):
    p = Path(path)
    if p.suffix == ".gz":
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        with gzip.open(p, "rb") as src, open(tmp.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return rasterio.open(tmp.name), Path(tmp.name)
    return rasterio.open(p), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--local-glob", default="data/raw/gosif/*.tif*")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pts = read_points(args.points)
    coords = list(zip(pts["lon"].astype(float), pts["lat"].astype(float)))

    files = []
    for f in sorted(glob(args.local_glob)):
        parsed = parse_file(f)
        if parsed is None:
            continue
        year, doy, date = parsed
        if args.start_year <= year <= args.end_year:
            files.append((f, year, doy, date))

    if not files:
        raise SystemExit("No GOSIF files matched requested years.")

    rows = []
    for i, (f, year, doy, date) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Sampling {f}")
        ds = None
        tmp = None
        try:
            ds, tmp = open_raster_maybe_gz(f)
            vals = []
            for v in ds.sample(coords):
                x = float(v[0]) if len(v) else np.nan
                if ds.nodata is not None and np.isclose(x, ds.nodata):
                    x = np.nan
                vals.append(x)
            d = pts[["point_id", "lat", "lon"]].copy()
            d["date"] = date.strftime("%Y-%m-%d")
            d["year"] = year
            d["doy"] = doy
            d["gpp_gosif"] = vals
            rows.append(d)
        finally:
            if ds is not None:
                ds.close()
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["gpp_gosif"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "rows", len(out), "points", out["point_id"].nunique())


if __name__ == "__main__":
    main()
