#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import pandas as pd
import numpy as np


def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for c in df.columns:
        cl = c.lower()
        for n in names:
            if n.lower() in cl:
                return c
    raise SystemExit(f"Cannot find one of {names}. Columns: {list(df.columns)}")


def extract_layer(resp, prop):
    vals = []
    try:
        layers = resp["properties"]["layers"]
    except Exception:
        return np.nan

    for layer in layers:
        if layer.get("name") != prop:
            continue
        for d in layer.get("depths", []):
            v = d.get("values", {}).get("mean")
            if v is not None:
                vals.append(float(v))
    if not vals:
        return np.nan
    # SoilGrids values for texture fractions are often g/kg, so divide later if needed.
    return float(np.nanmean(vals))


def query_soilgrids(lat, lon, timeout=30):
    params = [
        ("lon", lon),
        ("lat", lat),
        ("property", "sand"),
        ("property", "silt"),
        ("property", "clay"),
        ("depth", "0-5cm"),
        ("depth", "5-15cm"),
        ("depth", "15-30cm"),
        ("value", "mean"),
    ]
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "grassland-wue-research/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="data/raw/gee/stable_grassland_points.csv")
    ap.add_argument("--out", default="data/external/soilgrids_texture_by_point.csv")
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    pts = pd.read_csv(args.points)
    pts["point_id"] = pts["point_id"].astype(str)
    lat_col = pick_col(pts, ["lat", "latitude"])
    lon_col = pick_col(pts, ["lon", "longitude"])

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        old = pd.read_csv(out_path)
        if "point_id" in old.columns:
            done = {str(r["point_id"]): r for _, r in old.iterrows()}

    rows = []
    for i, r in pts.iterrows():
        pid = str(r["point_id"])
        if pid in done and pd.notna(done[pid].get("soil_sand_mean", np.nan)):
            rows.append(dict(done[pid]))
            continue

        lat = float(r[lat_col])
        lon = float(r[lon_col])
        print(f"[{i+1}/{len(pts)}] SoilGrids point_id={pid} lat={lat} lon={lon}", flush=True)

        row = {"point_id": pid, "lat": lat, "lon": lon}
        try:
            js = query_soilgrids(lat, lon)
            sand = extract_layer(js, "sand")
            silt = extract_layer(js, "silt")
            clay = extract_layer(js, "clay")

            # Convert g/kg to fraction if needed.
            row["soil_sand_mean"] = sand / 1000.0 if sand > 1 else sand
            row["soil_silt_mean"] = silt / 1000.0 if silt > 1 else silt
            row["soil_clay_mean"] = clay / 1000.0 if clay > 1 else clay

            # Simple derived hydraulic/texture descriptor. This is not a full pedotransfer model;
            # it is a reproducible control variable for reviewer-facing soil texture adjustment.
            row["soil_texture_coarse_index"] = row["soil_sand_mean"] - row["soil_clay_mean"]
            row["soil_texture_fine_index"] = row["soil_clay_mean"] + row["soil_silt_mean"]
            row["soilgrids_status"] = "ok"
        except Exception as e:
            row["soil_sand_mean"] = np.nan
            row["soil_silt_mean"] = np.nan
            row["soil_clay_mean"] = np.nan
            row["soil_texture_coarse_index"] = np.nan
            row["soil_texture_fine_index"] = np.nan
            row["soilgrids_status"] = f"error: {e}"

        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("WROTE", out_path, df.shape)
    print(df["soilgrids_status"].value_counts(dropna=False).head())


if __name__ == "__main__":
    main()
