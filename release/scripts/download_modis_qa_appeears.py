#!/usr/bin/env python
from __future__ import annotations

import getpass
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
API = "https://appeears.earthdatacloud.nasa.gov/api"

POINTS = ROOT / "data/raw/gee/stable_grassland_points.csv"
OUTDIR = ROOT / "data/raw/appeears_modis_qa"
PROCESSED = ROOT / "data/processed/modis_qa_by_point_8day.csv"
SUMMARY = ROOT / "results/qc/modis_qc_summary.csv"
TASK_JSON = OUTDIR / "modis_qa_task.json"
TASK_META = OUTDIR / "modis_qa_task_meta.json"


def login() -> str:
    username = os.environ.get("EARTHDATA_USERNAME") or input("Earthdata/AppEEARS username: ").strip()
    password = os.environ.get("EARTHDATA_PASSWORD") or getpass.getpass("Earthdata/AppEEARS password: ")
    r = requests.post(f"{API}/login", auth=(username, password))
    print("Login status:", r.status_code)
    print(r.text[:500])
    r.raise_for_status()
    return r.json()["token"]


def read_points():
    df = pd.read_csv(POINTS)
    lat_col = next((c for c in ["lat", "latitude", "LAT", "Latitude"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "LON", "Longitude"] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise SystemExit(f"Missing lat/lon columns in {POINTS}. Columns={list(df.columns)}")

    if "point_id" not in df.columns:
        df["point_id"] = np.arange(len(df))

    df = df[["point_id", lat_col, lon_col]].rename(columns={lat_col: "lat", lon_col: "lon"}).dropna()
    df["point_id"] = df["point_id"].astype(str)

    coords = []
    for _, r in df.iterrows():
        coords.append({
            "latitude": float(r["lat"]),
            "longitude": float(r["lon"]),
            "id": str(r["point_id"]),
            "category": "stable_grassland"
        })

    print(f"Loaded {len(coords)} points from {POINTS}")
    return coords


def submit_task(token: str) -> str:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    coords = read_points()

    task = {
        "task_type": "point",
        "task_name": "grassland_wue_modis_qa_2001_2024",
        "params": {
            "dates": [
                {
                    "startDate": "01-01-2001",
                    "endDate": "12-31-2024"
                }
            ],
            "layers": [
                {"product": "MOD17A2HGF.061", "layer": "Gpp_500m"},
                {"product": "MOD17A2HGF.061", "layer": "Psn_QC_500m"},
                {"product": "MOD16A2GF.061", "layer": "ET_500m"},
                {"product": "MOD16A2GF.061", "layer": "ET_QC_500m"}
            ],
            "coordinates": coords
        }
    }

    TASK_JSON.write_text(json.dumps(task, indent=2))

    r = requests.post(
        f"{API}/task",
        json=task,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )

    print("Task submit status:", r.status_code)
    print(r.text[:2000])
    r.raise_for_status()

    meta = r.json()
    TASK_META.write_text(json.dumps(meta, indent=2))

    task_id = meta.get("task_id")
    if not task_id:
        raise SystemExit(f"No task_id returned: {meta}")

    print("Submitted AppEEARS task_id:", task_id)
    return task_id


def wait_task(token: str, task_id: str):
    while True:
        r = requests.get(
            f"{API}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        print("Task check status:", r.status_code)
        r.raise_for_status()

        meta = r.json()
        status = meta.get("status")
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "AppEEARS task", task_id, "status:", status)

        TASK_META.write_text(json.dumps(meta, indent=2))

        if status == "done":
            return

        if status in {"error", "failed"}:
            raise SystemExit(json.dumps(meta, indent=2))

        time.sleep(600)


def download_bundle(token: str, task_id: str):
    r = requests.get(
        f"{API}/bundle/{task_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    print("Bundle status:", r.status_code)
    print(r.text[:1000])
    r.raise_for_status()

    bundle = r.json()
    (OUTDIR / "bundle.json").write_text(json.dumps(bundle, indent=2))

    files = bundle.get("files", [])
    if not files:
        raise SystemExit("Bundle has no files.")

    for f in files:
        file_id = f["file_id"]
        file_name = f["file_name"]
        out = OUTDIR / file_name

        if out.exists() and out.stat().st_size > 0:
            print("Already downloaded:", out)
            continue

        print("Downloading:", file_name)
        rr = requests.get(
            f"{API}/bundle/{task_id}/{file_id}",
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=True,
            stream=True
        )
        rr.raise_for_status()

        with open(out, "wb") as w:
            for chunk in rr.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    w.write(chunk)

    print("Downloaded AppEEARS bundle to", OUTDIR)


def ingest():
    csvs = [
        p for p in sorted(OUTDIR.glob("*.csv"))
        if "README" not in p.name.upper()
    ]

    if not csvs:
        raise SystemExit(f"No CSV files found in {OUTDIR}")

    dfs = []
    for p in csvs:
        print("Reading", p)
        d = pd.read_csv(p)
        d["source_file"] = p.name
        dfs.append(d)

    raw = pd.concat(dfs, ignore_index=True)
    raw.columns = [c.strip() for c in raw.columns]

    raw.to_csv(OUTDIR / "raw_combined_appeears_modis.csv", index=False)

    point_col = next((c for c in raw.columns if c.lower() in {"id", "point_id", "sample_id"}), None)
    date_col = next((c for c in raw.columns if "date" in c.lower()), None)

    gpp_col = next((c for c in raw.columns if "Gpp_500m" in c), None)
    psn_qc_col = next((c for c in raw.columns if "Psn_QC_500m" in c), None)
    et_col = next((c for c in raw.columns if "ET_500m" in c and "QC" not in c), None)
    et_qc_col = next((c for c in raw.columns if "ET_QC_500m" in c), None)

    if point_col is None:
        point_col = raw.columns[0]

    out = pd.DataFrame()
    out["point_id"] = raw[point_col].astype(str)
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.NaT
    out["modis_gpp_appeears"] = pd.to_numeric(raw[gpp_col], errors="coerce") if gpp_col else np.nan
    out["modis_gpp_psn_qc"] = pd.to_numeric(raw[psn_qc_col], errors="coerce") if psn_qc_col else np.nan
    out["modis_et_appeears"] = pd.to_numeric(raw[et_col], errors="coerce") if et_col else np.nan
    out["modis_et_qc"] = pd.to_numeric(raw[et_qc_col], errors="coerce") if et_qc_col else np.nan

    out["modis_gpp_qc_good"] = (out["modis_gpp_psn_qc"].fillna(255).astype(int) & 3) <= 1
    out["modis_et_qc_good"] = (out["modis_et_qc"].fillna(255).astype(int) & 3) <= 1

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED, index=False)

    summary = pd.DataFrame([
        {
            "product": "MODIS GPP",
            "qa_column": "Psn_QC_500m",
            "n": int(out["modis_gpp_psn_qc"].notna().sum()),
            "good_fraction": float(out["modis_gpp_qc_good"].mean()),
            "source": "NASA AppEEARS MOD17A2HGF.061 point extraction"
        },
        {
            "product": "MODIS ET",
            "qa_column": "ET_QC_500m",
            "n": int(out["modis_et_qc"].notna().sum()),
            "good_fraction": float(out["modis_et_qc_good"].mean()),
            "source": "NASA AppEEARS MOD16A2GF.061 point extraction"
        },
    ])

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)

    print("Wrote", PROCESSED)
    print("Wrote", SUMMARY)
    print(summary)


def main():
    token = login()

    if TASK_META.exists():
        meta = json.loads(TASK_META.read_text())
        task_id = meta.get("task_id")
        if not task_id:
            task_id = submit_task(token)
        else:
            print("Resuming existing AppEEARS task:", task_id)
    else:
        task_id = submit_task(token)

    wait_task(token, task_id)
    download_bundle(token, task_id)
    ingest()


if __name__ == "__main__":
    main()
