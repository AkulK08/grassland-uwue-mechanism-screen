#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def find_col(cols, patterns):
    ncols = {c: norm(c) for c in cols}
    for pat in patterns:
        r = re.compile(pat)
        for c, nc in ncols.items():
            if r.search(nc):
                return c
    return None


def modland_good_or_other(x):
    """
    Conservative MODIS QA rule:
    accept if low two MODLAND QA bits are 0 or 1.
    This implements the project rule: keep QA bits 0-1 / good-or-other.
    """
    try:
        if pd.isna(x):
            return np.nan
        v = int(float(x))
        return ((v & 3) <= 1)
    except Exception:
        return np.nan


def read_all_tables(folder: Path):
    paths = []
    for ext in ["*.csv", "*.txt"]:
        paths.extend(folder.rglob(ext))
    tables = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            df["_source_file"] = str(p)
            tables.append(df)
            print(f"READ {p} shape={df.shape}")
        except Exception as e:
            print(f"SKIP {p}: {e}")
    if not tables:
        raise SystemExit(f"No readable CSV/TXT tables found under {folder}")
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--extract-dir", default="data/raw/appeears_modis_qa")
    ap.add_argument("--points", default="data/raw/gee/stable_grassland_points.csv")
    ap.add_argument("--out", default="data/processed/modis_qa_by_point_8day.csv")
    ap.add_argument("--summary", default="results/qc/modis_qc_summary.csv")
    ap.add_argument("--doc", default="docs/modis_qa.md")
    ap.add_argument("--manifest", default="results/qc/modis_qa_appeears_ingest_manifest.json")
    args = ap.parse_args()

    zpath = Path(args.zip).expanduser()
    extract_dir = Path(args.extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("ZIP:", zpath)
    print("EXTRACT:", extract_dir)

    with zipfile.ZipFile(zpath) as z:
        z.extractall(extract_dir)
        names = z.namelist()

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps({"zip": str(zpath), "extract_dir": str(extract_dir), "zip_names": names[:500]}, indent=2))

    tables = read_all_tables(extract_dir)

    records = []
    for df in tables:
        cols = list(df.columns)
        date_col = find_col(cols, [r"^date$", r"datetime", r"calendar_date", r"acquisition"])
        lat_col = find_col(cols, [r"^lat$", r"latitude"])
        lon_col = find_col(cols, [r"^lon$", r"longitude"])
        id_col = find_col(cols, [r"point_id", r"site_id", r"sample_id", r"fid", r"^id$"])

        psn_col = find_col(cols, [r"psn.*qc", r"psn_qc", r"mod17.*qc"])
        et_col = find_col(cols, [r"et.*qc", r"et_qc", r"mod16.*qc"])

        # AppEEARS sometimes stores product/band in long format.
        band_col = find_col(cols, [r"band", r"layer", r"product"])
        value_col = find_col(cols, [r"^value$", r"pixel_value", r"mean"])

        if psn_col is None and et_col is None and band_col and value_col:
            long = df.copy()
            long["_band_norm"] = long[band_col].map(norm)
            base_cols = []
            for c in [date_col, lat_col, lon_col, id_col, "_source_file"]:
                if c and c in long.columns:
                    base_cols.append(c)
            psn = long[long["_band_norm"].str.contains("psn") & long["_band_norm"].str.contains("qc")]
            et = long[long["_band_norm"].str.contains("et") & long["_band_norm"].str.contains("qc")]
            if len(psn):
                tmp = psn[base_cols + [value_col]].rename(columns={value_col: "Psn_QC_500m"})
                records.append(tmp)
            if len(et):
                tmp = et[base_cols + [value_col]].rename(columns={value_col: "ET_QC_500m"})
                records.append(tmp)
            continue

        use = []
        for c in [date_col, lat_col, lon_col, id_col, psn_col, et_col, "_source_file"]:
            if c and c in df.columns and c not in use:
                use.append(c)
        if psn_col or et_col:
            tmp = df[use].copy()
            if psn_col and psn_col != "Psn_QC_500m":
                tmp = tmp.rename(columns={psn_col: "Psn_QC_500m"})
            if et_col and et_col != "ET_QC_500m":
                tmp = tmp.rename(columns={et_col: "ET_QC_500m"})
            records.append(tmp)

    if not records:
        print("Could not auto-detect QA columns.")
        print("All extracted table columns:")
        for df in tables:
            print(df["_source_file"].iloc[0])
            print(list(df.columns))
        raise SystemExit(2)

    out = pd.concat(records, ignore_index=True, sort=False)

    # Normalize date.
    date_col = find_col(out.columns, [r"^date$", r"datetime", r"calendar_date", r"acquisition"])
    if date_col is None:
        raise SystemExit("Could not find date column in AppEEARS QA output.")
    out["date"] = pd.to_datetime(out[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Normalize point_id.
    if "point_id" not in out.columns:
        id_col = find_col(out.columns, [r"site_id", r"sample_id", r"fid", r"^id$"])
        if id_col:
            out["point_id"] = out[id_col].astype(str)
        else:
            lat_col = find_col(out.columns, [r"^lat$", r"latitude"])
            lon_col = find_col(out.columns, [r"^lon$", r"longitude"])
            if lat_col is None or lon_col is None:
                raise SystemExit("No point_id or lat/lon columns found.")
            pts = pd.read_csv(args.points)
            pts["lat_round"] = pd.to_numeric(pts["lat"], errors="coerce").round(5)
            pts["lon_round"] = pd.to_numeric(pts["lon"], errors="coerce").round(5)
            pts["point_id"] = pts["point_id"].astype(str)
            out["lat_round"] = pd.to_numeric(out[lat_col], errors="coerce").round(5)
            out["lon_round"] = pd.to_numeric(out[lon_col], errors="coerce").round(5)
            out = out.merge(pts[["point_id", "lat_round", "lon_round"]], on=["lat_round", "lon_round"], how="left")
    else:
        out["point_id"] = out["point_id"].astype(str)

    for c in ["Psn_QC_500m", "ET_QC_500m"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["modis_gpp_qc_good"] = out["Psn_QC_500m"].map(modland_good_or_other)
    out["modis_et_qc_good"] = out["ET_QC_500m"].map(modland_good_or_other)

    keep_cols = ["point_id", "date", "Psn_QC_500m", "ET_QC_500m", "modis_gpp_qc_good", "modis_et_qc_good", "_source_file"]
    out = out[[c for c in keep_cols if c in out.columns]].dropna(subset=["date"])
    out = out.drop_duplicates()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    rows = []
    for name, flag in [
        ("MOD17A2HGF.061/Psn_QC_500m", "modis_gpp_qc_good"),
        ("MOD16A2GF.061/ET_QC_500m", "modis_et_qc_good"),
    ]:
        s = out[flag]
        rows.append({
            "product_qa_band": name,
            "qa_rule": "accept low two MODLAND QA bits 0 or 1: (QA & 3) <= 1",
            "n_observations": int(s.notna().sum()),
            "n_good_or_other": int((s == True).sum()),
            "n_bad": int((s == False).sum()),
            "n_missing": int(s.isna().sum()),
            "good_fraction": float((s == True).sum() / max(1, s.notna().sum())),
        })
    summ = pd.DataFrame(rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.summary, index=False)

    Path(args.doc).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc).write_text(
        "# MODIS QA verification from AppEEARS\n\n"
        f"Source zip: `{zpath}`\n\n"
        f"Extracted folder: `{extract_dir}`\n\n"
        "Products/bands targeted:\n\n"
        "- MOD17A2HGF.061 `Psn_QC_500m`\n"
        "- MOD16A2GF.061 `ET_QC_500m`\n\n"
        "QA rule used: accept observations whose low two MODLAND QA bits are 0 or 1, implemented as `(QA & 3) <= 1`.\n\n"
        "Outputs:\n\n"
        f"- `{args.out}`\n"
        f"- `{args.summary}`\n\n"
        "This file should be manually checked against the AppEEARS layer metadata before manuscript submission.\n"
    )

    print("WROTE", args.out, out.shape)
    print("WROTE", args.summary)
    print(summ)


if __name__ == "__main__":
    main()
