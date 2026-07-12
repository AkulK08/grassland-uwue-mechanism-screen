#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


GPP_PRODUCTS = ["modis", "gosif", "pml"]
ET_PRODUCTS = ["modis", "gleam", "pml"]


def first_existing(cols, candidates, required=True, label="column"):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    if required:
        raise SystemExit(f"Could not find {label}. Tried {candidates}. Existing columns: {list(cols)}")
    return None


def wide_modis_qa(path: Path, out: Path) -> pd.DataFrame:
    qa = pd.read_csv(path)
    qa["point_id"] = qa["point_id"].astype(str)
    qa["date"] = pd.to_datetime(qa["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for c in ["Psn_QC_500m", "ET_QC_500m"]:
        if c not in qa.columns:
            qa[c] = np.nan
        qa[c] = pd.to_numeric(qa[c], errors="coerce")

    if "modis_gpp_qc_good" not in qa.columns:
        qa["modis_gpp_qc_good"] = qa["Psn_QC_500m"].map(lambda x: np.nan if pd.isna(x) else ((int(x) & 3) <= 1))
    if "modis_et_qc_good" not in qa.columns:
        qa["modis_et_qc_good"] = qa["ET_QC_500m"].map(lambda x: np.nan if pd.isna(x) else ((int(x) & 3) <= 1))

    # Collapse long/product-stacked AppEEARS rows to one row per point/date.
    g = qa.groupby(["point_id", "date"], dropna=False).agg(
        Psn_QC_500m=("Psn_QC_500m", "first"),
        ET_QC_500m=("ET_QC_500m", "first"),
        modis_gpp_qc_good=("modis_gpp_qc_good", lambda s: True if (s == True).any() else (False if (s == False).any() else np.nan)),
        modis_et_qc_good=("modis_et_qc_good", lambda s: True if (s == True).any() else (False if (s == False).any() else np.nan)),
    ).reset_index()

    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(out, index=False)
    return g


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd


def build_one_matrix(input_csv: Path, output_csv: Path, qa_wide: pd.DataFrame | None, soil_csv: Path | None):
    df = pd.read_csv(input_csv)
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    if qa_wide is not None:
        df = df.merge(qa_wide, on=["point_id", "date"], how="left")

    if soil_csv and soil_csv.exists():
        soil = pd.read_csv(soil_csv)
        soil["point_id"] = soil["point_id"].astype(str)
        keep = [c for c in soil.columns if c == "point_id" or c.startswith("soil_") or c in ["sand", "silt", "clay"]]
        df = df.merge(soil[keep], on="point_id", how="left")

    vpd_col = first_existing(
        df.columns,
        ["vpd", "vpd_kpa", "vpd_mean", "era5_vpd", "vpd_era5", "vpd_daily", "vpd_8day"],
        label="VPD column"
    )
    sm_col = first_existing(
        df.columns,
        ["soil_moisture", "sm", "swvl", "swvl_rootzone", "rootzone_sm", "era5_sm", "sm_rootzone", "soil_moisture_rootzone"],
        label="soil moisture column"
    )

    df["vpd_for_metric"] = pd.to_numeric(df[vpd_col], errors="coerce")
    df["sm_for_stress"] = pd.to_numeric(df[sm_col], errors="coerce")

    # If VPD is in Pa, convert rough Pa -> kPa. If already kPa, leave it.
    if df["vpd_for_metric"].median(skipna=True) > 50:
        df["vpd_for_metric"] = df["vpd_for_metric"] / 1000.0

    # Use point-wise standardized anomalies where possible.
    df["vpd_z"] = df.groupby("point_id")["vpd_for_metric"].transform(zscore)
    df["sm_z"] = df.groupby("point_id")["sm_for_stress"].transform(zscore)
    df["compound_z"] = 0.5 * df["vpd_z"] - 0.5 * df["sm_z"]
    df["vpd_x_sm"] = df["vpd_z"] * df["sm_z"]

    for gpp in GPP_PRODUCTS:
        gcol = f"gpp_{gpp}"
        if gcol not in df.columns:
            continue
        df[gcol] = pd.to_numeric(df[gcol], errors="coerce")

    for et in ET_PRODUCTS:
        ecol = f"et_{et}"
        if ecol not in df.columns:
            continue
        df[ecol] = pd.to_numeric(df[ecol], errors="coerce")

    for gpp in GPP_PRODUCTS:
        gcol = f"gpp_{gpp}"
        if gcol not in df.columns:
            continue

        for et in ET_PRODUCTS:
            ecol = f"et_{et}"
            if ecol not in df.columns:
                continue

            base = f"{gpp}_{et}"
            valid = (
                (df[gcol] > 0) &
                (df[ecol] > 0.1) &
                (df["vpd_for_metric"] > 0) &
                np.isfinite(df[gcol]) &
                np.isfinite(df[ecol]) &
                np.isfinite(df["vpd_for_metric"])
            )

            # Enforce MODIS QA if the combo uses MODIS GPP or MODIS ET.
            if gpp == "modis" and "modis_gpp_qc_good" in df.columns:
                valid &= (df["modis_gpp_qc_good"] == True)
            if et == "modis" and "modis_et_qc_good" in df.columns:
                valid &= (df["modis_et_qc_good"] == True)

            df[f"raw_wue_{base}"] = np.where(valid, df[gcol] / df[ecol], np.nan)
            df[f"uwue_{base}"] = np.where(valid, df[gcol] * np.sqrt(df["vpd_for_metric"]) / df[ecol], np.nan)
            df[f"iwue_{base}"] = np.where(valid, df[gcol] * df["vpd_for_metric"] / df[ecol], np.nan)

            for metric in ["raw_wue", "uwue", "iwue"]:
                c = f"{metric}_{base}"
                df[f"log_{c}"] = np.where(df[c] > 0, np.log(df[c]), np.nan)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    manifest = {
        "input": str(input_csv),
        "output": str(output_csv),
        "rows": int(len(df)),
        "vpd_column_used": vpd_col,
        "soil_moisture_column_used": sm_col,
        "primary_metric": "log_uwue_<gpp>_<et>",
        "sensitivity_metrics": ["log_iwue_<gpp>_<et>", "log_raw_wue_<gpp>_<et>"],
        "modis_qa_enforced_for_modis_products": True,
        "et_floor_mm_per_8day": 0.1,
    }
    Path(str(output_csv) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/agents/merged_full_matrix_raw.csv")
    ap.add_argument("--co2", default="data/raw/agents/merged_full_matrix_co2corrected.csv")
    ap.add_argument("--qa", default="data/processed/modis_qa_by_point_8day.csv")
    ap.add_argument("--qa-wide", default="data/processed/modis_qa_by_point_8day_wide.csv")
    ap.add_argument("--soil", default="data/external/soilgrids_texture_by_point.csv")
    args = ap.parse_args()

    qa_wide = None
    if Path(args.qa).exists():
        qa_wide = wide_modis_qa(Path(args.qa), Path(args.qa_wide))
        print("WROTE", args.qa_wide, qa_wide.shape)
    else:
        print("WARNING: no MODIS QA file found; continuing without MODIS QA merge.")

    soil_csv = Path(args.soil)
    build_one_matrix(Path(args.raw), Path("data/processed/project_metric_matrix_raw.csv"), qa_wide, soil_csv)
    build_one_matrix(Path(args.co2), Path("data/processed/project_metric_matrix_co2corrected.csv"), qa_wide, soil_csv)


if __name__ == "__main__":
    main()
