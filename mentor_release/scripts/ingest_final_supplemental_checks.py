#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
SUP = ROOT / "data/raw/gee_supplemental"


def files(pattern: str):
    out = sorted(SUP.glob(pattern))
    if not out:
        raise SystemExit(f"Missing supplemental files matching {pattern}")
    return out


def write(path: str, text: str):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print("Wrote", path)


def write_csv(path: str, df: pd.DataFrame):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print("Wrote", path, "rows=", len(df), "cols=", len(df.columns))


def read_many(pattern: str) -> pd.DataFrame:
    dfs = []
    for f in files(pattern):
        d = pd.read_csv(f)
        d["source_file"] = f.name
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def load_gee_matrix() -> pd.DataFrame:
    dfs = []
    for f in sorted((ROOT / "data/raw/gee").glob("wue_timeseries_*.csv")):
        d = pd.read_csv(f)
        d["source_file"] = f.name
        dfs.append(d)
    if not dfs:
        raise SystemExit("Missing data/raw/gee/wue_timeseries_*.csv")
    out = pd.concat(dfs, ignore_index=True)
    out["point_id"] = out["point_id"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def main():
    gee = load_gee_matrix()

    # SMAP validation.
    smap = read_many("final_smap_l4_by_point_8day*.csv")
    smap["point_id"] = smap["point_id"].astype(str)
    smap["date"] = pd.to_datetime(smap["date"], errors="coerce")

    smap_band = "smap_sm_rootzone" if "smap_sm_rootzone" in smap.columns else None
    if smap_band is None:
        candidates = [c for c in smap.columns if "sm" in c.lower()]
        if not candidates:
            raise SystemExit(f"No SMAP band found. Columns={list(smap.columns)}")
        smap_band = candidates[0]

    if "soil_moisture" not in gee.columns:
        raise SystemExit("GEE time-series lacks soil_moisture column.")

    matched = gee[["point_id", "date", "soil_moisture"]].merge(
        smap[["point_id", "date", smap_band]],
        on=["point_id", "date"],
        how="inner",
    )
    matched["soil_moisture"] = pd.to_numeric(matched["soil_moisture"], errors="coerce")
    matched[smap_band] = pd.to_numeric(matched[smap_band], errors="coerce")
    matched = matched.dropna(subset=["soil_moisture", smap_band])

    if len(matched) < 100:
        raise SystemExit(f"Too few SMAP/ERA5 matched rows: {len(matched)}")

    smap_summary = pd.DataFrame([{
        "comparison": "ERA5_Land_soil_moisture_vs_SMAP_L4",
        "smap_band": smap_band,
        "n_matched": len(matched),
        "pearson_r": matched["soil_moisture"].corr(matched[smap_band]),
        "rmse": float(np.sqrt(np.mean((matched["soil_moisture"] - matched[smap_band]) ** 2))),
        "bias_era5_minus_smap": float((matched["soil_moisture"] - matched[smap_band]).mean()),
        "date_min": matched["date"].min().date(),
        "date_max": matched["date"].max().date(),
    }])
    write_csv("results/stress/smap_era5_comparison.csv", smap_summary)
    write_csv("data/processed/smap_era5_matched_points.csv", matched)

    # MODIS QA proof.
    qa = read_many("final_modis_qa_by_point_8day*.csv")
    qa["point_id"] = qa["point_id"].astype(str)
    qa["date"] = pd.to_datetime(qa["date"], errors="coerce")

    needed = ["modis_gpp_qc_good", "modis_et_qc_good"]
    missing = [c for c in needed if c not in qa.columns]
    if missing:
        raise SystemExit(f"MODIS QA columns missing: {missing}; columns={list(qa.columns)}")

    for c in needed:
        qa[c] = pd.to_numeric(qa[c], errors="coerce")

    qa_summary = pd.DataFrame([
        {
            "product": "MODIS GPP",
            "qa_column": "modis_gpp_qc_good",
            "rule": "MOD17A2HGF Psn_QC low two bits <= 1",
            "n": int(qa["modis_gpp_qc_good"].notna().sum()),
            "good_fraction": float((qa["modis_gpp_qc_good"] == 1).mean()),
            "bad_or_missing_fraction": float((qa["modis_gpp_qc_good"] != 1).mean()),
        },
        {
            "product": "MODIS ET",
            "qa_column": "modis_et_qc_good",
            "rule": "MOD16A2GF ET_QC low two bits <= 1",
            "n": int(qa["modis_et_qc_good"].notna().sum()),
            "good_fraction": float((qa["modis_et_qc_good"] == 1).mean()),
            "bad_or_missing_fraction": float((qa["modis_et_qc_good"] != 1).mean()),
        },
    ])
    write_csv("results/qc/modis_qc_summary.csv", qa_summary)
    write_csv("data/processed/modis_qa_by_point_8day.csv", qa)

    # Irrigation exclusion.
    irr_files = files("final_irrigation_mask_by_point*.csv")
    irr = pd.read_csv(irr_files[0])
    irr["point_id"] = irr["point_id"].astype(str)

    mask_col = "irrigation_or_agri_mask" if "irrigation_or_agri_mask" in irr.columns else None
    if mask_col is None:
        candidates = [c for c in irr.columns if c not in {"system:index", ".geo", "point_id", "lat", "lon"}]
        if not candidates:
            raise SystemExit(f"No irrigation mask column found. Columns={list(irr.columns)}")
        mask_col = candidates[0]

    irr[mask_col] = pd.to_numeric(irr[mask_col], errors="coerce").fillna(0)
    out = irr[["point_id", "lat", "lon", mask_col]].rename(columns={mask_col: "irrigation_or_agri_mask"})
    out["exclude_irrigated"] = out["irrigation_or_agri_mask"] > 0
    write_csv("data/external/irrigation_by_point.csv", out)

    excluded = set(out.loc[out["exclude_irrigated"], "point_id"].astype(str))
    n_total = len(out)
    n_excl = len(excluded)

    irr_summary = pd.DataFrame([{
        "filter": "GFSAD1000 irrigated major cropland class or user-supplied irrigation asset",
        "n_points": n_total,
        "n_excluded": n_excl,
        "excluded_fraction": n_excl / n_total if n_total else np.nan,
        "status": "applied",
    }])
    write_csv("results/qc/irrigation_exclusion_summary.csv", irr_summary)

    # Create irrigation-filtered GEE folder.
    out_dir = ROOT / "data/raw/gee_final_filtered_no_irrigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = pd.read_csv(ROOT / "data/raw/gee/stable_grassland_points.csv")
    pts["point_id"] = pts["point_id"].astype(str)
    pts2 = pts[~pts["point_id"].isin(excluded)].copy()
    pts2.to_csv(out_dir / "stable_grassland_points.csv", index=False)

    for f in sorted((ROOT / "data/raw/gee").glob("wue_timeseries_*.csv")):
        d = pd.read_csv(f)
        d["point_id"] = d["point_id"].astype(str)
        d2 = d[~d["point_id"].isin(excluded)].copy()
        d2.to_csv(out_dir / f.name, index=False)

    land_summary = pd.DataFrame([
        {"filter": "stable MCD12Q1 grassland sample", "status": "input", "n_points": n_total},
        {"filter": "irrigation/agriculture exclusion", "status": "applied", "n_points_removed": n_excl},
        {"filter": "final sample", "status": "written to data/raw/gee_final_filtered_no_irrigation", "n_points_after": len(pts2)},
    ])
    write_csv("results/qc/land_cover_filter_summary.csv", land_summary)

    write_csv("results/qc/burned_area_exclusion_summary.csv", pd.DataFrame([{
        "filter": "MCD64A1 burned area",
        "status": "burned flag included in point-time data and available for exclusion/QC in final preprocessing",
    }]))

    write(
        "docs/smap_validation.md",
        "# SMAP validation\n\nSMAP L4 root-zone soil moisture was sampled at stable grassland points for 8-day windows after 2015 and compared against ERA5-Land soil moisture. See `results/stress/smap_era5_comparison.csv`.\n",
    )
    write(
        "docs/modis_qa_verification.md",
        "# MODIS QA verification\n\nMODIS GPP QA uses MOD17A2HGF Psn_QC. MODIS ET QA uses MOD16A2GF ET_QC. Good observations are defined using low two bits <= 1. See `results/qc/modis_qc_summary.csv`.\n",
    )
    write(
        "docs/irrigation_exclusion.md",
        "# Irrigation exclusion\n\nAn irrigation/agriculture mask was sampled at stable points and used to write `data/raw/gee_final_filtered_no_irrigation/`. See `results/qc/irrigation_exclusion_summary.csv`.\n",
    )

    manifest = {
        "smap_validation": "results/stress/smap_era5_comparison.csv",
        "modis_qa_summary": "results/qc/modis_qc_summary.csv",
        "irrigation_summary": "results/qc/irrigation_exclusion_summary.csv",
        "filtered_gee_folder": "data/raw/gee_final_filtered_no_irrigation",
        "smap_files": len(files("final_smap_l4_by_point_8day*.csv")),
        "qa_files": len(files("final_modis_qa_by_point_8day*.csv")),
    }
    write("results/qc/final_supplemental_manifest.json", json.dumps(manifest, indent=2))
    print("\nFINAL SUPPLEMENTAL INGEST PASS")


if __name__ == "__main__":
    main()
