#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
os.chdir(ROOT)

START_YEAR = int(os.environ.get("FULL_START_YEAR", "2001"))
END_YEAR = int(os.environ.get("FULL_END_YEAR", "2024"))
ALLOW_SHORT_GEE = os.environ.get("ALLOW_SHORT_GEE", "0") == "1"


def run(cmd: str, check: bool = True):
    print("\n" + "=" * 90)
    print(cmd)
    print("=" * 90)
    return subprocess.run(cmd, shell=True, check=check)


def fail(msg: str):
    raise SystemExit("\nFINAL RUN STOPPED:\n" + msg + "\n")


def write(path: str, text: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print("WROTE", path)


def write_csv(path: str, df: pd.DataFrame):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print("WROTE", path, "rows=", len(df), "cols=", len(df.columns))


def require_file(path: str):
    if not Path(path).exists():
        fail(f"Missing required final input: {path}")
    print("PASS", path)


def require_glob(pattern: str, min_count: int):
    files = sorted(glob(pattern))
    if len(files) < min_count:
        fail(f"Missing required final input glob: {pattern}; found {len(files)}, expected >= {min_count}")
    print("PASS", pattern, "count=", len(files))
    return files


def ensure_dirs():
    for d in [
        "docs", "logs", "data/provenance", "data/processed", "data/raw/agents",
        "results/qc", "results/decomposition", "results/stress", "results/growing_season",
        "results/full_matrix/raw", "results/full_matrix/co2corrected", "results/bayesian",
        "results/aridity", "results/tower_validation", "results/traits",
        "results/final_memos", "results/final_figures", "results/manuscript",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


def verify_final_raw_inputs():
    print("\nFINAL INPUT VERIFICATION")

    require_file("data/raw/gee/stable_grassland_points.csv")
    gee_files = require_glob("data/raw/gee/wue_timeseries_*.csv", 1)
    require_glob("data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz", 1000)
    require_glob("data/raw/gleam/E_20*_GLEAM_v4.3a.nc", 24)

    require_file("data/external/aridity_by_point.csv")
    require_file("data/external/noaa_co2_8day.csv")

    require_file("data/external/liu_2021_psi50_0p1deg.nc")
    require_file("data/external/konings_gentine_isohydricity_0p1deg.nc")
    require_file("data/external/stocker_2023_rooting_depth_0p1deg.nc")

    require_file("data/raw/towers/fluxnet2015_grassland_sites.csv")
    require_file("data/raw/towers/ameriflux_grassland_sites.csv")
    require_file("data/raw/towers/icos_grassland_sites.csv")
    require_file("data/raw/towers/ozflux_grassland_sites.csv")

    dates = []
    for f in gee_files:
        try:
            d = pd.read_csv(f, usecols=["date"])
            dates.extend(pd.to_datetime(d["date"], errors="coerce").dropna().tolist())
        except Exception as e:
            print("WARNING could not inspect", f, e)

    if not dates:
        fail("Could not read dates from GEE files.")

    min_date, max_date = min(dates), max(dates)
    print("GEE DATE COVERAGE:", min_date.date(), "to", max_date.date())

    if min_date.year > START_YEAR or max_date.year < END_YEAR:
        message = (
            f"Your GEE point-time files cover {min_date.date()} to {max_date.date()}, "
            f"but the final full-paper run is set to {START_YEAR}–{END_YEAR}. "
            "That is not the full long-period paper. You need to export/download the missing "
            "GEE MODIS/PML/ERA5 point-time files for the full period before this can be honestly final."
        )
        if not ALLOW_SHORT_GEE:
            fail(message + "\n\nIf you intentionally want to run the available-period final computation anyway, run:\nALLOW_SHORT_GEE=1 caffeinate -dimsu ./scripts/final_paper_run_all.sh")
        print("WARNING:", message)

    return min_date, max_date


def patch_docs_and_static_outputs():
    print("\nPATCHING FINAL DOCS, QC SKELETONS, AND PROVENANCE")

    write(
        "docs/preregistration.md",
        f"""# OSF Preregistration Draft

## Project
Grassland ecosystem water-use efficiency response to compound atmospheric and soil-moisture stress.

## Hypotheses
1. Grassland WUE responses to compound VPD and soil-moisture stress can be classified as enhancement, saturation, or reversal.
2. Thresholds are accepted only when segmented-regression bootstrap intervals and Bayesian change-point credible intervals overlap.
3. Robust claims require consistency across at least two GPP products, two ET products, two stress metrics, and two growing-season definitions.
4. Tower validation decides whether satellite product disagreement can be resolved.
5. Trait analysis is conditional on Gates 1–3 passing.

## Product matrix
GPP: MODIS, GOSIF, PML.
ET: MODIS, GLEAM, PML.

## Stress metrics
z-score compound stress, joint percentile stress, copula-style joint stress, and VPD × soil-moisture interaction.

## Growing-season definitions
GPP > 20% annual peak, climate-based growing season, and all observations with seasonal fixed effects.

## Time span
Requested final span: {START_YEAR}–{END_YEAR}.

## Success criteria
Gate 1: defensible response shape with uncertainty.
Gate 2: consistency across products, stress definitions, and growing-season definitions.
Gate 3: tower validation selects product family or shows irreducible uncertainty.
Trait phase: traits explain >20% climate-residual variance and survive cross-validation/significance checks.

## OSF
Upload this file to OSF before treating the final multi-product matrix as confirmatory.
""",
    )

    provenance = pd.DataFrame([
        {"dataset": "Earth Engine MODIS/PML/ERA5 point-time files", "path": "data/raw/gee/wue_timeseries_*.csv", "status": "present; date coverage checked at runtime"},
        {"dataset": "stable grassland points", "path": "data/raw/gee/stable_grassland_points.csv", "status": "present"},
        {"dataset": "GOSIF GPP v2 8-day Mean", "path": "data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz", "status": "present"},
        {"dataset": "GLEAM daily actual evaporation", "path": "data/raw/gleam/E_20*_GLEAM_v4.3a.nc", "status": "present; v4.3a used because v3.8a unavailable from current SFTP listing"},
        {"dataset": "NOAA CO2 8-day", "path": "data/external/noaa_co2_8day.csv", "status": "present"},
        {"dataset": "aridity by point", "path": "data/external/aridity_by_point.csv", "status": "present"},
        {"dataset": "tower CSVs", "path": "data/raw/towers/*.csv", "status": "present"},
        {"dataset": "trait maps", "path": "data/external/*_0p1deg.nc", "status": "present"},
    ])
    write_csv("data/provenance/data_provenance.csv", provenance)

    write(
        "docs/vpd_processing.md",
        """# VPD processing

VPD is represented by the `vpd` column in the Earth Engine point-time files. It is derived from ERA5-Land temperature and dewpoint in the extraction workflow and aggregated to 8-day scale. The final methods section should reproduce the exact formula from the Earth Engine script.
""",
    )

    write(
        "docs/soil_moisture_processing.md",
        """# Soil moisture processing

Root-zone soil moisture is represented by the `soil_moisture` column in the Earth Engine point-time files. It is derived from ERA5-Land swvl1 and swvl2. The final methods section must state the weighting scheme used in the Earth Engine extraction workflow.
""",
    )

    write(
        "docs/land_cover_filtering.md",
        """# Land-cover filtering

Primary sample: stable MCD12Q1 IGBP class 10 grasslands.
Extensions: classes 8 and 9 can be added as savanna/woody-savanna extensions.
Disturbance filtering: burned periods/pixels are excluded using the `burned` flag where available.
Agricultural filtering: cropland/mixed pixels are excluded through land-cover selection. Irrigation exclusion requires an irrigation mask if available.
""",
    )

    write(
        "docs/growing_season_definitions.md",
        """# Growing-season definitions

1. Baseline: GPP > 20% annual peak.
2. Climate-based: temperature/precipitation growing-season mask.
3. Fixed-effects approach: all observations retained with seasonal/month fixed effects.
""",
    )

    write(
        "docs/compound_stress_metrics.md",
        """# Compound stress metrics

1. Baseline: standardized VPD minus standardized soil moisture.
2. Percentile: VPD above 75th percentile and soil moisture below 25th percentile.
3. Copula-style: empirical rank-based joint atmospheric/soil dryness.
4. Interaction: VPD, soil moisture, and VPD × soil moisture.
""",
    )

    write_csv("results/stress/smap_era5_comparison.csv", pd.DataFrame([{
        "requirement": "SMAP L4 post-2015 soil-moisture validation",
        "status": "not_run_unless_SMAP_layer_added",
        "interpretation": "Final project contains the placeholder/provenance row; add SMAP point-time data for a true validation comparison.",
    }]))

    write_csv("results/qc/land_cover_filter_summary.csv", pd.DataFrame([
        {"filter": "MCD12Q1 IGBP class 10", "status": "stable grassland point file used"},
        {"filter": "classes 8/9 savanna extension", "status": "optional extension; not primary run"},
        {"filter": "unstable land cover", "status": "excluded during point selection if stable_grassland_points was generated as documented"},
        {"filter": "cropland/mixed", "status": "excluded by class-10 primary selection"},
    ]))

    write_csv("results/qc/burned_area_exclusion_summary.csv", pd.DataFrame([{
        "filter": "MCD64A1 burned area",
        "status": "burned column excluded during metric preprocessing where present",
    }]))

    write_csv("results/qc/irrigation_exclusion_summary.csv", pd.DataFrame([{
        "filter": "irrigation",
        "status": "requires external irrigation mask; if no irrigation mask is available, report as limitation",
    }]))

    write_csv("results/qc/modis_qc_summary.csv", pd.DataFrame([
        {"product": "MODIS GPP", "rule": "QA bits 0 and 1", "status": "must be verified from Earth Engine extraction script"},
        {"product": "MODIS ET", "rule": "QA bits 0 and 1", "status": "must be verified from Earth Engine extraction script"},
        {"product": "ET floor", "rule": "ET >= 0.1 mm per 8-day", "status": "applied during log/WUE preprocessing"},
    ]))

    write_csv("results/growing_season/growing_season_summary.csv", pd.DataFrame([
        {"definition": "gpp_threshold", "required": True},
        {"definition": "climate_threshold", "required": True},
        {"definition": "month_fixed_effects", "required": True},
    ]))

    write_csv("results/stress/stress_metric_summary.csv", pd.DataFrame([
        {"metric": "zscore", "required": True},
        {"metric": "percentile", "required": True},
        {"metric": "copula", "required": True},
        {"metric": "interaction", "required": True},
    ]))

    write_csv("results/full_matrix/algorithm_dependency_table.csv", pd.DataFrame([
        {"product": "MODIS GPP MOD17A2H", "type": "GPP", "uses_vpd": "yes", "uses_temperature": "yes", "uses_radiation": "yes", "uses_soil_moisture": "no direct root-zone SM", "uses_lai": "yes/fPAR", "uses_fapar": "yes"},
        {"product": "GOSIF GPP", "type": "GPP", "uses_vpd": "no direct model input", "uses_temperature": "no direct model input", "uses_radiation": "implicit via SIF/photosynthesis", "uses_soil_moisture": "no direct model input", "uses_lai": "not same as MODIS LUE", "uses_fapar": "not primary"},
        {"product": "PML GPP", "type": "GPP", "uses_vpd": "yes", "uses_temperature": "yes", "uses_radiation": "yes", "uses_soil_moisture": "model dependent", "uses_lai": "yes", "uses_fapar": "possibly"},
        {"product": "MODIS ET MOD16A2", "type": "ET", "uses_vpd": "yes", "uses_temperature": "yes", "uses_radiation": "yes", "uses_soil_moisture": "not direct root-zone SM", "uses_lai": "yes", "uses_fapar": "yes"},
        {"product": "GLEAM ET", "type": "ET", "uses_vpd": "meteorological forcing/stress formulation", "uses_temperature": "yes", "uses_radiation": "yes", "uses_soil_moisture": "yes", "uses_lai": "limited/product dependent", "uses_fapar": "not primary"},
        {"product": "PML ET", "type": "ET", "uses_vpd": "yes", "uses_temperature": "yes", "uses_radiation": "yes", "uses_soil_moisture": "model dependent", "uses_lai": "yes", "uses_fapar": "possibly"},
    ]))


def run_product_agents_and_merge():
    print("\nSAMPLING GOSIF/GLEAM AND MERGING FINAL MATRICES")

    if not Path("data/raw/agents/gosif_point_timeseries.csv").exists():
        run(
            f"python scripts/gosif_point_agent.py "
            f"--points data/raw/gee/stable_grassland_points.csv "
            f"--start-year {START_YEAR} "
            f"--end-year {END_YEAR} "
            f"--local-glob 'data/raw/gosif/*.tif*' "
            f"--out data/raw/agents/gosif_point_timeseries.csv"
        )
    else:
        print("PASS existing GOSIF point series")

    if not Path("data/raw/agents/gleam_point_timeseries.csv").exists():
        run(
            f"python scripts/gleam_point_agent.py "
            f"--points data/raw/gee/stable_grassland_points.csv "
            f"--start-year {START_YEAR} "
            f"--end-year {END_YEAR} "
            f"--local-glob 'data/raw/gleam/E_20*_GLEAM_v4.3a.nc' "
            f"--var E "
            f"--out data/raw/agents/gleam_point_timeseries.csv"
        )
    else:
        print("PASS existing GLEAM point series")

    run(
        "python scripts/merge_full_matrix_from_point_files.py "
        "--gee-glob 'data/raw/gee/wue_timeseries_*.csv' "
        "--gosif data/raw/agents/gosif_point_timeseries.csv "
        "--gleam data/raw/agents/gleam_point_timeseries.csv "
        "--co2 data/external/noaa_co2_8day.csv "
        "--aridity data/external/aridity_by_point.csv "
        "--out-raw data/raw/agents/merged_full_matrix_raw.csv "
        "--out-co2 data/raw/agents/merged_full_matrix_co2corrected.csv"
    )


def response_metric_and_decomposition_outputs():
    print("\nCREATING RESPONSE METRIC AND DECOMPOSITION OUTPUTS")

    df = pd.read_csv("data/raw/agents/merged_full_matrix_raw.csv")
    pairs = [
        ("gpp_modis", "et_modis"),
        ("gpp_modis", "et_gleam"),
        ("gpp_modis", "et_pml"),
        ("gpp_gosif", "et_modis"),
        ("gpp_gosif", "et_gleam"),
        ("gpp_gosif", "et_pml"),
        ("gpp_pml", "et_modis"),
        ("gpp_pml", "et_gleam"),
        ("gpp_pml", "et_pml"),
    ]

    qc = []
    dec = []

    for gpp, et in pairs:
        d = df[["point_id", "date", gpp, et]].copy()
        d[gpp] = pd.to_numeric(d[gpp], errors="coerce")
        d[et] = pd.to_numeric(d[et], errors="coerce")
        before = len(d)
        d = d[(d[gpp] > 1e-6) & (d[et] >= 0.1)].copy()
        d["log_gpp"] = np.log(d[gpp])
        d["log_et"] = np.log(d[et])
        d["log_wue"] = np.log(d[gpp] / d[et])
        err = d["log_wue"] - (d["log_gpp"] - d["log_et"])

        qc.append({
            "gpp": gpp,
            "et": et,
            "n_before": before,
            "n_after": len(d),
            "dropped": before - len(d),
            "gpp_floor": 1e-6,
            "et_floor_mm_8day": 0.1,
            "max_abs_log_identity_error": float(np.nanmax(np.abs(err))) if len(d) else np.nan,
        })

        dec.append({
            "gpp": gpp,
            "et": et,
            "n": len(d),
            "sd_log_wue": d["log_wue"].std(),
            "sd_log_gpp": d["log_gpp"].std(),
            "sd_log_et": d["log_et"].std(),
            "corr_logwue_loggpp": d["log_wue"].corr(d["log_gpp"]),
            "corr_logwue_loget": d["log_wue"].corr(d["log_et"]),
        })

    write_csv("results/qc/log_transform_thresholds.csv", pd.DataFrame(qc))
    write_csv("results/decomposition/wue_decomposition_summary.csv", pd.DataFrame(dec))


def run_full_matrix_models():
    print("\nRUNNING FINAL RAW AND CO2-CORRECTED 3×3 MATRIX")

    run(
        "find results/tables -name 'point_*' -delete || true && "
        "wue points run-all "
        "--input-glob 'data/raw/agents/merged_full_matrix_raw.csv' "
        "--gpp-products MODIS,GOSIF,PML "
        "--et-products MODIS,GLEAM,PML "
        "--min-obs 50 "
        "--n-boot 1000 && "
        "mkdir -p results/full_matrix/raw && "
        "cp results/tables/point_gate2_* results/full_matrix/raw/"
    )

    run(
        "find results/tables -name 'point_*' -delete || true && "
        "wue points run-all "
        "--input-glob 'data/raw/agents/merged_full_matrix_co2corrected.csv' "
        "--gpp-products MODIS,GOSIF,PML "
        "--et-products MODIS,GLEAM,PML "
        "--min-obs 50 "
        "--n-boot 1000 && "
        "mkdir -p results/full_matrix/co2corrected && "
        "cp results/tables/point_gate2_* results/full_matrix/co2corrected/"
    )


def run_bayesian_aridity_towers_traits():
    print("\nRUNNING BAYESIAN, ARIDITY, TOWERS, TRAITS")

    run(
        "python -m wue_pipeline.workflows.bayesian_validate "
        "--fits results/full_matrix/raw/point_gate2_pixel_results.csv "
        "--timeseries data/raw/agents/merged_full_matrix_raw.csv "
        "--out results/bayesian/bayesian_threshold_agreement_raw.csv"
    )

    run(
        "python -m wue_pipeline.workflows.bayesian_validate "
        "--fits results/full_matrix/co2corrected/point_gate2_pixel_results.csv "
        "--timeseries data/raw/agents/merged_full_matrix_co2corrected.csv "
        "--out results/bayesian/bayesian_threshold_agreement_co2corrected.csv"
    )

    run(
        "python -m wue_pipeline.workflows.aridity_report "
        "--fits results/full_matrix/raw/point_gate2_pixel_results.csv "
        "--aridity data/external/aridity_by_point.csv "
        "--out results/aridity/aridity_quartile_summary_raw.csv"
    )

    run(
        "python -m wue_pipeline.workflows.aridity_report "
        "--fits results/full_matrix/co2corrected/point_gate2_pixel_results.csv "
        "--aridity data/external/aridity_by_point.csv "
        "--out results/aridity/aridity_quartile_summary_co2corrected.csv"
    )

    run(
        "python -m wue_pipeline.validation.prepare_towers "
        "--fluxnet data/raw/towers/fluxnet2015_grassland_sites.csv "
        "--ameriflux data/raw/towers/ameriflux_grassland_sites.csv "
        "--icos data/raw/towers/icos_grassland_sites.csv "
        "--ozflux data/raw/towers/ozflux_grassland_sites.csv "
        "--out data/processed/tower_validation_ready.csv"
    )

    run(
        "python -m wue_pipeline.validation.run_tower_response "
        "--tower data/processed/tower_validation_ready.csv "
        "--out results/tower_validation/tower_response_classes.csv"
    )

    run(
        "python -m wue_pipeline.validation.compare_satellite_tower "
        "--tower results/tower_validation/tower_response_classes.csv "
        "--satellite results/full_matrix/raw/point_gate2_pixel_results.csv "
        "--out results/tower_validation/satellite_tower_concordance.csv"
    )

    run(
        "python -m wue_pipeline.traits.prepare_trait_table "
        "--response results/tower_validation/satellite_tower_concordance.csv "
        "--psi50 data/external/liu_2021_psi50_0p1deg.nc "
        "--isohydricity data/external/konings_gentine_isohydricity_0p1deg.nc "
        "--rooting-depth data/external/stocker_2023_rooting_depth_0p1deg.nc "
        "--aridity data/external/aridity_by_point.csv "
        "--matrix data/raw/agents/merged_full_matrix_raw.csv "
        "--out data/processed/trait_model_table.csv"
    )

    run(
        "python -m wue_pipeline.traits.run_random_forest_shap "
        "--input data/processed/trait_model_table.csv "
        "--out results/traits/random_forest_shap_results.csv"
    )

    run(
        "python -m wue_pipeline.traits.run_bayesian_hierarchical "
        "--input data/processed/trait_model_table.csv "
        "--out results/traits/bayesian_hierarchical_trait_results.csv"
    )

    run(
        "python -m wue_pipeline.traits.cross_validate_traits "
        "--input data/processed/trait_model_table.csv "
        "--out results/traits/trait_cross_validation.csv"
    )


def final_interpretation_outputs():
    print("\nCREATING FINAL MEMOS, FIGURES, MANUSCRIPT")

    raw = pd.read_csv("results/full_matrix/raw/point_gate2_pixel_results.csv")
    valid = raw[~raw["response_class"].astype(str).isin(["insufficient_data", "invalid_x_range", "inconclusive"])]

    write(
        "results/final_memos/gate1_success_status.md",
        "# Gate 1 Success Status\n\n"
        f"Total fits: {len(raw)}\n\n"
        f"Defensible non-inconclusive fits: {len(valid)}\n\n"
        f"Success: {len(valid) > 0}\n\n"
        "Response class counts:\n\n"
        f"```\n{raw['response_class'].value_counts(dropna=False).to_string()}\n```\n",
    )

    rob = pd.read_csv("results/full_matrix/raw/point_gate2_robustness_matrix.csv")
    write(
        "results/final_memos/gate2_success_status.md",
        "# Gate 2 Success Status\n\n"
        f"Robustness rows: {len(rob)}\n\n"
        "Interpret consistency across ≥2 GPP products, ≥2 ET products, ≥2 stress definitions, and ≥2 growing-season definitions.\n\n"
        f"{rob.head(50).to_markdown(index=False)}\n",
    )

    if Path("results/tower_validation/concordance_by_product.csv").exists():
        tw = pd.read_csv("results/tower_validation/concordance_by_product.csv")
        write(
            "results/final_memos/gate3_success_status.md",
            "# Gate 3 Success Status\n\n"
            "Tower validation completed. Interpret whether towers select a product family or show irreducible uncertainty.\n\n"
            f"{tw.to_markdown(index=False)}\n",
        )

    if Path("results/traits/trait_cross_validation.csv").exists():
        cv = pd.read_csv("results/traits/trait_cross_validation.csv")
        write(
            "results/final_memos/trait_success_status.md",
            "# Trait Success Status\n\n"
            "Criterion: traits explain >20% climate-residual variance and at least one trait survives LOOCV/significance testing.\n\n"
            f"{cv.to_markdown(index=False)}\n",
        )

    run("python -m wue_pipeline.reporting.final_memos --out results/final_memos", check=False)
    run("python -m wue_pipeline.figures.final_figures --out results/final_figures", check=False)
    run("python -m wue_pipeline.reporting.final_manuscript --out results/manuscript/manuscript_skeleton.md", check=False)

    completion = {
        "raw_matrix": Path("results/full_matrix/raw/point_gate2_pixel_results.csv").exists(),
        "co2_matrix": Path("results/full_matrix/co2corrected/point_gate2_pixel_results.csv").exists(),
        "bayesian_raw": Path("results/bayesian/bayesian_threshold_agreement_raw.csv").exists(),
        "bayesian_co2": Path("results/bayesian/bayesian_threshold_agreement_co2corrected.csv").exists(),
        "aridity_raw": Path("results/aridity/aridity_quartile_summary_raw.csv").exists(),
        "aridity_co2": Path("results/aridity/aridity_quartile_summary_co2corrected.csv").exists(),
        "tower_validation": Path("results/tower_validation/satellite_tower_concordance.csv").exists(),
        "traits": Path("results/traits/trait_cross_validation.csv").exists(),
        "manuscript": Path("results/manuscript/manuscript_skeleton.md").exists(),
        "allow_short_gee": ALLOW_SHORT_GEE,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
    }
    write("results/final_memos/completion_manifest.json", json.dumps(completion, indent=2))


def final_required_output_check():
    print("\nFINAL OUTPUT CHECK")

    required = [
        "docs/preregistration.md",
        "data/provenance/data_provenance.csv",
        "results/qc/log_transform_thresholds.csv",
        "results/decomposition/wue_decomposition_summary.csv",
        "docs/vpd_processing.md",
        "docs/soil_moisture_processing.md",
        "docs/land_cover_filtering.md",
        "results/qc/modis_qc_summary.csv",
        "docs/growing_season_definitions.md",
        "docs/compound_stress_metrics.md",
        "data/raw/agents/merged_full_matrix_raw.csv",
        "data/raw/agents/merged_full_matrix_co2corrected.csv",
        "results/full_matrix/raw/point_gate2_pixel_results.csv",
        "results/full_matrix/raw/point_gate2_robustness_matrix.csv",
        "results/full_matrix/co2corrected/point_gate2_pixel_results.csv",
        "results/full_matrix/co2corrected/point_gate2_robustness_matrix.csv",
        "results/bayesian/bayesian_threshold_agreement_raw.csv",
        "results/bayesian/bayesian_threshold_agreement_co2corrected.csv",
        "results/full_matrix/algorithm_dependency_table.csv",
        "results/aridity/aridity_quartile_summary_raw.csv",
        "results/aridity/aridity_quartile_summary_co2corrected.csv",
        "results/tower_validation/tower_quality_screen.csv",
        "data/processed/tower_validation_ready.csv",
        "results/tower_validation/tower_response_classes.csv",
        "results/tower_validation/satellite_tower_concordance.csv",
        "results/tower_validation/concordance_by_product.csv",
        "data/processed/trait_model_table.csv",
        "results/traits/random_forest_shap_results.csv",
        "results/traits/bayesian_hierarchical_trait_results.csv",
        "results/traits/trait_cross_validation.csv",
        "results/final_memos/gate1_success_status.md",
        "results/final_memos/gate2_success_status.md",
        "results/final_memos/gate3_success_status.md",
        "results/final_memos/trait_success_status.md",
        "results/manuscript/manuscript_skeleton.md",
    ]

    missing = []
    for r in required:
        if Path(r).exists():
            print("PASS", r)
        else:
            print("MISSING", r)
            missing.append(r)

    if missing:
        fail("Missing final outputs:\n" + "\n".join(missing))

    print("\nFINAL COMPUTATIONAL PIPELINE COMPLETE.")


def main():
    ensure_dirs()
    verify_final_raw_inputs()
    patch_docs_and_static_outputs()
    run_product_agents_and_merge()
    response_metric_and_decomposition_outputs()
    run_full_matrix_models()
    run_bayesian_aridity_towers_traits()
    final_interpretation_outputs()
    final_required_output_check()

    run("git add scripts docs data/provenance results/final_memos requirements.txt renv.lock FULL_PAPER_REQUIRED_INPUTS.md || true", check=False)
    run("git commit -m 'Run final full paper computational pipeline' || true", check=False)
    run("git tag -f final-computational-pipeline || true", check=False)

    print("\nDONE. You can now move to analysis/writing, subject to any warnings in completion_manifest.json.")


if __name__ == "__main__":
    main()
