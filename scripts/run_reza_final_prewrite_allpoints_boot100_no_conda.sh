#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
mkdir -p logs results/reza_fullspec results/reza_final_prewrite

echo "============================================================"
echo "REZA FINAL PREWRITE ALL-POINTS BOOT100"
echo "Started: $(date)"
echo "Using Python: /Users/me/miniconda3/envs/grassland_wue_nature/bin/python"
echo "============================================================"

echo "STEP 1 build QA-integrated metric matrix"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_build_metric_matrix.py

echo "STEP 2 run fullspec all-point boot100"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_fullspec_analysis.py --metrics uwue,iwue,raw_wue --stress-defs zscore,percentile_joint,copula_joint,interaction_surface --growing-seasons gpp_threshold,climate_common,month_fixed --n-boot 100 --n-perm 49 --min-obs 50 --max-points 0

echo "STEP 3 postprocess"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_postprocess_fullspec.py

echo "STEP 4 freeze outputs"
cp -f results/reza_fullspec/fullspec_response_results_raw.csv results/reza_final_prewrite/fullspec_response_results_raw_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_response_results_co2corrected.csv results/reza_final_prewrite/fullspec_response_results_co2corrected_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_vpd_sm_surface_raw.csv results/reza_final_prewrite/fullspec_vpd_sm_surface_raw_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_vpd_sm_surface_co2corrected.csv results/reza_final_prewrite/fullspec_vpd_sm_surface_co2corrected_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_aridity_summary_raw.csv results/reza_final_prewrite/fullspec_aridity_summary_raw_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_aridity_summary_co2corrected.csv results/reza_final_prewrite/fullspec_aridity_summary_co2corrected_boot100_allpoints.csv
cp -f results/reza_fullspec/tower_arbiter_status.csv results/reza_final_prewrite/tower_arbiter_status_boot100_allpoints.csv
cp -f results/reza_fullspec/hierarchical_trait_proxy_results.csv results/reza_final_prewrite/hierarchical_trait_proxy_results_boot100_allpoints.csv
cp -f results/reza_fullspec/fullspec_implementation_manifest.csv results/reza_final_prewrite/fullspec_implementation_manifest_boot100_allpoints.csv

echo "============================================================"
echo "REZA FINAL PREWRITE ALL-POINTS BOOT100 FINISHED"
echo "Finished: $(date)"
echo "============================================================"
