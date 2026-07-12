#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
mkdir -p logs results/reza_final_nature_boot50 results/reza_final_nature_boot200 data/processed/final_nature

echo "============================================================"
echo "REZA FINAL NATURE SEQUENTIAL BOOT50 THEN BOOT200"
echo "Started: $(date)"
echo "Using Python: /Users/me/miniconda3/envs/grassland_wue_nature/bin/python"
echo "============================================================"

echo "STEP 0 rebuild current metric matrices"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_build_metric_matrix.py

echo "STEP 1 patch/stage repaired 3x3 matrices"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_stage_repair_3x3.py

echo "============================================================"
echo "BOOT50 RUN START"
echo "============================================================"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_analysis_BOOT50.py   --metrics uwue,iwue,raw_wue   --stress-defs zscore,percentile_joint,copula_joint,interaction_surface   --growing-seasons gpp_threshold,climate_common,month_fixed   --n-boot 50   --n-perm 49   --min-obs 50   --max-points 0

"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_postprocess_BOOT50.py
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_validate_generic.py --outdir results/reza_final_nature_boot50 --label BOOT50

echo "============================================================"
echo "BOOT50 PASSED. BOOT200 RUN START"
echo "============================================================"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_analysis_BOOT200.py   --metrics uwue,iwue,raw_wue   --stress-defs zscore,percentile_joint,copula_joint,interaction_surface   --growing-seasons gpp_threshold,climate_common,month_fixed   --n-boot 200   --n-perm 99   --min-obs 50   --max-points 0

"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_postprocess_BOOT200.py
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_validate_generic.py --outdir results/reza_final_nature_boot200 --label BOOT200

echo "============================================================"
echo "REZA FINAL NATURE SEQUENTIAL BOOT50 THEN BOOT200 FINISHED"
echo "Finished: $(date)"
echo "============================================================"
