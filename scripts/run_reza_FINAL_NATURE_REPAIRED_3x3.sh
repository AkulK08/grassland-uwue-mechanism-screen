#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
mkdir -p logs results/reza_final_nature data/processed/final_nature

N_BOOT=${N_BOOT:-200}
N_PERM=${N_PERM:-99}

echo "============================================================"
echo "REZA FINAL NATURE-LEVEL COMPUTATIONAL RUN, REPAIRED 3x3"
echo "Started: $(date)"
echo "Using Python: /Users/me/miniconda3/envs/grassland_wue_nature/bin/python"
echo "N_BOOT: $N_BOOT"
echo "N_PERM: $N_PERM"
echo "============================================================"

echo "STEP 1: stage and repair independent 3x3 metric matrices"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_stage_repair_3x3.py

echo "STEP 2: run all-point 3x3 fullspec analysis"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_analysis_3x3.py   --metrics uwue,iwue,raw_wue   --stress-defs zscore,percentile_joint,copula_joint,interaction_surface   --growing-seasons gpp_threshold,climate_common,month_fixed   --n-boot "$N_BOOT"   --n-perm "$N_PERM"   --min-obs 50   --max-points 0

echo "STEP 3: postprocess final outputs"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_postprocess_3x3.py

echo "STEP 4: validate, summarize, and fail if incomplete"
"/Users/me/miniconda3/envs/grassland_wue_nature/bin/python" scripts/reza_final_nature_validate_and_summarize.py

echo "============================================================"
echo "REZA FINAL NATURE-LEVEL COMPUTATIONAL RUN FINISHED"
echo "Finished: $(date)"
echo "============================================================"
