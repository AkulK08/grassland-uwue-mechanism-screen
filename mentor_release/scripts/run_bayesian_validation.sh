#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/bayesian logs

python -m wue_pipeline.workflows.bayesian_validate \
  --fits results/full_matrix/full_matrix_raw/point_gate2_pixel_results.csv \
  --timeseries data/raw/agents/merged_full_matrix_raw.csv \
  --out results/bayesian/bayesian_threshold_agreement_raw.csv

python -m wue_pipeline.workflows.bayesian_validate \
  --fits results/full_matrix/full_matrix_co2corrected/point_gate2_pixel_results.csv \
  --timeseries data/raw/agents/merged_full_matrix_co2corrected.csv \
  --out results/bayesian/bayesian_threshold_agreement_co2corrected.csv
