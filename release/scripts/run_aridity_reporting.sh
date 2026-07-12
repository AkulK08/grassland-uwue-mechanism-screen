#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/aridity logs

python -m wue_pipeline.workflows.aridity_report \
  --fits results/full_matrix/full_matrix_raw/point_gate2_pixel_results.csv \
  --aridity data/external/aridity_by_point.csv \
  --out results/aridity/aridity_quartile_summary_raw.csv

python -m wue_pipeline.workflows.aridity_report \
  --fits results/full_matrix/full_matrix_co2corrected/point_gate2_pixel_results.csv \
  --aridity data/external/aridity_by_point.csv \
  --out results/aridity/aridity_quartile_summary_co2corrected.csv
