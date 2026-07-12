#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/full_matrix/full_matrix_raw
mkdir -p results/full_matrix/full_matrix_co2corrected
mkdir -p logs

echo "Running full 3x3 matrix without CO2 correction..."
find results/tables -name "point_*" -delete || true

time wue points run-all \
  --input-glob "data/raw/agents/merged_full_matrix_raw.csv" \
  --gpp-products MODIS,GOSIF,PML \
  --et-products MODIS,GLEAM,PML \
  --min-obs 50 \
  --n-boot 1000

cp results/tables/point_gate2_pixel_results.csv results/full_matrix/full_matrix_raw/
cp results/tables/point_gate2_robustness_matrix.csv results/full_matrix/full_matrix_raw/

echo "Running full 3x3 matrix with CO2 correction..."
find results/tables -name "point_*" -delete || true

time wue points run-all \
  --input-glob "data/raw/agents/merged_full_matrix_co2corrected.csv" \
  --gpp-products MODIS,GOSIF,PML \
  --et-products MODIS,GLEAM,PML \
  --min-obs 50 \
  --n-boot 1000

cp results/tables/point_gate2_pixel_results.csv results/full_matrix/full_matrix_co2corrected/
cp results/tables/point_gate2_robustness_matrix.csv results/full_matrix/full_matrix_co2corrected/
