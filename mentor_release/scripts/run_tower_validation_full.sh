#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/tower_validation logs

python -m wue_pipeline.validation.prepare_towers \
  --fluxnet data/raw/towers/fluxnet2015_grassland_sites.csv \
  --ameriflux data/raw/towers/ameriflux_grassland_sites.csv \
  --icos data/raw/towers/icos_grassland_sites.csv \
  --ozflux data/raw/towers/ozflux_grassland_sites.csv \
  --out data/processed/tower_validation_ready.csv

python -m wue_pipeline.validation.run_tower_response \
  --tower data/processed/tower_validation_ready.csv \
  --out results/tower_validation/tower_response_classes.csv

python -m wue_pipeline.validation.compare_satellite_tower \
  --tower results/tower_validation/tower_response_classes.csv \
  --satellite results/full_matrix/full_matrix_raw/point_gate2_pixel_results.csv \
  --out results/tower_validation/satellite_tower_concordance_raw.csv

python -m wue_pipeline.validation.compare_satellite_tower \
  --tower results/tower_validation/tower_response_classes.csv \
  --satellite results/full_matrix/full_matrix_co2corrected/point_gate2_pixel_results.csv \
  --out results/tower_validation/satellite_tower_concordance_co2corrected.csv
