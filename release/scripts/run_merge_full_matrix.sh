#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p data/raw/agents logs

python scripts/merge_full_matrix_from_point_files.py \
  --gee-glob "data/raw/gee/wue_timeseries_*.csv" \
  --gosif data/raw/agents/gosif_point_timeseries.csv \
  --gleam data/raw/agents/gleam_point_timeseries.csv \
  --co2 data/external/noaa_co2_8day.csv \
  --aridity data/external/aridity_by_point.csv \
  --out-raw data/raw/agents/merged_full_matrix_raw.csv \
  --out-co2 data/raw/agents/merged_full_matrix_co2corrected.csv
