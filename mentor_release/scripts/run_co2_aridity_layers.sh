#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p data/external logs

python scripts/make_noaa_co2_8day.py \
  --start-date 2001-01-01 \
  --end-date 2024-12-31 \
  --out data/external/noaa_co2_8day.csv

python scripts/make_aridity_by_point.py \
  --points data/raw/gee/stable_grassland_points.csv \
  --aridity-raster data/external/cgiar_aridity_index_0p1deg.nc \
  --out data/external/aridity_by_point.csv
