#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p data/raw/agents data/tmp/gosif logs

echo "TODO: implement scripts/gosif_point_agent.py"
echo "Expected output:"
echo "  data/raw/agents/gosif_point_timeseries.csv"
echo "Required columns:"
echo "  point_id,date,gpp_gosif"

python scripts/gosif_point_agent.py \
  --points data/raw/gee/stable_grassland_points.csv \
  --start-year 2001 \
  --end-year 2024 \
  --out data/raw/agents/gosif_point_timeseries.csv \
  --tmp data/tmp/gosif
