#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p data/raw/agents data/tmp/gleam logs

if [ -z "${GLEAM_USER:-}" ] || [ -z "${GLEAM_PASS:-}" ]; then
  echo "ERROR: Set GLEAM_USER and GLEAM_PASS first."
  echo "Example:"
  echo "  export GLEAM_USER='your_username'"
  echo "  export GLEAM_PASS='your_password'"
  exit 1
fi

python scripts/gleam_point_agent.py \
  --points data/raw/gee/stable_grassland_points.csv \
  --start-year 2001 \
  --end-year 2024 \
  --out data/raw/agents/gleam_point_timeseries.csv \
  --tmp data/tmp/gleam
