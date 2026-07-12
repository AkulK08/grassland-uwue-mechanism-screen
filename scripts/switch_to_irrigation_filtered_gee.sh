#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo

if [ ! -d data/raw/gee_final_filtered_no_irrigation ]; then
  echo "Missing data/raw/gee_final_filtered_no_irrigation. Run scripts/download_final_supplemental_checks.sh first."
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="data/raw/gee_before_irrigation_filter_${STAMP}"

mv data/raw/gee "$BACKUP"
cp -R data/raw/gee_final_filtered_no_irrigation data/raw/gee

rm -f data/raw/agents/gosif_point_timeseries.csv
rm -f data/raw/agents/gleam_point_timeseries.csv
rm -f data/raw/agents/merged_full_matrix_raw.csv
rm -f data/raw/agents/merged_full_matrix_co2corrected.csv
rm -rf results/full_matrix/raw results/full_matrix/co2corrected results/bayesian results/aridity results/tower_validation results/traits
mkdir -p results/full_matrix/raw results/full_matrix/co2corrected results/bayesian results/aridity results/tower_validation results/traits

echo "Switched to irrigation-filtered GEE folder."
echo "Backup: $BACKUP"

python - <<'PY'
from pathlib import Path
import pandas as pd
files = sorted(Path("data/raw/gee").glob("wue_timeseries_*.csv"))
dates = []
pts = set()
rows = 0
for f in files:
    df = pd.read_csv(f, usecols=["point_id","date"])
    rows += len(df)
    pts.update(df["point_id"].astype(str))
    dates += pd.to_datetime(df["date"], errors="coerce").dropna().tolist()
print("files", len(files))
print("rows", rows)
print("points", len(pts))
print("date range", min(dates).date(), max(dates).date())
PY
