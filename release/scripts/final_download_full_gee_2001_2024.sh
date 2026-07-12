#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

export GEE_PROJECT="${GEE_PROJECT:-518502707617}"

DRIVE_FOLDER="grassland_wue_exports_FINAL_2001_2024"
LOCAL_TMP="data/raw/gee_FINAL_2001_2024_download"
LOCAL_FINAL="data/raw/gee"
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="logs/final_gee_export_download_${STAMP}.log"

mkdir -p logs "$LOCAL_TMP"

exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "FINAL GEE 2001-2024 EXPORT + DOWNLOAD"
echo "Started: $(date)"
echo "Project: $GEE_PROJECT"
echo "Drive folder: $DRIVE_FOLDER"
echo "Local temp: $LOCAL_TMP"
echo "============================================================"

echo ""
echo "1. Clean only the FINAL Drive folder, not your old folder."
rclone mkdir "gdrive:${DRIVE_FOLDER}" || true
rclone delete "gdrive:${DRIVE_FOLDER}" --include "*.csv" || true

echo ""
echo "2. Submit FINAL 2001-2024 Earth Engine export."
python scripts/gee_drive_export_wue.py \
  --project "$GEE_PROJECT" \
  --folder "$DRIVE_FOLDER" \
  --start-year 2001 \
  --end-year 2024 \
  --n-points 5000 \
  --n-shards 5 \
  --scale 10000 \
  --export-points \
  --export-years

echo ""
echo "3. Waiting for Earth Engine CSV exports to appear in Google Drive."
echo "Expected minimum: 120 time-series shard CSVs + 1 stable points CSV = 121 CSVs."
echo "This can take hours. Leave this running."

while true; do
  COUNT=$(rclone lsf "gdrive:${DRIVE_FOLDER}" --include "*.csv" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date): CSVs currently visible in Drive: $COUNT / 121"

  if command -v earthengine >/dev/null 2>&1; then
    echo "Recent Earth Engine task status:"
    earthengine task list | grep -E "wue_timeseries|wue_points" | head -40 || true
  fi

  if [ "$COUNT" -ge 121 ]; then
    echo "Enough CSVs visible in Drive. Proceeding to download."
    break
  fi

  sleep 600
done

echo ""
echo "4. Download final CSVs from Google Drive."
rm -rf "$LOCAL_TMP"
mkdir -p "$LOCAL_TMP"

rclone copy -P \
  "gdrive:${DRIVE_FOLDER}" \
  "$LOCAL_TMP" \
  --include "*.csv"

echo ""
echo "5. Validate downloaded final GEE dataset."
python - <<'PY'
from pathlib import Path
import pandas as pd
import sys

tmp = Path("data/raw/gee_FINAL_2001_2024_download")

points = list(tmp.glob("stable_grassland_points*.csv"))
ts = sorted(tmp.glob("wue_timeseries_*.csv"))

print("Stable point files:", len(points))
print("Time-series shard files:", len(ts))

if len(points) < 1:
    raise SystemExit("ERROR: stable_grassland_points CSV missing.")

if len(ts) < 120:
    raise SystemExit(f"ERROR: expected at least 120 time-series shard CSVs, found {len(ts)}.")

dates = []
n_rows = 0
n_points = set()

required_cols = {
    "point_id",
    "date",
    "gpp_modis",
    "et_modis",
    "gpp_pml",
    "et_pml",
    "vpd",
    "soil_moisture",
    "temperature",
    "precipitation",
    "lai",
    "burned",
}

for f in ts:
    df = pd.read_csv(f)
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: {f} missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates.extend(df["date"].dropna().tolist())
    n_rows += len(df)
    n_points.update(df["point_id"].astype(str).unique().tolist())

if not dates:
    raise SystemExit("ERROR: no valid dates found in downloaded GEE files.")

min_date = min(dates)
max_date = max(dates)

print("Rows:", n_rows)
print("Unique points:", len(n_points))
print("Date range:", min_date.date(), "to", max_date.date())

if min_date.year > 2001:
    raise SystemExit(f"ERROR: date range starts too late: {min_date.date()}")

if max_date.year < 2024:
    raise SystemExit(f"ERROR: date range ends too early: {max_date.date()}")

print("FINAL GEE VALIDATION PASS.")
PY

echo ""
echo "6. Replace old data/raw/gee only after validation passed."
BACKUP="data/raw/gee_backup_before_FINAL_2001_2024_${STAMP}"
mkdir -p data/raw
if [ -d "$LOCAL_FINAL" ]; then
  mv "$LOCAL_FINAL" "$BACKUP"
  echo "Old GEE folder backed up to: $BACKUP"
fi

mkdir -p "$LOCAL_FINAL"
cp "$LOCAL_TMP"/*.csv "$LOCAL_FINAL"/

echo ""
echo "7. Delete derived files that depended on the old 2021-2024 point set."
rm -f data/raw/agents/gosif_point_timeseries.csv
rm -f data/raw/agents/gleam_point_timeseries.csv
rm -f data/raw/agents/merged_full_matrix_raw.csv
rm -f data/raw/agents/merged_full_matrix_co2corrected.csv
rm -f data/processed/point_timeseries_prepared.csv

rm -rf results/full_matrix/raw
rm -rf results/full_matrix/co2corrected
rm -rf results/bayesian
rm -rf results/aridity
rm -rf results/tower_validation
rm -rf results/traits

mkdir -p results/full_matrix/raw results/full_matrix/co2corrected results/bayesian results/aridity results/tower_validation results/traits

echo ""
echo "8. Final local check."
echo "Stable points:"
ls -lh data/raw/gee/stable_grassland_points*.csv

echo "Time-series files:"
find data/raw/gee -maxdepth 1 -name "wue_timeseries_*.csv" | wc -l

python - <<'PY'
from pathlib import Path
import pandas as pd

files = sorted(Path("data/raw/gee").glob("wue_timeseries_*.csv"))
dates = []
rows = 0
pts = set()
for f in files:
    df = pd.read_csv(f, usecols=["point_id", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates.extend(df["date"].dropna().tolist())
    pts.update(df["point_id"].astype(str).unique())
    rows += len(df)

print("Rows:", rows)
print("Unique points:", len(pts))
print("Date range:", min(dates).date(), "to", max(dates).date())
PY

echo ""
echo "============================================================"
echo "FINAL GEE 2001-2024 DOWNLOAD COMPLETE"
echo "Finished: $(date)"
echo "Log: $LOG"
echo "============================================================"

echo ""
echo "Next command after this finishes:"
echo "export GLEAM_VAR=E"
echo "caffeinate -dimsu ./scripts/final_paper_run_all.sh"
