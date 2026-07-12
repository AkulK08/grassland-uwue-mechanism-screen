#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

export GEE_PROJECT="${GEE_PROJECT:-518502707617}"

FOLDER="grassland_wue_supplemental_FINAL"
LOCAL="data/raw/gee_supplemental"
LOG="logs/final_supplemental_checks_$(date +%Y%m%d_%H%M%S).log"
EXPECTED=35

mkdir -p "$LOCAL" logs

exec > >(tee -a "$LOG") 2>&1

echo "FINAL SUPPLEMENTAL CHECKS STARTED: $(date)"
echo "Project: $GEE_PROJECT"
echo "Drive folder: $FOLDER"
echo "Local folder: $LOCAL"
echo "Expected CSVs: $EXPECTED"
echo "  1 irrigation file"
echo "  10 SMAP yearly files, 2015-2024"
echo "  24 MODIS QA yearly files, 2001-2024"

echo ""
echo "Cleaning old supplemental CSVs from Drive folder."
rclone mkdir "gdrive:${FOLDER}" || true
rclone delete "gdrive:${FOLDER}" --include "*.csv" || true

echo ""
echo "Submitting Earth Engine supplemental yearly exports."
python scripts/export_final_supplemental_checks_gee.py \
  --project "$GEE_PROJECT" \
  --points-csv data/raw/gee/stable_grassland_points.csv \
  --folder "$FOLDER" \
  --start-year 2001 \
  --end-year 2024 \
  --scale 10000 \
  ${IRRIGATION_ASSET:+--irrigation-asset "$IRRIGATION_ASSET"}

echo ""
echo "Waiting for supplemental CSVs."
while true; do
  COUNT=$(rclone lsf "gdrive:${FOLDER}" --include "*.csv" 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date): supplemental CSVs visible: $COUNT / $EXPECTED"
  earthengine task list | grep -E "final_smap_l4|final_modis_qa|final_irrigation" | head -35 || true

  if [ "$COUNT" -ge "$EXPECTED" ]; then
    break
  fi

  sleep 300
done

echo ""
echo "Downloading supplemental CSVs."
rm -rf "$LOCAL"
mkdir -p "$LOCAL"
rclone copy -P "gdrive:${FOLDER}" "$LOCAL" --include "*.csv"

echo ""
echo "Downloaded:"
ls -lh "$LOCAL"

echo ""
echo "Ingesting supplemental checks."
python scripts/ingest_final_supplemental_checks.py

echo ""
echo "FINAL SUPPLEMENTAL CHECKS COMPLETE: $(date)"
echo "Log: $LOG"
