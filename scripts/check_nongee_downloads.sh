#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

echo "===== MODIS QA AppEEARS ====="
ls -lh data/processed/modis_qa_by_point_8day.csv results/qc/modis_qc_summary.csv 2>/dev/null || true

echo ""
echo "===== Irrigation AppEEARS ====="
ls -lh data/external/irrigation_by_point.csv results/qc/irrigation_exclusion_summary.csv 2>/dev/null || true

echo ""
echo "===== SMAP Earthdata downloads ====="
ls -lh data/raw/smap_l4/smap_l4_8day_download_manifest.csv 2>/dev/null || true
find data/raw/smap_l4 -maxdepth 1 -type f 2>/dev/null | wc -l

echo ""
echo "===== Existing local final core data ====="
find data/raw/gee -maxdepth 1 -name "wue_timeseries_*.csv" | wc -l
find data/raw/gosif -maxdepth 1 -name "GOSIF_GPP_*_Mean.tif.gz" | wc -l
find data/raw/gleam -maxdepth 1 -name "E_20*_GLEAM_v4.3a.nc" | wc -l
