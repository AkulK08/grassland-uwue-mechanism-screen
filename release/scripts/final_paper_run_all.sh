#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

mkdir -p logs
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="logs/final_paper_run_all_${STAMP}.log"

export GLEAM_VAR="${GLEAM_VAR:-E}"
export FULL_START_YEAR="${FULL_START_YEAR:-2001}"
export FULL_END_YEAR="${FULL_END_YEAR:-2024}"

exec > >(tee -a "$LOG") 2>&1

echo "FINAL PAPER RUN STARTED: $(date)"
echo "LOG: $LOG"
echo "FULL_START_YEAR=$FULL_START_YEAR"
echo "FULL_END_YEAR=$FULL_END_YEAR"
echo "ALLOW_SHORT_GEE=${ALLOW_SHORT_GEE:-0}"

python scripts/final_paper_run_all.py

echo "FINAL PAPER RUN FINISHED: $(date)"
