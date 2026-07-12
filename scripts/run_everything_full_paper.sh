#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p logs
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="logs/full_paper_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "START FULL PAPER PIPELINE: $(date)"

echo "1. Existing Earth Engine MODIS/PML files assumed downloaded."
echo "2. Running GOSIF agent..."
./scripts/run_gosif_agent.sh

echo "3. Running GLEAM agent..."
./scripts/run_gleam_agent.sh

echo "4. Running CO2 + aridity layers..."
./scripts/run_co2_aridity_layers.sh

echo "5. Merging full matrix..."
./scripts/run_merge_full_matrix.sh

echo "6. Running full 3x3 raw + CO2-corrected matrix..."
./scripts/run_full_3x3_matrix.sh

echo "7. Running Bayesian threshold validation..."
./scripts/run_bayesian_validation.sh

echo "8. Running aridity reporting..."
./scripts/run_aridity_reporting.sh

echo "9. Running tower validation..."
./scripts/run_tower_validation_full.sh

echo "10. Running conditional trait phase..."
./scripts/run_trait_phase_full.sh

echo "11. Generating final figures/memos/manuscript..."
./scripts/run_final_outputs.sh

echo "12. Git commit final code/status, not raw data/results..."
git add scripts src configs docs STATUS.md FULL_PAPER_REQUIRED_INPUTS.md || true
git commit -m "Add full paper pipeline stages" || true
git push || true

echo "FULL PAPER PIPELINE COMPLETE: $(date)"
echo "Log: $LOG"
