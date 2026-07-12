#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/traits logs

python -m wue_pipeline.traits.prepare_trait_table \
  --response results/tower_validation/satellite_tower_concordance_raw.csv \
  --psi50 data/external/liu_2021_psi50_0p1deg.nc \
  --isohydricity data/external/konings_gentine_isohydricity_0p1deg.nc \
  --rooting-depth data/external/stocker_2023_rooting_depth_0p1deg.nc \
  --aridity data/external/aridity_by_point.csv \
  --matrix data/raw/agents/merged_full_matrix_raw.csv \
  --out data/processed/trait_model_table.csv

python -m wue_pipeline.traits.run_random_forest_shap \
  --input data/processed/trait_model_table.csv \
  --out results/traits/random_forest_shap_results.csv

python -m wue_pipeline.traits.run_bayesian_hierarchical \
  --input data/processed/trait_model_table.csv \
  --out results/traits/bayesian_hierarchical_trait_results.csv

python -m wue_pipeline.traits.cross_validate_traits \
  --input data/processed/trait_model_table.csv \
  --out results/traits/trait_cross_validation.csv
