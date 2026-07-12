#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
conda activate grassland_wue_nature

mkdir -p results/final_figures results/final_memos results/manuscript logs

python -m wue_pipeline.reporting.memos \
  --full-matrix results/full_matrix \
  --bayesian results/bayesian \
  --aridity results/aridity \
  --tower results/tower_validation \
  --traits results/traits \
  --out results/final_memos

python -m wue_pipeline.figures.main_figures \
  --full-matrix results/full_matrix \
  --bayesian results/bayesian \
  --aridity results/aridity \
  --tower results/tower_validation \
  --traits results/traits \
  --out results/final_figures

python -m wue_pipeline.reporting.manuscript \
  --memos results/final_memos \
  --figures results/final_figures \
  --out results/manuscript/manuscript_skeleton.md
