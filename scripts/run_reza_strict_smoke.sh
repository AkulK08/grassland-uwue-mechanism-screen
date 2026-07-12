#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

mkdir -p logs results/reza_strict data/processed

echo "============================================================"
echo "REZA STRICT SMOKE TEST"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "STEP 1: verify required source files"
ls -lh \
  data/raw/agents/merged_full_matrix_raw.csv \
  data/raw/agents/merged_full_matrix_co2corrected.csv \
  data/processed/modis_qa_by_point_8day.csv \
  scripts/reza_build_metric_matrix.py \
  scripts/reza_strict_gate_analysis.py

echo ""
echo "STEP 2: build metric matrices and integrate AppEEARS MODIS QA"
python scripts/reza_build_metric_matrix.py

echo ""
echo "STEP 3: prove AppEEARS was integrated"
ls -lh \
  data/processed/modis_qa_by_point_8day_wide.csv \
  data/processed/reza_metric_matrix_raw.csv \
  data/processed/reza_metric_matrix_co2corrected.csv

python - <<'PY'
import pandas as pd

qa = pd.read_csv("data/processed/modis_qa_by_point_8day_wide.csv")
print("\nQA wide shape:", qa.shape)
print("QA wide columns:", list(qa.columns))
print("GPP QA good fraction:", qa["modis_gpp_qc_good"].mean())
print("ET QA good fraction:", qa["modis_et_qc_good"].mean())

df = pd.read_csv("data/processed/reza_metric_matrix_raw.csv", nrows=1000)
needed = [
    "modis_gpp_qc_good",
    "modis_et_qc_good",
    "log_uwue_modis_modis",
    "log_iwue_modis_modis",
    "log_raw_wue_modis_modis",
    "vpd_z",
    "sm_z",
    "vpd_x_sm",
]
print("\nMetric matrix first 1000 rows columns present:")
for c in needed:
    print(c, c in df.columns)
PY

echo ""
echo "STEP 4: run very short strict analysis smoke test, 5 points only"
python scripts/reza_strict_gate_analysis.py \
  --n-boot 10 \
  --n-perm 9 \
  --min-obs 50 \
  --max-points 5

echo ""
echo "STEP 5: summarize smoke outputs"
python - <<'PY'
from pathlib import Path
import pandas as pd

for p in [
    "results/reza_strict/strict_response_results_raw.csv",
    "results/reza_strict/strict_response_results_co2corrected.csv",
    "results/reza_strict/vpd_sm_surface_partial_effects_raw.csv",
    "results/reza_strict/vpd_sm_surface_partial_effects_co2corrected.csv",
]:
    print("\n====", p, "====")
    if not Path(p).exists():
        print("MISSING")
        continue
    df = pd.read_csv(p)
    print("shape:", df.shape)
    print(df.head())
    if "metric" in df.columns:
        print("metric counts:", df["metric"].value_counts(dropna=False).to_dict())
    if "response_class_strict" in df.columns:
        print("class counts:", df["response_class_strict"].value_counts(dropna=False).to_dict())
    if "accepted_transition" in df.columns:
        print("accepted transitions:", int(df["accepted_transition"].sum()))

print("\nDONE smoke test.")
PY

echo "============================================================"
echo "REZA STRICT SMOKE TEST FINISHED"
echo "Finished: $(date)"
echo "============================================================"
