#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

mkdir -p logs results/reza_fullspec data/processed data/external docs

echo "============================================================"
echo "REZA FULLSPEC SMOKE TEST"
echo "Started: $(date)"
echo "============================================================"

echo "STEP 1: source files"
ls -lh \
  data/raw/agents/merged_full_matrix_raw.csv \
  data/raw/agents/merged_full_matrix_co2corrected.csv \
  data/processed/modis_qa_by_point_8day.csv \
  scripts/reza_build_metric_matrix.py \
  scripts/reza_fullspec_analysis.py \
  scripts/reza_postprocess_fullspec.py

echo "STEP 2: optional SoilGrids"
if [ ! -s data/external/soilgrids_texture_by_point.csv ]; then
  python scripts/reza_fetch_soilgrids_texture.py \
    --points data/raw/gee/stable_grassland_points.csv \
    --out data/external/soilgrids_texture_by_point.csv || true
else
  echo "SoilGrids already exists"
fi

echo "STEP 3: build metric matrix with AppEEARS QA + soil texture if available"
python scripts/reza_build_metric_matrix.py

echo "STEP 4: run tiny fullspec analysis"
python scripts/reza_fullspec_analysis.py \
  --metrics uwue,iwue,raw_wue \
  --stress-defs zscore,percentile_joint,copula_joint,interaction_surface \
  --growing-seasons gpp_threshold,climate_common,month_fixed \
  --n-boot 10 \
  --n-perm 9 \
  --min-obs 50 \
  --max-points 3

echo "STEP 5: postprocess aridity / algorithm table / DAG / tower status / trait proxy"
python scripts/reza_postprocess_fullspec.py

echo "STEP 6: summarize outputs"
python - <<'PY'
from pathlib import Path
import pandas as pd

files = [
 "results/reza_fullspec/fullspec_response_results_raw.csv",
 "results/reza_fullspec/fullspec_response_results_co2corrected.csv",
 "results/reza_fullspec/fullspec_vpd_sm_surface_raw.csv",
 "results/reza_fullspec/fullspec_vpd_sm_surface_co2corrected.csv",
 "docs/algorithm_dependency_table.csv",
 "docs/trait_causal_dag.md",
 "results/reza_fullspec/tower_arbiter_status.csv",
 "results/reza_fullspec/hierarchical_trait_proxy_results.csv",
 "results/reza_fullspec/fullspec_implementation_manifest.csv",
]

for f in files:
    p = Path(f)
    print(f, "EXISTS" if p.exists() else "MISSING", p.stat().st_size if p.exists() else 0)

p = Path("results/reza_fullspec/fullspec_response_results_raw.csv")
if p.exists():
    df = pd.read_csv(p)
    print("raw shape", df.shape)
    for c in ["metric","stress_definition","growing_season","response_class_strict"]:
        print(c, df[c].value_counts(dropna=False).to_dict())

m = Path("results/reza_fullspec/fullspec_implementation_manifest.csv")
if m.exists():
    print(pd.read_csv(m).to_string(index=False))
PY

echo "============================================================"
echo "REZA FULLSPEC SMOKE TEST FINISHED"
echo "Finished: $(date)"
echo "============================================================"
