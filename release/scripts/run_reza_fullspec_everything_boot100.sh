#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source $(conda info --base)/etc/profile.d/conda.sh
conda activate grassland_wue_nature

mkdir -p logs results/reza_fullspec data/processed data/external docs

echo ============================================================
echo REZA FULLSPEC EVERYTHING BOOT100
echo Started $(date)
echo ============================================================

echo STEP 1 build QA integrated metric matrix
python scripts/reza_build_metric_matrix.py

echo STEP 2 run fullspec all metrics all stress definitions all growing seasons
python scripts/reza_fullspec_analysis.py \
  --metrics uwue,iwue,raw_wue \
  --stress-defs zscore,percentile_joint,copula_joint,interaction_surface \
  --growing-seasons gpp_threshold,climate_common,month_fixed \
  --n-boot 100 \
  --n-perm 49 \
  --min-obs 50

echo STEP 3 postprocess algorithm table DAG aridity tower trait manifest
python scripts/reza_postprocess_fullspec.py

echo STEP 4 final summary
python - <<'PY'
import pandas as pd
from pathlib import Path

print()
print("===== MANIFEST =====")
m = pd.read_csv("results/reza_fullspec/fullspec_implementation_manifest.csv")
print(m.to_string(index=False))
print("ALL_REQUIRED_FILES_EXIST_AND_NONEMPTY:", bool((m["exists"] & (m["size"] > 0)).all()))

for version in ["raw", "co2corrected"]:
    p = Path(f"results/reza_fullspec/fullspec_response_results_{version}.csv")
    print()
    print("====", version.upper(), "====")
    print("exists:", p.exists())
    if not p.exists():
        continue

    df = pd.read_csv(p)
    print("shape:", df.shape)
    print("unique_points:", df["point_id"].nunique())
    print("product_combos:", df[["gpp_product","et_product"]].drop_duplicates().shape[0])
    print("metrics:", sorted(df["metric"].unique()))
    print("stress_defs:", sorted(df["stress_definition"].unique()))
    print("growing_seasons:", sorted(df["growing_season"].unique()))
    print("accepted_transitions:", int(df["accepted_transition"].sum()), "/", len(df))

    print()
    print("response_classes")
    print(df["response_class_strict"].value_counts(dropna=False).to_string())

    print()
    print("accepted_transition_rate_by_product_combo")
    acc = (
        df.groupby(["gpp_product","et_product"])["accepted_transition"]
        .agg(["sum","count","mean"])
        .reset_index()
        .sort_values(["gpp_product","et_product"])
    )
    print(acc.to_string(index=False))

print()
print("===== TOWER STATUS =====")
print(pd.read_csv("results/reza_fullspec/tower_arbiter_status.csv").to_string(index=False))

print()
print("===== TRAIT PROXY =====")
print(pd.read_csv("results/reza_fullspec/hierarchical_trait_proxy_results.csv").to_string(index=False))
PY

echo ============================================================
echo REZA FULLSPEC EVERYTHING BOOT100 FINISHED
echo Finished $(date)
echo ============================================================
