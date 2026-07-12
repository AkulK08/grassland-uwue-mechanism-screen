#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/me/Downloads/grassland_wue_nature_repo"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

export GLEAM_VAR="${GLEAM_VAR:-E}"
export FULL_START_YEAR="${FULL_START_YEAR:-2001}"
export FULL_END_YEAR="${FULL_END_YEAR:-2024}"
unset ALLOW_SHORT_GEE

CKPT=".checkpoints/final_handoff_no_modisqa"
mkdir -p "$CKPT" logs results/qc data/raw/agents

echo "============================================================"
echo "CHECKPOINTED FINAL PIPELINE FROM HANDOFF, EXCLUDING MODIS QA"
echo "Started: $(date)"
echo "Root: $ROOT"
echo "Checkpoint dir: $CKPT"
echo "============================================================"

mark_done() {
  local name="$1"
  date > "$CKPT/${name}.done"
  echo "DONE: $name"
}

is_done() {
  local name="$1"
  [ -s "$CKPT/${name}.done" ]
}

require_file() {
  local path="$1"
  if [ ! -s "$path" ]; then
    echo "MISSING REQUIRED FILE: $path"
    exit 2
  fi
}

csv_rows() {
  python - "$1" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
try:
    print(len(pd.read_csv(p)))
except Exception:
    print(0)
PY
}

count_points_in_folder() {
  python - "$1" <<'PY'
from pathlib import Path
import pandas as pd, sys
folder = Path(sys.argv[1])
pts = set()
for f in sorted(folder.glob("wue_timeseries_*.csv")):
    try:
        df = pd.read_csv(f, usecols=["point_id"])
        pts.update(df["point_id"].astype(str))
    except Exception:
        pass
print(len(pts))
PY
}

count_shards_in_folder() {
  find "$1" -maxdepth 1 -name "wue_timeseries_*.csv" | wc -l | tr -d ' '
}

echo ""
echo "STEP 0: stop old stuck local Earth Engine supplemental waiter if present"
pkill -f "download_final_supplemental_checks.sh" || true
mark_done "00_stop_old_gee_waiter"

echo ""
echo "STEP 1: verify handoff inputs"
if is_done "01_verify_handoff_inputs"; then
  echo "SKIP: handoff input verification already completed"
else
  require_file data/raw/gee/stable_grassland_points.csv
  require_file data/external/irrigation_by_point.csv
  require_file results/qc/irrigation_exclusion_summary.csv

  require_file data/processed/smap_era5_matched_points.csv
  require_file results/stress/smap_era5_comparison.csv
  require_file results/qc/smap_validation_status.json
  require_file docs/smap_validation.md

  require_file data/external/noaa_co2_8day.csv
  require_file data/external/aridity_by_point.csv

  require_file data/external/liu_2021_psi50_0p1deg.nc
  require_file data/external/konings_gentine_isohydricity_0p1deg.nc
  require_file data/external/stocker_2023_rooting_depth_0p1deg.nc

  require_file data/raw/towers/fluxnet2015_grassland_sites.csv
  require_file data/raw/towers/ameriflux_grassland_sites.csv
  require_file data/raw/towers/icos_grassland_sites.csv
  require_file data/raw/towers/ozflux_grassland_sites.csv

  GEE_SHARDS=$(count_shards_in_folder data/raw/gee)
  if [ "$GEE_SHARDS" -lt 120 ]; then
    echo "Expected at least 120 GEE shards in data/raw/gee, found $GEE_SHARDS"
    exit 2
  fi

  GOSIF_COUNT=$(find data/raw/gosif -maxdepth 1 -name "GOSIF_GPP_*_Mean.tif.gz" | wc -l | tr -d ' ')
  if [ "$GOSIF_COUNT" -lt 1000 ]; then
    echo "Expected GOSIF files, found only $GOSIF_COUNT"
    exit 2
  fi

  GLEAM_COUNT=$(find data/raw/gleam -maxdepth 1 -name "E_20*_GLEAM_v4.3a.nc" | wc -l | tr -d ' ')
  if [ "$GLEAM_COUNT" -ne 24 ]; then
    echo "Expected 24 GLEAM v4.3a files, found $GLEAM_COUNT"
    exit 2
  fi

  python - <<'PY'
import json
from pathlib import Path
import pandas as pd

irr = pd.read_csv("data/external/irrigation_by_point.csv")
smap = pd.read_csv("results/stress/smap_era5_comparison.csv")

manifest = {
    "core_gee_folder": "data/raw/gee",
    "core_gee_shards": len(list(Path("data/raw/gee").glob("wue_timeseries_*.csv"))),
    "gosif_files": len(list(Path("data/raw/gosif").glob("GOSIF_GPP_*_Mean.tif.gz"))),
    "gleam_v4_3a_files": len(list(Path("data/raw/gleam").glob("E_20*_GLEAM_v4.3a.nc"))),
    "smap_validation": {
        "matched_points_file": "data/processed/smap_era5_matched_points.csv",
        "comparison_file": "results/stress/smap_era5_comparison.csv",
        "status_file": "results/qc/smap_validation_status.json",
        "rows_in_comparison": len(smap),
    },
    "irrigation": {
        "classification_file": "data/external/irrigation_by_point.csv",
        "summary_file": "results/qc/irrigation_exclusion_summary.csv",
        "points_total": int(len(irr)),
        "points_excluded": int(irr["exclude_irrigated"].astype(bool).sum()),
        "points_kept": int((~irr["exclude_irrigated"].astype(bool)).sum()),
    },
    "modis_qa_status": "NOT_INCLUDED_IN_THIS_RUN_PENDING_REAL_MODIS_QA_OUTPUTS",
}
Path("results/qc").mkdir(parents=True, exist_ok=True)
Path("results/qc/final_handoff_input_manifest_no_modisqa.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

  mark_done "01_verify_handoff_inputs"
fi

echo ""
echo "STEP 2: create or verify 199-point no-irrigation filtered GEE folder"
if is_done "02_create_filtered_no_irrigation"; then
  echo "SKIP: filtered no-irrigation folder already checkpointed"
else
  mkdir -p data/raw/gee_final_filtered_no_irrigation

  python - <<'PY'
from pathlib import Path
import pandas as pd
import shutil

src = Path("data/raw/gee")
dst = Path("data/raw/gee_final_filtered_no_irrigation")
dst.mkdir(parents=True, exist_ok=True)

irr = pd.read_csv("data/external/irrigation_by_point.csv")
keep = set(irr.loc[~irr["exclude_irrigated"].astype(bool), "point_id"].astype(str))

pts = pd.read_csv(src / "stable_grassland_points.csv")
pts["point_id"] = pts["point_id"].astype(str)
pts = pts[pts["point_id"].isin(keep)]
pts.to_csv(dst / "stable_grassland_points.csv", index=False)

for f in sorted(src.glob("wue_timeseries_*.csv")):
    df = pd.read_csv(f)
    df["point_id"] = df["point_id"].astype(str)
    df = df[df["point_id"].isin(keep)]
    df.to_csv(dst / f.name, index=False)

for f in src.glob("*.json"):
    shutil.copy2(f, dst / f.name)

points = set()
rows = 0
for f in sorted(dst.glob("wue_timeseries_*.csv")):
    df = pd.read_csv(f, usecols=["point_id"])
    rows += len(df)
    points.update(df["point_id"].astype(str))

print("filtered_shards:", len(list(dst.glob("wue_timeseries_*.csv"))))
print("filtered_rows:", rows)
print("filtered_unique_points:", len(points))

if len(points) != 199:
    raise SystemExit(f"Expected 199 non-irrigated points, got {len(points)}")
PY

  mark_done "02_create_filtered_no_irrigation"
fi

echo ""
echo "STEP 3: activate 199-point no-irrigation GEE dataset as primary data/raw/gee"
if is_done "03_activate_filtered_gee"; then
  echo "SKIP: filtered GEE dataset already activated"
else
  ACTIVE_POINTS=$(count_points_in_folder data/raw/gee)
  FILTERED_POINTS=$(count_points_in_folder data/raw/gee_final_filtered_no_irrigation)
  FILTERED_SHARDS=$(count_shards_in_folder data/raw/gee_final_filtered_no_irrigation)

  echo "active data/raw/gee points: $ACTIVE_POINTS"
  echo "filtered data/raw/gee_final_filtered_no_irrigation points: $FILTERED_POINTS"
  echo "filtered shards: $FILTERED_SHARDS"

  if [ "$FILTERED_POINTS" != "199" ]; then
    echo "Filtered no-irrigation folder does not have 199 points. Stop."
    exit 2
  fi

  if [ "$FILTERED_SHARDS" -lt 120 ]; then
    echo "Filtered no-irrigation folder has too few shards: $FILTERED_SHARDS"
    exit 2
  fi

  if [ "$ACTIVE_POINTS" = "199" ]; then
    echo "data/raw/gee already appears to be the 199-point no-irrigation dataset."
  else
    STAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP="data/raw/gee_before_199pt_handoff_${STAMP}"
    echo "Backing up current data/raw/gee to $BACKUP"
    mv data/raw/gee "$BACKUP"
    cp -R data/raw/gee_final_filtered_no_irrigation data/raw/gee
  fi

  NEW_ACTIVE_POINTS=$(count_points_in_folder data/raw/gee)
  if [ "$NEW_ACTIVE_POINTS" != "199" ]; then
    echo "Activation failed. data/raw/gee has $NEW_ACTIVE_POINTS points, expected 199."
    exit 2
  fi

  mark_done "03_activate_filtered_gee"
fi

echo ""
echo "STEP 4: precompute GOSIF point-time series with checkpoint"
if is_done "04_gosif_agent" && [ -s data/raw/agents/gosif_point_timeseries.csv ]; then
  echo "SKIP: GOSIF point-time series already complete"
else
  if [ -s data/raw/agents/gosif_point_timeseries.csv ]; then
    ROWS=$(csv_rows data/raw/agents/gosif_point_timeseries.csv)
    if [ "$ROWS" -gt 1000 ]; then
      echo "Existing GOSIF point-time file found with $ROWS rows; marking complete."
      mark_done "04_gosif_agent"
    else
      echo "Existing GOSIF file too small; regenerating."
      rm -f data/raw/agents/gosif_point_timeseries.csv
    fi
  fi

  if ! is_done "04_gosif_agent"; then
    python scripts/gosif_point_agent.py \
      --points data/raw/gee/stable_grassland_points.csv \
      --start-year 2001 \
      --end-year 2024 \
      --local-glob "data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz" \
      --out data/raw/agents/gosif_point_timeseries.csv

    require_file data/raw/agents/gosif_point_timeseries.csv
    mark_done "04_gosif_agent"
  fi
fi

echo ""
echo "STEP 5: precompute GLEAM point-time series with checkpoint"
if is_done "05_gleam_agent" && [ -s data/raw/agents/gleam_point_timeseries.csv ]; then
  echo "SKIP: GLEAM point-time series already complete"
else
  if [ -s data/raw/agents/gleam_point_timeseries.csv ]; then
    ROWS=$(csv_rows data/raw/agents/gleam_point_timeseries.csv)
    if [ "$ROWS" -gt 1000 ]; then
      echo "Existing GLEAM point-time file found with $ROWS rows; marking complete."
      mark_done "05_gleam_agent"
    else
      echo "Existing GLEAM file too small; regenerating."
      rm -f data/raw/agents/gleam_point_timeseries.csv
    fi
  fi

  if ! is_done "05_gleam_agent"; then
    python scripts/gleam_point_agent.py \
      --points data/raw/gee/stable_grassland_points.csv \
      --start-year 2001 \
      --end-year 2024 \
      --local-glob "data/raw/gleam/E_20*_GLEAM_v4.3a.nc" \
      --var "$GLEAM_VAR" \
      --out data/raw/agents/gleam_point_timeseries.csv

    require_file data/raw/agents/gleam_point_timeseries.csv
    mark_done "05_gleam_agent"
  fi
fi

echo ""
echo "STEP 6: run final paper pipeline with checkpoint"
FINAL_REQUIRED=(
  "results/full_matrix/raw/point_gate2_pixel_results.csv"
  "results/full_matrix/raw/point_gate2_robustness_matrix.csv"
  "results/full_matrix/co2corrected/point_gate2_pixel_results.csv"
  "results/full_matrix/co2corrected/point_gate2_robustness_matrix.csv"
  "results/bayesian/bayesian_threshold_agreement_raw.csv"
  "results/aridity/aridity_quartile_summary_raw.csv"
  "results/tower_validation/tower_response_classes.csv"
  "results/tower_validation/concordance_by_product.csv"
  "results/traits/random_forest_shap_results.csv"
  "results/traits/bayesian_hierarchical_trait_results.csv"
  "results/traits/trait_cross_validation.csv"
  "results/final_memos/gate1_success_status.md"
  "results/final_memos/gate2_success_status.md"
  "results/final_memos/gate3_success_status.md"
  "results/final_memos/trait_success_status.md"
  "results/manuscript/manuscript_skeleton.md"
)

ALL_PRESENT=1
for f in "${FINAL_REQUIRED[@]}"; do
  if [ ! -s "$f" ]; then
    ALL_PRESENT=0
  fi
done

if is_done "06_final_paper_run" && [ "$ALL_PRESENT" = "1" ]; then
  echo "SKIP: final paper outputs already present and checkpointed"
else
  echo "Running final_paper_run_all.sh"
  echo "This step may take a long time. Previous GOSIF/GLEAM/filtered-data steps are checkpointed."
  ./scripts/final_paper_run_all.sh

  MISSING=0
  for f in "${FINAL_REQUIRED[@]}"; do
    if [ ! -s "$f" ]; then
      echo "MISSING FINAL OUTPUT: $f"
      MISSING=1
    else
      echo "OK: $f"
    fi
  done

  if [ "$MISSING" = "1" ]; then
    echo "Final pipeline ran but some required non-MODIS-QA outputs are missing."
    echo "Checkpoint 06_final_paper_run was NOT written. Fix the missing outputs and rerun this script."
    exit 2
  fi

  mark_done "06_final_paper_run"
fi

echo ""
echo "STEP 7: write final no-MODIS-QA completion manifest"
python - <<'PY'
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

def exists(p):
    return Path(p).exists() and Path(p).stat().st_size > 0

final_outputs = [
    "results/full_matrix/raw/point_gate2_pixel_results.csv",
    "results/full_matrix/raw/point_gate2_robustness_matrix.csv",
    "results/full_matrix/co2corrected/point_gate2_pixel_results.csv",
    "results/full_matrix/co2corrected/point_gate2_robustness_matrix.csv",
    "results/bayesian/bayesian_threshold_agreement_raw.csv",
    "results/aridity/aridity_quartile_summary_raw.csv",
    "results/tower_validation/tower_response_classes.csv",
    "results/tower_validation/concordance_by_product.csv",
    "results/traits/random_forest_shap_results.csv",
    "results/traits/bayesian_hierarchical_trait_results.csv",
    "results/traits/trait_cross_validation.csv",
    "results/final_memos/gate1_success_status.md",
    "results/final_memos/gate2_success_status.md",
    "results/final_memos/gate3_success_status.md",
    "results/final_memos/trait_success_status.md",
    "results/manuscript/manuscript_skeleton.md",
]

irr = pd.read_csv("data/external/irrigation_by_point.csv")
smap = pd.read_csv("results/stress/smap_era5_comparison.csv")

manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "status": "complete_except_modis_qa",
    "primary_point_set": "199 non-irrigated stable grassland points",
    "excluded_from_completion_claim": [
        "MODIS GPP/ET QA-bit proof/classification"
    ],
    "included_completed_inputs": {
        "core_gee": "data/raw/gee now activated as 199-point no-irrigation copy",
        "gosif": "data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz",
        "gleam": "data/raw/gleam/E_2001_GLEAM_v4.3a.nc through E_2024_GLEAM_v4.3a.nc",
        "co2": "data/external/noaa_co2_8day.csv",
        "aridity": "data/external/aridity_by_point.csv",
        "smap_validation": "results/stress/smap_era5_comparison.csv",
        "irrigation_classification": "data/external/irrigation_by_point.csv",
        "trait_maps": [
            "data/external/liu_2021_psi50_0p1deg.nc",
            "data/external/konings_gentine_isohydricity_0p1deg.nc",
            "data/external/stocker_2023_rooting_depth_0p1deg.nc"
        ],
        "towers": [
            "data/raw/towers/fluxnet2015_grassland_sites.csv",
            "data/raw/towers/ameriflux_grassland_sites.csv",
            "data/raw/towers/icos_grassland_sites.csv",
            "data/raw/towers/ozflux_grassland_sites.csv"
        ]
    },
    "irrigation_summary": {
        "points_before": int(len(irr)),
        "points_excluded": int(irr["exclude_irrigated"].astype(bool).sum()),
        "points_after": int((~irr["exclude_irrigated"].astype(bool)).sum()),
    },
    "smap_summary_rows": smap.to_dict(orient="records"),
    "final_outputs": {p: exists(p) for p in final_outputs},
    "modis_qa_outputs": {
        "data/processed/modis_qa_by_point_8day.csv": exists("data/processed/modis_qa_by_point_8day.csv"),
        "results/qc/modis_qc_summary.csv": exists("results/qc/modis_qc_summary.csv"),
        "docs/modis_qa.md": exists("docs/modis_qa.md"),
    },
}
Path("results/qc").mkdir(parents=True, exist_ok=True)
Path("results/qc/final_checkpointed_no_modisqa_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

mark_done "07_manifest"

echo ""
echo "============================================================"
echo "CHECKPOINTED FINAL PIPELINE COMPLETE EXCEPT MODIS QA"
echo "Finished: $(date)"
echo "Manifest: results/qc/final_checkpointed_no_modisqa_manifest.json"
echo "============================================================"
