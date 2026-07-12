#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo

PY_SCRIPT="scripts/stage1b6j_full_direct_earthdata_extract_resumable.py"
MANIFEST="results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02r_bbox_unique_download_manifest.csv"
SHARD_ROOT="data/raw_local/no_gee_direct_point_extract_full/_shards"
QUEUE_OUT="results/stage1b6j_remaining_queue"
QUEUE_TABLE="$QUEUE_OUT/tables/Table_PRODUCT02bt_remaining_queue_status.csv"

mkdir -p "$QUEUE_OUT/tables" "$QUEUE_OUT/text" logs "$SHARD_ROOT"

if pgrep -f "stage1b6j_full_direct_earthdata_extract_resumable.py" >/dev/null 2>&1; then
  echo "ERROR: Another Stage 1B.6J extraction process is already running."
  echo "Stop it first with Control+C in that terminal, then rerun this queue."
  exit 1
fi

if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: Missing $PY_SCRIPT"
  exit 1
fi

if [ ! -f "$MANIFEST" ]; then
  echo "ERROR: Missing $MANIFEST"
  exit 1
fi

expected_count() {
  local product="$1"
  python - <<PY
import pandas as pd
m = pd.read_csv("$MANIFEST")
print(int((m["product_group"] == "$product").sum()))
PY
}

done_count() {
  local product="$1"
  find "$SHARD_ROOT/$product" -name "*.csv" 2>/dev/null | wc -l | tr -d " "
}

write_status() {
  python - <<'PY'
from pathlib import Path
import pandas as pd
from datetime import datetime

manifest = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02r_bbox_unique_download_manifest.csv")
shard_root = Path("data/raw_local/no_gee_direct_point_extract_full/_shards")
out = Path("results/stage1b6j_remaining_queue/tables/Table_PRODUCT02bt_remaining_queue_status.csv")
out.parent.mkdir(parents=True, exist_ok=True)

products = [
    "MODIS_ET_MOD16",
    "MODIS_GPP_MOD17",
    "MCD64A1_BURNED_AREA",
    "MODIS_LAI_MCD15",
]

m = pd.read_csv(manifest)
rows = []
for product in products:
    expected = int((m["product_group"] == product).sum())
    done = len(list((shard_root / product).glob("*.csv"))) if (shard_root / product).exists() else 0
    rows.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "product_group": product,
        "done_shards": done,
        "expected_shards": expected,
        "remaining_shards": max(expected - done, 0),
        "percent_done": round(100 * done / expected, 4) if expected else 0,
        "status": "DONE" if expected and done >= expected else "INCOMPLETE",
    })

df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(df.to_string(index=False))
PY
}

run_one_product_until_complete() {
  local product="$1"
  local expected
  local before
  local after
  local stamp
  local log

  expected=$(expected_count "$product")

  if [ "$expected" -eq 0 ]; then
    echo "WARNING: Expected shard count is 0 for $product. Skipping."
    return 0
  fi

  while true; do
    before=$(done_count "$product")

    echo ""
    echo "============================================================"
    echo "PRODUCT: $product"
    echo "DONE: $before / $expected"
    echo "============================================================"

    if [ "$before" -ge "$expected" ]; then
      echo "$product is already complete. Skipping."
      return 0
    fi

    stamp=$(date +"%Y%m%d_%H%M%S")
    log="logs/stage1b6j_QUEUE_${product}_${stamp}.log"

    echo "Starting/resuming $product ..."
    echo "Log: $log"

    MAX_GRANULES_PER_PRODUCT=0 \
    KEEP_HDF=0 \
    PRODUCT_FILTER="$product" \
    python -u "$PY_SCRIPT" 2>&1 | tee "$log"

    after=$(done_count "$product")

    echo ""
    echo "After run: $product $after / $expected"
    write_status

    if [ "$after" -ge "$expected" ]; then
      echo "$product complete."
      return 0
    fi

    if [ "$after" -le "$before" ]; then
      echo "ERROR: $product did not make progress."
      echo "Before: $before"
      echo "After:  $after"
      echo "Check the latest log: $log"
      exit 2
    fi

    echo "$product still incomplete but made progress. Continuing/resuming..."
  done
}

echo ""
echo "===== INITIAL QUEUE STATUS ====="
write_status

# MODIS ET is already done, so we do not include it in the remaining queue.
# Run the scientific priority product first, then QA/covariate products.
PRODUCT_QUEUE=(
  "MODIS_GPP_MOD17"
  "MCD64A1_BURNED_AREA"
  "MODIS_LAI_MCD15"
)

for product in "${PRODUCT_QUEUE[@]}"; do
  run_one_product_until_complete "$product"
done

echo ""
echo "===== FINAL REBUILD / COVERAGE LOCK ====="
stamp=$(date +"%Y%m%d_%H%M%S")
MAX_GRANULES_PER_PRODUCT=0 \
KEEP_HDF=0 \
PRODUCT_FILTER=ALL \
REBUILD_COMBINED=1 \
python -u "$PY_SCRIPT" 2>&1 | tee "logs/stage1b6j_QUEUE_FINAL_REBUILD_${stamp}.log"

echo ""
echo "===== FINAL QUEUE STATUS ====="
write_status

echo ""
echo "===== STAGE 1B.6J COMPLETION DECISION ====="
cat results/stage1b6j_full_direct_earthdata_extract/tables/Table_PRODUCT02bj_full_direct_completion_decision.csv 2>/dev/null || true

echo ""
echo "Queue complete."
