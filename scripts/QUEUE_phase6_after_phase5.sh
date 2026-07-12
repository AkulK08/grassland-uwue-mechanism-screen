#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

mkdir -p logs results/trait_framework/phase6 results/trait_framework/phase7

echo "===== QUEUED PHASE 6/7 ====="
echo "Waiting for Phase 5 to finish..."

while pgrep -f "python.*scripts/phase5_causal_trait_models.py" >/dev/null; do
  echo "$(date): Phase 5 still running..."
  sleep 60
done

echo ""
echo "===== PHASE 5 ENDED ====="
echo "$(date)"

if [ ! -f results/trait_framework/phase5/phase5_model_manifest.json ]; then
  echo "ERROR: Phase 5 ended, but phase5_model_manifest.json is missing."
  echo "Not starting Phase 6/7."
  exit 1
fi

echo ""
echo "===== FAST-PATCHING PHASE 6/7 ====="
python - <<'PY'
from pathlib import Path

p = Path("scripts/phase6_robustness_phase7_interpretation.py")
s = p.read_text()

backup = Path("scripts/phase6_robustness_phase7_interpretation_BEFORE_FAST_QUEUE_PATCH.py")
if not backup.exists():
    backup.write_text(s)
    print("WROTE backup:", backup)

s = s.replace("MAX_CV = 5", "MAX_CV = 3")
s = s.replace("n_estimators=300", "n_estimators=75")

p.write_text(s)
print("PATCHED Phase 6/7: MAX_CV=3 and n_estimators=75")
PY

echo ""
echo "===== STARTING PHASE 6/7 ====="

LOG="logs/phase6_phase7_AFTER_PHASE5_FAST_$(date +%Y%m%d_%H%M%S).log"
python -u scripts/phase6_robustness_phase7_interpretation.py 2>&1 | tee "$LOG"

echo ""
echo "LOG: $LOG"

echo ""
echo "===== PHASE 6 FILES ====="
ls -lh results/trait_framework/phase6

echo ""
echo "===== PHASE 7 FILES ====="
ls -lh results/trait_framework/phase7
