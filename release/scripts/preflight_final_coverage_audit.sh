#!/usr/bin/env bash
set -euo pipefail

cd /Users/me/Downloads/grassland_wue_nature_repo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

mkdir -p results/qc logs

OUT="results/qc/preflight_final_coverage_audit.txt"
: > "$OUT"

pass(){ echo "PASS: $*" | tee -a "$OUT"; }
warn(){ echo "WARNING: $*" | tee -a "$OUT"; }
fail(){ echo "FAIL: $*" | tee -a "$OUT"; FAILED=1; }
section(){ echo "" | tee -a "$OUT"; echo "============================================================" | tee -a "$OUT"; echo "$*" | tee -a "$OUT"; echo "============================================================" | tee -a "$OUT"; }

FAILED=0

section "1. Raw final data inputs"

[ -f data/raw/gee/stable_grassland_points.csv ] && pass "stable_grassland_points.csv exists" || fail "missing data/raw/gee/stable_grassland_points.csv"

GEE_COUNT=$(find data/raw/gee -maxdepth 1 -name "wue_timeseries_*.csv" | wc -l | tr -d ' ')
[ "$GEE_COUNT" -ge 120 ] && pass "GEE final shard count >=120: $GEE_COUNT" || fail "GEE shard count too low: $GEE_COUNT"

GOSIF_COUNT=$(find data/raw/gosif -maxdepth 1 -name "GOSIF_GPP_*_Mean.tif.gz" | wc -l | tr -d ' ')
[ "$GOSIF_COUNT" -ge 1000 ] && pass "GOSIF files present: $GOSIF_COUNT" || fail "GOSIF files too few: $GOSIF_COUNT"

GLEAM_COUNT=$(find data/raw/gleam -maxdepth 1 -name "E_20*_GLEAM_v4.3a.nc" | wc -l | tr -d ' ')
[ "$GLEAM_COUNT" -eq 24 ] && pass "GLEAM v4.3a daily E files present: 24" || fail "GLEAM file count should be 24, found $GLEAM_COUNT"

[ -f data/external/noaa_co2_8day.csv ] && pass "CO2 file exists" || fail "missing data/external/noaa_co2_8day.csv"
[ -f data/external/aridity_by_point.csv ] && pass "aridity_by_point.csv exists" || fail "missing data/external/aridity_by_point.csv"

[ -f data/external/liu_2021_psi50_0p1deg.nc ] && pass "psi50 trait map exists" || fail "missing psi50 trait map"
[ -f data/external/konings_gentine_isohydricity_0p1deg.nc ] && pass "isohydricity trait map exists" || fail "missing isohydricity trait map"
[ -f data/external/stocker_2023_rooting_depth_0p1deg.nc ] && pass "rooting-depth trait map exists" || fail "missing rooting-depth trait map"

[ -f data/raw/towers/fluxnet2015_grassland_sites.csv ] && pass "FLUXNET tower file exists" || fail "missing FLUXNET tower file"
[ -f data/raw/towers/ameriflux_grassland_sites.csv ] && pass "AmeriFlux tower file exists" || fail "missing AmeriFlux tower file"
[ -f data/raw/towers/icos_grassland_sites.csv ] && pass "ICOS tower file exists" || fail "missing ICOS tower file"
[ -f data/raw/towers/ozflux_grassland_sites.csv ] && pass "OzFlux tower file exists" || fail "missing OzFlux tower file"

section "2. Final date coverage and required GEE columns"

python - <<'PY' | tee -a "$OUT"
from pathlib import Path
import pandas as pd
import sys

files = sorted(Path("data/raw/gee").glob("wue_timeseries_*.csv"))
required = {
    "point_id", "date",
    "gpp_modis", "et_modis",
    "gpp_pml", "et_pml",
    "vpd", "soil_moisture",
    "temperature", "precipitation", "lai", "burned",
}

bad = []
dates = []
pts = set()
rows = 0

for f in files:
    df = pd.read_csv(f)
    miss = required - set(df.columns)
    if miss:
        bad.append((f.name, sorted(miss)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates.extend(df["date"].dropna().tolist())
    pts.update(df["point_id"].astype(str).unique())
    rows += len(df)

if bad:
    print("FAIL: missing required columns:")
    for name, miss in bad[:20]:
        print(name, miss)
    sys.exit(2)

if not dates:
    print("FAIL: no valid dates")
    sys.exit(2)

mn, mx = min(dates), max(dates)
print(f"PASS: rows={rows}")
print(f"PASS: unique_points={len(pts)}")
print(f"PASS: date_range={mn.date()} to {mx.date()}")

if mn.year > 2001 or mx.year < 2024:
    print("FAIL: date range does not cover 2001-2024")
    sys.exit(2)

print("PASS: GEE covers 2001-2024 and has required columns")
PY
if [ "${PIPESTATUS[0]}" -ne 0 ]; then FAILED=1; fi

section "3. Required generator scripts"

for f in \
  scripts/final_paper_run_all.py \
  scripts/final_paper_run_all.sh \
  scripts/export_final_supplemental_checks_gee.py \
  scripts/download_final_supplemental_checks.sh \
  scripts/ingest_final_supplemental_checks.py \
  scripts/switch_to_irrigation_filtered_gee.sh \
  scripts/gosif_point_agent.py \
  scripts/gleam_point_agent.py \
  scripts/merge_full_matrix_from_point_files.py \
  scripts/requirements_01_30.sh
do
  [ -f "$f" ] && pass "$f exists" || fail "$f missing"
done

section "4. Python syntax / import check"

for f in \
  scripts/final_paper_run_all.py \
  scripts/export_final_supplemental_checks_gee.py \
  scripts/ingest_final_supplemental_checks.py \
  scripts/gosif_point_agent.py \
  scripts/gleam_point_agent.py \
  scripts/merge_full_matrix_from_point_files.py
do
  if [ -f "$f" ]; then
    python -m py_compile "$f" && pass "$f compiles" || fail "$f does not compile"
  fi
done

section "5. CLI/module availability"

command -v wue >/dev/null 2>&1 && pass "wue CLI exists" || fail "wue CLI missing"
wue --help >/dev/null 2>&1 && pass "wue --help works" || fail "wue --help failed"
wue points --help >/dev/null 2>&1 && pass "wue points command exists" || fail "wue points command missing"

python - <<'PY' | tee -a "$OUT"
import importlib, sys

mods = [
    "wue_pipeline.workflows.bayesian_validate",
    "wue_pipeline.workflows.aridity_report",
    "wue_pipeline.validation.prepare_towers",
    "wue_pipeline.validation.run_tower_response",
    "wue_pipeline.validation.compare_satellite_tower",
    "wue_pipeline.traits.prepare_trait_table",
    "wue_pipeline.traits.run_random_forest_shap",
    "wue_pipeline.traits.run_bayesian_hierarchical",
    "wue_pipeline.traits.cross_validate_traits",
    "wue_pipeline.reporting.final_memos",
    "wue_pipeline.figures.final_figures",
    "wue_pipeline.reporting.final_manuscript",
]

bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print("PASS:", m)
    except Exception as e:
        print("FAIL:", m, repr(e))
        bad.append(m)

sys.exit(1 if bad else 0)
PY
if [ "${PIPESTATUS[0]}" -ne 0 ]; then FAILED=1; fi

section "6. Supplemental check readiness"

if [ -f results/qc/final_supplemental_manifest.json ]; then
  pass "supplemental manifest already exists"
  cat results/qc/final_supplemental_manifest.json | tee -a "$OUT"
else
  warn "supplemental manifest does not exist yet; this is okay only before running download_final_supplemental_checks.sh"
fi

if [ -d data/raw/gee_final_filtered_no_irrigation ]; then
  FILT_COUNT=$(find data/raw/gee_final_filtered_no_irrigation -maxdepth 1 -name "wue_timeseries_*.csv" | wc -l | tr -d ' ')
  [ "$FILT_COUNT" -ge 120 ] && pass "irrigation-filtered GEE folder exists with $FILT_COUNT shards" || fail "irrigation-filtered GEE folder exists but has too few shards: $FILT_COUNT"
else
  warn "irrigation-filtered GEE folder not created yet; run supplemental checks before final paper run"
fi

section "7. Bullet coverage map"

cat <<'TXT' | tee -a "$OUT"
Covered by final_paper_run_all.sh:
- WUE/log response metrics
- GPP and ET filtering
- WUE decomposition
- full 3x3 GPP x ET matrix
- raw and CO2-corrected runs
- segmented/bootstrap response shapes
- Bayesian threshold agreement
- stress-metric robustness
- growing-season robustness
- aridity quartile analysis
- tower screening, tower response, satellite/tower concordance
- trait table, RF/SHAP, Bayesian trait model, cross-validation
- final memos, figures, manuscript skeleton

Covered by download_final_supplemental_checks.sh:
- SMAP vs ERA5 soil-moisture validation
- MODIS GPP/ET QA proof
- irrigation/agriculture exclusion
- filtered final GEE dataset

Procedural only, not computationally verified here:
- OSF upload
- final human interpretation/prose
TXT

section "8. Final preflight verdict"

if [ "$FAILED" -eq 0 ]; then
  echo "PREFLIGHT PASS: the repo has the raw inputs, code, commands, and generator coverage needed to run the full final project." | tee -a "$OUT"
  echo "Next run order:" | tee -a "$OUT"
  echo "  caffeinate -dimsu ./scripts/download_final_supplemental_checks.sh" | tee -a "$OUT"
  echo "  ./scripts/switch_to_irrigation_filtered_gee.sh" | tee -a "$OUT"
  echo "  export GLEAM_VAR=E; unset ALLOW_SHORT_GEE; caffeinate -dimsu ./scripts/final_paper_run_all.sh" | tee -a "$OUT"
else
  echo "PREFLIGHT FAIL: fix the FAIL lines above before running the full final project." | tee -a "$OUT"
  exit 2
fi

echo ""
echo "Audit saved to: $OUT"
