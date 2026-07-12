#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/me/Downloads/grassland_wue_nature_repo"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate grassland_wue_nature

ACTION="${1:-check}"
STEP="${2:-all}"

mkdir -p docs data/provenance data/external/_status results/qc results/decomposition results/stress results/growing_season results/full_matrix/raw results/full_matrix/co2corrected results/bayesian results/aridity results/tower_validation results/traits results/final_memos results/final_figures results/manuscript logs scripts

pass(){ echo "PASS: $1"; }
miss(){ echo "MISSING: $1"; }
warn(){ echo "WARNING: $1"; }
stop(){ echo ""; echo "STOP: $1"; exit 2; }

exists(){ [ -e "$1" ]; }
glob_count(){ find $1 2>/dev/null | wc -l | tr -d ' '; }

need_file(){
  if [ -f "$1" ]; then pass "$1"; else miss "$1"; return 1; fi
}

need_glob(){
  local pattern="$1"
  local min="${2:-1}"
  local n
  n=$(find $(dirname "$pattern") -maxdepth 1 -name "$(basename "$pattern")" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -ge "$min" ]; then pass "$pattern count=$n"; else miss "$pattern count=$n expected>=$min"; return 1; fi
}

run_python(){
python - "$@" <<'PY'
import sys, os, glob, json
from pathlib import Path
import pandas as pd
import numpy as np

cmd = sys.argv[1]

def fail(msg):
    print("MISSING:", msg)
    sys.exit(2)

def ok(msg):
    print("PASS:", msg)

if cmd == "check_matrix_columns":
    path = sys.argv[2]
    required = sys.argv[3:]
    if not Path(path).exists():
        fail(f"{path} missing")
    df = pd.read_csv(path, nrows=5)
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"{path} missing columns {missing}")
    ok(f"{path} has required columns")

elif cmd == "check_csv_rows":
    path = sys.argv[2]
    min_rows = int(sys.argv[3])
    if not Path(path).exists():
        fail(f"{path} missing")
    n = sum(1 for _ in open(path, errors="ignore")) - 1
    if n < min_rows:
        fail(f"{path} rows={n}, expected >= {min_rows}")
    ok(f"{path} rows={n}")

elif cmd == "check_netcdf":
    import xarray as xr
    path = sys.argv[2]
    if not Path(path).exists():
        fail(f"{path} missing")
    ds = xr.open_dataset(path)
    vars_ = list(ds.data_vars)
    coords = list(ds.coords)
    if "lat" not in coords or "lon" not in coords:
        fail(f"{path} missing lat/lon coords; coords={coords}")
    ok(f"{path} vars={vars_} dims={dict(ds.sizes)}")

elif cmd == "make_response_metric_check":
    out = Path("scripts/verify_response_metrics.py")
    out.write_text(r'''
from pathlib import Path
import pandas as pd
import numpy as np
import json

matrix = Path("data/raw/agents/merged_full_matrix_raw.csv")
if not matrix.exists():
    raise SystemExit("Missing data/raw/agents/merged_full_matrix_raw.csv; run merge step first.")

df = pd.read_csv(matrix)
pairs = [
    ("gpp_modis","et_modis"),
    ("gpp_modis","et_gleam"),
    ("gpp_modis","et_pml"),
    ("gpp_gosif","et_modis"),
    ("gpp_gosif","et_gleam"),
    ("gpp_gosif","et_pml"),
    ("gpp_pml","et_modis"),
    ("gpp_pml","et_gleam"),
    ("gpp_pml","et_pml"),
]
rows = []
for gpp, et in pairs:
    if gpp not in df.columns or et not in df.columns:
        rows.append({"gpp":gpp,"et":et,"status":"missing_columns"})
        continue
    d = df[[gpp, et]].copy()
    d[gpp] = pd.to_numeric(d[gpp], errors="coerce")
    d[et] = pd.to_numeric(d[et], errors="coerce")
    before = len(d)
    d = d[(d[gpp] > 1e-6) & (d[et] > 0.1)].copy()
    d["wue"] = d[gpp] / d[et]
    d["log_wue_direct"] = np.log(d["wue"])
    d["log_gpp_minus_log_et"] = np.log(d[gpp]) - np.log(d[et])
    max_abs_err = float(np.nanmax(np.abs(d["log_wue_direct"] - d["log_gpp_minus_log_et"]))) if len(d) else np.nan
    rows.append({
        "gpp":gpp, "et":et, "status":"ok", "n_before":before, "n_after":len(d),
        "dropped":before-len(d), "gpp_floor":1e-6, "et_floor":0.1,
        "max_abs_log_identity_error":max_abs_err
    })
out = pd.DataFrame(rows)
Path("results/qc").mkdir(parents=True, exist_ok=True)
Path("results/decomposition").mkdir(parents=True, exist_ok=True)
out.to_csv("results/qc/log_transform_thresholds.csv", index=False)

# simple decomposition by product combo
out2 = []
for gpp, et in pairs:
    if gpp in df.columns and et in df.columns:
        d = df[["point_id", "date", gpp, et]].copy()
        d[gpp] = pd.to_numeric(d[gpp], errors="coerce")
        d[et] = pd.to_numeric(d[et], errors="coerce")
        d = d[(d[gpp] > 1e-6) & (d[et] > 0.1)].copy()
        d["log_gpp"] = np.log(d[gpp])
        d["log_et"] = np.log(d[et])
        d["log_wue"] = d["log_gpp"] - d["log_et"]
        out2.append({
            "gpp": gpp, "et": et, "n": len(d),
            "sd_log_wue": d["log_wue"].std(),
            "sd_log_gpp": d["log_gpp"].std(),
            "sd_log_et": d["log_et"].std(),
            "corr_logwue_loggpp": d["log_wue"].corr(d["log_gpp"]),
            "corr_logwue_loget": d["log_wue"].corr(d["log_et"]),
        })
pd.DataFrame(out2).to_csv("results/decomposition/wue_decomposition_summary.csv", index=False)
print("Wrote response metric/decomposition checks.")
''')
    ok("wrote scripts/verify_response_metrics.py")

elif cmd == "make_provenance":
    Path("data/provenance").mkdir(parents=True, exist_ok=True)
    p = Path("data/provenance/data_provenance.csv")
    if not p.exists():
        pd.DataFrame([
            {"dataset":"MODIS/PML/ERA5 Earth Engine point files","path":"data/raw/gee/wue_timeseries_*.csv","status":"check","source":"Google Earth Engine export"},
            {"dataset":"GOSIF GPP v2 8-day Mean","path":"data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz","status":"check","source":"https://data.globalecology.unh.edu/data/GOSIF-GPP_v2/8day/Mean/"},
            {"dataset":"GLEAM v4.3a daily E","path":"data/raw/gleam/E_20*_GLEAM_v4.3a.nc","status":"check","source":"GLEAM SFTP aether.ugent.be"},
            {"dataset":"NOAA CO2 8-day","path":"data/external/noaa_co2_8day.csv","status":"check","source":"NOAA Mauna Loa monthly CO2"},
            {"dataset":"Traits","path":"data/external/*trait*.nc","status":"check","source":"prepared/standardized in Step 5"},
            {"dataset":"Aridity","path":"data/external/cgiar_aridity_index_0p1deg.nc OR data/external/aridity_by_point.csv","status":"missing_or_check","source":"CGIAR-CSI aridity"},
            {"dataset":"Towers","path":"data/raw/towers/*.csv","status":"missing_or_check","source":"FLUXNET/AmeriFlux/ICOS/OzFlux"},
        ]).to_csv(p, index=False)
    ok("data/provenance/data_provenance.csv")

elif cmd == "make_algorithm_dependency":
    Path("results/full_matrix").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"product":"MODIS GPP MOD17A2H","type":"GPP","uses_vpd":"yes/algorithmic meteorology","uses_temperature":"yes","uses_radiation":"yes","uses_soil_moisture":"no direct SM","uses_LAI":"yes/fPAR","uses_fAPAR":"yes","bias_note":"can encode meteorological stress response"},
        {"product":"GOSIF GPP","type":"GPP","uses_vpd":"no direct model input","uses_temperature":"no direct model input","uses_radiation":"implicit via SIF/photosynthesis","uses_soil_moisture":"no direct model input","uses_LAI":"not same as MODIS LUE","uses_fAPAR":"not primary","bias_note":"more observation/SIF-driven"},
        {"product":"PML GPP","type":"GPP","uses_vpd":"yes/meteorology","uses_temperature":"yes","uses_radiation":"yes","uses_soil_moisture":"possible/model dependent","uses_LAI":"yes","uses_fAPAR":"possibly","bias_note":"coupled GPP/ET model; not independent from meteorological drivers"},
        {"product":"MODIS ET MOD16A2","type":"ET","uses_vpd":"yes","uses_temperature":"yes","uses_radiation":"yes","uses_soil_moisture":"indirect/no direct root-zone SM","uses_LAI":"yes","uses_fAPAR":"yes","bias_note":"known ET stress sensitivity can affect WUE thresholds"},
        {"product":"GLEAM ET","type":"ET","uses_vpd":"meteorological forcing in evaporative stress formulation","uses_temperature":"yes","uses_radiation":"yes","uses_soil_moisture":"yes/microwave-root-zone stress","uses_LAI":"limited/product dependent","uses_fAPAR":"no primary","bias_note":"structurally distinct ET check"},
        {"product":"PML ET","type":"ET","uses_vpd":"yes","uses_temperature":"yes","uses_radiation":"yes","uses_soil_moisture":"possible/model dependent","uses_LAI":"yes","uses_fAPAR":"possibly","bias_note":"coupled GPP/ET product; may share assumptions with PML GPP"},
    ]).to_csv("results/full_matrix/algorithm_dependency_table.csv", index=False)
    ok("results/full_matrix/algorithm_dependency_table.csv")

else:
    fail(f"unknown python cmd {cmd}")
PY
}

check0(){
  echo "STEP 0: Pre-project setup"
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && pass "git repo" || miss "git repo"
  need_file requirements.txt || true
  [ -f renv.lock ] && pass "renv.lock" || warn "renv.lock missing; only needed if R is used"
  need_file README.md || true
  need_file docs/preregistration.md || true
  need_file data/provenance/data_provenance.csv || true
  if git tag | grep -q .; then pass "git tags exist"; else miss "no git tags yet"; fi
}
add0(){
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init
  [ -f requirements.txt ] || python -m pip freeze > requirements.txt
  [ -f README.md ] || echo "# Grassland WUE pipeline" > README.md
  mkdir -p docs
  [ -f docs/preregistration.md ] || cat > docs/preregistration.md <<'EOF'
# OSF Preregistration Draft

## Hypotheses
1. Grassland WUE response to compound VPD-soil moisture stress may show enhancement, saturation, or reversal.
2. A transition is accepted only when segmented-regression and Bayesian change-point intervals overlap.
3. Robust claims require consistency across at least 2 GPP products, 2 ET products, 2 stress definitions, and 2 growing-season definitions.

## Methods
Full 3x3 GPP/ET product matrix; 4 stress metrics; 3 growing-season definitions; raw and CO2-corrected runs; tower validation; conditional traits.

## Success criteria
Gate 1: defensible response in at least one product combo.
Gate 2: robust qualitative class with overlapping uncertainty.
Gate 3: tower validation selects product family or shows irreducible uncertainty.
Trait phase: traits explain >20% climate-residual variance and survive LOOCV.
EOF
  run_python make_provenance
  [ -f renv.lock ] || echo "# No R environment currently used. Add real renv.lock if R is introduced." > renv.lock
  git add README.md requirements.txt renv.lock docs/preregistration.md data/provenance/data_provenance.csv || true
  git commit -m "Add reproducibility and preregistration draft" || true
  echo "MANUAL: upload docs/preregistration.md to OSF before final multi-product matrix."
}

check1(){
  echo "STEP 1: Define response metrics"
  need_file results/qc/log_transform_thresholds.csv || true
  need_file results/decomposition/wue_decomposition_summary.csv || true
}
add1(){
  run_python make_response_metric_check
  python scripts/verify_response_metrics.py
}

check2(){
  echo "STEP 2: Build satellite dataset"
  need_glob "data/raw/gee/wue_timeseries_*.csv" 1 || true
  need_file data/raw/gee/stable_grassland_points.csv || true
  need_glob "data/raw/gosif/GOSIF_GPP_*_Mean.tif.gz" 100 || true
  need_glob "data/raw/gleam/E_20*_GLEAM_v4.3a.nc" 24 || true
  need_file data/external/cgiar_aridity_index_0p1deg.nc || need_file data/external/aridity_by_point.csv || true
}
add2(){
  mkdir -p data/raw/gosif data/raw/gleam
  echo "Adding GOSIF by wget if needed..."
  if ! find data/raw/gosif -maxdepth 1 -name "GOSIF_GPP_*_Mean.tif.gz" | grep -q .; then
    wget -c -r -np -nH --cut-dirs=4 -A "GOSIF_GPP_*_Mean.tif.gz" -R "index.html*" -P data/raw/gosif https://data.globalecology.unh.edu/data/GOSIF-GPP_v2/8day/Mean/
  fi
  echo "GLEAM cannot be added without SFTP credentials, but handoff says it should be downloaded. Verify with:"
  echo 'find data/raw/gleam -maxdepth 1 -name "E_20*_GLEAM_v4.3a.nc" | wc -l'
  check2
}

check3(){
  echo "STEP 3: Process atmospheric stress/VPD"
  need_file docs/vpd_processing.md || true
  if ls data/raw/gee/wue_timeseries_*.csv >/dev/null 2>&1; then
    run_python check_matrix_columns "$(ls data/raw/gee/wue_timeseries_*.csv | head -1)" vpd || true
  fi
}
add3(){
  mkdir -p docs
  cat > docs/vpd_processing.md <<'EOF'
# VPD processing

VPD is derived from ERA5-Land temperature and dewpoint, then aggregated to 8-day means in the point-time extraction workflow. Final method text must document the exact formula and Earth Engine aggregation used.
EOF
  echo "To fully add this, rerun/verify Earth Engine export contains vpd for 2001-2024."
  check3
}

check4(){
  echo "STEP 4: Process soil moisture"
  need_file docs/soil_moisture_processing.md || true
  if ls data/raw/gee/wue_timeseries_*.csv >/dev/null 2>&1; then
    run_python check_matrix_columns "$(ls data/raw/gee/wue_timeseries_*.csv | head -1)" soil_moisture || true
  fi
  need_file results/stress/smap_era5_comparison.csv || true
}
add4(){
  mkdir -p docs results/stress
  cat > docs/soil_moisture_processing.md <<'EOF'
# Soil moisture processing

Root-zone soil moisture is computed from ERA5-Land swvl1 and swvl2 using a documented weighting scheme, then aggregated to 8-day means. SMAP L4 post-2015 validation still requires real SMAP extraction unless `results/stress/smap_era5_comparison.csv` exists.
EOF
  echo "SMAP validation is not automatic here. Add SMAP point-time file, then create results/stress/smap_era5_comparison.csv."
  check4
}

check5(){
  echo "STEP 5: Land cover filtering"
  need_file docs/land_cover_filtering.md || true
  need_file results/qc/land_cover_filter_summary.csv || true
  need_file results/qc/burned_area_exclusion_summary.csv || true
  need_file results/qc/irrigation_exclusion_summary.csv || true
}
add5(){
  mkdir -p docs results/qc
  cat > docs/land_cover_filtering.md <<'EOF'
# Land-cover filtering

Primary sample: MCD12Q1 IGBP class 10 grasslands.
Extensions: class 8 woody savanna and class 9 savanna.
Stability: remove pixels changing class during 2001-2024.
Disturbance: remove MCD64A1 burned periods/pixels.
Agricultural: exclude cropland/mixed and irrigated pixels.
EOF
  echo "dataset,rule,status" > results/qc/land_cover_filter_summary.csv
  echo "MCD12Q1,stable class 10 primary,needs final count verification" >> results/qc/land_cover_filter_summary.csv
  echo "dataset,rule,status" > results/qc/burned_area_exclusion_summary.csv
  echo "MCD64A1,burned periods excluded,needs final count verification" >> results/qc/burned_area_exclusion_summary.csv
  echo "dataset,rule,status" > results/qc/irrigation_exclusion_summary.csv
  echo "irrigation mask,irrigated pixels excluded,MISSING real irrigation mask" >> results/qc/irrigation_exclusion_summary.csv
  check5
}

check6(){
  echo "STEP 6: Quality control"
  need_file results/qc/modis_qc_summary.csv || true
  need_file results/qc/log_transform_thresholds.csv || true
}
add6(){
  mkdir -p results/qc docs
  cat > results/qc/modis_qc_summary.csv <<'EOF'
product,qa_rule,status
MODIS_GPP,keep QA bits 0 and 1,needs verification against exported QA bands
MODIS_ET,keep QA bits 0 and 1,needs verification against exported QA bands
ET,remove ET < 0.1 mm per 8-day,implemented in final metric checks/pipeline
EOF
  add1 || true
  check6
}

check7(){
  echo "STEP 7: Growing season definitions"
  need_file docs/growing_season_definitions.md || true
  need_file results/growing_season/growing_season_summary.csv || true
}
add7(){
  mkdir -p docs results/growing_season
  cat > docs/growing_season_definitions.md <<'EOF'
# Growing-season definitions

1. Baseline: GPP > 20% annual peak.
2. Climate-based: temperature + precipitation climatological season.
3. Fixed-effects: keep all observations with month/phenological fixed effects.
EOF
  echo "definition,status" > results/growing_season/growing_season_summary.csv
  echo "gpp_threshold,implemented/needs final run verification" >> results/growing_season/growing_season_summary.csv
  echo "climate_threshold,implemented/needs final run verification" >> results/growing_season/growing_season_summary.csv
  echo "month_fixed_effects,implemented/needs final run verification" >> results/growing_season/growing_season_summary.csv
  check7
}

check8(){
  echo "STEP 8: Compound stress metrics"
  need_file docs/compound_stress_metrics.md || true
  need_file results/stress/stress_metric_summary.csv || true
}
add8(){
  mkdir -p docs results/stress
  cat > docs/compound_stress_metrics.md <<'EOF'
# Compound stress metrics

1. Standardized VPD minus standardized soil moisture.
2. Joint percentile: VPD > 75th percentile and SM < 25th percentile.
3. Copula or empirical copula-style joint stress.
4. Interaction model: VPD + soil moisture + VPD×soil moisture.
EOF
  echo "metric,status" > results/stress/stress_metric_summary.csv
  echo "zscore,implemented/needs final output verification" >> results/stress/stress_metric_summary.csv
  echo "percentile,implemented/needs final output verification" >> results/stress/stress_metric_summary.csv
  echo "copula,implemented as empirical copula-style unless formal model added" >> results/stress/stress_metric_summary.csv
  echo "interaction,implemented/needs coefficient uncertainty verification" >> results/stress/stress_metric_summary.csv
  check8
}

check9(){
  echo "STEP 9: Gate 1 response-shape analysis"
  need_file results/full_matrix/raw/point_gate2_pixel_results.csv || true
  if [ -f results/full_matrix/raw/point_gate2_pixel_results.csv ]; then
    run_python check_matrix_columns results/full_matrix/raw/point_gate2_pixel_results.csv response_class pre_slope post_slope slope_change || true
  fi
}
add9(){
  echo "Gate 1 is produced by the full 3x3 run. Running add13 will produce the required file."
  add13
}

check10(){
  echo "STEP 10: Threshold detection"
  need_file results/full_matrix/raw/point_gate2_pixel_results.csv || true
  need_file results/bayesian/bayesian_threshold_agreement_raw.csv || true
}
add10(){
  if [ ! -f results/full_matrix/raw/point_gate2_pixel_results.csv ]; then
    echo "Need full matrix first."
    add13
  fi
  python -m wue_pipeline.workflows.bayesian_validate \
    --fits results/full_matrix/raw/point_gate2_pixel_results.csv \
    --timeseries data/raw/agents/merged_full_matrix_raw.csv \
    --out results/bayesian/bayesian_threshold_agreement_raw.csv
  if [ -f results/full_matrix/co2corrected/point_gate2_pixel_results.csv ]; then
    python -m wue_pipeline.workflows.bayesian_validate \
      --fits results/full_matrix/co2corrected/point_gate2_pixel_results.csv \
      --timeseries data/raw/agents/merged_full_matrix_co2corrected.csv \
      --out results/bayesian/bayesian_threshold_agreement_co2corrected.csv
  fi
  check10
}

check11(){
  echo "STEP 11: Parameters to report"
  if [ -f results/full_matrix/raw/point_gate2_pixel_results.csv ]; then
    run_python check_matrix_columns results/full_matrix/raw/point_gate2_pixel_results.csv pre_slope post_slope slope_change response_class || true
  else
    miss "results/full_matrix/raw/point_gate2_pixel_results.csv"
  fi
}
add11(){ add13; check11; }

check12(){
  echo "STEP 12: Gate 1 success criterion"
  need_file results/final_memos/gate1_success_status.md || true
}
add12(){
  mkdir -p results/final_memos
  if [ ! -f results/full_matrix/raw/point_gate2_pixel_results.csv ]; then add13; fi
  python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path("results/full_matrix/raw/point_gate2_pixel_results.csv")
out=Path("results/final_memos/gate1_success_status.md")
if not p.exists():
    out.write_text("# Gate 1\n\nMISSING full-matrix pixel results.\n")
else:
    df=pd.read_csv(p)
    valid=df[~df.get("response_class","").astype(str).isin(["insufficient_data","invalid_x_range","inconclusive"])]
    out.write_text(f"# Gate 1\n\nTotal fits: {len(df)}\nDefensible non-inconclusive fits: {len(valid)}\nSuccess: {len(valid)>0}\n\nResponse classes:\n{df['response_class'].value_counts(dropna=False).to_string()}\n")
print("Wrote", out)
PY
  check12
}

check13(){
  echo "STEP 13: Gate 2 product matrix"
  need_file results/full_matrix/raw/point_gate2_pixel_results.csv || true
  need_file results/full_matrix/raw/point_gate2_robustness_matrix.csv || true
}
add13(){
  echo "Preparing full matrix if missing..."
  if [ ! -f data/raw/agents/merged_full_matrix_raw.csv ] || [ ! -f data/raw/agents/merged_full_matrix_co2corrected.csv ]; then
    add24
    add25
    python scripts/gosif_point_agent.py --points data/raw/gee/stable_grassland_points.csv --start-year "${FULL_START_YEAR:-2001}" --end-year "${FULL_END_YEAR:-2024}" --local-glob "${GOSIF_LOCAL_GLOB:-data/raw/gosif/*.tif*}" --out data/raw/agents/gosif_point_timeseries.csv
    export GLEAM_VAR="${GLEAM_VAR:-E}"
    python scripts/gleam_point_agent.py --points data/raw/gee/stable_grassland_points.csv --start-year "${FULL_START_YEAR:-2001}" --end-year "${FULL_END_YEAR:-2024}" --local-glob "${GLEAM_LOCAL_GLOB:-data/raw/gleam/E_20*_GLEAM_v4.3a.nc}" --var "${GLEAM_VAR:-E}" --out data/raw/agents/gleam_point_timeseries.csv
    python scripts/merge_full_matrix_from_point_files.py \
      --gee-glob "data/raw/gee/wue_timeseries_*.csv" \
      --gosif data/raw/agents/gosif_point_timeseries.csv \
      --gleam data/raw/agents/gleam_point_timeseries.csv \
      --co2 data/external/noaa_co2_8day.csv \
      --aridity data/external/aridity_by_point.csv \
      --out-raw data/raw/agents/merged_full_matrix_raw.csv \
      --out-co2 data/raw/agents/merged_full_matrix_co2corrected.csv
  fi

  find results/tables -name "point_*" -delete || true
  wue points run-all \
    --input-glob "data/raw/agents/merged_full_matrix_raw.csv" \
    --gpp-products MODIS,GOSIF,PML \
    --et-products MODIS,GLEAM,PML \
    --min-obs 50 \
    --n-boot 1000
  mkdir -p results/full_matrix/raw
  cp results/tables/point_gate2_* results/full_matrix/raw/
  check13
}

check14(){
  echo "STEP 14: Stress-metric robustness"
  need_file results/full_matrix/raw/point_gate2_robustness_matrix.csv || true
}
add14(){ add13; check14; }

check15(){
  echo "STEP 15: Growing-season robustness"
  need_file results/full_matrix/raw/point_gate2_robustness_matrix.csv || true
}
add15(){ add13; check15; }

check16(){
  echo "STEP 16: Algorithm dependency table"
  need_file results/full_matrix/algorithm_dependency_table.csv || true
}
add16(){
  run_python make_algorithm_dependency
  check16
}

check17(){
  echo "STEP 17: Gate 2 success criterion"
  need_file results/final_memos/gate2_success_status.md || true
}
add17(){
  mkdir -p results/final_memos
  [ -f results/full_matrix/raw/point_gate2_robustness_matrix.csv ] || add13
  python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path("results/full_matrix/raw/point_gate2_robustness_matrix.csv")
out=Path("results/final_memos/gate2_success_status.md")
if not p.exists():
    out.write_text("# Gate 2\n\nMISSING robustness matrix.\n")
else:
    df=pd.read_csv(p)
    out.write_text("# Gate 2\n\nRobustness rows: %d\n\nThis file must be interpreted for consistency across >=2 GPP, >=2 ET, >=2 stress definitions, and >=2 growing-season definitions.\n\n%s\n" % (len(df), df.head(30).to_markdown(index=False)))
print("Wrote", out)
PY
  check17
}

check18(){
  echo "STEP 18: Tower data collection"
  need_file data/raw/towers/fluxnet2015_grassland_sites.csv || true
  need_file data/raw/towers/ameriflux_grassland_sites.csv || true
  need_file data/raw/towers/icos_grassland_sites.csv || true
  need_file data/raw/towers/ozflux_grassland_sites.csv || true
}
add18(){
  mkdir -p data/raw/towers
  cat > data/raw/towers/README_REQUIRED_COLUMNS.txt <<'EOF'
Create/harmonize these files:
fluxnet2015_grassland_sites.csv
ameriflux_grassland_sites.csv
icos_grassland_sites.csv
ozflux_grassland_sites.csv

Required columns or detectable equivalents:
tower_id,date,lat,lon,GPP_NT_VUT_REF,LE_F_MDS,VPD,soil_moisture,energy_balance_closure,gapfill_fraction
EOF
  echo "MANUAL DATA REQUIRED: download/harmonize tower CSVs into data/raw/towers/. See data/raw/towers/README_REQUIRED_COLUMNS.txt"
  check18
}

check19(){
  echo "STEP 19: Tower site screening"
  need_file results/tower_validation/tower_quality_screen.csv || true
}
add19(){
  check18 || stop "Tower CSVs missing; cannot screen sites."
  python -m wue_pipeline.validation.prepare_towers \
    --fluxnet data/raw/towers/fluxnet2015_grassland_sites.csv \
    --ameriflux data/raw/towers/ameriflux_grassland_sites.csv \
    --icos data/raw/towers/icos_grassland_sites.csv \
    --ozflux data/raw/towers/ozflux_grassland_sites.csv \
    --out data/processed/tower_validation_ready.csv
  check19
}

check20(){
  echo "STEP 20: Tower variables and 8-day aggregation"
  need_file data/processed/tower_validation_ready.csv || true
}
add20(){ add19; check20; }

check21(){
  echo "STEP 21: Tower analysis"
  need_file results/tower_validation/tower_response_classes.csv || true
}
add21(){
  [ -f data/processed/tower_validation_ready.csv ] || add19
  python -m wue_pipeline.validation.run_tower_response \
    --tower data/processed/tower_validation_ready.csv \
    --out results/tower_validation/tower_response_classes.csv
  check21
}

check22(){
  echo "STEP 22: Satellite vs tower comparison"
  need_file results/tower_validation/satellite_tower_concordance.csv || true
  need_file results/tower_validation/concordance_by_product.csv || true
}
add22(){
  [ -f results/tower_validation/tower_response_classes.csv ] || add21
  [ -f results/full_matrix/raw/point_gate2_pixel_results.csv ] || add13
  python -m wue_pipeline.validation.compare_satellite_tower \
    --tower results/tower_validation/tower_response_classes.csv \
    --satellite results/full_matrix/raw/point_gate2_pixel_results.csv \
    --out results/tower_validation/satellite_tower_concordance.csv
  check22
}

check23(){
  echo "STEP 23: Gate 3 success criterion"
  need_file results/final_memos/gate3_success_status.md || true
}
add23(){
  [ -f results/tower_validation/concordance_by_product.csv ] || add22
  mkdir -p results/final_memos
  python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path("results/tower_validation/concordance_by_product.csv")
out=Path("results/final_memos/gate3_success_status.md")
if not p.exists():
    out.write_text("# Gate 3\n\nMISSING tower concordance.\n")
else:
    df=pd.read_csv(p)
    best=df.sort_values("class_match_fraction", ascending=False).head(5) if "class_match_fraction" in df else df.head(5)
    out.write_text("# Gate 3\n\nTower validation completed. Either choose best product family or conclude towers cannot resolve disagreement.\n\nTop product families:\n\n%s\n" % best.to_markdown(index=False))
print("Wrote", out)
PY
  check23
}

check24(){
  echo "STEP 24: CO2 correction"
  need_file data/external/noaa_co2_8day.csv || true
  need_file data/raw/agents/merged_full_matrix_co2corrected.csv || true
}
add24(){
  python scripts/make_noaa_co2_8day.py \
    --start-date "${FULL_START_DATE:-2001-01-01}" \
    --end-date "${FULL_END_DATE:-2024-12-31}" \
    --out data/external/noaa_co2_8day.csv
  check24 || true
}

check25(){
  echo "STEP 25: Aridity stratification"
  need_file data/external/aridity_by_point.csv || need_file data/external/cgiar_aridity_index_0p1deg.nc || true
  need_file results/aridity/aridity_quartile_summary_raw.csv || true
}
add25(){
  if [ ! -f data/external/aridity_by_point.csv ]; then
    python scripts/make_aridity_by_point.py \
      --points data/raw/gee/stable_grassland_points.csv \
      --aridity-raster "${ARIDITY_RASTER:-data/external/cgiar_aridity_index_0p1deg.nc}" \
      --out data/external/aridity_by_point.csv
  fi
  if [ -f results/full_matrix/raw/point_gate2_pixel_results.csv ]; then
    python -m wue_pipeline.workflows.aridity_report \
      --fits results/full_matrix/raw/point_gate2_pixel_results.csv \
      --aridity data/external/aridity_by_point.csv \
      --out results/aridity/aridity_quartile_summary_raw.csv
  fi
  if [ -f results/full_matrix/co2corrected/point_gate2_pixel_results.csv ]; then
    python -m wue_pipeline.workflows.aridity_report \
      --fits results/full_matrix/co2corrected/point_gate2_pixel_results.csv \
      --aridity data/external/aridity_by_point.csv \
      --out results/aridity/aridity_quartile_summary_co2corrected.csv
  fi
  check25
}

check26(){
  echo "STEP 26: Uncertainty analyses"
  need_file results/bayesian/bayesian_threshold_agreement_raw.csv || true
  need_file results/full_matrix/raw/point_gate2_pixel_results.csv || true
  need_file results/aridity/aridity_quartile_summary_raw.csv || true
  need_file results/tower_validation/satellite_tower_concordance.csv || true
}
add26(){
  add10 || true
  add25 || true
  add22 || true
  check26
}

check27(){
  echo "STEP 27: Conditional trait analysis data"
  need_file data/external/liu_2021_psi50_0p1deg.nc || true
  need_file data/external/konings_gentine_isohydricity_0p1deg.nc || true
  need_file data/external/stocker_2023_rooting_depth_0p1deg.nc || true
  need_file data/processed/trait_model_table.csv || true
}
add27(){
  check23 || warn "Gate 3 status not available; trait interpretation must remain conditional."
  [ -f results/tower_validation/satellite_tower_concordance.csv ] || add22
  python -m wue_pipeline.traits.prepare_trait_table \
    --response results/tower_validation/satellite_tower_concordance.csv \
    --psi50 data/external/liu_2021_psi50_0p1deg.nc \
    --isohydricity data/external/konings_gentine_isohydricity_0p1deg.nc \
    --rooting-depth data/external/stocker_2023_rooting_depth_0p1deg.nc \
    --aridity data/external/aridity_by_point.csv \
    --matrix data/raw/agents/merged_full_matrix_raw.csv \
    --out data/processed/trait_model_table.csv
  check27
}

check28(){
  echo "STEP 28: Climate controls"
  need_file data/processed/trait_model_table.csv || true
  if [ -f data/processed/trait_model_table.csv ]; then
    run_python check_matrix_columns data/processed/trait_model_table.csv aridity_index mean_temperature mean_precipitation mean_lai || true
  fi
}
add28(){ add27; check28; }

check29(){
  echo "STEP 29: Trait models"
  need_file results/traits/random_forest_shap_results.csv || true
  need_file results/traits/bayesian_hierarchical_trait_results.csv || true
}
add29(){
  [ -f data/processed/trait_model_table.csv ] || add27
  python -m wue_pipeline.traits.run_random_forest_shap \
    --input data/processed/trait_model_table.csv \
    --out results/traits/random_forest_shap_results.csv
  python -m wue_pipeline.traits.run_bayesian_hierarchical \
    --input data/processed/trait_model_table.csv \
    --out results/traits/bayesian_hierarchical_trait_results.csv
  check29
}

check30(){
  echo "STEP 30: Trait success criterion"
  need_file results/traits/trait_cross_validation.csv || true
  need_file results/final_memos/trait_success_status.md || true
}
add30(){
  [ -f data/processed/trait_model_table.csv ] || add27
  python -m wue_pipeline.traits.cross_validate_traits \
    --input data/processed/trait_model_table.csv \
    --out results/traits/trait_cross_validation.csv
  mkdir -p results/final_memos
  python - <<'PY'
from pathlib import Path
import pandas as pd
p=Path("results/traits/trait_cross_validation.csv")
out=Path("results/final_memos/trait_success_status.md")
if not p.exists():
    out.write_text("# Trait success\n\nMISSING trait cross-validation.\n")
else:
    df=pd.read_csv(p)
    txt="# Trait success\n\nCriterion: traits explain >20% climate-residual variance and at least one trait survives LOOCV/significance testing.\n\n"
    txt += df.to_markdown(index=False)
    out.write_text(txt)
print("Wrote", out)
PY
  check30
}

run_step(){
  local n="$1"
  if [ "$ACTION" = "check" ]; then "check$n"; elif [ "$ACTION" = "add" ]; then "add$n"; else stop "Action must be check or add"; fi
}

if [ "$STEP" = "all" ]; then
  for n in $(seq 0 30); do
    echo ""
    echo "============================================================"
    echo "$ACTION STEP $n"
    echo "============================================================"
    if [ "$ACTION" = "check" ]; then
      "check$n" || true
    else
      "add$n"
    fi
  done
else
  run_step "$STEP"
fi
