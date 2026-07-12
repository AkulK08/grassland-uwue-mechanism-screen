#!/usr/bin/env bash

cd /Users/me/Downloads/grassland_wue_nature_repo || exit 1

BAR_WIDTH=40
SLEEP_SEC=30

bar() {
  local pct=$1
  local filled=$((pct * BAR_WIDTH / 100))
  local empty=$((BAR_WIDTH - filled))
  printf "["
  printf "%0.s#" $(seq 1 $filled 2>/dev/null)
  printf "%0.s-" $(seq 1 $empty 2>/dev/null)
  printf "] %3d%%" "$pct"
}

exists() {
  [ -s "$1" ]
}

count_done() {
  local done=0

  exists ".checkpoints/final_handoff_no_modisqa/01_verify_handoff_inputs.done" && done=$((done+1))
  exists ".checkpoints/final_handoff_no_modisqa/02_create_filtered_no_irrigation.done" && done=$((done+1))
  exists ".checkpoints/final_handoff_no_modisqa/03_activate_filtered_gee.done" && done=$((done+1))
  exists ".checkpoints/final_handoff_no_modisqa/04_gosif_agent.done" && done=$((done+1))
  exists ".checkpoints/final_handoff_no_modisqa/05_gleam_agent.done" && done=$((done+1))

  exists "data/raw/agents/merged_full_matrix_raw.csv" && done=$((done+1))
  exists "data/raw/agents/merged_full_matrix_co2corrected.csv" && done=$((done+1))

  exists "results/full_matrix/raw/point_gate2_pixel_results.csv" && done=$((done+1))
  exists "results/full_matrix/raw/point_gate2_robustness_matrix.csv" && done=$((done+1))

  exists "results/full_matrix/co2corrected/point_gate2_pixel_results.csv" && done=$((done+1))
  exists "results/full_matrix/co2corrected/point_gate2_robustness_matrix.csv" && done=$((done+1))

  exists "results/bayesian/bayesian_threshold_agreement_raw.csv" && done=$((done+1))
  exists "results/bayesian/bayesian_threshold_agreement_co2corrected.csv" && done=$((done+1))

  exists "results/aridity/aridity_quartile_summary_raw.csv" && done=$((done+1))
  exists "results/aridity/aridity_quartile_summary_co2corrected.csv" && done=$((done+1))

  exists "results/tower_validation/tower_response_classes.csv" && done=$((done+1))
  exists "results/tower_validation/concordance_by_product.csv" && done=$((done+1))

  exists "results/traits/random_forest_shap_results.csv" && done=$((done+1))
  exists "results/traits/bayesian_hierarchical_trait_results.csv" && done=$((done+1))
  exists "results/traits/trait_cross_validation.csv" && done=$((done+1))

  exists "results/final_memos/gate1_success_status.md" && done=$((done+1))
  exists "results/final_memos/gate2_success_status.md" && done=$((done+1))
  exists "results/final_memos/gate3_success_status.md" && done=$((done+1))
  exists "results/final_memos/trait_success_status.md" && done=$((done+1))

  exists "results/manuscript/manuscript_skeleton.md" && done=$((done+1))
  exists "results/qc/final_checkpointed_no_modisqa_manifest.json" && done=$((done+1))

  echo "$done"
}

stage_name() {
  if exists "results/qc/final_checkpointed_no_modisqa_manifest.json"; then
    echo "COMPLETE except MODIS QA"
  elif exists "results/manuscript/manuscript_skeleton.md"; then
    echo "Writing final manifest"
  elif exists "results/final_memos/gate3_success_status.md"; then
    echo "Writing final memos/manuscript"
  elif exists "results/traits/trait_cross_validation.csv"; then
    echo "Trait analysis complete; final reporting"
  elif exists "results/tower_validation/concordance_by_product.csv"; then
    echo "Tower validation / trait analysis"
  elif exists "results/aridity/aridity_quartile_summary_raw.csv"; then
    echo "Aridity / tower validation"
  elif exists "results/bayesian/bayesian_threshold_agreement_raw.csv"; then
    echo "Bayesian / aridity analysis"
  elif exists "results/full_matrix/co2corrected/point_gate2_pixel_results.csv"; then
    echo "CO2-corrected matrix complete; Bayesian next"
  elif exists "results/full_matrix/raw/point_gate2_pixel_results.csv"; then
    echo "Raw matrix complete; CO2-corrected run next"
  elif pgrep -fl "wue points run-all" >/dev/null; then
    echo "Running slow WUE bootstrap fitting"
  elif exists "data/raw/agents/merged_full_matrix_raw.csv"; then
    echo "Merged matrices written; waiting for fitting outputs"
  elif exists ".checkpoints/final_handoff_no_modisqa/05_gleam_agent.done"; then
    echo "GOSIF/GLEAM done; preparing final run"
  elif exists ".checkpoints/final_handoff_no_modisqa/04_gosif_agent.done"; then
    echo "GLEAM extraction"
  elif exists ".checkpoints/final_handoff_no_modisqa/03_activate_filtered_gee.done"; then
    echo "GOSIF extraction"
  else
    echo "Startup / verification"
  fi
}

while true; do
  clear

  TOTAL=26
  DONE=$(count_done)
  PCT=$((DONE * 100 / TOTAL))
  STAGE=$(stage_name)

  echo "============================================================"
  echo "Final checkpointed pipeline progress"
  echo "============================================================"
  echo ""
  bar "$PCT"
  echo "  ($DONE / $TOTAL milestones)"
  echo ""
  echo "Current stage: $STAGE"
  echo "Time: $(date)"
  echo ""

  echo "===== Running processes ====="
  pgrep -fl "run_final_checkpointed|final_paper_run_all|wue points run-all|python|caffeinate" || echo "No final run process found."
  echo ""

  PID=$(pgrep -f "wue points run-all" | tail -1 || true)
  if [ -n "$PID" ]; then
    echo "===== Active fitting process ====="
    ps -p "$PID" -o pid,etime,%cpu,%mem,command
    echo ""
  fi

  echo "===== Checkpoints ====="
  ls .checkpoints/final_handoff_no_modisqa 2>/dev/null | sort || echo "No checkpoint folder yet."
  echo ""

  echo "===== Key outputs present ====="
  for f in \
    data/raw/agents/merged_full_matrix_raw.csv \
    data/raw/agents/merged_full_matrix_co2corrected.csv \
    results/full_matrix/raw/point_gate2_pixel_results.csv \
    results/full_matrix/co2corrected/point_gate2_pixel_results.csv \
    results/bayesian/bayesian_threshold_agreement_raw.csv \
    results/aridity/aridity_quartile_summary_raw.csv \
    results/tower_validation/concordance_by_product.csv \
    results/traits/trait_cross_validation.csv \
    results/manuscript/manuscript_skeleton.md \
    results/qc/final_checkpointed_no_modisqa_manifest.json
  do
    if [ -s "$f" ]; then
      echo "OK      $f"
    else
      echo "WAITING $f"
    fi
  done
  echo ""

  echo "===== Latest final log lines ====="
  LATEST=$(ls -t logs/final_paper_run_all_*.log 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "$LATEST"
    tail -25 "$LATEST"
  else
    echo "No final_paper_run_all log yet."
  fi

  echo ""
  echo "Refreshes every $SLEEP_SEC seconds. Press Ctrl+C to stop monitor only."
  echo "The actual pipeline keeps running in the background."

  if [ -s "results/qc/final_checkpointed_no_modisqa_manifest.json" ]; then
    echo ""
    echo "FINAL MANIFEST EXISTS. RUN IS COMPLETE EXCEPT MODIS QA."
    exit 0
  fi

  sleep "$SLEEP_SEC"
done
