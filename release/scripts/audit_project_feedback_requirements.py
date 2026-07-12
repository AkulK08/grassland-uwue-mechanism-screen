#!/usr/bin/env python
from pathlib import Path
import re
import json
import pandas as pd

ROOT = Path(".")
checks = []

def has_file(p):
    return Path(p).exists() and Path(p).stat().st_size > 0

def grep(pattern, paths):
    rgx = re.compile(pattern, re.I | re.M)
    hits = []
    for base in paths:
        for p in Path(base).rglob("*.py"):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            if rgx.search(txt):
                hits.append(str(p))
    return hits

def add(item, status, evidence, required_fix=""):
    checks.append({
        "item": item,
        "status": status,
        "evidence": evidence,
        "required_fix": required_fix
    })

# Data wiring
add("GOSIF data present", "PASS" if has_file("data/raw/agents/gosif_point_timeseries.csv") else "FAIL",
    "data/raw/agents/gosif_point_timeseries.csv",
    "Run gosif_point_agent.py and mask fill values >=60000.")

add("GLEAM data present", "PASS" if has_file("data/raw/agents/gleam_point_timeseries.csv") else "FAIL",
    "data/raw/agents/gleam_point_timeseries.csv",
    "Run gleam_point_agent.py with GLEAM_VAR=E.")

add("Merged 3x3 matrix present", "PASS" if has_file("data/raw/agents/merged_full_matrix_raw.csv") else "FAIL",
    "data/raw/agents/merged_full_matrix_raw.csv",
    "Run merge_full_matrix_from_point_files.py.")

add("MODIS QA from AppEEARS", "PASS" if has_file("data/processed/modis_qa_by_point_8day.csv") and has_file("results/qc/modis_qc_summary.csv") else "FAIL",
    "data/processed/modis_qa_by_point_8day.csv + results/qc/modis_qc_summary.csv",
    "Run scripts/ingest_appeears_modis_qa_zip.py.")

add("SMAP validation", "PASS" if has_file("results/stress/smap_era5_comparison.csv") else "FAIL",
    "results/stress/smap_era5_comparison.csv",
    "Do not redownload if present.")

add("Irrigation filtered dataset", "PASS" if has_file("data/external/irrigation_by_point.csv") else "FAIL",
    "data/external/irrigation_by_point.csv",
    "Create/use data/raw/gee_final_filtered_no_irrigation with 199 points.")

# Code/method features
features = {
    "uWUE primary metric": (r"uwue|underlying_wue|sqrt\(.*vpd|np\.sqrt\(.*vpd", "Implement uWUE = GPP*sqrt(VPD)/ET as primary; raw WUE and iWUE sensitivity."),
    "iWUE sensitivity": (r"iwue|inherent_wue|gpp.*vpd.*et", "Implement iWUE = GPP*VPD/ET as sensitivity."),
    "breakpoint existence test": (r"bic|davies|supf|score_test|no[-_ ]?break|linear_null", "Add BIC + supF/Davies-style test vs no-break linear model."),
    "Bayesian overlap hard gate": (r"bayes_overlap.*(and|required|if)|require.*bayes|hard.*gate", "Require segmented/Bayesian interval overlap for reporting transitions/classes."),
    "block bootstrap": (r"block_boot|block bootstrap|moving_block|year_block|season_block", "Replace iid bootstrap with block bootstrap by year/season block."),
    "2D VPD x SM surface": (r"vpd.*soil_moisture.*interaction|response_surface|vpd_x_sm|partial_effect", "Add 2-D surface / decoupled interaction and partial effects."),
    "soil texture covariate": (r"soilgrids|sand|silt|clay|soil_texture|hydraulic_conduct", "Add SoilGrids sand/silt/clay and derived texture/conductivity descriptor."),
    "mutual exclusive classifier": (r"classify_response|response_class", "Inspect manually: must require significant pre/post slopes and slope-change logic."),
    "tower arbiter": (r"tower.*concordance|compare_satellite_tower|tower_response", "Implement quality-screened tower classification and product-family arbitration."),
    "hierarchical trait model": (r"hierarchical|partial_pool|pymc|stan|bayesian_hierarchical", "Make trait phase partial-pooling/hierarchical, RF+SHAP descriptive only."),
    "causal DAG": (r"\bdag\b|causal|estimand|confound", "Add DAG/estimand doc and climate/soil texture controls.")
}
for item, (pat, fix) in features.items():
    hits = grep(pat, ["src", "scripts"])
    add(item, "PASS_OR_PARTIAL" if hits else "FAIL", "; ".join(hits[:10]), fix)

out = Path("results/qc/project_feedback_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(checks, indent=2))

print("\nproject FEEDBACK AUDIT")
for c in checks:
    print(f"{c['status']:15} {c['item']}")
    if c["evidence"]:
        print(f"  evidence: {c['evidence']}")
    if c["status"] == "FAIL":
        print(f"  fix: {c['required_fix']}")
print("\nWROTE", out)
