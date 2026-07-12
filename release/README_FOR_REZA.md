# Grassland uWUE mechanism screen — mentor release

This folder contains the cleaned scripts and result tables for the updated mechanism screen.

## Main decision output

Read first:

- `results/stage1b6bl_individual_reza_mechanism_decision/text/READ_ME_individual_reza_mechanism_decision.txt`

Main tables:

- `results/stage1b6bl_individual_reza_mechanism_decision/tables/DEDUPLICATED_MECHANISM_FAMILY_DECISION_TABLE.csv`
- `results/stage1b6bl_individual_reza_mechanism_decision/tables/SATELLITE_REZA_PASS_NO_TOWER_NO_C4_FOCUS.csv`
- `results/stage1b6bl_individual_reza_mechanism_decision/tables/ONE_GATE_SHORT_NO_C4_FOCUS.csv`
- `results/stage1b6bl_individual_reza_mechanism_decision/tables/INDIVIDUAL_REZA_DECISION_PROGRAMMING_AUDIT.json`

## Core result

No mechanism currently passes the full tower-inclusive Reza standard because there is not yet a true independent same-feature tower-derived uWUE-response test.

Fifteen mechanisms pass all satellite-side gates: full controls, FDR, bootstrap/LOO stability, clean land-cover sensitivity, exact product robustness, and GOSIF × GLEAM/product-dependence checking.

The strongest overall family is a cold-temperature threshold. The strongest more ecological/canopy candidate is a sparse-canopy / low-LAI threshold.

## Programming note

The final decision screen does not globally filter points based on unrelated missing traits. Each mechanism uses complete cases only for its own outcome, focal term, moderator if any, and required controls.
