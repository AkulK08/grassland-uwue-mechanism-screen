#!/usr/bin/env python
from pathlib import Path
import json
import pandas as pd

Path("docs").mkdir(exist_ok=True, parents=True)
Path("results/reza_strict").mkdir(exist_ok=True, parents=True)

Path("docs/reza_method_patch_notes.md").write_text("""# Reza feedback method patch notes

This repo patch addresses the mentor feedback by changing the analysis target from a raw WUE threshold on a collapsed one-dimensional compound-stress index to a stricter, pre-specified framework.

## Primary response metric

Primary metric: underlying WUE, `uWUE = GPP * sqrt(VPD) / ET`.

Sensitivity metrics:

- raw `WUE = GPP / ET`
- inherent `iWUE = GPP * VPD / ET`

All three are stored in log space for fitting, with `log_uwue_*` as the primary response.

## MODIS QA

The AppEEARS MODIS QA table is converted to a wide one-row-per-point-date table and merged into the metric matrix. MODIS GPP combinations require `Psn_QC_500m` good-or-other, and MODIS ET combinations require `ET_QC_500m` good-or-other.

## Threshold acceptance

A transition is accepted only if all conditions hold:

1. segmented model improves over no-break linear model by BIC margin >= 6;
2. supF-style permutation p-value <= 0.05;
3. block-bootstrap transition interval overlaps the Bayesian/profile transition interval.

If these conditions are not met, the breakpoint is not treated as reportable.

## Block bootstrap

Uncertainty intervals are computed with year-block bootstrap rather than iid row bootstrap.

## Response classes

The strict response classifier is mutually exclusive:

- `breakdown`: accepted transition, pre-slope CI > 0, post-slope CI < 0, slope-change CI < 0.
- `saturation`: accepted transition, pre-slope CI > 0, post-slope CI includes 0, slope-change CI < 0.
- `enhancement_no_accepted_breakpoint`: no accepted transition but positive linear slope.
- `inconclusive`: all other cases.

## 2-D VPD x soil moisture surface

For each fit, a decoupled surface is also fit:

`log(metric) ~ VPD_z + SM_z + VPD_z:SM_z`

The VPD partial effect, SM partial effect, and interaction coefficient are reported separately.

## Soil texture

SoilGrids sand, silt, and clay are fetched and merged as required covariates where available.

## Tower and trait phase

Tower validation should use the strict response classes and the same response metric definitions. Trait attribution is framed as a causal/partial-pooling problem, not as a random-forest-only variable-importance exercise.
""")

Path("docs/trait_causal_dag.md").write_text("""# Trait causal DAG and estimand

## Causal question

Target estimand: the effect of plant hydraulic/stomatal strategy on the grassland uWUE response-shape metric, holding climate and soil texture fixed.

## DAG, text version

Climate/aridity affects:

- hydraulic traits
- vegetation composition
- soil moisture regime
- VPD regime
- response shape

Soil texture affects:

- soil moisture availability
- rooting-zone storage
- drought exposure
- response shape

Species composition affects:

- hydraulic traits
- rooting depth
- response shape

Hydraulic/stomatal traits affect:

- stomatal regulation
- carbon-water coupling
- post-transition slope or slope change

Therefore, climate and soil texture are required controls. Random forest + SHAP can describe predictive patterns, but the causal claim must come from a model that estimates trait effects conditional on climate and soil texture.

## Required controls

- aridity
- mean annual precipitation
- mean annual temperature
- growing-season LAI
- sand/silt/clay or derived soil texture descriptor

## Preferred model

Partial-pooling/hierarchical model where response shape is estimated jointly with trait effects. The current strict pipeline prepares the required response metric and covariates; a full PyMC/Stan implementation should be treated as the next major trait-phase task.
""")

# Write a simple current-status manifest.
files = {
    "strict_raw": "results/reza_strict/strict_response_results_raw.csv",
    "strict_co2": "results/reza_strict/strict_response_results_co2corrected.csv",
    "surface_raw": "results/reza_strict/vpd_sm_surface_partial_effects_raw.csv",
    "surface_co2": "results/reza_strict/vpd_sm_surface_partial_effects_co2corrected.csv",
    "qa_wide": "data/processed/modis_qa_by_point_8day_wide.csv",
    "metric_raw": "data/processed/reza_metric_matrix_raw.csv",
    "metric_co2": "data/processed/reza_metric_matrix_co2corrected.csv",
    "soil": "data/external/soilgrids_texture_by_point.csv",
    "dag": "docs/trait_causal_dag.md",
}
manifest = {k: {"path": v, "exists": Path(v).exists(), "size": Path(v).stat().st_size if Path(v).exists() else 0} for k, v in files.items()}
Path("results/reza_strict/reza_patch_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
