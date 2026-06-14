"""Generate preregistration and manuscript skeletons."""

from __future__ import annotations
from ..config import ProjectConfig


def write_manuscript_files(cfg: ProjectConfig) -> None:
    prereg = """# OSF Pre-Registration Template

## Title

Ecosystem water-use efficiency response to compound atmospheric–soil moisture stress in global grasslands

## Primary hypothesis

Grassland ecosystem log(WUE) exhibits a structured response to compound high VPD and low soil-moisture stress that can be classified as enhancement, saturation, or reversal using pre- and post-transition slopes.

## Primary response variable

CO2-corrected log(WUE) = log(GPP) - log(ET) + log(CO2_ref) - log(CO2_t).

## Primary analysis

Segmented regression of log(WUE) against the z-score compound stress index using MODIS GPP and MODIS ET for Gate 1. Thresholds are reported only when bootstrap and Bayesian intervals overlap.

## Robustness tests

- 3 x 3 GPP/ET product matrix.
- Four stress definitions: z-score, percentile, copula, interaction.
- Three growing-season definitions: GPP threshold, climate threshold, month fixed effects.

## Validation

Flux tower validation uses screened grassland and savanna eddy-covariance sites.

## Trait analysis

Conditional on Gates 1-3. Dependent variable: slope change or post-transition slope. Predictors: psi50, isohydricity, rooting depth. Controls: aridity, MAP, MAT, LAI.
"""
    skeleton = """# Manuscript Skeleton

# Ecosystem water-use efficiency response to compound atmospheric-soil moisture stress in global grasslands

## Abstract

[Fill after gates complete.]

## Introduction

1. Ecosystem WUE as a window into carbon-water coupling.
2. Compound atmospheric and soil-moisture stress as a coupled climate hazard.
3. Need to separate enhancement, saturation, and true reversal.
4. Product identifiability and tower validation as prerequisites for trait inference.

## Results

### Gate 1: Response-shape characterization

Report Figure 2 and Table S1.

### Gate 2: Cross-product robustness

Report Figure 3, algorithm dependency table, robustness matrix, and sensitivity ranges.

### Gate 3: Tower validation

Report Figure 4, tower screening table, and product-family concordance.

### Conditional Phase 4: Trait predictors of response shape

Report Figure 5 if Gates 1-3 support trait analysis.

## Discussion

1. Interpret response shape.
2. Product dependence and algorithmic imprinting.
3. Tower validation and footprint mismatch.
4. Trait interpretation or negative trait finding.
5. Implications for Earth system model parameterization.

## Methods

Full product versions, preprocessing, quality flags, masks, CO2 correction, statistical models, uncertainty propagation, and reproducibility information.
"""
    cfg.file("manuscript", "preregistration_osf_template.md").write_text(prereg, encoding="utf-8")
    cfg.file("manuscript", "manuscript_skeleton.md").write_text(skeleton, encoding="utf-8")
