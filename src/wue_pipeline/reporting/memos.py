"""Gate memo and manuscript text generation."""

from __future__ import annotations
import pandas as pd
from ..config import ProjectConfig


def _write(path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_gate1_memo(cfg: ProjectConfig, results: pd.DataFrame):
    counts = results["response_class"].value_counts(dropna=False).to_dict() if len(results) else {}
    text = f"""# Gate 1 Memo: Response-Shape Characterization

## Objective

Gate 1 tests whether a defensible response signal exists in at least one product combination and establishes the operational vocabulary for enhancement, saturation, and reversal.

## Product combination

- GPP product: {results['gpp_product'].iloc[0] if len(results) else 'NA'}
- ET product: {results['et_product'].iloc[0] if len(results) else 'NA'}
- Stress definition: zscore
- Growing season: GPP threshold

## Classification counts

{counts}

## Decision rule

Proceed to Gate 2 if at least one response class is defensible with pre- and post-transition slopes and uncertainty intervals reported. Reversal/breakdown is only defensible when the post-transition upper confidence bound is below zero.

## Files

- `results/tables/gate1_pixel_results.csv`
- `results/tables/gate1_aridity_summary.csv`
"""
    return _write(cfg.file("memos", "gate1_memo.md"), text)


def write_gate2_memo(cfg: ProjectConfig, summary: pd.DataFrame, success: pd.DataFrame):
    success_text = success.to_markdown(index=False) if len(success) else "No response class passed the formal Gate 2 criterion."
    text = f"""# Gate 2 Memo: Cross-Product Robustness

## Objective

Gate 2 evaluates whether the response-shape classification is robust across product choice, compound-stress definition, growing-season definition, and aridity strata.

## Formal success criteria

The same qualitative response shape must be found across at least two GPP products, at least two ET products, at least two stress definitions, and at least two growing-season definitions.

## Success table

{success_text}

## Interpretation guide

If Gate 2 fails, the result should be framed as product/method dependence rather than ecosystem-level breakdown. If Gate 2 passes, proceed to tower validation.

## Files

- `results/tables/gate2_pixel_results.csv`
- `results/tables/gate2_robustness_matrix.csv`
- `results/tables/gate2_success_criteria.csv`
- `results/tables/algorithm_dependency_table.csv`
"""
    return _write(cfg.file("memos", "gate2_memo.md"), text)


def write_gate3_memo(cfg: ProjectConfig, screen: pd.DataFrame, tower_class: pd.DataFrame, validation: pd.DataFrame, concordance: pd.DataFrame):
    n_good = int(screen["passes_quality"].sum()) if len(screen) else 0
    conc_text = concordance.to_markdown(index=False) if len(concordance) else "Tower-satellite concordance could not be estimated."
    text = f"""# Gate 3 Memo: Tower-Based Validation

## Objective

Gate 3 uses independent eddy-covariance measurements to evaluate whether satellite-derived response shapes correspond to in situ grassland carbon-water behavior.

## Tower screening

- Candidate sites: {len(screen)}
- Sites passing quality screen: {n_good}

## Concordance by product family

{conc_text}

## Decision rule

If one product family has clear tower concordance, use it as the validated product family for Phase 4. If towers cannot resolve product disagreement, report that limitation directly and frame the paper around satellite-product identifiability.

## Files

- `results/tables/gate3_tower_quality_screen.csv`
- `results/tables/gate3_tower_response_classes.csv`
- `results/tables/gate3_tower_validation.csv`
"""
    return _write(cfg.file("memos", "gate3_memo.md"), text)


def write_phase4_memo(cfg: ProjectConfig, summary: pd.DataFrame, shap_importance: pd.DataFrame):
    summ = summary.to_markdown(index=False) if len(summary) else "No Phase 4 summary available."
    shap_txt = shap_importance.head(10).to_markdown(index=False) if len(shap_importance) else "No SHAP table available."
    text = f"""# Phase 4 Memo: Trait Analysis

## Objective

Phase 4 tests whether spatial variation in the tower-validated response metric is predicted by plant hydraulic and stomatal traits after controlling for climate.

## Variance explained

{summ}

## Predictor importance

{shap_txt}

## Decision rule

Trait inference is treated as successful if traits explain at least the configured threshold of climate-residual variance and at least one trait is consistently important across model families.
"""
    return _write(cfg.file("memos", "phase4_memo.md"), text)
