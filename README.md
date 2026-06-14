# Grassland WUE Nature Pipeline

This repository is a full production-oriented codebase for the project:

**Ecosystem Water-Use Efficiency Response to Compound Atmospheric–Soil Moisture Stress in Global Grasslands**

It implements the gated analysis requested in the research blueprint:

1. **Gate 1:** response-shape characterization with log(WUE) decomposition, segmented regression, bootstrap uncertainty, optional Bayesian change-point analysis, and shape classification.
2. **Gate 2:** cross-product robustness over the 3×3 GPP/ET product matrix, four compound-stress definitions, and three growing-season definitions.
3. **Gate 3:** eddy-covariance tower validation, quality screening, tower WUE/CSI construction, and tower-satellite concordance.
4. **Conditional Phase 4:** plant trait analysis using validated response metrics, climate residualization, random forest + SHAP, and Bayesian trait regression.
5. **Reporting:** manuscript-ready tables, figures, gate memos, pre-registration template, and manuscript skeleton.

The repository has two modes:

- **Demo mode:** creates synthetic but structurally realistic data and runs the entire workflow end-to-end. This is useful for testing the pipeline, figure generation, and reproducibility.
- **Production mode:** uses real datasets listed in `configs/production_template.yaml`. You supply paths to locally downloaded MODIS, GOSIF, PML-V2, GLEAM, ERA5-Land, SMAP, MCD12Q1, MCD64A1, tower, aridity, LAI, and trait datasets.

## What this repository can and cannot do

This repository implements the analysis logic, file layout, computations, uncertainty reporting, and figure/table generation required for the paper. It cannot invent unavailable satellite, tower, or trait data. In production mode, you must provide the raw data files and credentials required by each data provider. Once those data are present in the expected locations, the pipeline can run the full analysis.

## Install

```bash
conda env create -f environment.yml
conda activate grassland_wue_nature
pip install -e .
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Demo end-to-end run

```bash
make install
make demo CONFIG=configs/demo.yaml
make run-all CONFIG=configs/demo.yaml
```

This creates demo data, processes it, runs all gates, and writes outputs under `results/`.

## Production run

1. Copy the production template:

```bash
cp configs/production_template.yaml configs/production.yaml
```

2. Edit all paths under `data_catalog`.

3. Run:

```bash
make run-all CONFIG=configs/production.yaml
```

## Main outputs

- `results/tables/gate1_pixel_results.csv`
- `results/tables/gate1_aridity_summary.csv`
- `results/tables/gate2_robustness_matrix.csv`
- `results/tables/gate2_success_criteria.csv`
- `results/tables/gate3_tower_validation.csv`
- `results/tables/phase4_trait_results.csv`
- `results/figures/figure1_conceptual_response.png`
- `results/figures/figure2_gate1_response.png`
- `results/figures/figure3_product_matrix.png`
- `results/figures/figure4_tower_validation.png`
- `results/figures/figure5_trait_importance.png`
- `results/memos/gate1_memo.md`
- `results/memos/gate2_memo.md`
- `results/memos/gate3_memo.md`
- `results/manuscript/manuscript_skeleton.md`
- `results/manuscript/preregistration_osf_template.md`

## Repository layout

```text
configs/                  YAML configs for demo and production
src/wue_pipeline/          Installable Python package
  cli.py                   Command-line interface
  config.py                YAML parsing and settings
  constants.py             Product/stress/growing-season definitions
  io/                      Data adapters and demo data generation
  processing/              Regridding, masks, VPD, WUE, CSI, growing season
  models/                  Segmented, Bayesian, robustness, tower, trait models
  workflows/               Gate 1, Gate 2, Gate 3, Phase 4, reports
  figures/                 Manuscript figures
  reporting/               Memos, tables, manuscript skeleton
scripts/                   Convenience shell scripts
tests/                     Unit tests
docs/                      Methods, data contract, runbook
```

## Design principles

- All main quantitative outputs include uncertainty intervals.
- Product dependence is not hidden; it is explicitly measured.
- Reversal/breakdown is only assigned when the post-transition slope is significantly negative.
- Saturation is separated from true reversal.
- The analysis is stratified by aridity quartile.
- Tower validation is treated as the arbiter when products disagree.
- Trait analysis is conditional on successful gates.
