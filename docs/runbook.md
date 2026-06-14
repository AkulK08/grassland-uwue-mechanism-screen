# Runbook

## Demo

```bash
conda env create -f environment.yml
conda activate grassland_wue_nature
pip install -e .
wue run-all --config configs/demo.yaml --make-demo
```

## Production

1. Prepare real data on a common 8-day, 0.1-degree grid.
2. Copy `configs/production_template.yaml` to `configs/production.yaml`.
3. Fill every data path under `data_catalog`.
4. Run:

```bash
wue preprocess --config configs/production.yaml
wue gate1 --config configs/production.yaml
wue gate2 --config configs/production.yaml
wue gate3 --config configs/production.yaml
wue phase4 --config configs/production.yaml
wue figures --config configs/production.yaml
wue manuscript --config configs/production.yaml
```

## Key result checks

- Gate 1 should produce slope intervals and a response class for at least one product combination.
- Gate 2 should identify whether the same response class appears across at least two GPP products, two ET products, two stress metrics, and two growing-season definitions.
- Gate 3 should report product-family tower concordance.
- Phase 4 should report trait-unique variance after climate controls.
