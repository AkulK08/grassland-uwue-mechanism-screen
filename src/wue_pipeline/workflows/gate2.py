"""Gate 2 cross-product robustness workflow."""

from __future__ import annotations
import logging
import pandas as pd
import xarray as xr

from ..config import ProjectConfig
from ..constants import GPP_PRODUCTS, ET_PRODUCTS, STRESS_DEFINITIONS, GROWING_SEASONS, ALGORITHM_DEPENDENCY
from ..utils import safe_to_csv
from ..models.robustness import response_fraction_table, evaluate_gate2_success, product_sensitivity_iqr
from ..reporting.memos import write_gate2_memo
from ._fit import _spatial_points, sample_points, fit_pixels_for_combo

log = logging.getLogger(__name__)


def run_gate2(cfg: ProjectConfig) -> None:
    log.info("Running Gate 2")
    gpps = cfg.products.get("gpp", GPP_PRODUCTS)
    ets = cfg.products.get("et", ET_PRODUCTS)
    stress_defs = cfg.products.get("stress_definitions", STRESS_DEFINITIONS)
    growing_seasons = cfg.products.get("growing_seasons", GROWING_SEASONS)
    stress = xr.open_dataset(cfg.file("processed", "stress_indices.nc"))
    met = xr.open_dataset(cfg.file("processed", "met.nc"))
    # Use common point sample across all combinations.
    first = xr.open_dataset(cfg.file("processed", f"components_{gpps[0]}_{ets[0]}.nc"))["log_wue"]
    points = _spatial_points(first)
    points = sample_points(points, int(cfg.analysis.get("pilot_sample_size", 500)), cfg.seed)
    all_rows = []
    for gp in gpps:
        gpp = xr.open_dataset(cfg.file("processed", f"gpp_{gp}.nc"))["gpp"]
        for ep in ets:
            comp = xr.open_dataset(cfg.file("processed", f"components_{gp}_{ep}.nc"))
            for sd in stress_defs:
                for gs in growing_seasons:
                    df = fit_pixels_for_combo(
                        comp, stress, met, gpp, points, gp, ep, sd, gs,
                        int(cfg.analysis.get("min_observations", 50)), int(cfg.analysis.get("bootstrap_iterations", 1000)), cfg.seed
                    )
                    all_rows.append(df)
    results = pd.concat(all_rows, ignore_index=True)
    anc = xr.open_dataset(cfg.file("processed", "ancillary.nc"))["aridity_index"].to_dataframe().reset_index()
    results = results.merge(anc, on=["lat", "lon"], how="left")
    results["aridity_quartile"] = pd.qcut(results["aridity_index"], q=int(cfg.analysis.get("aridity_bins", 4)), labels=False, duplicates="drop") + 1
    safe_to_csv(results, cfg.file("tables", "gate2_pixel_results.csv"))
    summary = response_fraction_table(results, ["gpp_product", "et_product", "stress_definition", "growing_season", "aridity_quartile"])
    safe_to_csv(summary, cfg.file("tables", "gate2_robustness_matrix.csv"))
    success = evaluate_gate2_success(summary)
    safe_to_csv(success, cfg.file("tables", "gate2_success_criteria.csv"))
    sensitivity = product_sensitivity_iqr(results)
    safe_to_csv(sensitivity, cfg.file("tables", "gate2_uncertainty_ranges.csv"))
    safe_to_csv(pd.DataFrame(ALGORITHM_DEPENDENCY), cfg.file("tables", "algorithm_dependency_table.csv"))
    write_gate2_memo(cfg, summary, success)
    log.info("Gate 2 complete")
