"""Conditional Phase 4 trait analysis workflow."""

from __future__ import annotations
import logging
import pandas as pd
import xarray as xr

from ..config import ProjectConfig
from ..io.loaders import load_traits
from ..utils import safe_to_csv
from ..models.traits import prepare_trait_frame, random_forest_trait_analysis
from ..models.bayesian import bayesian_trait_regression
from ..reporting.memos import write_phase4_memo

log = logging.getLogger(__name__)


def run_phase4(cfg: ProjectConfig) -> None:
    log.info("Running Phase 4")
    # Use best product from tower concordance if available; otherwise use all Gate 2 records.
    gate2 = pd.read_csv(cfg.file("tables", "gate2_pixel_results.csv"))
    try:
        conc = pd.read_csv(cfg.file("tables", "gate3_concordance_by_product.csv"))
        if len(conc):
            best = conc.sort_values("concordance_rate", ascending=False).iloc[0]
            response = gate2[(gate2["gpp_product"] == best["gpp_product"]) & (gate2["et_product"] == best["et_product"]) & (gate2["stress_definition"] == "zscore") & (gate2["growing_season"] == "gpp_threshold")].copy()
        else:
            response = gate2[(gate2["stress_definition"] == "zscore") & (gate2["growing_season"] == "gpp_threshold")].copy()
    except Exception:
        response = gate2[(gate2["stress_definition"] == "zscore") & (gate2["growing_season"] == "gpp_threshold")].copy()
    traits = load_traits(cfg).to_dataframe().reset_index()
    ancillary = xr.open_dataset(cfg.file("processed", "ancillary.nc"))
    met = xr.open_dataset(cfg.file("processed", "met.nc"))
    climate = ancillary.to_dataframe().reset_index()
    # Add MAP/MAT from met if not present.
    map_df = met["precipitation"].mean("time").rename("map").to_dataframe().reset_index()
    mat_df = met["temperature"].mean("time").rename("mat").to_dataframe().reset_index()
    climate = climate.merge(map_df, on=["lat", "lon"], how="left").merge(mat_df, on=["lat", "lon"], how="left")
    if "lai" in climate and "lai_mean" not in climate:
        climate["lai"] = climate["lai"]
    df = prepare_trait_frame(response, traits, climate)
    # Ensure configured columns exist.
    y_col = cfg.trait.get("dependent_metric", "slope_change")
    trait_cols = cfg.trait.get("predictors", ["psi50_abs", "isohydricity", "rooting_depth"])
    climate_cols = cfg.trait.get("climate_covariates", ["aridity_index", "map", "mat", "lai"])
    rf = random_forest_trait_analysis(df, y_col, trait_cols, climate_cols, int(cfg.trait.get("random_forest_estimators", 500)), cfg.seed)
    summary = pd.DataFrame([{k: v for k, v in rf.items() if isinstance(v, (float, int, str, bool))}])
    safe_to_csv(summary, cfg.file("tables", "phase4_trait_results.csv"))
    safe_to_csv(rf["permutation_importance"], cfg.file("tables", "phase4_permutation_importance.csv"))
    safe_to_csv(rf["shap_importance"], cfg.file("tables", "phase4_shap_importance.csv"))
    if cfg.trait.get("bayesian_enabled", False):
        bayes = bayesian_trait_regression(df, y_col, trait_cols + climate_cols, seed=cfg.seed)
        safe_to_csv(bayes, cfg.file("tables", "phase4_bayesian_trait_regression.csv"))
    write_phase4_memo(cfg, summary, rf["shap_importance"])
    log.info("Phase 4 complete")
