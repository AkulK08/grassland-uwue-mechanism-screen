"""Gate 1 response-shape characterization workflow."""

from __future__ import annotations
import logging
import pandas as pd
import xarray as xr

from ..config import ProjectConfig
from ..utils import safe_to_csv
from ..reporting.memos import write_gate1_memo
from ._fit import _spatial_points, sample_points, fit_pixels_for_combo

log = logging.getLogger(__name__)


def run_gate1(cfg: ProjectConfig) -> None:
    log.info("Running Gate 1")
    gp = cfg.products.get("gpp", ["MODIS"])[0]
    ep = cfg.products.get("et", ["MODIS"])[0]
    comp = xr.open_dataset(cfg.file("processed", f"components_{gp}_{ep}.nc"))
    stress = xr.open_dataset(cfg.file("processed", "stress_indices.nc"))
    met = xr.open_dataset(cfg.file("processed", "met.nc"))
    gpp = xr.open_dataset(cfg.file("processed", f"gpp_{gp}.nc"))["gpp"]
    points = _spatial_points(comp["log_wue"])
    points = sample_points(points, int(cfg.analysis.get("pilot_sample_size", 500)), cfg.seed)
    res = fit_pixels_for_combo(
        comp, stress, met, gpp, points, gp, ep, "zscore", "gpp_threshold",
        int(cfg.analysis.get("min_observations", 50)), int(cfg.analysis.get("bootstrap_iterations", 1000)), cfg.seed
    )
    anc = xr.open_dataset(cfg.file("processed", "ancillary.nc"))
    aridity_rows = anc["aridity_index"].to_dataframe().reset_index()
    res = res.merge(aridity_rows, on=["lat", "lon"], how="left")
    res["aridity_quartile"] = pd.qcut(res["aridity_index"], q=int(cfg.analysis.get("aridity_bins", 4)), labels=False, duplicates="drop") + 1
    safe_to_csv(res, cfg.file("tables", "gate1_pixel_results.csv"))
    summary = res.groupby(["response_class", "aridity_quartile"], dropna=False).size().reset_index(name="n")
    safe_to_csv(summary, cfg.file("tables", "gate1_aridity_summary.csv"))
    write_gate1_memo(cfg, res)
    log.info("Gate 1 complete")
