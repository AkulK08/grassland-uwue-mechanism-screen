"""Preprocessing workflow."""

from __future__ import annotations
import logging
import xarray as xr

from ..config import ProjectConfig
from ..io.loaders import load_gpp, load_et, load_met, load_masks, load_ancillary, load_co2
from ..constants import GPP_PRODUCTS, ET_PRODUCTS
from ..processing.grid import standardize_dims
from ..processing.stress import stress_dataset
from ..processing.masks import combined_analysis_mask
from ..processing.wue import compute_log_components, apply_co2_correction
from ..utils import safe_to_netcdf

log = logging.getLogger(__name__)


def run_preprocess(cfg: ProjectConfig) -> None:
    log.info("Starting preprocessing")
    gpps = cfg.products.get("gpp", GPP_PRODUCTS)
    ets = cfg.products.get("et", ET_PRODUCTS)
    met = standardize_dims(load_met(cfg))
    masks = standardize_dims(load_masks(cfg))
    ancillary = standardize_dims(load_ancillary(cfg))
    co2 = load_co2(cfg)
    analysis_mask = combined_analysis_mask(
        masks,
        cfg.spatial.get("grassland_classes", [10]),
        cfg.spatial.get("biome_extension_classes", [8, 9]),
        float(cfg.spatial.get("irrigation_fraction_max", 0.10)),
        include_extensions=False,
    )
    stress = stress_dataset(met["vpd"], met["soil_moisture"])
    stress = stress.where(analysis_mask)
    safe_to_netcdf(stress, cfg.file("processed", "stress_indices.nc"))
    safe_to_netcdf(xr.Dataset({"analysis_mask": analysis_mask}), cfg.file("processed", "analysis_mask.nc"))
    safe_to_netcdf(ancillary.where(analysis_mask), cfg.file("processed", "ancillary.nc"))
    safe_to_netcdf(met.where(analysis_mask), cfg.file("processed", "met.nc"))
    for gp in gpps:
        gpp = standardize_dims(load_gpp(cfg, gp)).where(analysis_mask)
        safe_to_netcdf(xr.Dataset({"gpp": gpp}), cfg.file("processed", f"gpp_{gp}.nc"))
    for ep in ets:
        et = standardize_dims(load_et(cfg, ep)).where(analysis_mask)
        safe_to_netcdf(xr.Dataset({"et": et}), cfg.file("processed", f"et_{ep}.nc"))
    # Precompute log components for each combination.
    for gp in gpps:
        gpp = xr.open_dataset(cfg.file("processed", f"gpp_{gp}.nc"))["gpp"]
        for ep in ets:
            et = xr.open_dataset(cfg.file("processed", f"et_{ep}.nc"))["et"]
            gpp, et = xr.align(gpp, et, join="inner")
            comp = compute_log_components(gpp, et, float(cfg.analysis.get("gpp_floor", 0.05)), float(cfg.analysis.get("et_floor", 0.1)))
            comp["log_wue_co2"] = apply_co2_correction(comp["log_wue"], co2, float(cfg.analysis.get("co2_reference_ppm", 398.0)))
            comp = comp.where(analysis_mask)
            safe_to_netcdf(comp, cfg.file("processed", f"components_{gp}_{ep}.nc"))
    log.info("Preprocessing complete")
