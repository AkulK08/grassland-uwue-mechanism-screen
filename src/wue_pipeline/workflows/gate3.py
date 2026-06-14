"""Gate 3 tower validation workflow."""

from __future__ import annotations
import logging
import pandas as pd

from ..config import ProjectConfig
from ..io.loaders import load_towers
from ..utils import safe_to_csv
from ..models.tower import prepare_tower_table, quality_screen_sites, tower_csi, classify_tower_sites, validate_tower_vs_satellite
from ..reporting.memos import write_gate3_memo

log = logging.getLogger(__name__)


def run_gate3(cfg: ProjectConfig) -> None:
    log.info("Running Gate 3")
    tower_raw = load_towers(cfg)
    tower = prepare_tower_table(tower_raw)
    q = cfg.analysis.get("quality", {})
    screen = quality_screen_sites(
        tower,
        min_years=int(q.get("tower_min_years", 3)),
        max_gap_fraction=float(q.get("max_gap_fraction", 0.30)),
        ebr_min=float(q.get("energy_balance_min", 0.70)),
        ebr_max=float(q.get("energy_balance_max", 1.30)),
    )
    safe_to_csv(screen, cfg.file("tables", "gate3_tower_quality_screen.csv"))
    good_sites = screen.loc[screen["passes_quality"], "site_id"]
    tower_good = tower[tower["site_id"].isin(good_sites)].copy()
    tower_good = tower_csi(tower_good)
    tower_class = classify_tower_sites(
        tower_good,
        min_obs=int(cfg.analysis.get("min_observations", 50)),
        n_boot=int(cfg.analysis.get("bootstrap_iterations", 1000)),
        seed=cfg.seed,
    )
    safe_to_csv(tower_class, cfg.file("tables", "gate3_tower_response_classes.csv"))
    sat = pd.read_csv(cfg.file("tables", "gate2_pixel_results.csv"))
    validation = validate_tower_vs_satellite(tower_class, sat)
    safe_to_csv(validation, cfg.file("tables", "gate3_tower_validation.csv"))
    if len(validation):
        concordance = validation.groupby(["gpp_product", "et_product"])["concordant"].mean().reset_index(name="concordance_rate")
    else:
        concordance = pd.DataFrame(columns=["gpp_product", "et_product", "concordance_rate"])
    safe_to_csv(concordance, cfg.file("tables", "gate3_concordance_by_product.csv"))
    write_gate3_memo(cfg, screen, tower_class, validation, concordance)
    log.info("Gate 3 complete")
