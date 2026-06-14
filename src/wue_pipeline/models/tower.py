"""Flux tower quality screening and tower-satellite validation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..constants import LAMBDA_MJ_PER_KG, SECONDS_PER_DAY
from .segmented import segmented_with_uncertainty


def prepare_tower_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    # Convert LE W m-2 to approximate ET mm per 8-day composite.
    # W m-2 = J s-1 m-2. For daily water equivalent: LE * seconds / lambda_J_per_kg.
    lambda_j = LAMBDA_MJ_PER_KG * 1e6
    out["ET_8day"] = out["LE_F_MDS"].astype(float) * SECONDS_PER_DAY * 8.0 / lambda_j
    out["GPP_8day"] = out["GPP_NT_VUT_REF"].astype(float)
    out["log_wue"] = np.log(out["GPP_8day"].where(out["GPP_8day"] > 0)) - np.log(out["ET_8day"].where(out["ET_8day"] > 0))
    return out


def quality_screen_sites(df: pd.DataFrame, min_years: int = 3, max_gap_fraction: float = 0.30, ebr_min: float = 0.70, ebr_max: float = 1.30) -> pd.DataFrame:
    rows = []
    d = df.copy()
    d["year"] = pd.to_datetime(d["time"]).dt.year
    if "LE_F_MDS_QC" not in d:
        d["LE_F_MDS_QC"] = 0
    for site, g in d.groupby("site_id"):
        years_ok = g.groupby("year").filter(lambda x: (x["LE_F_MDS_QC"] > 0).mean() <= max_gap_fraction)["year"].nunique()
        gap_fraction = float((g["LE_F_MDS_QC"] > 0).mean())
        energy_available = {"H_F_MDS", "LE_F_MDS", "NETRAD", "G"}.issubset(g.columns)
        if energy_available:
            denom = (g["NETRAD"].astype(float) - g["G"].astype(float)).replace(0, np.nan)
            ebr = ((g["H_F_MDS"].astype(float) + g["LE_F_MDS"].astype(float)) / denom).replace([np.inf, -np.inf], np.nan)
            ebr_median = float(ebr.median())
        else:
            ebr_median = np.nan
        passes = years_ok >= min_years and gap_fraction <= max_gap_fraction and (np.isnan(ebr_median) or (ebr_min <= ebr_median <= ebr_max))
        rows.append({"site_id": site, "years_ok": years_ok, "gap_fraction": gap_fraction, "energy_balance_ratio": ebr_median, "passes_quality": passes})
    return pd.DataFrame(rows)


def tower_csi(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SWC_F_MDS" not in out:
        out["SWC_F_MDS"] = np.nan
    pieces = []
    for site, g in out.groupby("site_id"):
        g = g.copy()
        g["vpd_z"] = (g["VPD_F_MDS"] - g["VPD_F_MDS"].mean()) / g["VPD_F_MDS"].std()
        g["sm_z"] = (g["SWC_F_MDS"] - g["SWC_F_MDS"].mean()) / g["SWC_F_MDS"].std()
        g["csi_zscore"] = 0.5 * g["vpd_z"] - 0.5 * g["sm_z"]
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def classify_tower_sites(df: pd.DataFrame, min_obs: int, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for site, g in df.groupby("site_id"):
        fit = segmented_with_uncertainty(g["csi_zscore"].values, g["log_wue"].values, min_obs=min_obs, n_boot=n_boot, seed=seed)
        row = fit.to_dict()
        row["site_id"] = site
        row["latitude"] = float(g["latitude"].iloc[0])
        row["longitude"] = float(g["longitude"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def match_towers_to_pixels(towers: pd.DataFrame, pixel_results: pd.DataFrame) -> pd.DataFrame:
    if not {"lat", "lon"}.issubset(pixel_results.columns):
        # Fall back to grid indices if lat/lon absent.
        pixel_results = pixel_results.copy()
        pixel_results["lat"] = pixel_results.get("lat", pixel_results.get("y", 0))
        pixel_results["lon"] = pixel_results.get("lon", pixel_results.get("x", 0))
    coords = pixel_results[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(coords[["lat", "lon"]].to_numpy())
    dist, idx = tree.query(towers[["latitude", "longitude"]].to_numpy())
    match = towers.copy()
    match["matched_lat"] = coords.iloc[idx]["lat"].values
    match["matched_lon"] = coords.iloc[idx]["lon"].values
    match["match_distance_deg"] = dist
    return match


def validate_tower_vs_satellite(tower_class: pd.DataFrame, sat_results: pd.DataFrame, product_family_filter: dict | None = None) -> pd.DataFrame:
    s = sat_results.copy()
    if product_family_filter:
        for k, v in product_family_filter.items():
            s = s[s[k] == v]
    matched = match_towers_to_pixels(tower_class, s)
    rows = []
    for _, tw in matched.iterrows():
        candidates = s[(np.isclose(s["lat"], tw["matched_lat"])) & (np.isclose(s["lon"], tw["matched_lon"]))]
        for _, sat in candidates.iterrows():
            rows.append({
                "site_id": tw["site_id"],
                "tower_class": tw["response_class"],
                "satellite_class": sat["response_class"],
                "concordant": tw["response_class"] == sat["response_class"],
                "gpp_product": sat.get("gpp_product"),
                "et_product": sat.get("et_product"),
                "stress_definition": sat.get("stress_definition"),
                "growing_season": sat.get("growing_season"),
                "tower_post_slope": tw.get("post_slope"),
                "satellite_post_slope": sat.get("post_slope"),
                "match_distance_deg": tw["match_distance_deg"],
            })
    return pd.DataFrame(rows)
