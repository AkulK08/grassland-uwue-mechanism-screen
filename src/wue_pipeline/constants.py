"""Project-wide constants and enumerations."""

from __future__ import annotations

GPP_PRODUCTS = ["MODIS", "GOSIF", "PML"]
ET_PRODUCTS = ["MODIS", "GLEAM", "PML"]
STRESS_DEFINITIONS = ["zscore", "percentile", "copula", "interaction"]
GROWING_SEASONS = ["gpp_threshold", "climate_threshold", "month_fixed_effects"]
RESPONSE_CLASSES = ["enhancement", "saturation", "reversal", "inconclusive", "insufficient_data"]

ALGORITHM_DEPENDENCY = [
    {
        "product": "MODIS_MOD17",
        "family": "GPP",
        "resolution": "500 m / 8-day",
        "formulation": "light-use-efficiency",
        "uses_vpd": True,
        "uses_soil_moisture": False,
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": True,
        "artifact_risk": "Direct VPD scalar in GPP algorithm; VPD-response can be partly algorithmic.",
    },
    {
        "product": "GOSIF_v2",
        "family": "GPP",
        "resolution": "0.05 deg / 8-day",
        "formulation": "solar-induced-fluorescence upscaling",
        "uses_vpd": "indirect/upscaling",
        "uses_soil_moisture": "indirect/upscaling",
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": False,
        "artifact_risk": "More independent carbon cross-check, but upscaling still uses environmental predictors.",
    },
    {
        "product": "PML_V2_GPP",
        "family": "GPP",
        "resolution": "500 m / 8-day",
        "formulation": "Penman-Monteith-Leuning coupled GPP-ET",
        "uses_vpd": True,
        "uses_soil_moisture": True,
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": True,
        "artifact_risk": "Coupled conductance model can impose carbon-water covariance.",
    },
    {
        "product": "MODIS_MOD16",
        "family": "ET",
        "resolution": "500 m / 8-day",
        "formulation": "Penman-Monteith ET",
        "uses_vpd": True,
        "uses_soil_moisture": False,
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": True,
        "artifact_risk": "Aerodynamic demand term can imprint VPD response into ET.",
    },
    {
        "product": "GLEAM_v3_8a",
        "family": "ET",
        "resolution": "0.25 deg / daily",
        "formulation": "Priestley-Taylor with evaporative stress",
        "uses_vpd": False,
        "uses_soil_moisture": True,
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": False,
        "artifact_risk": "Structurally distinct ET discriminator; coarser resolution adds mixed-pixel uncertainty.",
    },
    {
        "product": "PML_V2_ET",
        "family": "ET",
        "resolution": "500 m / 8-day",
        "formulation": "Penman-Monteith-Leuning coupled GPP-ET",
        "uses_vpd": True,
        "uses_soil_moisture": True,
        "uses_temperature": True,
        "uses_radiation": True,
        "uses_lai_fapar": True,
        "artifact_risk": "VPD and conductance terms can impose stress response in both numerator and denominator.",
    },
]

TOWER_REQUIRED_COLUMNS = [
    "site_id", "time", "latitude", "longitude", "igbp", "GPP_NT_VUT_REF", "LE_F_MDS",
    "H_F_MDS", "NETRAD", "G", "VPD_F_MDS", "TA_F_MDS"
]

LAMBDA_MJ_PER_KG = 2.45
SECONDS_PER_DAY = 86400.0
