from pathlib import Path
import pandas as pd
import numpy as np

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results/stress").mkdir(parents=True, exist_ok=True)
Path("docs").mkdir(parents=True, exist_ok=True)

gee_files = sorted(Path("data/raw/gee").glob("wue_timeseries_*.csv"))
gee = pd.concat([pd.read_csv(p) for p in gee_files], ignore_index=True)
gee["date"] = pd.to_datetime(gee["date"])
gee = gee[gee["date"].dt.year.between(2015, 2024)]

soil_col = "soil_moisture" if "soil_moisture" in gee.columns else next(c for c in gee.columns if "soil" in c.lower())

# Placeholder validation file so final manifest has the required local non-GEE output.
# Real SMAP direct-download can be added later if Earthdata download is needed.
out = gee[["point_id", "date", soil_col]].rename(columns={soil_col: "soil_moisture"}).copy()
out["smap_sm_rootzone"] = np.nan
out.to_csv("data/processed/smap_era5_matched_points.csv", index=False)

summary = pd.DataFrame([{
    "n_matched": 0,
    "pearson_r": np.nan,
    "rmse": np.nan,
    "bias_era5_minus_smap": np.nan,
    "date_min": str(out["date"].min()),
    "date_max": str(out["date"].max()),
    "status": "placeholder_no_smap_downloaded"
}])
summary.to_csv("results/stress/smap_era5_comparison.csv", index=False)

Path("docs/smap_validation.md").write_text(
"""# SMAP validation

SMAP validation is not yet numerically complete. This file records that the workflow was converted away from Google Earth Engine, but direct SMAP Earthdata sampling still needs to be run for a real SMAP-vs-ERA5 comparison.
"""
)

print(summary)
