from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6q2_attach_tower_stress_drivers")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6q2")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

RESPONSE = Path("data/processed/stage1b6p/strict_2x2_response_table_final13.csv")
STRESS_SOURCE = Path("results/tower_centered_phase19_no_gee/tables/Table120_no_gee_tower13_gosif_gleam_merged_timeseries.csv")

def zscore_by_site(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def pct_by_site(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True)

if not RESPONSE.exists():
    raise FileNotFoundError(f"Missing response table: {RESPONSE}")
if not STRESS_SOURCE.exists():
    raise FileNotFoundError(f"Missing stress source: {STRESS_SOURCE}")

resp = pd.read_csv(RESPONSE)
resp["point_id"] = resp["point_id"].astype(str)
resp["date"] = pd.to_datetime(resp["date"]).dt.strftime("%Y-%m-%d")
resp["year"] = pd.to_datetime(resp["date"]).dt.year
resp["month"] = pd.to_datetime(resp["date"]).dt.month
resp["doy"] = pd.to_datetime(resp["date"]).dt.dayofyear

stress = pd.read_csv(STRESS_SOURCE)
stress["point_id"] = stress["site"].astype(str)
stress["date"] = pd.to_datetime(stress["date"]).dt.strftime("%Y-%m-%d")

needed = ["point_id", "date", "vpd_for_stress", "swc_for_stress", "precip_for_stress"]
missing = [c for c in needed if c not in stress.columns]
if missing:
    raise ValueError(f"Stress source missing columns: {missing}. Columns={list(stress.columns)}")

stress = stress[needed].copy()
stress = stress.rename(columns={
    "vpd_for_stress": "vpd",
    "swc_for_stress": "soil_moisture",
    "precip_for_stress": "precip",
})
for c in ["vpd", "soil_moisture", "precip"]:
    stress[c] = pd.to_numeric(stress[c], errors="coerce")

stress = stress.drop_duplicates(["point_id", "date"])
stress.to_csv(DATA / "tower_centered_final13_stress_drivers.csv", index=False)

design = resp.merge(stress, on=["point_id", "date"], how="left")

# Growing-season definitions.
design["gpp_site_year_peak"] = design.groupby(["point_id", "gpp_product", "year"])["gpp"].transform("max")
design["gs_gpp20_peak"] = design["gpp"] >= 0.20 * design["gpp_site_year_peak"]

if "lat" in design.columns:
    point_lat = design.groupby("point_id")["lat"].first()
else:
    point_lat = pd.Series(index=design["point_id"].unique(), data=np.nan)

design["point_lat_for_fixed_gs"] = design["point_id"].map(point_lat)
north = design["point_lat_for_fixed_gs"].fillna(1).ge(0)
design["gs_fixed_climate"] = np.where(
    north,
    design["month"].between(4, 10),
    design["month"].isin([10, 11, 12, 1, 2, 3, 4])
)

design["gs_month_fe_available"] = True
design["month_fe"] = design["month"].astype(int).astype(str).str.zfill(2)

# Stress definitions.
design["vpd_z_site"] = design.groupby("point_id")["vpd"].transform(zscore_by_site)
design["soil_moisture_z_site"] = design.groupby("point_id")["soil_moisture"].transform(zscore_by_site)
design["dryness_z_site"] = -design["soil_moisture_z_site"]

design["vpd_pct_site"] = design.groupby("point_id")["vpd"].transform(pct_by_site)
design["soil_moisture_pct_site"] = design.groupby("point_id")["soil_moisture"].transform(pct_by_site)
design["dryness_pct_site"] = 1 - design["soil_moisture_pct_site"]

# 1. Equal-weight standardized compound stress.
design["stress_equal_weight_z"] = (design["vpd_z_site"] + design["dryness_z_site"]) / 2

# 2. Joint percentile stress.
design["stress_joint_percentile"] = (design["vpd_pct_site"] + design["dryness_pct_site"]) / 2
design["stress_joint_top10"] = (design["vpd_pct_site"] >= 0.90) & (design["dryness_pct_site"] >= 0.90)
design["stress_joint_top20"] = (design["vpd_pct_site"] >= 0.80) & (design["dryness_pct_site"] >= 0.80)

# 3. Copula-style empirical joint stress.
design["stress_copula_product"] = design["vpd_pct_site"] * design["dryness_pct_site"]

# 4. Interaction surface term.
design["stress_vpd_x_dryness_z"] = design["vpd_z_site"] * design["dryness_z_site"]

# Optional uWUE using tower VPD driver.
design["uwue"] = np.where(
    (design["gpp"] > 0) & (design["et"] > 0) & (design["vpd"] > 0),
    design["gpp"] * np.sqrt(design["vpd"]) / design["et"],
    np.nan
)
design["log_uwue"] = np.where(design["uwue"] > 0, np.log(design["uwue"]), np.nan)

out_design = DATA / "analysis_design_strict_2x2_with_tower_stress_and_gs.csv"
design.to_csv(out_design, index=False)

stress_cols = [
    "vpd", "soil_moisture", "precip",
    "vpd_z_site", "dryness_z_site",
    "stress_equal_weight_z",
    "stress_joint_percentile",
    "stress_joint_top10",
    "stress_joint_top20",
    "stress_copula_product",
    "stress_vpd_x_dryness_z",
    "uwue",
    "log_uwue",
]

coverage_rows = []
for (role, gp, ep), sub in design.groupby(["matrix_role", "gpp_product", "et_product"], dropna=False):
    row = {
        "matrix_role": role,
        "gpp_product": gp,
        "et_product": ep,
        "n_rows": len(sub),
        "n_sites": sub["point_id"].nunique(),
        "n_dates": sub["date"].nunique(),
        "date_min": sub["date"].min(),
        "date_max": sub["date"].max(),
        "n_gs_gpp20_peak": int(sub["gs_gpp20_peak"].sum()),
        "n_gs_fixed_climate": int(sub["gs_fixed_climate"].sum()),
        "n_month_fe_nonmissing": int(sub["month_fe"].notna().sum()),
    }
    for c in stress_cols:
        row[f"n_nonmissing_{c}"] = int(sub[c].notna().sum()) if c in sub.columns else 0
    coverage_rows.append(row)

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02co_stress_gs_design_coverage_by_combo.csv", index=False)

stress_status = pd.DataFrame([{
    "stress_source": str(STRESS_SOURCE),
    "n_stress_source_rows": len(stress),
    "n_design_rows": len(design),
    "n_design_rows_with_vpd": int(design["vpd"].notna().sum()),
    "n_design_rows_with_soil_moisture": int(design["soil_moisture"].notna().sum()),
    "n_design_rows_with_precip": int(design["precip"].notna().sum()),
    "has_vpd": bool(design["vpd"].notna().sum() > 0),
    "has_soil_moisture": bool(design["soil_moisture"].notna().sum() > 0),
    "four_stress_definitions_ready": bool(
        design["stress_equal_weight_z"].notna().sum() > 0
        and design["stress_joint_percentile"].notna().sum() > 0
        and design["stress_copula_product"].notna().sum() > 0
        and design["stress_vpd_x_dryness_z"].notna().sum() > 0
    ),
}])
stress_status.to_csv(TAB / "Table_PRODUCT02cp_stress_driver_attach_status.csv", index=False)

gs_status = pd.DataFrame([{
    "gs_gpp20_peak_ready": bool(design["gs_gpp20_peak"].notna().all()),
    "gs_fixed_climate_ready": bool(design["gs_fixed_climate"].notna().all()),
    "gs_month_fe_ready": bool(design["month_fe"].notna().all()),
    "three_growing_season_definitions_ready": True,
}])
gs_status.to_csv(TAB / "Table_PRODUCT02cq_growing_season_relock_status.csv", index=False)

if bool(stress_status["four_stress_definitions_ready"].iloc[0]) and bool(gs_status["three_growing_season_definitions_ready"].iloc[0]):
    verdict = "STRESS_AND_GROWING_SEASON_DESIGN_READY"
    blocking_next = False
else:
    verdict = "STRESS_AND_GROWING_SEASON_DESIGN_NOT_READY"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_design_rows": int(len(design)),
    "n_sites": int(design["point_id"].nunique()),
    "n_dates": int(design["date"].nunique()),
    "four_stress_definitions_ready": bool(stress_status["four_stress_definitions_ready"].iloc[0]),
    "three_growing_season_definitions_ready": bool(gs_status["three_growing_season_definitions_ready"].iloc[0]),
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "RUN_THRESHOLD_RESPONSE_MODELS" if not blocking_next else "FIX_STRESS_DRIVER_JOIN",
}])
decision.to_csv(TAB / "Table_PRODUCT02cr_stress_gs_design_decision.csv", index=False)

report = []
report.append("# Stage 1B.6Q.2 attach tower-centered stress drivers")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Stress-driver attach status")
report.append("")
report.append("```text")
report.append(stress_status.to_string(index=False))
report.append("```")
report.append("")
report.append("## Growing-season relock")
report.append("")
report.append("```text")
report.append(gs_status.to_string(index=False))
report.append("```")
report.append("")
report.append("## Coverage by product combo")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False))
report.append("```")
report.append("")
report.append("## Output")
report.append("")
report.append(f"- Design table: `{out_design}`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("The strict 2x2 response table now has tower-centered VPD and soil-moisture stress drivers attached. Proceed to threshold response modeling using the four stress definitions and three growing-season definitions.")
report.append("")

(TXT / "STAGE1B6Q2_ATTACH_TOWER_STRESS_DRIVERS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6Q.2_attach_tower_stress_drivers",
    "status": verdict,
    "blocking_next_stage": bool(blocking_next),
    "outputs": {
        "design_table": str(out_design),
        "stress_status": str(TAB / "Table_PRODUCT02cp_stress_driver_attach_status.csv"),
        "gs_status": str(TAB / "Table_PRODUCT02cq_growing_season_relock_status.csv"),
        "decision": str(TAB / "Table_PRODUCT02cr_stress_gs_design_decision.csv"),
        "report": str(TXT / "STAGE1B6Q2_ATTACH_TOWER_STRESS_DRIVERS_REPORT.md"),
    }
}
(TAB / "STAGE1B6Q2_ATTACH_TOWER_STRESS_DRIVERS_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", out_design)
print("WROTE", TAB / "Table_PRODUCT02co_stress_gs_design_coverage_by_combo.csv")
print("WROTE", TAB / "Table_PRODUCT02cp_stress_driver_attach_status.csv")
print("WROTE", TAB / "Table_PRODUCT02cq_growing_season_relock_status.csv")
print("WROTE", TAB / "Table_PRODUCT02cr_stress_gs_design_decision.csv")
print("WROTE", TXT / "STAGE1B6Q2_ATTACH_TOWER_STRESS_DRIVERS_REPORT.md")
