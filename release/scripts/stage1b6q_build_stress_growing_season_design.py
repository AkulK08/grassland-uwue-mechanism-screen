from pathlib import Path
from datetime import datetime
import json
import glob
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6q_stress_growing_season_design")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6q")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

RESPONSE = Path("data/processed/stage1b6p/strict_2x2_response_table_final13.csv")

CANDIDATE_PATTERNS = [
    "data/**/*.csv",
    "results/**/*.csv",
]

STRESS_KEYWORDS = [
    "vpd", "swc", "soil", "sm", "smap", "era5", "precip", "ppt",
    "water", "stress", "aridity", "tower"
]

def safe_read_head(path, n=5):
    try:
        return pd.read_csv(path, nrows=n)
    except Exception:
        return None

def find_candidate_files():
    files = []
    for pat in CANDIDATE_PATTERNS:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))

    rows = []
    for f in files:
        p = Path(f)
        low = str(p).lower()
        if not any(k in low for k in STRESS_KEYWORDS):
            continue
        head = safe_read_head(p, 20)
        if head is None:
            rows.append({
                "path": str(p),
                "read_ok": False,
                "columns": "",
                "size_bytes": p.stat().st_size if p.exists() else None,
            })
            continue
        cols = list(head.columns)
        cols_low = ";".join(cols).lower()
        rows.append({
            "path": str(p),
            "read_ok": True,
            "columns": ";".join(cols),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "has_point_id": "point_id" in cols or "site_id" in cols or "id" in cols,
            "has_date": "date" in cols or "time" in cols,
            "has_vpd_col": "vpd" in cols_low,
            "has_soil_moisture_col": any(x in cols_low for x in ["swc", "soil", "smap", "sm_root", "sm_surface", "soil_moisture"]),
            "has_precip_col": any(x in cols_low for x in ["precip", "ppt", "rain"]),
        })
    return pd.DataFrame(rows)

def standardize_driver_file(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    id_col = None
    for k in ["point_id", "site_id", "site", "id"]:
        if k in cols:
            id_col = cols[k]
            break

    date_col = None
    for k in ["date", "time", "datetime", "timestamp"]:
        if k in cols:
            date_col = cols[k]
            break

    if not id_col or not date_col:
        return None

    out = pd.DataFrame()
    out["point_id"] = df[id_col].astype(str)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Find VPD-ish column.
    vpd_col = None
    for c in df.columns:
        cl = c.lower()
        if "vpd" in cl:
            vpd_col = c
            break

    sm_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["swc", "soil_moisture", "smap", "sm_root", "rootzone", "soil_water", "theta"]):
            sm_col = c
            break

    precip_col = None
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["precip", "ppt", "rain"]):
            precip_col = c
            break

    keep_any = False
    if vpd_col:
        out["vpd"] = pd.to_numeric(df[vpd_col], errors="coerce")
        keep_any = True
    if sm_col:
        out["soil_moisture"] = pd.to_numeric(df[sm_col], errors="coerce")
        keep_any = True
    if precip_col:
        out["precip"] = pd.to_numeric(df[precip_col], errors="coerce")
        keep_any = True

    if not keep_any:
        return None

    out = out.dropna(subset=["date"])
    out = out.drop_duplicates(["point_id", "date"])
    return out

def zscore_by_site(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def rank_pct_by_site(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True)

# Load response.
errors = []
if not RESPONSE.exists():
    raise FileNotFoundError(f"Missing response table: {RESPONSE}")

resp = pd.read_csv(RESPONSE)
resp["date"] = pd.to_datetime(resp["date"]).dt.strftime("%Y-%m-%d")
resp["point_id"] = resp["point_id"].astype(str)
resp["month"] = pd.to_datetime(resp["date"]).dt.month
resp["year"] = pd.to_datetime(resp["date"]).dt.year
resp["doy"] = pd.to_datetime(resp["date"]).dt.dayofyear

# Candidate stress-driver inventory.
inventory = find_candidate_files()
inventory.to_csv(TAB / "Table_PRODUCT02ci_stress_driver_candidate_inventory.csv", index=False)

# Try to locate best joinable driver files.
driver_tables = []
used_driver_paths = []

if len(inventory):
    inv2 = inventory[
        inventory.get("read_ok", False).eq(True)
        & inventory.get("has_point_id", False).eq(True)
        & inventory.get("has_date", False).eq(True)
        & (inventory.get("has_vpd_col", False).eq(True) | inventory.get("has_soil_moisture_col", False).eq(True) | inventory.get("has_precip_col", False).eq(True))
    ].copy()

    for _, r in inv2.iterrows():
        try:
            d = standardize_driver_file(r["path"])
            if d is not None and len(d):
                driver_tables.append(d)
                used_driver_paths.append(r["path"])
        except Exception as e:
            errors.append({"stage": "driver_read", "path": r["path"], "error": repr(e)})

if driver_tables:
    drivers = pd.concat(driver_tables, ignore_index=True, sort=False)
    # Collapse duplicate source rows by site-date.
    agg = {}
    for c in ["vpd", "soil_moisture", "precip"]:
        if c in drivers.columns:
            agg[c] = "mean"
    drivers = drivers.groupby(["point_id", "date"], as_index=False).agg(agg)
else:
    drivers = pd.DataFrame(columns=["point_id", "date"])

drivers.to_csv(DATA / "stress_driver_candidates_joined_by_point_date.csv", index=False)

design = resp.merge(drivers, on=["point_id", "date"], how="left")

# Growing-season definitions.
# GS1: product-specific GPP >20% site-year-product peak.
design["gpp_site_year_peak"] = design.groupby(["point_id", "gpp_product", "year"])["gpp"].transform("max")
design["gs_gpp20_peak"] = design["gpp"] >= 0.20 * design["gpp_site_year_peak"]

# GS2: broad fixed grassland growing season. Northern Hemisphere Apr-Oct, Southern Hemisphere Oct-Apr.
# All current target sites are northern/equatorial-ish except none in southern lat in final 13, but compute from lat if present.
# Removed unused lat_lookup line that caused duplicate point_id reset_index error.
if "lat" in resp.columns:
    point_lat = resp.groupby("point_id")["lat"].first()
else:
    point_lat = pd.Series(index=design["point_id"].unique(), data=np.nan)

design["point_lat_for_fixed_gs"] = design["point_id"].map(point_lat)
north = design["point_lat_for_fixed_gs"].fillna(1).ge(0)
design["gs_fixed_climate"] = np.where(
    north,
    design["month"].between(4, 10),
    design["month"].isin([10, 11, 12, 1, 2, 3, 4])
)

# GS3: phenology/month fixed effects are not a row filter; keep all rows and include month FE.
design["gs_month_fe_available"] = True
design["month_fe"] = design["month"].astype("int64").astype(str).str.zfill(2)

# Stress definitions only if VPD + soil moisture exist.
has_vpd = "vpd" in design.columns and design["vpd"].notna().sum() > 0
has_sm = "soil_moisture" in design.columns and design["soil_moisture"].notna().sum() > 0

if has_vpd:
    design["vpd_z_site"] = design.groupby("point_id")["vpd"].transform(zscore_by_site)
    design["vpd_pct_site"] = design.groupby("point_id")["vpd"].transform(rank_pct_by_site)

if has_sm:
    design["soil_moisture_z_site"] = design.groupby("point_id")["soil_moisture"].transform(zscore_by_site)
    design["soil_moisture_pct_site"] = design.groupby("point_id")["soil_moisture"].transform(rank_pct_by_site)
    # Low soil moisture = high dryness.
    design["dryness_z_site"] = -design["soil_moisture_z_site"]
    design["dryness_pct_site"] = 1 - design["soil_moisture_pct_site"]

if has_vpd and has_sm:
    # 1. Equal-weight standardized VPD + dry soil stress.
    design["stress_equal_weight_z"] = (design["vpd_z_site"] + design["dryness_z_site"]) / 2

    # 2. Joint percentile stress: both high VPD and dry soil.
    design["stress_joint_percentile"] = (design["vpd_pct_site"] + design["dryness_pct_site"]) / 2
    design["stress_joint_top10"] = (design["vpd_pct_site"] >= 0.90) & (design["dryness_pct_site"] >= 0.90)
    design["stress_joint_top20"] = (design["vpd_pct_site"] >= 0.80) & (design["dryness_pct_site"] >= 0.80)

    # 3. Simple empirical copula-style product.
    design["stress_copula_product"] = design["vpd_pct_site"] * design["dryness_pct_site"]

    # 4. Interaction surface terms.
    design["stress_vpd_x_dryness_z"] = design["vpd_z_site"] * design["dryness_z_site"]
else:
    for c in [
        "stress_equal_weight_z", "stress_joint_percentile",
        "stress_joint_top10", "stress_joint_top20",
        "stress_copula_product", "stress_vpd_x_dryness_z"
    ]:
        design[c] = np.nan

# Save outputs.
design_out = DATA / "analysis_design_strict_2x2_with_gs_and_stress_candidates.csv"
design.to_csv(design_out, index=False)

stress_cols = [
    "vpd", "soil_moisture", "vpd_z_site", "dryness_z_site",
    "stress_equal_weight_z", "stress_joint_percentile",
    "stress_joint_top10", "stress_joint_top20",
    "stress_copula_product", "stress_vpd_x_dryness_z"
]
stress_cols_present = [c for c in stress_cols if c in design.columns]

gs_cols = ["gs_gpp20_peak", "gs_fixed_climate", "gs_month_fe_available", "month_fe"]

coverage_rows = []
for role, sub in design.groupby(["matrix_role", "gpp_product", "et_product"], dropna=False):
    matrix_role, gpp_product, et_product = role
    row = {
        "matrix_role": matrix_role,
        "gpp_product": gpp_product,
        "et_product": et_product,
        "n_rows": len(sub),
        "n_sites": sub["point_id"].nunique(),
        "n_dates": sub["date"].nunique(),
        "date_min": sub["date"].min(),
        "date_max": sub["date"].max(),
        "n_gs_gpp20": int(sub["gs_gpp20_peak"].sum()),
        "n_gs_fixed": int(sub["gs_fixed_climate"].sum()),
        "n_month_fe_nonmissing": int(sub["month_fe"].notna().sum()),
    }
    for c in stress_cols_present:
        row[f"n_nonmissing_{c}"] = int(sub[c].notna().sum())
    coverage_rows.append(row)

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02cj_design_coverage_by_product_combo.csv", index=False)

stress_status = pd.DataFrame([{
    "has_vpd": bool(has_vpd),
    "has_soil_moisture": bool(has_sm),
    "used_driver_paths": ";".join(used_driver_paths),
    "n_driver_rows_joined": int(len(drivers)),
    "n_design_rows": int(len(design)),
    "n_design_rows_with_vpd": int(design["vpd"].notna().sum()) if "vpd" in design.columns else 0,
    "n_design_rows_with_soil_moisture": int(design["soil_moisture"].notna().sum()) if "soil_moisture" in design.columns else 0,
    "stress_definitions_ready": bool(has_vpd and has_sm),
}])
stress_status.to_csv(TAB / "Table_PRODUCT02ck_stress_driver_join_status.csv", index=False)

gs_status = pd.DataFrame([{
    "gs_gpp20_peak_ready": True,
    "gs_fixed_climate_ready": True,
    "gs_month_fe_ready": True,
    "three_growing_season_definitions_ready": True,
}])
gs_status.to_csv(TAB / "Table_PRODUCT02cl_growing_season_status.csv", index=False)

err_df = pd.DataFrame(errors)
err_df.to_csv(TAB / "Table_PRODUCT02cm_design_errors.csv", index=False)

if bool(stress_status["stress_definitions_ready"].iloc[0]) and bool(gs_status["three_growing_season_definitions_ready"].iloc[0]) and len(err_df) == 0:
    verdict = "STRESS_AND_GROWING_SEASON_DESIGN_READY"
    blocking_next = False
elif bool(gs_status["three_growing_season_definitions_ready"].iloc[0]) and not bool(stress_status["stress_definitions_ready"].iloc[0]):
    verdict = "GROWING_SEASON_READY_BUT_STRESS_DRIVERS_MISSING"
    blocking_next = True
else:
    verdict = "DESIGN_NOT_READY"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_design_rows": int(len(design)),
    "growing_season_ready": bool(gs_status["three_growing_season_definitions_ready"].iloc[0]),
    "stress_definitions_ready": bool(stress_status["stress_definitions_ready"].iloc[0]),
    "has_vpd": bool(has_vpd),
    "has_soil_moisture": bool(has_sm),
    "n_errors": int(len(err_df)),
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "RUN_THRESHOLD_RESPONSE_MODELS" if not blocking_next else "ATTACH_OR_BUILD_VPD_SOIL_MOISTURE_DRIVERS",
}])
decision.to_csv(TAB / "Table_PRODUCT02cn_stress_growing_season_design_decision.csv", index=False)

report = []
report.append("# Stage 1B.6Q stress and growing-season design")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Stress-driver join status")
report.append("")
report.append("```text")
report.append(stress_status.to_string(index=False))
report.append("```")
report.append("")
report.append("## Growing-season status")
report.append("")
report.append("```text")
report.append(gs_status.to_string(index=False))
report.append("```")
report.append("")
report.append("## Design coverage by product combo")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage.")
report.append("```")
report.append("")
report.append("## Candidate stress-driver inventory")
report.append("")
report.append("```text")
report.append(inventory.head(80).to_string(index=False) if len(inventory) else "No candidate stress-driver files found.")
report.append("```")
report.append("")
report.append("## Errors")
report.append("")
report.append("```text")
report.append(err_df.to_string(index=False) if len(err_df) else "No errors.")
report.append("```")
report.append("")
report.append("## Output")
report.append("")
report.append(f"- Design table: `{design_out}`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Do not run the final compound-stress response model unless both VPD and soil-moisture drivers are joined. Growing-season definitions can be ready before stress drivers.")
report.append("")

(TXT / "STAGE1B6Q_STRESS_GROWING_SEASON_DESIGN_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6Q_stress_growing_season_design",
    "status": verdict,
    "blocking_next_stage": bool(blocking_next),
    "outputs": {
        "design_table": str(design_out),
        "stress_status": str(TAB / "Table_PRODUCT02ck_stress_driver_join_status.csv"),
        "gs_status": str(TAB / "Table_PRODUCT02cl_growing_season_status.csv"),
        "decision": str(TAB / "Table_PRODUCT02cn_stress_growing_season_design_decision.csv"),
        "report": str(TXT / "STAGE1B6Q_STRESS_GROWING_SEASON_DESIGN_REPORT.md"),
    }
}
(TAB / "STAGE1B6Q_STRESS_GROWING_SEASON_DESIGN_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", design_out)
print("WROTE", TAB / "Table_PRODUCT02cj_design_coverage_by_product_combo.csv")
print("WROTE", TAB / "Table_PRODUCT02ck_stress_driver_join_status.csv")
print("WROTE", TAB / "Table_PRODUCT02cl_growing_season_status.csv")
print("WROTE", TAB / "Table_PRODUCT02cn_stress_growing_season_design_decision.csv")
print("WROTE", TXT / "STAGE1B6Q_STRESS_GROWING_SEASON_DESIGN_REPORT.md")
