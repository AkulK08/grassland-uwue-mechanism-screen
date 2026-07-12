from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import glob

OUT = Path("results/stage1b6k_parallel_mentor_readiness")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

def exists_any(patterns):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))
    hits = sorted(set(hits))
    return hits

def file_status(patterns, min_count=1):
    hits = exists_any(patterns)
    return {
        "exists": len(hits) >= min_count,
        "n_hits": len(hits),
        "examples": "; ".join(hits[:8]),
    }

checks = []

def add(requirement, category, status, source_plan, blocking, notes=""):
    checks.append({
        "category": category,
        "mentor_requirement": requirement,
        "status": status,
        "source_plan": source_plan,
        "blocking_for_final_claim": blocking,
        "notes": notes,
    })

# Product matrix checks.
direct_modis = {
    "MODIS_GPP_MOD17": file_status(["data/raw_local/no_gee_direct_point_extract_full/MODIS_GPP_MOD17_FULL_direct_earthdata_point_samples.csv"]),
    "MODIS_ET_MOD16": file_status(["data/raw_local/no_gee_direct_point_extract_full/MODIS_ET_MOD16_FULL_direct_earthdata_point_samples.csv"]),
    "MODIS_LAI_MCD15": file_status(["data/raw_local/no_gee_direct_point_extract_full/MODIS_LAI_MCD15_FULL_direct_earthdata_point_samples.csv"]),
    "MCD64A1_BURNED_AREA": file_status(["data/raw_local/no_gee_direct_point_extract_full/MCD64A1_BURNED_AREA_FULL_direct_earthdata_point_samples.csv"]),
}

gosif = file_status([
    "data/raw/tower_centered_phase19/agents/gosif_tower13_point_timeseries.csv",
    "data/raw/gosif_gpp_v2/**/*.tif*",
    "data/raw/gosif/**/*.tif*",
])
gleam = file_status([
    "data/raw/tower_centered_phase19/agents/gleam_tower13_point_timeseries.csv",
    "data/raw/gleam/**/*.nc",
])
pml = file_status([
    "data/**/*PML*.nc",
    "data/**/*pml*.nc",
    "data/**/*PML*.csv",
    "data/**/*pml*.csv",
])
era5 = file_status([
    "data/**/*era5*.nc",
    "data/**/*ERA5*.nc",
    "/Users/me/Downloads/untitled folder/data_raw/era5/*.nc",
])
smap = file_status([
    "data/raw/smap_8day/**/*.h5",
    "data/raw/smap/**/*.h5",
    "data/**/*SMAP*.h5",
])
tower = file_status([
    "data/raw/towers/**/*.zip",
    "data/raw/towers/**/*.csv",
    "results/tower_validation_broad_inventory/**/*.csv",
])
traits_p50 = file_status([
    "data/external/**/*P50*.nc",
    "data/external/**/*p50*.nc",
    "data/external/**/*MDF_P50*.nc",
])
traits_iso = file_status([
    "data/external/**/*isohydricity*.nc",
    "data/external/**/*Isohydricity*.nc",
])
traits_root = file_status([
    "data/external/**/*rooting*.nc",
    "data/external/**/*Rooting*.nc",
])
soilgrids = file_status([
    "data/external/soilgrids_texture_by_point.csv",
    "data/**/*soilgrids*.csv",
    "data/**/*SoilGrids*.csv",
])
aridity = file_status([
    "data/external/**/*aridity*.zip",
    "data/external/**/*aridity*.tif",
    "data/external/**/*Global-AI*.zip",
])
landcover = file_status([
    "data/**/*mcd12*.tif",
    "data/**/*MCD12*.tif",
    "/Users/me/Downloads/untitled folder/data_raw/modis/mcd12*.tif",
])
phase19 = file_status([
    "results/tower_centered_phase19_no_gee/tables/Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv",
    "results/tower_centered_phase19_no_gee/tables/Table123_no_gee_validation_summary.csv",
])
stage1b6j_decision = file_status([
    "results/stage1b6j_full_direct_earthdata_extract/tables/Table_PRODUCT02bj_full_direct_completion_decision.csv"
])

add("Primary grassland/savanna/open target set locked", "target_design", "DONE",
    "Stage 1B.6G primary 13 target lock", False,
    "Primary inference is not all49 and not selected only by strong effect.")

add("Strict GRA sensitivity set locked", "target_design", "DONE",
    "Stage 1B.6G strict GRA sensitivity lock", False,
    "Used as sensitivity, not main sample.")

add("All49 broad tower set not used as primary", "target_design", "DONE",
    "Stage 1B.6G all49 contrast-only lock", False,
    "Prevents dilution of grassland question.")

add("MODIS GPP available for product matrix", "product_matrix",
    "DONE" if direct_modis["MODIS_GPP_MOD17"]["exists"] else "PENDING_STAGE_1B6J",
    "Direct Earthdata HDF point extraction, no GEE/AppEEARS",
    not direct_modis["MODIS_GPP_MOD17"]["exists"],
    direct_modis["MODIS_GPP_MOD17"]["examples"])

add("MODIS ET available for product matrix", "product_matrix",
    "DONE" if direct_modis["MODIS_ET_MOD16"]["exists"] else "PENDING_STAGE_1B6J",
    "Direct Earthdata HDF point extraction, no GEE/AppEEARS",
    not direct_modis["MODIS_ET_MOD16"]["exists"],
    direct_modis["MODIS_ET_MOD16"]["examples"])

add("GOSIF GPP available for product matrix", "product_matrix",
    "DONE" if gosif["exists"] else "MISSING",
    "Existing local GOSIF tower-centered or raw files",
    not gosif["exists"],
    gosif["examples"])

add("GLEAM ET available for product matrix", "product_matrix",
    "DONE" if gleam["exists"] else "MISSING",
    "Existing local GLEAM tower-centered or raw files",
    not gleam["exists"],
    gleam["examples"])

add("PML GPP/ET available for product matrix", "product_matrix",
    "CHECK_NEEDED" if pml["exists"] else "MISSING",
    "Existing local PML files; must verify whether demo/processed vs full product",
    not pml["exists"],
    pml["examples"])

add("3x3 matrix can run after MODIS/MCD completion", "product_matrix",
    "PENDING_STAGE_1B6J",
    "MODIS direct + GOSIF + GLEAM + PML",
    True,
    "Cannot claim full 3x3 until MODIS direct extraction and PML verification are complete.")

add("Four stress definitions prepared", "stress_metrics",
    "PREP_REQUIRED",
    "Tower VPD/SWC/precip plus ERA5/SMAP sensitivity",
    True,
    "equal-weight z stress; percentile joint stress; copula/joint stress; VPD×SM interaction surface.")

add("ERA5 available for climate/VPD sensitivity", "stress_metrics",
    "CHECK_NEEDED" if era5["exists"] else "MISSING",
    "Existing local ERA5 files",
    not era5["exists"],
    era5["examples"])

add("SMAP available for soil-moisture validation/sensitivity", "stress_metrics",
    "CHECK_NEEDED" if smap["exists"] else "MISSING",
    "Existing local SMAP files",
    not smap["exists"],
    smap["examples"])

add("Three growing-season definitions prepared", "growing_season",
    "PREP_REQUIRED",
    "GPP-threshold season, climate/fixed season, phenology/month FE",
    True,
    "LAI direct extraction helps canopy/phenology covariates but GPP-based and fixed-month seasons can be implemented immediately.")

add("LAI/canopy covariate available", "qa_covariates",
    "DONE" if direct_modis["MODIS_LAI_MCD15"]["exists"] else "PENDING_STAGE_1B6J",
    "Direct Earthdata MCD15A2H point extraction",
    not direct_modis["MODIS_LAI_MCD15"]["exists"],
    direct_modis["MODIS_LAI_MCD15"]["examples"])

add("Burned-area exclusion available", "qa_covariates",
    "DONE" if direct_modis["MCD64A1_BURNED_AREA"]["exists"] else "PENDING_STAGE_1B6J",
    "Direct Earthdata MCD64A1 point extraction",
    not direct_modis["MCD64A1_BURNED_AREA"]["exists"],
    direct_modis["MCD64A1_BURNED_AREA"]["examples"])

add("Landcover filter available", "qa_covariates",
    "CHECK_NEEDED" if landcover["exists"] else "MISSING",
    "Local MCD12Q1 landcover",
    not landcover["exists"],
    landcover["examples"])

add("Aridity stratification available", "qa_covariates",
    "CHECK_NEEDED" if aridity["exists"] else "MISSING",
    "Local aridity product",
    not aridity["exists"],
    aridity["examples"])

add("Tower response validation available", "tower_validation",
    "DONE" if tower["exists"] or phase19["exists"] else "MISSING",
    "Existing AmeriFlux/FLUXNET tower validation outputs",
    not (tower["exists"] or phase19["exists"]),
    tower["examples"] + " " + phase19["examples"])

add("Tower-vs-satellite arbitration ready", "tower_validation",
    "PENDING_STAGE_1B6J",
    "Run after full MODIS/MCD direct extraction and PML verification",
    True,
    "Existing GOSIF/GLEAM partial validation is not enough for final mentor claim.")

add("P50/xylem trait available", "trait_mechanism",
    "CHECK_NEEDED" if traits_p50["exists"] else "MISSING",
    "Local trait raster/table",
    not traits_p50["exists"],
    traits_p50["examples"])

add("Isohydricity trait available", "trait_mechanism",
    "CHECK_NEEDED" if traits_iso["exists"] else "MISSING",
    "Local trait raster/table",
    not traits_iso["exists"],
    traits_iso["examples"])

add("Rooting depth trait available", "trait_mechanism",
    "CHECK_NEEDED" if traits_root["exists"] else "MISSING",
    "Local trait raster/table",
    not traits_root["exists"],
    traits_root["examples"])

add("Soil texture controls available", "trait_mechanism",
    "DONE" if soilgrids["exists"] else "MISSING",
    "Existing SoilGrids point table acceptable for tower-centered trait model",
    not soilgrids["exists"],
    soilgrids["examples"])

add("Hierarchical trait model with climate/soil adjustment", "trait_mechanism",
    "PREP_REQUIRED",
    "Run after full response table and covariates are joined",
    True,
    "Must not claim causal proof; report climate-residual trait contribution and uncertainty.")

add("Final gridded MODIS xarray/Zarr cube", "optional_gridded_extension",
    "NOT_REQUIRED_FOR_TOWER_CENTERED_VALIDATION",
    "Storage-dependent extension only",
    False,
    "Only required for mapped/global pixel claims, not for tower-centered mentor validation.")

df = pd.DataFrame(checks)
df.to_csv(TAB / "Table_PRODUCT02bk_parallel_mentor_requirement_readiness.csv", index=False)

summary = df.groupby(["category", "status"]).size().reset_index(name="n")
summary.to_csv(TAB / "Table_PRODUCT02bl_parallel_mentor_readiness_summary.csv", index=False)

blocking = df[df["blocking_for_final_claim"].astype(bool)].copy()
blocking.to_csv(TAB / "Table_PRODUCT02bm_current_blockers_for_final_claim.csv", index=False)

report = []
report.append("# Stage 1B.6K parallel mentor-readiness prep")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Meaning")
report.append("")
report.append("This stage does not use GEE or AppEEARS and does not start another download. It audits what can be prepared while Stage 1B.6J direct Earthdata extraction is running.")
report.append("")
report.append("## Readiness summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Current blockers for final strict claim")
report.append("")
report.append("```text")
report.append(blocking.to_string(index=False) if len(blocking) else "No current blockers.")
report.append("```")
report.append("")
report.append("## Full mentor requirement table")
report.append("")
report.append("```text")
report.append(df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Do not make the final full mentor-standard claim until all blocking rows are resolved. The gridded MODIS cube is not blocking unless the claim becomes spatially gridded/global.")
report.append("")

(TXT / "STAGE1B6K_PARALLEL_MENTOR_READINESS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6K_parallel_mentor_readiness",
    "status": "complete",
    "n_requirements": int(len(df)),
    "n_blockers": int(len(blocking)),
    "outputs": {
        "readiness": str(TAB / "Table_PRODUCT02bk_parallel_mentor_requirement_readiness.csv"),
        "summary": str(TAB / "Table_PRODUCT02bl_parallel_mentor_readiness_summary.csv"),
        "blockers": str(TAB / "Table_PRODUCT02bm_current_blockers_for_final_claim.csv"),
        "report": str(TXT / "STAGE1B6K_PARALLEL_MENTOR_READINESS_REPORT.md"),
    }
}

(TAB / "STAGE1B6K_PARALLEL_MENTOR_READINESS_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bk_parallel_mentor_requirement_readiness.csv")
print("WROTE", TAB / "Table_PRODUCT02bl_parallel_mentor_readiness_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02bm_current_blockers_for_final_claim.csv")
print("WROTE", TXT / "STAGE1B6K_PARALLEL_MENTOR_READINESS_REPORT.md")
