from pathlib import Path
import json
import pandas as pd
from datetime import datetime

OUT = Path("results/stage1b5_no_gee_acquisition")
TAB = OUT / "tables"
TXT = OUT / "text"
DL = Path("data/raw_local/no_gee_downloads")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DL.mkdir(parents=True, exist_ok=True)

strict_path = Path("results/stage1b_strict_readiness_lock/tables/Table_PRODUCT02f_STRICT_READINESS_LOCK.csv")
if not strict_path.exists():
    raise FileNotFoundError("Run Stage 1B strict readiness lock first.")

strict = pd.read_csv(strict_path)

products = [
    {
        "product_group": "MODIS_GPP_MOD17",
        "official_product_options": "MOD17A2HGF.061 preferred; MOD17A2H.061 acceptable if gap-filled GF unavailable",
        "needed_for": "3x3 product matrix GPP",
        "current_status_from_strict_lock": strict.loc[strict.product_group.eq("MODIS_GPP_MOD17"), "status"].iloc[0],
        "local_source_now": strict.loc[strict.product_group.eq("MODIS_GPP_MOD17"), "best_path"].iloc[0],
        "no_gee_acquisition_route": "NASA Earthdata/LP DAAC through earthaccess or AppEEARS download; no GEE",
        "raw_gridded_required_for_full_xarray": True,
        "point_csv_acceptable_for_tower_only": True,
        "download_priority": 1,
        "target_dir": str(DL / "modis_mod17_gpp"),
    },
    {
        "product_group": "MODIS_ET_MOD16",
        "official_product_options": "MOD16A2GF.061 preferred; MOD16A2.061 acceptable if gap-filled GF unavailable",
        "needed_for": "3x3 product matrix ET",
        "current_status_from_strict_lock": strict.loc[strict.product_group.eq("MODIS_ET_MOD16"), "status"].iloc[0],
        "local_source_now": strict.loc[strict.product_group.eq("MODIS_ET_MOD16"), "best_path"].iloc[0],
        "no_gee_acquisition_route": "NASA Earthdata/LP DAAC through earthaccess or AppEEARS download; no GEE",
        "raw_gridded_required_for_full_xarray": True,
        "point_csv_acceptable_for_tower_only": True,
        "download_priority": 1,
        "target_dir": str(DL / "modis_mod16_et"),
    },
    {
        "product_group": "MODIS_LAI_MOD15",
        "official_product_options": "MCD15A2H.061 preferred for combined Terra+Aqua LAI/FPAR; MOD15A2H.061 acceptable",
        "needed_for": "growing-season definition, LAI covariate, canopy control",
        "current_status_from_strict_lock": strict.loc[strict.product_group.eq("MODIS_LAI_MOD15"), "status"].iloc[0],
        "local_source_now": strict.loc[strict.product_group.eq("MODIS_LAI_MOD15"), "best_path"].iloc[0] if pd.notna(strict.loc[strict.product_group.eq("MODIS_LAI_MOD15"), "best_path"].iloc[0]) else "",
        "no_gee_acquisition_route": "NASA Earthdata/LP DAAC through earthaccess/AppEEARS; no GEE",
        "raw_gridded_required_for_full_xarray": True,
        "point_csv_acceptable_for_tower_only": True,
        "download_priority": 2,
        "target_dir": str(DL / "modis_mcd15_lai"),
    },
    {
        "product_group": "MCD64A1_BURNED_AREA",
        "official_product_options": "MCD64A1.061 monthly burned area",
        "needed_for": "burned/disturbed observation exclusion",
        "current_status_from_strict_lock": strict.loc[strict.product_group.eq("MCD64A1_BURNED_AREA"), "status"].iloc[0],
        "local_source_now": strict.loc[strict.product_group.eq("MCD64A1_BURNED_AREA"), "best_path"].iloc[0] if pd.notna(strict.loc[strict.product_group.eq("MCD64A1_BURNED_AREA"), "best_path"].iloc[0]) else "",
        "no_gee_acquisition_route": "NASA LAADS/LP DAAC/Earthdata download; no GEE",
        "raw_gridded_required_for_full_xarray": True,
        "point_csv_acceptable_for_tower_only": True,
        "download_priority": 3,
        "target_dir": str(DL / "modis_mcd64_burned_area"),
    },
    {
        "product_group": "SOILGRIDS_TEXTURE",
        "official_product_options": "SoilGrids sand, silt, clay rasters/depth layers",
        "needed_for": "soil texture causal controls",
        "current_status_from_strict_lock": strict.loc[strict.product_group.eq("SOILGRIDS_TEXTURE"), "status"].iloc[0],
        "local_source_now": strict.loc[strict.product_group.eq("SOILGRIDS_TEXTURE"), "best_path"].iloc[0],
        "no_gee_acquisition_route": "ISRIC SoilGrids web services/downloads; no GEE",
        "raw_gridded_required_for_full_xarray": True,
        "point_csv_acceptable_for_tower_only": True,
        "download_priority": 2,
        "target_dir": str(DL / "soilgrids_texture"),
    },
]

plan = pd.DataFrame(products)
plan.to_csv(TAB / "Table_PRODUCT02h_no_gee_acquisition_plan.csv", index=False)

for d in plan["target_dir"]:
    Path(d).mkdir(parents=True, exist_ok=True)

earthaccess_script = r'''#!/usr/bin/env python
from pathlib import Path
import earthaccess
import json
import pandas as pd

OUT = Path("data/raw_local/no_gee_downloads")
OUT.mkdir(parents=True, exist_ok=True)

# This script searches only. It does not download unless you set DO_DOWNLOAD = True.
# First run it as search-only to see counts.
DO_DOWNLOAD = False

PRODUCTS = [
    {"short_name": "MOD17A2HGF", "version": "061", "out": OUT / "modis_mod17_gpp"},
    {"short_name": "MOD16A2GF", "version": "061", "out": OUT / "modis_mod16_et"},
    {"short_name": "MCD15A2H",  "version": "061", "out": OUT / "modis_mcd15_lai"},
    {"short_name": "MCD64A1",   "version": "061", "out": OUT / "modis_mcd64_burned_area"},
]

# Start with the tower-analysis years. Expand later if needed.
TEMPORAL = ("2001-01-01", "2024-12-31")

# Global bounding box for search. Downloading global all tiles is huge.
# Later we should restrict by MODIS h/v tiles around tower or sampled grassland points.
BOUNDING_BOX = (-180, -90, 180, 90)

auth = earthaccess.login(strategy="netrc")

rows = []
for p in PRODUCTS:
    print("")
    print("Searching", p["short_name"], p["version"])
    results = earthaccess.search_data(
        short_name=p["short_name"],
        version=p["version"],
        temporal=TEMPORAL,
        bounding_box=BOUNDING_BOX,
        count=20,
    )
    print("Returned first", len(results), "records")
    p["out"].mkdir(parents=True, exist_ok=True)

    for r in results[:20]:
        try:
            size = r.size()
        except Exception:
            size = None
        rows.append({
            "short_name": p["short_name"],
            "version": p["version"],
            "size": size,
            "summary": str(r)[:1000],
        })

    if DO_DOWNLOAD and results:
        earthaccess.download(results, local_path=str(p["out"]))

pd.DataFrame(rows).to_csv("results/stage1b5_no_gee_acquisition/tables/Table_PRODUCT02i_earthaccess_search_preview.csv", index=False)
print("")
print("WROTE results/stage1b5_no_gee_acquisition/tables/Table_PRODUCT02i_earthaccess_search_preview.csv")
'''

Path("scripts/stage1b5_earthaccess_search_preview.py").write_text(earthaccess_script)

report = []
report.append("# Stage 1B.5 no-GEE acquisition plan")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Current rule")
report.append("")
report.append("Every stage is finished only when missing products are downloaded locally, accepted as point/table inputs for a specific analysis, or explicitly marked as a limitation. Google Earth Engine is not used.")
report.append("")
report.append("## Products needing acquisition or decision")
report.append("")
report.append("```text")
report.append(plan.to_string(index=False))
report.append("```")
report.append("")
report.append("## Immediate next step")
report.append("")
report.append("Run the earthaccess search preview script. It searches NASA Earthdata for the missing MODIS/MCD products but does not download yet.")
report.append("")
report.append("```bash")
report.append("python -u scripts/stage1b5_earthaccess_search_preview.py 2>&1 | tee logs/stage1b5_earthaccess_search_preview.log")
report.append("cat results/stage1b5_no_gee_acquisition/tables/Table_PRODUCT02i_earthaccess_search_preview.csv")
report.append("```")
report.append("")
report.append("## Decision after search preview")
report.append("")
report.append("If products are found, restrict downloads to needed MODIS tiles or point/sampling region before large downloads. If products are not found through earthaccess, use AppEEARS/LP DAAC/LAADS manual downloads, still no GEE.")
report.append("")

(TXT / "STAGE1B5_NO_GEE_ACQUISITION_PLAN.md").write_text("\n".join(report))

summary = {
    "stage": "1B.5_no_gee_acquisition_plan",
    "status": "plan_created",
    "products_needing_acquisition_or_decision": products,
    "outputs": {
        "plan_csv": str(TAB / "Table_PRODUCT02h_no_gee_acquisition_plan.csv"),
        "report": str(TXT / "STAGE1B5_NO_GEE_ACQUISITION_PLAN.md"),
        "earthaccess_search_script": "scripts/stage1b5_earthaccess_search_preview.py"
    }
}
(TAB / "STAGE1B5_NO_GEE_ACQUISITION_PLAN_SUMMARY.json").write_text(json.dumps(summary, indent=2))

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02h_no_gee_acquisition_plan.csv")
print("WROTE", TXT / "STAGE1B5_NO_GEE_ACQUISITION_PLAN.md")
print("WROTE scripts/stage1b5_earthaccess_search_preview.py")
