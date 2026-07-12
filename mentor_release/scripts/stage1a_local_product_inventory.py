from pathlib import Path
import os
import re
import json
import pandas as pd
from datetime import datetime

OUT = Path("results/stage1a_local_product_inventory")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    Path(".").resolve(),
    Path("/Users/me/Downloads"),
    Path("/Users/me/Desktop"),
    Path("/Volumes"),
]

SKIP_DIR_NAMES = {
    ".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
    ".venv", "venv", "env", "site-packages", ".conda",
    "miniconda3", "anaconda3", "Library", "Applications",
}

ALLOWED_SUFFIXES = {
    ".hdf", ".h5", ".hdf5",
    ".nc", ".nc4", ".cdf",
    ".tif", ".tiff", ".vrt",
    ".gz", ".zip",
    ".csv", ".parquet", ".feather",
    ".json", ".txt",
}

PRODUCT_RULES = {
    "MODIS_GPP_MOD17": {
        "required_for_full_pipeline": True,
        "variable_group": "GPP",
        "keys_any": ["mod17", "mod17a2", "mod17a2h", "gpp"],
        "keys_strong": ["mod17"],
        "expected_variables": "Gpp, Psn_QC",
        "purpose": "MODIS GPP product for 3x3 product matrix",
    },
    "MODIS_ET_MOD16": {
        "required_for_full_pipeline": True,
        "variable_group": "ET",
        "keys_any": ["mod16", "mod16a2", "mod16a2gf"],
        "keys_strong": ["mod16"],
        "expected_variables": "ET, ET_QC",
        "purpose": "MODIS ET product for 3x3 product matrix",
    },
    "GOSIF_GPP": {
        "required_for_full_pipeline": True,
        "variable_group": "GPP",
        "keys_any": ["gosif", "gosif_gpp"],
        "keys_strong": ["gosif"],
        "expected_variables": "GOSIF GPP",
        "purpose": "SIF-based GPP product for 3x3 product matrix",
    },
    "GLEAM_ET": {
        "required_for_full_pipeline": True,
        "variable_group": "ET",
        "keys_any": ["gleam", "e_20", "evaporation"],
        "keys_strong": ["gleam"],
        "expected_variables": "E / ET",
        "purpose": "Structurally independent ET product for 3x3 product matrix",
    },
    "PML_GPP_ET": {
        "required_for_full_pipeline": True,
        "variable_group": "GPP_AND_ET",
        "keys_any": ["pml", "pml_v2", "pml-v2", "pml_v22", "pmlv2"],
        "keys_strong": ["pml"],
        "expected_variables": "GPP, ET",
        "purpose": "PML GPP and ET product for 3x3 product matrix",
    },
    "ERA5_LAND": {
        "required_for_full_pipeline": True,
        "variable_group": "STRESS",
        "keys_any": ["era5", "era5_land", "era5-land", "temperature_2m", "dewpoint", "swvl1", "swvl2", "vpd"],
        "keys_strong": ["era5"],
        "expected_variables": "temperature_2m, dewpoint_temperature_2m, swvl1, swvl2",
        "purpose": "VPD and root-zone soil moisture stress variables",
    },
    "SMAP_L4": {
        "required_for_full_pipeline": False,
        "variable_group": "STRESS_CHECK",
        "keys_any": ["smap", "spl4", "smap_l4", "sm_rootzone"],
        "keys_strong": ["smap", "spl4"],
        "expected_variables": "root-zone soil moisture",
        "purpose": "Post-2015 soil moisture validation check",
    },
    "MODIS_LAI_MOD15": {
        "required_for_full_pipeline": True,
        "variable_group": "CANOPY",
        "keys_any": ["mod15", "mod15a2", "lai", "lai_500m"],
        "keys_strong": ["mod15", "lai"],
        "expected_variables": "Lai_500m",
        "purpose": "LAI covariate and growing-season/phenology support",
    },
    "MCD12Q1_LANDCOVER": {
        "required_for_full_pipeline": True,
        "variable_group": "LAND_COVER",
        "keys_any": ["mcd12", "mcd12q1", "landcover", "land_cover", "igbp"],
        "keys_strong": ["mcd12", "landcover"],
        "expected_variables": "IGBP land-cover class",
        "purpose": "Grassland/savanna/open filtering and land-cover stability",
    },
    "MCD64A1_BURNED_AREA": {
        "required_for_full_pipeline": True,
        "variable_group": "DISTURBANCE",
        "keys_any": ["mcd64", "mcd64a1", "burndate", "burned", "burn"],
        "keys_strong": ["mcd64", "burn"],
        "expected_variables": "BurnDate",
        "purpose": "Exclude disturbed/burned observations",
    },
    "CGIAR_ARIDITY": {
        "required_for_full_trait_analysis": True,
        "variable_group": "CLIMATE_COVARIATE",
        "keys_any": ["aridity", "cgiar", "csi", "ai_et0", "global_aridity"],
        "keys_strong": ["aridity"],
        "expected_variables": "aridity index",
        "purpose": "Required climate confounder and aridity quartiles",
    },
    "SOILGRIDS_TEXTURE": {
        "required_for_full_trait_analysis": True,
        "variable_group": "SOIL_COVARIATE",
        "keys_any": ["soilgrids", "sand", "silt", "clay", "soil_texture"],
        "keys_strong": ["soilgrids", "sand", "silt", "clay"],
        "expected_variables": "sand, silt, clay",
        "purpose": "Required soil texture controls",
    },
    "TRAIT_P50_XYLEM": {
        "required_for_full_trait_analysis": True,
        "variable_group": "TRAIT",
        "keys_any": ["p50", "xylem", "vulnerability", "hydraulic_trait", "liu"],
        "keys_strong": ["p50", "xylem"],
        "expected_variables": "P50 / xylem vulnerability",
        "purpose": "Hydraulic vulnerability trait",
    },
    "TRAIT_ISOHYDRICITY": {
        "required_for_full_trait_analysis": True,
        "variable_group": "TRAIT",
        "keys_any": ["isohydric", "anisohydric", "konings", "gentine"],
        "keys_strong": ["isohydric"],
        "expected_variables": "isohydricity / anisohydricity",
        "purpose": "Stomatal strategy trait",
    },
    "TRAIT_ROOTING_DEPTH": {
        "required_for_full_trait_analysis": True,
        "variable_group": "TRAIT",
        "keys_any": ["rooting", "root_depth", "rooting_depth", "stocker", "water_storage"],
        "keys_strong": ["rooting", "root_depth"],
        "expected_variables": "effective rooting depth / root-zone storage",
        "purpose": "Root-zone water storage trait",
    },
    "FLUX_TOWER_DATA": {
        "required_for_full_pipeline": True,
        "variable_group": "TOWER",
        "keys_any": ["fluxnet", "ameriflux", "icos", "ozflux", "gpp_nt_vut_ref", "le_f_mds"],
        "keys_strong": ["fluxnet", "ameriflux", "gpp_nt_vut_ref"],
        "expected_variables": "GPP, LE/ET, VPD, SWC, quality flags",
        "purpose": "Tower validation reference",
    },
}

YEAR_RE = re.compile(r"(19[8-9][0-9]|20[0-3][0-9])")
DOY_RE = re.compile(r"(?:A|_)(20[0-3][0-9])([0-3][0-9][0-9])")

def classify_file(path_str):
    low = path_str.lower()
    matches = []
    for product, rule in PRODUCT_RULES.items():
        strong = rule.get("keys_strong", [])
        any_keys = rule.get("keys_any", [])
        strong_hit = any(k.lower() in low for k in strong)
        any_hit = any(k.lower() in low for k in any_keys)
        if strong_hit or any_hit:
            score = 2 if strong_hit else 1
            matches.append((product, score))
    return matches

def infer_format(p):
    suffixes = [s.lower() for s in p.suffixes]
    joined = "".join(suffixes)
    if ".hdf" in suffixes:
        return "HDF/HDF-EOS"
    if ".h5" in suffixes or ".hdf5" in suffixes:
        return "HDF5"
    if ".nc" in suffixes or ".nc4" in suffixes or ".cdf" in suffixes:
        return "NetCDF"
    if ".tif" in suffixes or ".tiff" in suffixes:
        return "GeoTIFF"
    if ".csv" in suffixes:
        return "CSV"
    if ".parquet" in suffixes:
        return "Parquet"
    if ".zip" in suffixes:
        return "ZIP archive"
    if ".gz" in suffixes:
        return "Gzip-compressed"
    return joined or "unknown"

def extract_years(name):
    years = []
    for y in YEAR_RE.findall(name):
        yy = int(y)
        if 1980 <= yy <= 2035:
            years.append(yy)
    return sorted(set(years))

def extract_doy(name):
    m = DOY_RE.search(name)
    if m:
        return int(m.group(2))
    return None

rows = []
visited = set()

for root in SEARCH_ROOTS:
    if not root.exists():
        continue

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)

        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIR_NAMES
            and not d.startswith(".")
            and "cache" not in d.lower()
            and "backup" not in d.lower()
        ]

        for fn in filenames:
            p = dp / fn
            try:
                resolved = str(p.resolve())
            except Exception:
                resolved = str(p)

            if resolved in visited:
                continue
            visited.add(resolved)

            low_name = fn.lower()
            suffixes = [s.lower() for s in p.suffixes]
            if not suffixes:
                continue
            if not any(low_name.endswith(s) for s in ALLOWED_SUFFIXES):
                continue

            matches = classify_file(str(p))
            if not matches:
                continue

            try:
                size = p.stat().st_size
            except Exception:
                size = None

            years = extract_years(str(p))
            doy = extract_doy(str(p))

            for product, score in matches:
                rule = PRODUCT_RULES[product]
                rows.append({
                    "product_group": product,
                    "match_score": score,
                    "variable_group": rule.get("variable_group", ""),
                    "path": str(p),
                    "filename": fn,
                    "file_format_guess": infer_format(p),
                    "suffixes": "".join(suffixes),
                    "size_bytes": size,
                    "years_in_filename": ",".join(map(str, years)),
                    "min_year_in_filename": min(years) if years else None,
                    "max_year_in_filename": max(years) if years else None,
                    "doy_in_filename": doy,
                    "expected_variables": rule.get("expected_variables", ""),
                    "purpose": rule.get("purpose", ""),
                })

inventory = pd.DataFrame(rows)

if len(inventory):
    inventory = inventory.sort_values(
        ["product_group", "match_score", "path"],
        ascending=[True, False, True]
    )
else:
    inventory = pd.DataFrame(columns=[
        "product_group", "match_score", "variable_group", "path", "filename",
        "file_format_guess", "suffixes", "size_bytes", "years_in_filename",
        "min_year_in_filename", "max_year_in_filename", "doy_in_filename",
        "expected_variables", "purpose"
    ])

inventory.to_csv(TAB / "Table_PRODUCT01_local_product_inventory.csv", index=False)

summary_rows = []
for product, rule in PRODUCT_RULES.items():
    sub = inventory[inventory["product_group"].eq(product)].copy()
    n = len(sub)
    years = sorted(set(pd.to_numeric(sub["min_year_in_filename"], errors="coerce").dropna().astype(int).tolist() + pd.to_numeric(sub["max_year_in_filename"], errors="coerce").dropna().astype(int).tolist()))
    status = "FOUND_LOCAL_CANDIDATES" if n > 0 else "MISSING_OR_NOT_FOUND"
    if n > 0 and product in ["MODIS_GPP_MOD17", "MODIS_ET_MOD16", "PML_GPP_ET", "ERA5_LAND", "TRAIT_P50_XYLEM", "TRAIT_ISOHYDRICITY", "TRAIT_ROOTING_DEPTH"]:
        status = "FOUND_BUT_NEEDS_FORMAT_INSPECTION"
    summary_rows.append({
        "product_group": product,
        "variable_group": rule.get("variable_group", ""),
        "required_for_full_pipeline": bool(rule.get("required_for_full_pipeline", False)),
        "required_for_full_trait_analysis": bool(rule.get("required_for_full_trait_analysis", False)),
        "n_candidate_files": int(n),
        "status": status,
        "year_min_guess": min(years) if years else None,
        "year_max_guess": max(years) if years else None,
        "expected_variables": rule.get("expected_variables", ""),
        "purpose": rule.get("purpose", ""),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "Table_PRODUCT01b_product_group_summary.csv", index=False)

missing = summary[
    (summary["status"].eq("MISSING_OR_NOT_FOUND"))
    & (
        summary["required_for_full_pipeline"].eq(True)
        | summary["required_for_full_trait_analysis"].eq(True)
    )
].copy()
missing.to_csv(TAB / "Table_PRODUCT01c_missing_or_weak_products.csv", index=False)

next_steps = []
for _, r in summary.iterrows():
    product = r["product_group"]
    status = r["status"]
    n = r["n_candidate_files"]
    if status == "MISSING_OR_NOT_FOUND":
        next_steps.append({
            "product_group": product,
            "next_action": "Download or move local files into data/raw_local for this product."
        })
    elif "FORMAT_INSPECTION" in status:
        next_steps.append({
            "product_group": product,
            "next_action": "Inspect file format/subdatasets, scale factors, CRS, time parsing, and QA bands before extraction."
        })
    else:
        next_steps.append({
            "product_group": product,
            "next_action": "Candidate files found; likely ready for extraction script after spot-check."
        })

next_df = pd.DataFrame(next_steps)
next_df.to_csv(TAB / "Table_PRODUCT01d_next_actions.csv", index=False)

report = []
report.append("# Stage 1A local product inventory")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Purpose")
report.append("")
report.append("This inventory checks whether local files can replace Google Earth Engine for the full satellite, reanalysis, soil, land-cover, and trait pipeline.")
report.append("")
report.append("## Product group summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Missing or not found products required for full pipeline/trait analysis")
report.append("")
report.append("```text")
report.append(missing.to_string(index=False) if len(missing) else "No required product groups are completely missing by filename search.")
report.append("```")
report.append("")
report.append("## Next actions")
report.append("")
report.append("```text")
report.append(next_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## First 150 candidate files")
report.append("")
report.append("```text")
if len(inventory):
    report.append(inventory.head(150).to_string(index=False))
else:
    report.append("No candidate files found.")
report.append("```")
report.append("")
report.append("## Stage 1A completion rule")
report.append("")
report.append("Stage 1A is complete when this report identifies which product groups are present, missing, or need format inspection. Stage 1B begins by inspecting subdatasets/CRS/scale factors and converting each product to xarray/Zarr.")
report.append("")

(TXT / "STAGE1A_LOCAL_PRODUCT_INVENTORY_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1A_local_product_inventory",
    "status": "complete",
    "n_candidate_files": int(len(inventory)),
    "n_product_groups_found": int((summary["n_candidate_files"] > 0).sum()),
    "n_product_groups_missing_required": int(len(missing)),
    "outputs": {
        "inventory": str(TAB / "Table_PRODUCT01_local_product_inventory.csv"),
        "summary": str(TAB / "Table_PRODUCT01b_product_group_summary.csv"),
        "missing": str(TAB / "Table_PRODUCT01c_missing_or_weak_products.csv"),
        "next_actions": str(TAB / "Table_PRODUCT01d_next_actions.csv"),
        "report": str(TXT / "STAGE1A_LOCAL_PRODUCT_INVENTORY_REPORT.md"),
    }
}
(TAB / "STAGE1A_LOCAL_PRODUCT_INVENTORY_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT01_local_product_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT01b_product_group_summary.csv")
print("WROTE", TAB / "Table_PRODUCT01c_missing_or_weak_products.csv")
print("WROTE", TAB / "Table_PRODUCT01d_next_actions.csv")
print("WROTE", TXT / "STAGE1A_LOCAL_PRODUCT_INVENTORY_REPORT.md")
print("WROTE", TAB / "STAGE1A_LOCAL_PRODUCT_INVENTORY_SUMMARY.json")
