from pathlib import Path
import os
import re
import json
import pandas as pd
from datetime import datetime

OUT = Path("results/stage1b_strict_readiness_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

ROOTS = [
    Path(".").resolve(),
    Path("/Users/me/Downloads"),
    Path("/Users/me/Desktop"),
    Path("/Volumes"),
]

SKIP_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "node_modules",
    ".venv", "venv", "env", ".conda", "miniconda3", "anaconda3",
    "Library", "Applications"
}

RAW_EXTS = {".hdf", ".h5", ".hdf5", ".nc", ".nc4", ".tif", ".tiff", ".vrt"}
TABLE_EXTS = {".csv", ".parquet", ".feather"}
ARCHIVE_EXTS = {".zip", ".gz"}

PRODUCTS = {
    "MODIS_GPP_MOD17": {
        "must_have": [r"mod17", r"mod17a2", r"mod17a2h"],
        "reject": [r"gosif", r"gleam", r"pml", r"mod16", r"mod15", r"mcd12", r"mcd64"],
        "required": True,
        "purpose": "MODIS GPP for 3x3 product matrix",
    },
    "MODIS_ET_MOD16": {
        "must_have": [r"mod16", r"mod16a2", r"mod16a2gf"],
        "reject": [r"gosif", r"gleam", r"pml", r"mod17", r"mod15", r"mcd12", r"mcd64"],
        "required": True,
        "purpose": "MODIS ET for 3x3 product matrix",
    },
    "GOSIF_GPP": {
        "must_have": [r"gosif"],
        "reject": [r"mod17", r"mod16", r"pml"],
        "required": True,
        "purpose": "SIF-based GPP for 3x3 product matrix",
    },
    "GLEAM_ET": {
        "must_have": [r"gleam"],
        "reject": [r"mod16", r"mod17", r"pml"],
        "required": True,
        "purpose": "GLEAM ET for 3x3 product matrix",
    },
    "PML_GPP_ET": {
        "must_have": [r"pml"],
        "reject": [r"gosif_gpp_", r"gleam_v"],
        "required": True,
        "purpose": "PML GPP and ET for 3x3 product matrix",
    },
    "ERA5_LAND": {
        "must_have": [r"era5", r"t2m", r"d2m", r"swvl", r"vpd"],
        "reject": [],
        "required": True,
        "purpose": "VPD and soil moisture stress",
    },
    "SMAP_L4": {
        "must_have": [r"smap", r"spl4", r"sm_rootzone"],
        "reject": [],
        "required": False,
        "purpose": "Post-2015 soil moisture check",
    },
    "MODIS_LAI_MOD15": {
        "must_have": [r"mod15", r"mod15a2", r"lai"],
        "reject": [r"mod16", r"mod17", r"gosif", r"gleam", r"pml"],
        "required": True,
        "purpose": "LAI/growing-season/canopy covariate",
    },
    "MCD12Q1_LANDCOVER": {
        "must_have": [r"mcd12", r"mcd12q1", r"lc_type", r"landcover", r"igbp"],
        "reject": [r"mod16", r"mod17", r"gosif", r"gleam", r"pml"],
        "required": True,
        "purpose": "Land cover and stable grassland filtering",
    },
    "MCD64A1_BURNED_AREA": {
        "must_have": [r"mcd64", r"mcd64a1", r"burn", r"burndate"],
        "reject": [r"mod16", r"mod17", r"gosif", r"gleam", r"pml"],
        "required": True,
        "purpose": "Burned/disturbed pixel exclusion",
    },
    "CGIAR_ARIDITY": {
        "must_have": [r"aridity", r"cgiar", r"ai_et0", r"global-ai"],
        "reject": [],
        "required": True,
        "purpose": "Aridity covariate/quartiles",
    },
    "SOILGRIDS_TEXTURE": {
        "must_have": [r"soilgrids", r"sand", r"silt", r"clay", r"soil_texture"],
        "reject": [],
        "required": True,
        "purpose": "Soil texture controls",
    },
    "TRAIT_P50_XYLEM": {
        "must_have": [r"p50", r"xylem", r"vulnerability"],
        "reject": [],
        "required": True,
        "purpose": "Hydraulic vulnerability trait",
    },
    "TRAIT_ISOHYDRICITY": {
        "must_have": [r"isohydric", r"anisohydric", r"konings"],
        "reject": [],
        "required": True,
        "purpose": "Stomatal strategy trait",
    },
    "TRAIT_ROOTING_DEPTH": {
        "must_have": [r"rooting", r"root_depth", r"rooting_depth", r"stocker", r"water_storage"],
        "reject": [],
        "required": True,
        "purpose": "Root-zone storage trait",
    },
    "FLUX_TOWER_DATA": {
        "must_have": [r"fluxnet", r"ameriflux", r"icos", r"ozflux", r"gpp_nt_vut_ref", r"le_f_mds"],
        "reject": [],
        "required": True,
        "purpose": "Tower validation reference",
    },
}

DERIVED_MARKERS = [
    "/results/",
    "/bootstrap_runs/",
    "/inspection/",
    "/final_nonwriting_lock/",
    "/handoff_to_other_chat/",
    "/logs/",
]

def suffix_kind(path):
    suffixes = [s.lower() for s in Path(path).suffixes]
    if any(s in RAW_EXTS for s in suffixes):
        return "RAW_GRID_OR_ARRAY"
    if any(s in TABLE_EXTS for s in suffixes):
        return "TABLE_OR_POINT_SAMPLE"
    if any(s in ARCHIVE_EXTS for s in suffixes):
        return "ARCHIVE"
    return "OTHER"

def path_role(path):
    low = str(path).lower()
    if any(m in low for m in DERIVED_MARKERS):
        return "DERIVED_PREVIOUS_OUTPUT"
    if "/data/raw" in low:
        return "RAW_DATA_DIR"
    if "/data/external/_downloads" in low:
        return "EXTERNAL_DOWNLOAD_DIR"
    if "/data/external" in low:
        return "EXTERNAL_TABLE_OR_RASTER_DIR"
    if "/data/processed" in low:
        return "PROCESSED_DATA_DIR"
    if "/downloads/" in low or "/desktop/" in low or "/volumes/" in low:
        return "LOCAL_FILE_SYSTEM"
    return "UNKNOWN"

def score_product(product, path):
    rule = PRODUCTS[product]
    low = str(path).lower()
    must = [p for p in rule["must_have"] if re.search(p, low)]
    reject = [p for p in rule["reject"] if re.search(p, low)]
    if not must or reject:
        return 0, must, reject

    kind = suffix_kind(path)
    role = path_role(path)

    score = len(must) * 10

    if kind == "RAW_GRID_OR_ARRAY":
        score += 30
    elif kind == "ARCHIVE":
        score += 20
    elif kind == "TABLE_OR_POINT_SAMPLE":
        score += 10

    if role in {"RAW_DATA_DIR", "EXTERNAL_DOWNLOAD_DIR"}:
        score += 30
    elif role == "LOCAL_FILE_SYSTEM":
        score += 20
    elif role == "EXTERNAL_TABLE_OR_RASTER_DIR":
        score += 15
    elif role == "PROCESSED_DATA_DIR":
        score += 5
    elif role == "DERIVED_PREVIOUS_OUTPUT":
        score -= 40

    return score, must, reject

rows = []
seen = set()

for root in ROOTS:
    if not root.exists():
        continue

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".") and "cache" not in d.lower()
        ]

        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                path = str(p.resolve())
            except Exception:
                path = str(p)

            if path in seen:
                continue
            seen.add(path)

            lower = path.lower()
            if not any(lower.endswith(ext) for ext in RAW_EXTS | TABLE_EXTS | ARCHIVE_EXTS):
                continue

            try:
                size = p.stat().st_size
            except Exception:
                size = None

            for product in PRODUCTS:
                score, must, reject = score_product(product, path)
                if score <= 0:
                    continue

                rows.append({
                    "product_group": product,
                    "path": path,
                    "filename": fn,
                    "strict_score": score,
                    "kind": suffix_kind(path),
                    "role": path_role(path),
                    "size_bytes": size,
                    "matched_required_patterns": ",".join(must),
                    "rejected_patterns": ",".join(reject),
                })

df = pd.DataFrame(rows)
if len(df):
    df = df.sort_values(["product_group", "strict_score", "size_bytes"], ascending=[True, False, False])
else:
    df = pd.DataFrame(columns=[
        "product_group", "path", "filename", "strict_score", "kind", "role",
        "size_bytes", "matched_required_patterns", "rejected_patterns"
    ])

df.to_csv(TAB / "Table_PRODUCT02e_strict_candidate_files.csv", index=False)

summary_rows = []
for product, rule in PRODUCTS.items():
    sub = df[df["product_group"].eq(product)].copy()
    raw = sub[sub["kind"].eq("RAW_GRID_OR_ARRAY")]
    archive = sub[sub["kind"].eq("ARCHIVE")]
    point = sub[sub["kind"].eq("TABLE_OR_POINT_SAMPLE")]
    derived = sub[sub["role"].eq("DERIVED_PREVIOUS_OUTPUT")]

    best = sub.iloc[0].to_dict() if len(sub) else {}

    if len(raw) > 0 and raw.iloc[0]["role"] != "DERIVED_PREVIOUS_OUTPUT":
        status = "READY_RAW_GRID_OR_ARRAY"
        next_action = "Use this as input for Stage 1C local extractor/converter."
    elif len(archive) > 0 and archive.iloc[0]["role"] != "DERIVED_PREVIOUS_OUTPUT":
        status = "READY_ARCHIVE_NEEDS_UNZIP_OR_INTERNAL_READ"
        next_action = "Use zip/internal files for Stage 1C extraction."
    elif len(point) > 0 and point.iloc[0]["role"] != "DERIVED_PREVIOUS_OUTPUT":
        status = "POINT_SAMPLE_OR_TABLE_ONLY"
        next_action = "Can use for tower/pixel point validation, but not full gridded xarray cube."
    elif len(sub) > 0 and len(derived) == len(sub):
        status = "ONLY_DERIVED_OUTPUTS"
        next_action = "Need raw files or accept this as previous-output-only limitation."
    else:
        status = "MISSING_STRICT_MATCH"
        next_action = "Download or locate true product files."

    summary_rows.append({
        "product_group": product,
        "required": rule["required"],
        "purpose": rule["purpose"],
        "n_strict_candidates": int(len(sub)),
        "n_raw_grid_or_array": int(len(raw)),
        "n_archives": int(len(archive)),
        "n_tables_or_point_samples": int(len(point)),
        "status": status,
        "best_path": best.get("path", ""),
        "best_kind": best.get("kind", ""),
        "best_role": best.get("role", ""),
        "next_action": next_action,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "Table_PRODUCT02f_STRICT_READINESS_LOCK.csv", index=False)

missing_or_limited = summary[
    summary["status"].isin([
        "POINT_SAMPLE_OR_TABLE_ONLY",
        "ONLY_DERIVED_OUTPUTS",
        "MISSING_STRICT_MATCH"
    ])
].copy()
missing_or_limited.to_csv(TAB / "Table_PRODUCT02g_products_needing_download_or_decision.csv", index=False)

report = []
report.append("# Stage 1B strict readiness lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Why this was needed")
report.append("")
report.append("The first Stage 1B inspection was too permissive. For example, MODIS_GPP_MOD17 was incorrectly matched to GOSIF_GPP files because both filenames contain GPP. This strict lock requires product-specific identifiers such as MOD17 for MODIS GPP.")
report.append("")
report.append("## Strict readiness summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Products needing download or decision")
report.append("")
report.append("```text")
report.append(missing_or_limited.to_string(index=False) if len(missing_or_limited) else "No products are missing or limited.")
report.append("```")
report.append("")
report.append("## Best strict candidates by product")
report.append("")
report.append("```text")
cols = ["product_group", "kind", "role", "strict_score", "size_bytes", "path"]
if len(df):
    bests = df.groupby("product_group", as_index=False).head(5)
    report.append(bests[cols].to_string(index=False))
else:
    report.append("No strict candidates found.")
report.append("```")
report.append("")
report.append("## Completion decision")
report.append("")
report.append("Stage 1B is fully complete after this strict lock. Stage 1C should only use products marked READY_RAW_GRID_OR_ARRAY, READY_ARCHIVE_NEEDS_UNZIP_OR_INTERNAL_READ, or intentionally accepted POINT_SAMPLE_OR_TABLE_ONLY.")
report.append("")

(TXT / "STAGE1B_STRICT_READINESS_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B_strict_readiness_lock",
    "status": "complete",
    "n_products": len(PRODUCTS),
    "n_ready_raw": int(summary["status"].eq("READY_RAW_GRID_OR_ARRAY").sum()),
    "n_ready_archive": int(summary["status"].eq("READY_ARCHIVE_NEEDS_UNZIP_OR_INTERNAL_READ").sum()),
    "n_point_only": int(summary["status"].eq("POINT_SAMPLE_OR_TABLE_ONLY").sum()),
    "n_missing_or_derived_only": int(summary["status"].isin(["ONLY_DERIVED_OUTPUTS", "MISSING_STRICT_MATCH"]).sum()),
    "outputs": {
        "strict_candidates": str(TAB / "Table_PRODUCT02e_strict_candidate_files.csv"),
        "strict_readiness_lock": str(TAB / "Table_PRODUCT02f_STRICT_READINESS_LOCK.csv"),
        "needs_download_or_decision": str(TAB / "Table_PRODUCT02g_products_needing_download_or_decision.csv"),
        "report": str(TXT / "STAGE1B_STRICT_READINESS_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B_STRICT_READINESS_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02e_strict_candidate_files.csv")
print("WROTE", TAB / "Table_PRODUCT02f_STRICT_READINESS_LOCK.csv")
print("WROTE", TAB / "Table_PRODUCT02g_products_needing_download_or_decision.csv")
print("WROTE", TXT / "STAGE1B_STRICT_READINESS_LOCK_REPORT.md")
print("WROTE", TAB / "STAGE1B_STRICT_READINESS_LOCK_SUMMARY.json")
