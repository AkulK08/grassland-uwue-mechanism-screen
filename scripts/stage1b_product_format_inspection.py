from pathlib import Path
import os
import json
import zipfile
import traceback
import pandas as pd
from datetime import datetime

OUT = Path("results/stage1b_product_format_inspection")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

INV_PATH = Path("results/stage1a_local_product_inventory/tables/Table_PRODUCT01_local_product_inventory.csv")
if not INV_PATH.exists():
    raise FileNotFoundError(f"Missing Stage 1A inventory: {INV_PATH}")

inventory = pd.read_csv(INV_PATH, low_memory=False)

PRODUCT_EXPECTED_HINTS = {
    "MODIS_GPP_MOD17": ["gpp", "psn_qc", "mod17"],
    "MODIS_ET_MOD16": ["et", "et_qc", "mod16"],
    "GOSIF_GPP": ["gosif", "gpp"],
    "GLEAM_ET": ["gleam", "e", "et", "evap"],
    "PML_GPP_ET": ["pml", "gpp", "et", "ec", "ei", "es"],
    "ERA5_LAND": ["temperature", "dewpoint", "swvl", "vpd", "t2m", "d2m", "soil"],
    "SMAP_L4": ["smap", "sm_rootzone", "soil", "sm"],
    "MODIS_LAI_MOD15": ["lai", "mod15", "fpar"],
    "MCD12Q1_LANDCOVER": ["mcd12", "igbp", "landcover", "lc_type"],
    "MCD64A1_BURNED_AREA": ["mcd64", "burn", "burndate"],
    "CGIAR_ARIDITY": ["aridity", "ai", "et0"],
    "SOILGRIDS_TEXTURE": ["sand", "silt", "clay", "soilgrids"],
    "TRAIT_P50_XYLEM": ["p50", "xylem", "vulnerability"],
    "TRAIT_ISOHYDRICITY": ["isohydric", "anisohydric", "konings"],
    "TRAIT_ROOTING_DEPTH": ["root", "rooting", "depth", "water_storage"],
    "FLUX_TOWER_DATA": ["gpp", "le", "vpd", "swc", "nee", "fluxnet", "ameriflux"],
}

RAW_FORMAT_PRIORITY = {
    "HDF/HDF-EOS": 1,
    "HDF5": 1,
    "NetCDF": 1,
    "GeoTIFF": 1,
    "ZIP archive": 2,
    "Gzip-compressed": 2,
    "Parquet": 3,
    "CSV": 4,
}

DERIVED_PATH_MARKERS = [
    "/results/",
    "/bootstrap_runs/",
    "/inspection/",
    "/final_nonwriting_lock/",
    "/handoff_to_other_chat/",
    "/logs/",
]

def import_optional(name):
    try:
        return __import__(name)
    except Exception:
        return None

rasterio = import_optional("rasterio")
xarray = import_optional("xarray")
h5py = import_optional("h5py")
pyarrow_parquet = None
try:
    import pyarrow.parquet as pq
    pyarrow_parquet = pq
except Exception:
    pyarrow_parquet = None

def safe_str(x, maxlen=1200):
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    s = s.replace("\n", " | ")
    return s[:maxlen]

def path_role(path):
    low = str(path).lower()
    if any(marker in low for marker in DERIVED_PATH_MARKERS):
        return "DERIVED_OR_PREVIOUS_OUTPUT"
    if "/data/raw" in low or "/data/external" in low or "/data/" in low:
        return "DATA_DIRECTORY"
    if "/downloads/" in low or "/desktop/" in low or "/volumes/" in low:
        return "LOCAL_FILE_SYSTEM"
    return "UNKNOWN"

def inspect_csv(p):
    out = {}
    try:
        df = pd.read_csv(p, nrows=5, low_memory=False)
        out["read_status"] = "READABLE"
        out["n_columns"] = len(df.columns)
        out["columns_or_variables"] = ", ".join(map(str, df.columns[:80]))
        out["dims_shape"] = f"sample_rows=5, columns={len(df.columns)}"
    except Exception as e:
        out["read_status"] = "READ_ERROR"
        out["error"] = safe_str(e)
    return out

def inspect_parquet(p):
    out = {}
    try:
        if pyarrow_parquet is not None:
            pf = pyarrow_parquet.ParquetFile(str(p))
            schema_names = pf.schema.names
            out["read_status"] = "READABLE"
            out["n_columns"] = len(schema_names)
            out["columns_or_variables"] = ", ".join(map(str, schema_names[:80]))
            out["dims_shape"] = f"rows={pf.metadata.num_rows}, columns={len(schema_names)}"
        else:
            df = pd.read_parquet(p)
            out["read_status"] = "READABLE"
            out["n_columns"] = len(df.columns)
            out["columns_or_variables"] = ", ".join(map(str, df.columns[:80]))
            out["dims_shape"] = f"rows={len(df)}, columns={len(df.columns)}"
    except Exception as e:
        out["read_status"] = "READ_ERROR"
        out["error"] = safe_str(e)
    return out

def inspect_zip(p):
    out = {}
    try:
        with zipfile.ZipFile(p, "r") as z:
            names = z.namelist()
        out["read_status"] = "READABLE_ARCHIVE"
        out["n_columns"] = None
        out["columns_or_variables"] = ", ".join(names[:60])
        out["dims_shape"] = f"zip_members={len(names)}"
    except Exception as e:
        out["read_status"] = "READ_ERROR"
        out["error"] = safe_str(e)
    return out

def inspect_rasterio(p):
    out = {}
    if rasterio is None:
        out["read_status"] = "SKIPPED_RASTERIO_NOT_INSTALLED"
        return out
    try:
        with rasterio.open(str(p)) as src:
            subs = list(src.subdatasets or [])
            out["read_status"] = "READABLE"
            out["crs"] = safe_str(src.crs)
            out["bounds"] = safe_str(src.bounds)
            out["dims_shape"] = f"width={src.width}, height={src.height}, count={src.count}"
            out["columns_or_variables"] = ", ".join([d for d in (src.descriptions or []) if d])[:1200]
            out["subdatasets"] = " || ".join(subs[:80])
            out["n_subdatasets"] = len(subs)
            out["dtype"] = ",".join(map(str, src.dtypes))
    except Exception as e:
        out["read_status"] = "READ_ERROR"
        out["error"] = safe_str(e)
    return out

def inspect_xarray(p):
    out = {}
    if xarray is None:
        out["read_status"] = "SKIPPED_XARRAY_NOT_INSTALLED"
        return out

    engines = [None, "netcdf4", "h5netcdf"]
    last_err = None
    for engine in engines:
        try:
            kwargs = {"decode_times": False}
            if engine:
                kwargs["engine"] = engine
            ds = xarray.open_dataset(str(p), **kwargs)
            try:
                vars_ = list(ds.data_vars)
                coords_ = list(ds.coords)
                dims_ = dict(ds.sizes)
                attrs_ = dict(ds.attrs)
                out["read_status"] = "READABLE"
                out["xarray_engine"] = engine or "default"
                out["columns_or_variables"] = ", ".join(vars_[:100])
                out["coords"] = ", ".join(coords_[:100])
                out["dims_shape"] = safe_str(dims_)
                out["attrs"] = safe_str({k: attrs_.get(k) for k in list(attrs_)[:20]})
            finally:
                ds.close()
            return out
        except Exception as e:
            last_err = e

    out["read_status"] = "READ_ERROR"
    out["error"] = safe_str(last_err)
    return out

def inspect_h5py(p):
    out = {}
    if h5py is None:
        out["read_status"] = "SKIPPED_H5PY_NOT_INSTALLED"
        return out

    try:
        datasets = []
        groups = []
        def visitor(name, obj):
            if len(datasets) + len(groups) >= 120:
                return
            try:
                import h5py as _h5py
                if isinstance(obj, _h5py.Dataset):
                    datasets.append(f"{name} shape={obj.shape} dtype={obj.dtype}")
                else:
                    groups.append(name)
            except Exception:
                pass
        with h5py.File(str(p), "r") as f:
            f.visititems(visitor)
        out["read_status"] = "READABLE"
        out["columns_or_variables"] = " || ".join(datasets[:80])
        out["groups"] = " || ".join(groups[:80])
        out["dims_shape"] = f"h5_datasets={len(datasets)}, h5_groups={len(groups)}"
    except Exception as e:
        out["read_status"] = "READ_ERROR"
        out["error"] = safe_str(e)
    return out

def inspect_file(p, fmt):
    p = Path(p)
    low = str(p).lower()

    base = {
        "path": str(p),
        "exists": p.exists(),
        "file_size_bytes_current": None,
        "read_status": "NOT_INSPECTED",
        "columns_or_variables": "",
        "coords": "",
        "dims_shape": "",
        "crs": "",
        "bounds": "",
        "subdatasets": "",
        "n_subdatasets": None,
        "dtype": "",
        "attrs": "",
        "groups": "",
        "error": "",
    }

    if not p.exists():
        base["read_status"] = "MISSING_AT_INSPECTION_TIME"
        return base

    try:
        base["file_size_bytes_current"] = p.stat().st_size
    except Exception:
        pass

    if low.endswith(".csv"):
        base.update(inspect_csv(p))
    elif low.endswith(".parquet"):
        base.update(inspect_parquet(p))
    elif low.endswith(".zip"):
        base.update(inspect_zip(p))
    elif low.endswith(".nc") or low.endswith(".nc4") or low.endswith(".cdf"):
        base.update(inspect_xarray(p))
    elif low.endswith(".tif") or low.endswith(".tiff") or low.endswith(".vrt"):
        base.update(inspect_rasterio(p))
    elif low.endswith(".hdf") or low.endswith(".h5") or low.endswith(".hdf5"):
        r = inspect_rasterio(p)
        if r.get("read_status") == "READABLE" and (r.get("subdatasets") or r.get("columns_or_variables")):
            base.update(r)
        else:
            h = inspect_h5py(p)
            if h.get("read_status") == "READABLE":
                base.update(h)
            else:
                base.update(r)
                base["h5py_error"] = h.get("error", "")
    elif low.endswith(".gz"):
        base["read_status"] = "COMPRESSED_GZ_NOT_OPENED"
    else:
        base["read_status"] = "UNKNOWN_EXTENSION_NOT_OPENED"

    return base

def relevance_hit(product, text):
    hints = PRODUCT_EXPECTED_HINTS.get(product, [])
    low = str(text).lower()
    hits = [h for h in hints if h.lower() in low]
    return hits

def sample_candidates(inv, product, max_files=12):
    sub = inv[inv["product_group"].eq(product)].copy()
    if len(sub) == 0:
        return sub

    sub["role"] = sub["path"].map(path_role)
    sub["format_priority"] = sub["file_format_guess"].map(RAW_FORMAT_PRIORITY).fillna(99)
    sub["is_derived"] = sub["role"].eq("DERIVED_OR_PREVIOUS_OUTPUT")
    sub["size_rank"] = pd.to_numeric(sub["size_bytes"], errors="coerce").fillna(0)

    # Prefer raw/real gridded files first, then large data files, then derived CSVs.
    sub = sub.sort_values(
        ["is_derived", "format_priority", "match_score", "size_rank"],
        ascending=[True, True, False, False]
    )

    # Keep diversity by file format/path.
    return sub.head(max_files)

all_products = list(PRODUCT_EXPECTED_HINTS.keys())
inspect_rows = []
shortlist_rows = []

for product in all_products:
    cand = sample_candidates(inventory, product, max_files=14)
    if len(cand) == 0:
        inspect_rows.append({
            "product_group": product,
            "path": "",
            "file_format_guess": "",
            "role": "",
            "read_status": "NO_CANDIDATE_FILES",
            "columns_or_variables": "",
            "dims_shape": "",
            "crs": "",
            "subdatasets": "",
            "n_subdatasets": None,
            "relevance_hits": "",
            "usable_for_next_stage_guess": False,
            "why": "No files matched this product group in Stage 1A.",
        })
        continue

    for _, row in cand.iterrows():
        p = Path(row["path"])
        fmt = row.get("file_format_guess", "")
        inspected = inspect_file(p, fmt)
        combined_text = " ".join([
            str(row.get("filename", "")),
            str(row.get("path", "")),
            str(inspected.get("columns_or_variables", "")),
            str(inspected.get("subdatasets", "")),
            str(inspected.get("groups", "")),
            str(inspected.get("attrs", "")),
        ])
        hits = relevance_hit(product, combined_text)
        role = path_role(p)
        status = inspected.get("read_status", "")

        usable = (
            status in {"READABLE", "READABLE_ARCHIVE"}
            and len(hits) > 0
            and role != "DERIVED_OR_PREVIOUS_OUTPUT"
            and inspected.get("file_size_bytes_current", 0) not in [0, None]
        )

        why = []
        if status not in {"READABLE", "READABLE_ARCHIVE"}:
            why.append(f"not_readable_or_not_opened={status}")
        if not hits:
            why.append("no_expected_variable_keyword_hit")
        if role == "DERIVED_OR_PREVIOUS_OUTPUT":
            why.append("derived_or_previous_output_not_raw")
        if inspected.get("file_size_bytes_current", 0) in [0, None]:
            why.append("zero_or_unknown_size")

        outrow = {
            "product_group": product,
            "source_inventory_filename": row.get("filename", ""),
            "path": str(p),
            "role": role,
            "file_format_guess": fmt,
            "match_score": row.get("match_score", ""),
            "size_bytes": inspected.get("file_size_bytes_current", ""),
            "read_status": status,
            "columns_or_variables": inspected.get("columns_or_variables", ""),
            "coords": inspected.get("coords", ""),
            "dims_shape": inspected.get("dims_shape", ""),
            "crs": inspected.get("crs", ""),
            "bounds": inspected.get("bounds", ""),
            "subdatasets": inspected.get("subdatasets", ""),
            "n_subdatasets": inspected.get("n_subdatasets", ""),
            "dtype": inspected.get("dtype", ""),
            "attrs": inspected.get("attrs", ""),
            "groups": inspected.get("groups", ""),
            "relevance_hits": ",".join(hits),
            "usable_for_next_stage_guess": bool(usable),
            "why_not_usable_or_caution": "; ".join(why),
            "error": inspected.get("error", ""),
        }
        inspect_rows.append(outrow)
        if usable:
            shortlist_rows.append(outrow)

inspection = pd.DataFrame(inspect_rows)
shortlist = pd.DataFrame(shortlist_rows)

inspection.to_csv(TAB / "Table_PRODUCT02_format_inspection_by_file.csv", index=False)
shortlist.to_csv(TAB / "Table_PRODUCT02c_raw_candidate_shortlist.csv", index=False)

summary_rows = []
for product in all_products:
    sub = inspection[inspection["product_group"].eq(product)]
    usable = sub[sub["usable_for_next_stage_guess"].eq(True)]
    readable = sub[sub["read_status"].isin(["READABLE", "READABLE_ARCHIVE"])]
    derived = sub[sub["role"].eq("DERIVED_OR_PREVIOUS_OUTPUT")]

    if len(usable) > 0:
        status = "USABLE_RAW_CANDIDATE_FOUND"
        action = "Proceed to Stage 1C/1D extraction/conversion for this product after checking scale factor and units."
    elif len(readable) > 0 and len(derived) == len(readable):
        status = "ONLY_DERIVED_OUTPUTS_FOUND"
        action = "Need true raw/local product files, not old result CSVs."
    elif len(readable) > 0:
        status = "READABLE_BUT_NEEDS_MANUAL_REVIEW"
        action = "Readable files exist but keyword/role checks were inconclusive; inspect report rows."
    elif len(sub) > 0:
        status = "CANDIDATES_NOT_READABLE_IN_CURRENT_ENV"
        action = "Install missing readers or locate different raw files."
    else:
        status = "NO_CANDIDATES"
        action = "Download required product."

    summary_rows.append({
        "product_group": product,
        "n_inspected_files": int(len(sub)),
        "n_readable_files": int(len(readable)),
        "n_usable_raw_candidates": int(len(usable)),
        "status": status,
        "best_candidate_path": usable["path"].iloc[0] if len(usable) else "",
        "best_candidate_format": usable["file_format_guess"].iloc[0] if len(usable) else "",
        "action": action,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "Table_PRODUCT02b_format_readiness_summary.csv", index=False)

machine = {
    "stage": "1B_product_format_inspection",
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_products": len(all_products),
    "n_products_with_usable_raw_candidate": int((summary["n_usable_raw_candidates"] > 0).sum()),
    "outputs": {
        "inspection_by_file": str(TAB / "Table_PRODUCT02_format_inspection_by_file.csv"),
        "readiness_summary": str(TAB / "Table_PRODUCT02b_format_readiness_summary.csv"),
        "raw_candidate_shortlist": str(TAB / "Table_PRODUCT02c_raw_candidate_shortlist.csv"),
        "report": str(TXT / "STAGE1B_PRODUCT_FORMAT_INSPECTION_REPORT.md"),
    }
}
(TAB / "STAGE1B_PRODUCT_FORMAT_INSPECTION_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

report = []
report.append("# Stage 1B product format inspection")
report.append("")
report.append(f"Generated: {machine['generated']}")
report.append("")
report.append("## What this step does")
report.append("")
report.append("Stage 1A only found candidate filenames. Stage 1B opens a representative shortlist for each product group and checks whether the files are readable, raw/useful, and contain relevant variables/subdatasets/columns.")
report.append("")
report.append("## Readiness summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Raw candidate shortlist")
report.append("")
report.append("```text")
if len(shortlist):
    cols = ["product_group", "file_format_guess", "read_status", "relevance_hits", "dims_shape", "crs", "path"]
    report.append(shortlist[cols].head(80).to_string(index=False))
else:
    report.append("No usable raw candidates identified by automatic checks.")
report.append("```")
report.append("")
report.append("## Products requiring manual attention")
report.append("")
needs = summary[~summary["status"].eq("USABLE_RAW_CANDIDATE_FOUND")]
report.append("```text")
report.append(needs.to_string(index=False) if len(needs) else "All product groups have at least one usable raw candidate.")
report.append("```")
report.append("")
report.append("## Stage 1B completion rule")
report.append("")
report.append("Stage 1B is complete when every required product has either a usable raw candidate or a clear download/manual-fix action. The next coding stage is to build product-specific local extractors/converters into xarray/Zarr or tower-centered CSV.")
report.append("")

(TXT / "STAGE1B_PRODUCT_FORMAT_INSPECTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02_format_inspection_by_file.csv")
print("WROTE", TAB / "Table_PRODUCT02b_format_readiness_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02c_raw_candidate_shortlist.csv")
print("WROTE", TXT / "STAGE1B_PRODUCT_FORMAT_INSPECTION_REPORT.md")
print("WROTE", TAB / "STAGE1B_PRODUCT_FORMAT_INSPECTION_SUMMARY.json")
