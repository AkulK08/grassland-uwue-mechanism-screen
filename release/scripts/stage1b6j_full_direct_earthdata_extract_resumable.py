from pathlib import Path
from datetime import datetime
import os
import re
import json
import math
import pandas as pd
import numpy as np
import requests

OUT = Path("results/stage1b6j_full_direct_earthdata_extract")
TAB = OUT / "tables"
TXT = OUT / "text"
OUTCSV = Path("data/raw_local/no_gee_direct_point_extract_full")
SHARDS = OUTCSV / "_shards"
TMP = Path("tmp/stage1b6j_hdf_stream")

TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
OUTCSV.mkdir(parents=True, exist_ok=True)
SHARDS.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02r_bbox_unique_download_manifest.csv")
POINTS = Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv")

PRODUCT_FILTER = os.environ.get("PRODUCT_FILTER", "ALL").strip()
MAX_GRANULES_PER_PRODUCT = int(os.environ.get("MAX_GRANULES_PER_PRODUCT", "0"))
KEEP_HDF = os.environ.get("KEEP_HDF", "0").strip() == "1"
REBUILD_COMBINED = os.environ.get("REBUILD_COMBINED", "1").strip() == "1"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not POINTS.exists():
    raise FileNotFoundError(f"Missing final points: {POINTS}")

manifest = pd.read_csv(MANIFEST)
points = pd.read_csv(POINTS)

manifest.columns = [str(c).strip() for c in manifest.columns]
points.columns = [str(c).strip() for c in points.columns]

for c in ["id", "latitude", "longitude"]:
    if c not in points.columns:
        raise ValueError(f"Point file missing {c}. Found: {list(points.columns)}")

points["id"] = points["id"].astype(str)
points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
points = points.dropna(subset=["latitude", "longitude"]).copy()

PRODUCT_SDS = {
    "MODIS_GPP_MOD17": {
        "short_name": "MOD17A2HGF",
        "layers": {
            "Gpp_500m": ["Gpp_500m"],
            "Psn_QC_500m": ["Psn_QC_500m"],
        },
        "scale": {
            "Gpp_500m": 0.0001,
            "Psn_QC_500m": 1.0,
        },
        "offset": {
            "Gpp_500m": 0.0,
            "Psn_QC_500m": 0.0,
        },
    },
    "MODIS_ET_MOD16": {
        "short_name": "MOD16A2GF",
        "layers": {
            "ET_500m": ["ET_500m"],
            "ET_QC_500m": ["ET_QC_500m"],
        },
        "scale": {
            "ET_500m": 0.1,
            "ET_QC_500m": 1.0,
        },
        "offset": {
            "ET_500m": 0.0,
            "ET_QC_500m": 0.0,
        },
    },
    "MODIS_LAI_MCD15": {
        "short_name": "MCD15A2H",
        "layers": {
            "Lai_500m": ["Lai_500m"],
            "FparLai_QC": ["FparLai_QC"],
        },
        "scale": {
            "Lai_500m": 0.1,
            "FparLai_QC": 1.0,
        },
        "offset": {
            "Lai_500m": 0.0,
            "FparLai_QC": 0.0,
        },
    },
    "MCD64A1_BURNED_AREA": {
        "short_name": "MCD64A1",
        "layers": {
            "Burn_Date": ["Burn Date", "Burn_Date"],
            "QA": [":QA", "QA"],
        },
        "scale": {
            "Burn_Date": 1.0,
            "QA": 1.0,
        },
        "offset": {
            "Burn_Date": 0.0,
            "QA": 0.0,
        },
    },
}

def parse_date_from_modis_filename(fn):
    m = re.search(r"\.A(\d{4})(\d{3})\.", str(fn))
    if not m:
        return "", "", ""
    year = int(m.group(1))
    doy = int(m.group(2))
    dt = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    return str(dt.date()), year, doy

def tile_from_filename(fn):
    m = re.search(r"\.(h\d{2}v\d{2})\.", str(fn))
    return m.group(1) if m else ""

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))

def sites_for_manifest_row(row):
    s = str(row.get("sites_hit", "") or "")
    vals = [x.strip() for x in s.split(";") if x.strip()]
    vals = [x for x in vals if x in set(points["id"])]
    return vals

def select_subdatasets(hdf_path, product_group):
    import rasterio
    wanted = PRODUCT_SDS[product_group]["layers"]
    selected = {}

    with rasterio.open(str(hdf_path)) as src:
        subs = list(src.subdatasets or [])

    for out_name, patterns in wanted.items():
        for sd in subs:
            low = sd.lower()
            ok = False
            for pat in patterns:
                if pat.startswith(":"):
                    ok = pat[1:].lower() == low.split(":")[-1].lower()
                else:
                    ok = pat.lower().replace("_", " ") in low.replace("_", " ") or pat.lower() in low
                if ok:
                    selected[out_name] = sd
                    break
            if out_name in selected:
                break

    return selected, subs

def download_file(session, url, outpath):
    if outpath.exists() and outpath.stat().st_size > 0:
        return "ALREADY_PRESENT", outpath.stat().st_size, ""

    try:
        with session.get(url, stream=True, timeout=240) as resp:
            resp.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return "DOWNLOADED", outpath.stat().st_size, ""
    except Exception as e:
        return "DOWNLOAD_ERROR", 0, str(e)[:1200]

def sample_layer(sd_path, point_df, product_group, layer_name):
    import rasterio
    from pyproj import Transformer

    rows = []

    scale = PRODUCT_SDS[product_group]["scale"].get(layer_name, 1.0)
    offset = PRODUCT_SDS[product_group]["offset"].get(layer_name, 0.0)

    with rasterio.open(sd_path) as src:
        crs = src.crs
        nodata = src.nodata
        tags = src.tags()

        try:
            tag_scale = tags.get("scale_factor", None)
            if tag_scale not in [None, ""]:
                scale = float(tag_scale)
        except Exception:
            pass

        try:
            tag_offset = tags.get("add_offset", None)
            if tag_offset not in [None, ""]:
                offset = float(tag_offset)
        except Exception:
            pass

        if crs is not None:
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        else:
            transformer = None

        for _, p in point_df.iterrows():
            lon = float(p["longitude"])
            lat = float(p["latitude"])

            try:
                if transformer is not None:
                    x, y = transformer.transform(lon, lat)
                else:
                    x, y = lon, lat

                row, col = src.index(x, y)

                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    rows.append({
                        "id": p["id"],
                        "latitude": lat,
                        "longitude": lon,
                        "row": row,
                        "col": col,
                        "raw_value": np.nan,
                        "scaled_value": np.nan,
                        "scale_factor": scale,
                        "add_offset": offset,
                        "nodata": nodata,
                        "sample_status": "OUTSIDE_RASTER",
                    })
                    continue

                arr = src.read(1, window=((row, row + 1), (col, col + 1)))
                raw = arr[0, 0]

                if nodata is not None and raw == nodata:
                    scaled = np.nan
                    status = "NODATA"
                elif not np.isfinite(raw):
                    scaled = np.nan
                    status = "NONFINITE"
                else:
                    scaled = float(raw) * scale + offset
                    status = "OK"

                rows.append({
                    "id": p["id"],
                    "latitude": lat,
                    "longitude": lon,
                    "row": row,
                    "col": col,
                    "raw_value": float(raw) if np.isfinite(raw) else np.nan,
                    "scaled_value": scaled,
                    "scale_factor": scale,
                    "add_offset": offset,
                    "nodata": nodata,
                    "sample_status": status,
                })
            except Exception as e:
                rows.append({
                    "id": p["id"],
                    "latitude": lat,
                    "longitude": lon,
                    "row": "",
                    "col": "",
                    "raw_value": np.nan,
                    "scaled_value": np.nan,
                    "scale_factor": scale,
                    "add_offset": offset,
                    "nodata": nodata,
                    "sample_status": "SAMPLE_ERROR",
                    "error": str(e)[:500],
                })

    return pd.DataFrame(rows)

try:
    import earthaccess
    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()
    auth_status = "EARTHACCESS_SESSION_OK"
except Exception as e:
    session = requests.Session()
    auth_status = "EARTHACCESS_SESSION_ERROR_FALLBACK_REQUESTS"
    auth_error = str(e)[:1000]

manifest = manifest[manifest["product_group"].isin(PRODUCT_SDS.keys())].copy()
manifest = manifest.dropna(subset=["url", "filename"])
manifest = manifest[manifest["url"].astype(str).str.contains(".hdf", regex=False)].copy()

if PRODUCT_FILTER != "ALL":
    keep = [x.strip() for x in PRODUCT_FILTER.split(",") if x.strip()]
    manifest = manifest[manifest["product_group"].isin(keep)].copy()

run_rows = []
sds_rows = []

for product_group, sub in manifest.groupby("product_group"):
    sub = sub.sort_values("filename").copy()

    if MAX_GRANULES_PER_PRODUCT > 0:
        sub = sub.head(MAX_GRANULES_PER_PRODUCT)

    product_shard_dir = SHARDS / product_group
    product_shard_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(sub)
    n_done_before = 0
    n_processed_now = 0

    for idx, (_, row) in enumerate(sub.iterrows(), start=1):
        fn = str(row["filename"])
        url = str(row["url"])
        tile = str(row.get("tile", "")) or tile_from_filename(fn)
        date, year, doy = parse_date_from_modis_filename(fn)

        shard_path = product_shard_dir / f"{safe_name(fn)}.csv"

        if shard_path.exists() and shard_path.stat().st_size > 0:
            n_done_before += 1
            if idx % 250 == 0:
                print(f"{product_group}: {idx}/{n_total} checked, already done {n_done_before}")
            continue

        site_ids = sites_for_manifest_row(row)
        if not site_ids:
            run_rows.append({
                "product_group": product_group,
                "filename": fn,
                "date": date,
                "year": year,
                "doy": doy,
                "tile": tile,
                "status": "SKIPPED_NO_FINAL_TARGET_SITE_IN_GRANULE",
                "n_sites_for_granule": 0,
            })
            continue

        psub = points[points["id"].isin(site_ids)].copy()
        tmp_path = TMP / product_group / fn
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        dl_status, size_bytes, dl_error = download_file(session, url, tmp_path)

        base_log = {
            "product_group": product_group,
            "filename": fn,
            "date": date,
            "year": year,
            "doy": doy,
            "tile": tile,
            "n_sites_for_granule": len(psub),
            "download_status": dl_status,
            "size_bytes": size_bytes,
            "download_error": dl_error,
        }

        if dl_status == "DOWNLOAD_ERROR":
            run_rows.append({**base_log, "status": "DOWNLOAD_ERROR"})
            continue

        try:
            selected, all_subs = select_subdatasets(tmp_path, product_group)
            sds_rows.append({
                "product_group": product_group,
                "filename": fn,
                "n_subdatasets": len(all_subs),
                "selected_layers": ";".join(selected.keys()),
                "all_expected_layers_found": set(PRODUCT_SDS[product_group]["layers"]).issubset(set(selected.keys())),
            })
        except Exception as e:
            run_rows.append({**base_log, "status": "SUBDATASET_SELECT_ERROR", "error": str(e)[:1000]})
            if not KEEP_HDF and tmp_path.exists():
                tmp_path.unlink()
            continue

        if not set(PRODUCT_SDS[product_group]["layers"]).issubset(set(selected.keys())):
            run_rows.append({
                **base_log,
                "status": "MISSING_EXPECTED_SUBDATASET",
                "selected_layers": ";".join(selected.keys()),
                "expected_layers": ";".join(PRODUCT_SDS[product_group]["layers"].keys()),
            })
            if not KEEP_HDF and tmp_path.exists():
                tmp_path.unlink()
            continue

        layer_dfs = []
        for layer_name, sd_path in selected.items():
            try:
                sampled = sample_layer(sd_path, psub, product_group, layer_name)
                sampled["product_group"] = product_group
                sampled["filename"] = fn
                sampled["date"] = date
                sampled["year"] = year
                sampled["doy"] = doy
                sampled["tile"] = tile
                sampled["layer"] = layer_name
                sampled["url"] = url
                layer_dfs.append(sampled)
            except Exception as e:
                run_rows.append({
                    **base_log,
                    "status": "LAYER_SAMPLE_ERROR",
                    "layer": layer_name,
                    "error": str(e)[:1000],
                })

        if layer_dfs:
            shard = pd.concat(layer_dfs, ignore_index=True)
            shard.to_csv(shard_path, index=False)
            n_processed_now += 1
            run_rows.append({
                **base_log,
                "status": "SHARD_WRITTEN",
                "shard_path": str(shard_path),
                "n_rows_written": len(shard),
                "layers_written": ";".join(sorted(shard["layer"].unique())),
                "ok_rows": int(shard["sample_status"].eq("OK").sum()),
            })

        if not KEEP_HDF and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        if idx % 100 == 0:
            print(f"{product_group}: processed/checkpoint {idx}/{n_total}; new {n_processed_now}; already {n_done_before}")

run_log = pd.DataFrame(run_rows)
run_log.to_csv(TAB / "Table_PRODUCT02bf_full_direct_run_log.csv", index=False)

sds_log = pd.DataFrame(sds_rows)
sds_log.to_csv(TAB / "Table_PRODUCT02bg_full_direct_subdataset_log.csv", index=False)

combined_rows = []
coverage_rows = []

for product_group in PRODUCT_SDS:
    shard_dir = SHARDS / product_group
    shard_files = sorted(shard_dir.glob("*.csv")) if shard_dir.exists() else []

    if REBUILD_COMBINED and shard_files:
        parts = []
        for sp in shard_files:
            try:
                parts.append(pd.read_csv(sp))
            except Exception:
                pass
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            combined_path = OUTCSV / f"{product_group}_FULL_direct_earthdata_point_samples.csv"
            combined.to_csv(combined_path, index=False)

            combined_rows.append({
                "product_group": product_group,
                "combined_csv": str(combined_path),
                "n_rows": len(combined),
                "n_shards": len(shard_files),
                "n_sites": combined["id"].nunique(),
                "n_dates": combined["date"].nunique(),
                "layers": ";".join(sorted(combined["layer"].dropna().unique())),
                "ok_rows": int(combined["sample_status"].eq("OK").sum()),
            })
        else:
            combined = pd.DataFrame()
            combined_path = OUTCSV / f"{product_group}_FULL_direct_earthdata_point_samples.csv"
    else:
        combined_path = OUTCSV / f"{product_group}_FULL_direct_earthdata_point_samples.csv"
        if combined_path.exists():
            combined = pd.read_csv(combined_path)
        else:
            combined = pd.DataFrame()

    expected_layers = sorted(PRODUCT_SDS[product_group]["layers"].keys())

    if combined.empty:
        coverage_rows.append({
            "product_group": product_group,
            "combined_csv_exists": False,
            "n_rows": 0,
            "n_sites": 0,
            "n_dates": 0,
            "layers": "",
            "n_missing_sites": len(points),
            "missing_sites": ";".join(points["id"]),
            "coverage_status": "NO_COMBINED_OUTPUT",
        })
        continue

    missing_sites = sorted(set(points["id"]) - set(combined["id"].astype(str)))
    layers_present = sorted(set(combined["layer"].dropna().astype(str)))
    missing_layers = sorted(set(expected_layers) - set(layers_present))

    if product_group == "MODIS_LAI_MCD15":
        min_expected_dates = 1000
    elif product_group == "MCD64A1_BURNED_AREA":
        min_expected_dates = 250
    else:
        min_expected_dates = 1000

    site_date = combined.groupby("id")["date"].nunique().to_dict()
    weak_sites = [sid for sid in points["id"] if site_date.get(sid, 0) < min_expected_dates]

    if not missing_sites and not missing_layers and not weak_sites:
        status = "FULL_DIRECT_POINT_EXTRACTION_COMPLETE"
    else:
        status = "INCOMPLETE_NEEDS_RERUN_OR_REVIEW"

    coverage_rows.append({
        "product_group": product_group,
        "combined_csv_exists": True,
        "combined_csv": str(combined_path),
        "n_rows": len(combined),
        "n_sites": combined["id"].nunique(),
        "n_dates": combined["date"].nunique(),
        "date_min": combined["date"].min(),
        "date_max": combined["date"].max(),
        "layers": ";".join(layers_present),
        "n_missing_sites": len(missing_sites),
        "missing_sites": ";".join(missing_sites),
        "missing_layers": ";".join(missing_layers),
        "n_weak_date_sites": len(weak_sites),
        "weak_date_sites": ";".join(weak_sites),
        "min_expected_dates_per_site": min_expected_dates,
        "coverage_status": status,
    })

combined_summary = pd.DataFrame(combined_rows)
combined_summary.to_csv(TAB / "Table_PRODUCT02bh_full_direct_combined_output_summary.csv", index=False)

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02bi_full_direct_coverage_lock.csv", index=False)

all_complete = len(coverage) > 0 and coverage["coverage_status"].eq("FULL_DIRECT_POINT_EXTRACTION_COMPLETE").all()

decision = pd.DataFrame([{
    "stage": "1B.6J_full_direct_earthdata_extract",
    "no_gee": True,
    "no_appeears": True,
    "uses_direct_earthdata_hdf": True,
    "stores_full_hdf_tiles": KEEP_HDF,
    "full_point_extraction_complete": bool(all_complete),
    "decision": "COMPLETE_FOR_TOWER_CENTERED_PRODUCT_ARBITRATION" if all_complete else "NOT_COMPLETE_YET_RERUN_RESUMABLE_SCRIPT",
}])
decision.to_csv(TAB / "Table_PRODUCT02bj_full_direct_completion_decision.csv", index=False)

report = []
report.append("# Stage 1B.6J full direct Earthdata point extraction")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append(f"PRODUCT_FILTER: {PRODUCT_FILTER}")
report.append(f"MAX_GRANULES_PER_PRODUCT: {MAX_GRANULES_PER_PRODUCT}")
report.append(f"KEEP_HDF: {KEEP_HDF}")
report.append(f"Auth/session status: {auth_status}")
report.append("")
report.append("## Combined output summary")
report.append("")
report.append("```text")
report.append(combined_summary.to_string(index=False) if len(combined_summary) else "No combined outputs yet.")
report.append("```")
report.append("")
report.append("## Coverage lock")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False))
report.append("```")
report.append("")
report.append("## Completion decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("This stage is complete only when every required MODIS/MCD product has complete direct point time series for the final 13 grassland/open target points, with expected layers, dates, and QA layers present. No GEE or AppEEARS is used.")
report.append("")

(TXT / "STAGE1B6J_FULL_DIRECT_EARTHDATA_EXTRACT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6J_full_direct_earthdata_point_extract_no_gee_no_appeears",
    "status": "complete" if all_complete else "incomplete_rerun_resumable",
    "no_gee": True,
    "no_appeears": True,
    "outputs": {
        "run_log": str(TAB / "Table_PRODUCT02bf_full_direct_run_log.csv"),
        "subdataset_log": str(TAB / "Table_PRODUCT02bg_full_direct_subdataset_log.csv"),
        "combined_summary": str(TAB / "Table_PRODUCT02bh_full_direct_combined_output_summary.csv"),
        "coverage_lock": str(TAB / "Table_PRODUCT02bi_full_direct_coverage_lock.csv"),
        "completion_decision": str(TAB / "Table_PRODUCT02bj_full_direct_completion_decision.csv"),
        "report": str(TXT / "STAGE1B6J_FULL_DIRECT_EARTHDATA_EXTRACT_REPORT.md"),
        "output_dir": str(OUTCSV),
    }
}
(TAB / "STAGE1B6J_FULL_DIRECT_EARTHDATA_EXTRACT_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bf_full_direct_run_log.csv")
print("WROTE", TAB / "Table_PRODUCT02bg_full_direct_subdataset_log.csv")
print("WROTE", TAB / "Table_PRODUCT02bh_full_direct_combined_output_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02bi_full_direct_coverage_lock.csv")
print("WROTE", TAB / "Table_PRODUCT02bj_full_direct_completion_decision.csv")
print("WROTE", TXT / "STAGE1B6J_FULL_DIRECT_EARTHDATA_EXTRACT_REPORT.md")
