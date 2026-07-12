from pathlib import Path
from datetime import datetime
import os
import re
import json
import time
import math
import pandas as pd
import numpy as np
import requests

OUT = Path("results/stage1b6h_direct_earthdata_point_extract")
TAB = OUT / "tables"
TXT = OUT / "text"
OUTCSV = Path("data/raw_local/no_gee_direct_point_extract")
TMP = Path("tmp/stage1b6h_hdf_stream")

TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
OUTCSV.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02r_bbox_unique_download_manifest.csv")
POINTS = Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv")

MAX_GRANULES_PER_PRODUCT = int(os.environ.get("MAX_GRANULES_PER_PRODUCT", "5"))
KEEP_HDF = os.environ.get("KEEP_HDF", "0").strip() == "1"
START_FROM = os.environ.get("START_FROM", "").strip()

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not POINTS.exists():
    raise FileNotFoundError(f"Missing points: {POINTS}")

manifest = pd.read_csv(MANIFEST)
points = pd.read_csv(POINTS)

points.columns = [str(c).strip() for c in points.columns]
manifest.columns = [str(c).strip() for c in manifest.columns]

for c in ["id", "latitude", "longitude"]:
    if c not in points.columns:
        raise ValueError(f"Point file missing {c}. Found {list(points.columns)}")

points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
points = points.dropna(subset=["latitude", "longitude"]).copy()

PRODUCT_SDS = {
    "MODIS_GPP_MOD17": {
        "short_name": "MOD17A2HGF",
        "wanted": {
            "Gpp_500m": ["Gpp_500m"],
            "Psn_QC_500m": ["Psn_QC_500m"],
        }
    },
    "MODIS_ET_MOD16": {
        "short_name": "MOD16A2GF",
        "wanted": {
            "ET_500m": ["ET_500m"],
            "ET_QC_500m": ["ET_QC_500m"],
        }
    },
    "MODIS_LAI_MCD15": {
        "short_name": "MCD15A2H",
        "wanted": {
            "Lai_500m": ["Lai_500m"],
            "FparLai_QC": ["FparLai_QC"],
        }
    },
    "MCD64A1_BURNED_AREA": {
        "short_name": "MCD64A1",
        "wanted": {
            "Burn_Date": ["Burn Date", "Burn_Date"],
            "QA": [":QA", "QA"],
        }
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

def get_tile(fn):
    m = re.search(r"\.(h\d{2}v\d{2})\.", str(fn))
    return m.group(1) if m else ""

def sites_for_manifest_row(row):
    s = str(row.get("sites_hit", "") or "")
    vals = [x.strip() for x in s.split(";") if x.strip()]
    if vals:
        return vals
    return list(points["id"].astype(str))

def select_subdatasets(hdf_path, product_group):
    import rasterio
    wanted = PRODUCT_SDS[product_group]["wanted"]
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
        with session.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return "DOWNLOADED", outpath.stat().st_size, ""
    except Exception as e:
        return "DOWNLOAD_ERROR", 0, str(e)[:1000]

def sample_subdataset(sd_path, point_df):
    import rasterio
    from pyproj import Transformer

    rows = []

    with rasterio.open(sd_path) as src:
        crs = src.crs
        transform = src.transform
        nodata = src.nodata
        tags = src.tags()

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
                        "sample_status": "OUTSIDE_RASTER",
                        "nodata": nodata,
                        "scale_factor_tag": tags.get("scale_factor", ""),
                        "add_offset_tag": tags.get("add_offset", ""),
                    })
                    continue

                arr = src.read(1, window=((row, row + 1), (col, col + 1)))
                val = arr[0, 0]

                if nodata is not None and val == nodata:
                    status = "NODATA"
                else:
                    status = "OK"

                rows.append({
                    "id": p["id"],
                    "latitude": lat,
                    "longitude": lon,
                    "row": row,
                    "col": col,
                    "raw_value": float(val) if np.isfinite(val) else np.nan,
                    "sample_status": status,
                    "nodata": nodata,
                    "scale_factor_tag": tags.get("scale_factor", ""),
                    "add_offset_tag": tags.get("add_offset", ""),
                })
            except Exception as e:
                rows.append({
                    "id": p["id"],
                    "latitude": lat,
                    "longitude": lon,
                    "row": "",
                    "col": "",
                    "raw_value": np.nan,
                    "sample_status": "SAMPLE_ERROR",
                    "nodata": nodata,
                    "scale_factor_tag": tags.get("scale_factor", ""),
                    "add_offset_tag": tags.get("add_offset", ""),
                    "error": str(e)[:500],
                })

    return pd.DataFrame(rows)

# Login/session.
try:
    import earthaccess
    earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()
    auth_status = "EARTHACCESS_SESSION_OK"
except Exception as e:
    session = requests.Session()
    auth_status = "EARTHACCESS_SESSION_ERROR_FALLBACK_REQUESTS"
    auth_error = str(e)[:1000]

run_rows = []
sample_rows = []
sds_rows = []

manifest = manifest[manifest["product_group"].isin(PRODUCT_SDS.keys())].copy()
manifest = manifest.dropna(subset=["url", "filename"])
manifest = manifest[manifest["url"].astype(str).str.contains(".hdf", regex=False)].copy()

if START_FROM:
    manifest = manifest[manifest["filename"].astype(str) >= START_FROM].copy()

for product_group, sub in manifest.groupby("product_group"):
    sub = sub.sort_values("filename").copy()

    if MAX_GRANULES_PER_PRODUCT > 0:
        sub = sub.head(MAX_GRANULES_PER_PRODUCT)

    product_out_rows = []

    for i, row in sub.iterrows():
        fn = str(row["filename"])
        url = str(row["url"])
        tile = str(row.get("tile", "")) or get_tile(fn)
        date, year, doy = parse_date_from_modis_filename(fn)

        tmp_path = TMP / product_group / fn
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        dl_status, size_bytes, dl_error = download_file(session, url, tmp_path)

        run_rows.append({
            "product_group": product_group,
            "filename": fn,
            "date": date,
            "year": year,
            "doy": doy,
            "tile": tile,
            "download_status": dl_status,
            "size_bytes": size_bytes,
            "download_error": dl_error,
        })

        if dl_status == "DOWNLOAD_ERROR":
            continue

        try:
            selected, all_subs = select_subdatasets(tmp_path, product_group)
            sds_rows.append({
                "product_group": product_group,
                "filename": fn,
                "n_subdatasets": len(all_subs),
                "selected_layers": ";".join(selected.keys()),
                "all_subdatasets": " || ".join(all_subs[:80]),
            })
        except Exception as e:
            run_rows[-1]["subdataset_error"] = str(e)[:1000]
            if not KEEP_HDF and tmp_path.exists():
                tmp_path.unlink()
            continue

        site_ids = sites_for_manifest_row(row)
        psub = points[points["id"].astype(str).isin(site_ids)].copy()
        if len(psub) == 0:
            psub = points.copy()

        for layer_name, sd_path in selected.items():
            try:
                sampled = sample_subdataset(sd_path, psub)
                sampled["product_group"] = product_group
                sampled["filename"] = fn
                sampled["date"] = date
                sampled["year"] = year
                sampled["doy"] = doy
                sampled["tile"] = tile
                sampled["layer"] = layer_name
                sampled["url"] = url
                product_out_rows.append(sampled)
            except Exception as e:
                sample_rows.append({
                    "product_group": product_group,
                    "filename": fn,
                    "date": date,
                    "tile": tile,
                    "layer": layer_name,
                    "id": "",
                    "sample_status": "LAYER_SAMPLE_ERROR",
                    "error": str(e)[:1000],
                })

        if not KEEP_HDF and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    if product_out_rows:
        prod_df = pd.concat(product_out_rows, ignore_index=True)
        prod_path = OUTCSV / f"{product_group}_direct_earthdata_point_samples.csv"
        prod_df.to_csv(prod_path, index=False)

        sample_rows.append({
            "product_group": product_group,
            "output_csv": str(prod_path),
            "n_rows": len(prod_df),
            "n_files": prod_df["filename"].nunique(),
            "n_dates": prod_df["date"].nunique(),
            "n_sites": prod_df["id"].nunique(),
            "layers": ";".join(sorted(prod_df["layer"].unique())),
            "ok_rows": int(prod_df["sample_status"].eq("OK").sum()),
        })

run_log = pd.DataFrame(run_rows)
run_log.to_csv(TAB / "Table_PRODUCT02ax_direct_extract_run_log.csv", index=False)

sds_df = pd.DataFrame(sds_rows)
sds_df.to_csv(TAB / "Table_PRODUCT02ay_direct_extract_subdataset_log.csv", index=False)

summary = pd.DataFrame(sample_rows)
summary.to_csv(TAB / "Table_PRODUCT02az_direct_extract_output_summary.csv", index=False)

coverage_rows = []
for product_group in PRODUCT_SDS:
    prod_path = OUTCSV / f"{product_group}_direct_earthdata_point_samples.csv"
    if not prod_path.exists():
        coverage_rows.append({
            "product_group": product_group,
            "csv_exists": False,
            "n_rows": 0,
            "n_sites": 0,
            "n_dates": 0,
            "layers": "",
            "coverage_status": "NO_OUTPUT",
        })
        continue

    df = pd.read_csv(prod_path)
    coverage_rows.append({
        "product_group": product_group,
        "csv_exists": True,
        "n_rows": len(df),
        "n_sites": df["id"].nunique() if "id" in df.columns else 0,
        "n_dates": df["date"].nunique() if "date" in df.columns else 0,
        "layers": ";".join(sorted(df["layer"].dropna().unique())) if "layer" in df.columns else "",
        "coverage_status": "SMOKE_OK" if MAX_GRANULES_PER_PRODUCT > 0 else "FULL_RUN_OUTPUT_CREATED",
    })

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02ba_direct_extract_coverage_summary.csv", index=False)

mode = "SMOKE_LIMITED" if MAX_GRANULES_PER_PRODUCT > 0 else "FULL_ALL_MANIFEST_GRANULES"

report = []
report.append("# Stage 1B.6H direct Earthdata point extraction, no AppEEARS")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append(f"Mode: {mode}")
report.append(f"MAX_GRANULES_PER_PRODUCT: {MAX_GRANULES_PER_PRODUCT}")
report.append(f"Auth/session status: {auth_status}")
report.append("")
report.append("## Output summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False) if len(summary) else "No output samples created.")
report.append("```")
report.append("")
report.append("## Coverage summary")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False))
report.append("```")
report.append("")
report.append("## Run log")
report.append("")
report.append("```text")
report.append(run_log.head(80).to_string(index=False) if len(run_log) else "No run rows.")
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Smoke mode is complete only if every product creates readable point-sample CSVs. Full mode is complete only if every product has complete 2001-2024 point time series at the final strict 13 target points.")
report.append("")

(TXT / "STAGE1B6H_DIRECT_EARTHDATA_POINT_EXTRACT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6H_direct_earthdata_point_extract_no_appeears",
    "status": "complete",
    "mode": mode,
    "max_granules_per_product": MAX_GRANULES_PER_PRODUCT,
    "auth_status": auth_status,
    "outputs": {
        "run_log": str(TAB / "Table_PRODUCT02ax_direct_extract_run_log.csv"),
        "subdataset_log": str(TAB / "Table_PRODUCT02ay_direct_extract_subdataset_log.csv"),
        "output_summary": str(TAB / "Table_PRODUCT02az_direct_extract_output_summary.csv"),
        "coverage_summary": str(TAB / "Table_PRODUCT02ba_direct_extract_coverage_summary.csv"),
        "report": str(TXT / "STAGE1B6H_DIRECT_EARTHDATA_POINT_EXTRACT_REPORT.md"),
        "output_dir": str(OUTCSV),
    }
}

(TAB / "STAGE1B6H_DIRECT_EARTHDATA_POINT_EXTRACT_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02ax_direct_extract_run_log.csv")
print("WROTE", TAB / "Table_PRODUCT02ay_direct_extract_subdataset_log.csv")
print("WROTE", TAB / "Table_PRODUCT02az_direct_extract_output_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02ba_direct_extract_coverage_summary.csv")
print("WROTE", TXT / "STAGE1B6H_DIRECT_EARTHDATA_POINT_EXTRACT_REPORT.md")
