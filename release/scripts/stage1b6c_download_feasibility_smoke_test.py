from pathlib import Path
from datetime import datetime
import os
import re
import json
import shutil
import pandas as pd
import requests

OUT = Path("results/stage1b6c_download_feasibility")
TAB = OUT / "tables"
TXT = OUT / "text"
SMOKE = Path("data/raw_local/no_gee_downloads/_smoke_test")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
SMOKE.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02r_bbox_unique_download_manifest.csv")
SUMMARY = Path("results/stage1b6b_bbox_modis_search/tables/Table_PRODUCT02s_bbox_search_download_summary.csv")

if not MANIFEST.exists():
    raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
if not SUMMARY.exists():
    raise FileNotFoundError(f"Missing summary: {SUMMARY}")

manifest = pd.read_csv(MANIFEST)
summary = pd.read_csv(SUMMARY)

# Disk space check.
usage = shutil.disk_usage(".")
disk = {
    "total_gb": usage.total / 1e9,
    "used_gb": usage.used / 1e9,
    "free_gb": usage.free / 1e9,
}

estimated_total_gb = float(summary["estimated_size_mb_all_found"].sum()) / 1024.0
recommended_free_gb = estimated_total_gb * 1.25

disk_df = pd.DataFrame([{
    "free_gb": disk["free_gb"],
    "estimated_download_gb": estimated_total_gb,
    "recommended_free_gb_25pct_buffer": recommended_free_gb,
    "enough_space_for_full_main13_strict_download": disk["free_gb"] >= recommended_free_gb,
}])
disk_df.to_csv(TAB / "Table_PRODUCT02v_disk_space_feasibility.csv", index=False)

# Pick one small-ish file per product for smoke test.
sample_rows = []
for product_group, sub in manifest.groupby("product_group"):
    sub = sub.copy()
    sub["size_mb_num"] = pd.to_numeric(sub["size_mb"], errors="coerce")
    sub = sub.dropna(subset=["url"])
    sub = sub[sub["url"].astype(str).str.contains(".hdf", regex=False)]
    if len(sub) == 0:
        continue

    # Avoid zero-size weird entries; choose smallest nonzero file to test.
    sub2 = sub[sub["size_mb_num"] > 0].sort_values("size_mb_num")
    if len(sub2) == 0:
        sub2 = sub.sort_values("filename")
    sample_rows.append(sub2.iloc[0].to_dict())

samples = pd.DataFrame(sample_rows)
samples.to_csv(TAB / "Table_PRODUCT02w_smoke_test_selected_files.csv", index=False)

download_rows = []

# Use requests with Earthdata cookies from earthaccess if possible.
# Fall back to writing download commands if direct request fails.
try:
    import earthaccess
    auth = earthaccess.login(strategy="netrc")
    session = earthaccess.get_requests_https_session()
except Exception as e:
    session = requests.Session()
    auth = None
    print("WARNING: earthaccess session unavailable:", e)

for _, r in samples.iterrows():
    pg = r["product_group"]
    url = str(r["url"])
    fn = str(r["filename"])
    outdir = SMOKE / pg
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / fn

    status = "NOT_ATTEMPTED"
    error = ""

    if outpath.exists() and outpath.stat().st_size > 0:
        status = "ALREADY_PRESENT"
    else:
        try:
            print(f"Downloading smoke-test file for {pg}: {fn}")
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(outpath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            status = "DOWNLOADED"
        except Exception as e:
            status = "DOWNLOAD_ERROR"
            error = str(e)[:1000]

    size = outpath.stat().st_size if outpath.exists() else 0
    download_rows.append({
        "product_group": pg,
        "filename": fn,
        "url": url,
        "outpath": str(outpath),
        "status": status,
        "size_bytes_after": size,
        "error": error,
    })

download_df = pd.DataFrame(download_rows)
download_df.to_csv(TAB / "Table_PRODUCT02x_smoke_test_download_log.csv", index=False)

# Inspect HDF subdatasets with rasterio/GDAL.
inspect_rows = []
try:
    import rasterio
except Exception as e:
    rasterio = None
    rasterio_error = str(e)

for _, r in download_df.iterrows():
    pg = r["product_group"]
    path = Path(r["outpath"])

    if not path.exists() or path.stat().st_size == 0:
        inspect_rows.append({
            "product_group": pg,
            "path": str(path),
            "read_status": "FILE_MISSING_OR_EMPTY",
            "n_subdatasets": 0,
            "subdatasets": "",
            "expected_keywords_found": "",
            "error": "",
        })
        continue

    if rasterio is None:
        inspect_rows.append({
            "product_group": pg,
            "path": str(path),
            "read_status": "RASTERIO_NOT_AVAILABLE",
            "n_subdatasets": 0,
            "subdatasets": "",
            "expected_keywords_found": "",
            "error": rasterio_error,
        })
        continue

    try:
        with rasterio.open(str(path)) as src:
            subs = list(src.subdatasets or [])
            desc = list(src.descriptions or [])
            text = " ".join(subs + [str(x) for x in desc]).lower()

        expected = {
            "MODIS_GPP_MOD17": ["gpp", "psn", "qc"],
            "MODIS_ET_MOD16": ["et", "qc"],
            "MODIS_LAI_MCD15": ["lai", "fpar", "qc"],
            "MCD64A1_BURNED_AREA": ["burn", "burndate", "qa"],
        }.get(pg, [])

        hits = [k for k in expected if k in text]

        inspect_rows.append({
            "product_group": pg,
            "path": str(path),
            "read_status": "READABLE",
            "n_subdatasets": len(subs),
            "subdatasets": " || ".join(subs[:80]),
            "expected_keywords_found": ",".join(hits),
            "error": "",
        })
    except Exception as e:
        inspect_rows.append({
            "product_group": pg,
            "path": str(path),
            "read_status": "READ_ERROR",
            "n_subdatasets": 0,
            "subdatasets": "",
            "expected_keywords_found": "",
            "error": str(e)[:1000],
        })

inspect_df = pd.DataFrame(inspect_rows)
inspect_df.to_csv(TAB / "Table_PRODUCT02y_smoke_test_hdf_subdataset_inspection.csv", index=False)

# Decision table.
decision_rows = []
for _, r in summary.iterrows():
    pg = r["product_group"]
    n_to_download = int(r["n_to_download"])
    est_gb = float(r["estimated_size_mb_all_found"]) / 1024.0

    smoke_sub = download_df[download_df["product_group"].eq(pg)]
    insp_sub = inspect_df[inspect_df["product_group"].eq(pg)]

    smoke_ok = len(smoke_sub) > 0 and smoke_sub["status"].isin(["DOWNLOADED", "ALREADY_PRESENT"]).any()
    read_ok = len(insp_sub) > 0 and insp_sub["read_status"].eq("READABLE").any()
    keyword_ok = len(insp_sub) > 0 and insp_sub["expected_keywords_found"].fillna("").astype(str).str.len().gt(0).any()

    if disk_df["enough_space_for_full_main13_strict_download"].iloc[0] and smoke_ok and read_ok and keyword_ok:
        decision = "FULL_DOWNLOAD_ALLOWED"
        next_action = "Run full DO_DOWNLOAD=1 command for this product or all products."
    elif not disk_df["enough_space_for_full_main13_strict_download"].iloc[0]:
        decision = "DO_NOT_FULL_DOWNLOAD_YET_STORAGE_INSUFFICIENT_OR_TOO_RISKY"
        next_action = "Use existing AppEEARS point CSV for tower-only MODIS, or attach external drive / free space before strict gridded download."
    elif not smoke_ok:
        decision = "DOWNLOAD_AUTH_OR_NETWORK_PROBLEM"
        next_action = "Fix Earthdata authentication/download before full download."
    elif not read_ok or not keyword_ok:
        decision = "HDF_READ_OR_SUBDATASET_PROBLEM"
        next_action = "Fix GDAL/rasterio/HDF reading or inspect subdataset names manually."
    else:
        decision = "MANUAL_REVIEW"

    decision_rows.append({
        "product_group": pg,
        "n_to_download": n_to_download,
        "estimated_gb": est_gb,
        "smoke_download_ok": smoke_ok,
        "hdf_read_ok": read_ok,
        "expected_subdataset_keyword_ok": keyword_ok,
        "decision": decision,
        "next_action": next_action,
    })

decision_df = pd.DataFrame(decision_rows)
decision_df.to_csv(TAB / "Table_PRODUCT02z_full_download_decision.csv", index=False)

report = []
report.append("# Stage 1B.6C download feasibility and HDF smoke test")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Disk feasibility")
report.append("")
report.append("```text")
report.append(disk_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Smoke-test selected files")
report.append("")
report.append("```text")
report.append(samples[["product_group", "filename", "size_mb", "url"]].to_string(index=False) if len(samples) else "No sample files selected.")
report.append("```")
report.append("")
report.append("## Smoke-test download log")
report.append("")
report.append("```text")
report.append(download_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## HDF subdataset inspection")
report.append("")
report.append("```text")
report.append(inspect_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Full download decision")
report.append("")
report.append("```text")
report.append(decision_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Full strict acquisition is not complete until the raw gridded HDF files are present locally or a deliberate tower-only point-sample limitation is accepted.")
report.append("")

(TXT / "STAGE1B6C_DOWNLOAD_FEASIBILITY_SMOKE_TEST_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6C_download_feasibility_smoke_test",
    "status": "complete",
    "estimated_total_download_gb": estimated_total_gb,
    "free_gb": disk["free_gb"],
    "enough_space": bool(disk_df["enough_space_for_full_main13_strict_download"].iloc[0]),
    "outputs": {
        "disk": str(TAB / "Table_PRODUCT02v_disk_space_feasibility.csv"),
        "samples": str(TAB / "Table_PRODUCT02w_smoke_test_selected_files.csv"),
        "download_log": str(TAB / "Table_PRODUCT02x_smoke_test_download_log.csv"),
        "subdataset_inspection": str(TAB / "Table_PRODUCT02y_smoke_test_hdf_subdataset_inspection.csv"),
        "decision": str(TAB / "Table_PRODUCT02z_full_download_decision.csv"),
        "report": str(TXT / "STAGE1B6C_DOWNLOAD_FEASIBILITY_SMOKE_TEST_REPORT.md"),
    }
}
(TAB / "STAGE1B6C_DOWNLOAD_FEASIBILITY_SMOKE_TEST_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02v_disk_space_feasibility.csv")
print("WROTE", TAB / "Table_PRODUCT02w_smoke_test_selected_files.csv")
print("WROTE", TAB / "Table_PRODUCT02x_smoke_test_download_log.csv")
print("WROTE", TAB / "Table_PRODUCT02y_smoke_test_hdf_subdataset_inspection.csv")
print("WROTE", TAB / "Table_PRODUCT02z_full_download_decision.csv")
print("WROTE", TXT / "STAGE1B6C_DOWNLOAD_FEASIBILITY_SMOKE_TEST_REPORT.md")
