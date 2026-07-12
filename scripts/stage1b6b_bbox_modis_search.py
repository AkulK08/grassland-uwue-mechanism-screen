from pathlib import Path
from datetime import datetime
import os
import re
import json
import pandas as pd
import earthaccess

OUT = Path("results/stage1b6b_bbox_modis_search")
TAB = OUT / "tables"
TXT = OUT / "text"
BASE_DL = Path("data/raw_local/no_gee_downloads")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
BASE_DL.mkdir(parents=True, exist_ok=True)

DO_DOWNLOAD = os.environ.get("DO_DOWNLOAD", "0").strip() == "1"
TARGET_SCOPE = os.environ.get("TARGET_SCOPE", "main13").strip()
START_YEAR = int(os.environ.get("START_YEAR", "2001"))
END_YEAR = int(os.environ.get("END_YEAR", "2024"))
BBOX_PAD_DEG = float(os.environ.get("BBOX_PAD_DEG", "0.03"))
COUNT_PER_SITE_PRODUCT = int(os.environ.get("COUNT_PER_SITE_PRODUCT", "5000"))

coord_options = {
    "main13": [
        Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv"),
        Path("results/tower_satellite_extraction_targets_FINAL/main_13_coordinates.csv"),
    ],
    "strict5": [
        Path("results/tower_satellite_extraction_targets_FINAL/SENSITIVITY_strict_GRA_coordinates.csv"),
    ],
    "all49": [
        Path("results/tower_satellite_extraction_targets_FINAL/CONTRAST_all_49_tower_coordinates.csv"),
    ],
}

coord_path = None
for p in coord_options.get(TARGET_SCOPE, coord_options["main13"]):
    if p.exists():
        coord_path = p
        break

if coord_path is None:
    raise FileNotFoundError(f"Could not find coordinate file for TARGET_SCOPE={TARGET_SCOPE}")

coords = pd.read_csv(coord_path)

lat_col = None
lon_col = None
id_col = None

for c in coords.columns:
    cl = c.lower()
    if cl in ["latitude", "lat"]:
        lat_col = c
    if cl in ["longitude", "lon", "long"]:
        lon_col = c
    if cl in ["target_id", "tower_id", "site", "site_id", "id", "point_id"]:
        id_col = c

if lat_col is None or lon_col is None:
    raise ValueError(f"Could not find lat/lon columns in {coord_path}: {list(coords.columns)}")

if id_col is None:
    coords["target_id"] = [f"target_{i:03d}" for i in range(len(coords))]
    id_col = "target_id"

coords = coords[[id_col, lat_col, lon_col]].rename(columns={id_col: "target_id", lat_col: "lat", lon_col: "lon"})
coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
coords = coords.dropna(subset=["lat", "lon"]).drop_duplicates()

coords.to_csv(TAB / "Table_PRODUCT02p_bbox_search_coordinates.csv", index=False)

PRODUCTS = [
    {
        "product_group": "MODIS_GPP_MOD17",
        "short_name": "MOD17A2HGF",
        "version": "061",
        "temporal": (f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"),
        "target_dir": BASE_DL / "modis_mod17_gpp",
    },
    {
        "product_group": "MODIS_ET_MOD16",
        "short_name": "MOD16A2GF",
        "version": "061",
        "temporal": (f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"),
        "target_dir": BASE_DL / "modis_mod16_et",
    },
    {
        "product_group": "MODIS_LAI_MCD15",
        "short_name": "MCD15A2H",
        "version": "061",
        "temporal": ("2002-07-04", f"{END_YEAR}-12-31"),
        "target_dir": BASE_DL / "modis_mcd15_lai",
    },
    {
        "product_group": "MCD64A1_BURNED_AREA",
        "short_name": "MCD64A1",
        "version": "061",
        "temporal": (f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"),
        "target_dir": BASE_DL / "modis_mcd64_burned_area",
    },
]

for p in PRODUCTS:
    p["target_dir"].mkdir(parents=True, exist_ok=True)

def get_links(result):
    try:
        links = result.data_links()
        if links:
            return links
    except Exception:
        pass
    return re.findall(r"https?://[^'\"]+?\.hdf", str(result))

def get_filename(result):
    links = get_links(result)
    for link in links:
        m = re.search(r"/([^/]+\.hdf)", link)
        if m:
            return m.group(1)
    return ""

def get_tile(filename):
    m = re.search(r"\.(h\d{2}v\d{2})\.", filename)
    return m.group(1) if m else ""

def get_size(result):
    try:
        return float(result.size())
    except Exception:
        try:
            return float(result.size)
        except Exception:
            return None

auth = earthaccess.login(strategy="netrc")

rows = []
unique_results = {}

for product in PRODUCTS:
    pg = product["product_group"]
    short = product["short_name"]
    version = product["version"]
    temporal = product["temporal"]

    for _, site in coords.iterrows():
        site_id = str(site["target_id"])
        lat = float(site["lat"])
        lon = float(site["lon"])

        bbox = (
            max(-180, lon - BBOX_PAD_DEG),
            max(-90, lat - BBOX_PAD_DEG),
            min(180, lon + BBOX_PAD_DEG),
            min(90, lat + BBOX_PAD_DEG),
        )

        try:
            results = list(earthaccess.search_data(
                short_name=short,
                version=version,
                temporal=temporal,
                bounding_box=bbox,
                count=COUNT_PER_SITE_PRODUCT,
            ))
            status = "FOUND" if results else "ZERO_RESULTS"
            err = ""
        except Exception as e:
            results = []
            status = "ERROR"
            err = str(e)[:1000]

        print(f"{pg} {short} {site_id}: {len(results)} granules")

        for r in results:
            fn = get_filename(r)
            links = get_links(r)
            url = links[0] if links else ""
            tile = get_tile(fn)
            key = (pg, fn or url)

            if key not in unique_results:
                unique_results[key] = {
                    "result": r,
                    "product_group": pg,
                    "short_name": short,
                    "version": version,
                    "filename": fn,
                    "tile": tile,
                    "url": url,
                    "size_mb": get_size(r),
                    "target_dir": str(product["target_dir"]),
                    "sites": set(),
                }
            unique_results[key]["sites"].add(site_id)

        rows.append({
            "product_group": pg,
            "short_name": short,
            "version": version,
            "target_id": site_id,
            "lat": lat,
            "lon": lon,
            "bbox": str(bbox),
            "temporal_start": temporal[0],
            "temporal_end": temporal[1],
            "search_status": status,
            "n_results": len(results),
            "error": err,
        })

site_search = pd.DataFrame(rows)
site_search.to_csv(TAB / "Table_PRODUCT02q_bbox_search_by_site_product.csv", index=False)

manifest_rows = []
download_results_by_product = {}

for key, item in unique_results.items():
    pg = item["product_group"]
    fn = item["filename"]
    target_dir = Path(item["target_dir"])
    already = bool(fn and (target_dir / fn).exists())

    manifest_rows.append({
        "product_group": pg,
        "short_name": item["short_name"],
        "version": item["version"],
        "filename": fn,
        "tile": item["tile"],
        "url": item["url"],
        "size_mb": item["size_mb"],
        "target_dir": str(target_dir),
        "n_sites_hit": len(item["sites"]),
        "sites_hit": ";".join(sorted(item["sites"])),
        "already_present": already,
    })

    if not already:
        download_results_by_product.setdefault(pg, []).append(item["result"])

manifest = pd.DataFrame(manifest_rows)
if len(manifest):
    manifest = manifest.sort_values(["product_group", "tile", "filename"])
manifest.to_csv(TAB / "Table_PRODUCT02r_bbox_unique_download_manifest.csv", index=False)

summary_rows = []
for product in PRODUCTS:
    pg = product["product_group"]
    sub_site = site_search[site_search["product_group"].eq(pg)]
    sub_man = manifest[manifest["product_group"].eq(pg)] if len(manifest) else pd.DataFrame()

    n_zero_sites = int((sub_site["n_results"] == 0).sum()) if len(sub_site) else 0
    n_error_sites = int((sub_site["search_status"] == "ERROR").sum()) if len(sub_site) else 0
    n_files = int(sub_man["filename"].nunique()) if len(sub_man) and "filename" in sub_man else 0
    n_tiles = int(sub_man["tile"].replace("", pd.NA).dropna().nunique()) if len(sub_man) and "tile" in sub_man else 0
    n_already = int(sub_man["already_present"].sum()) if len(sub_man) and "already_present" in sub_man else 0
    n_to_download = len(download_results_by_product.get(pg, []))
    size_total = float(pd.to_numeric(sub_man["size_mb"], errors="coerce").sum()) if len(sub_man) and "size_mb" in sub_man else 0.0

    summary_rows.append({
        "product_group": pg,
        "short_name": product["short_name"],
        "n_sites": int(len(coords)),
        "n_sites_zero_results": n_zero_sites,
        "n_sites_errors": n_error_sites,
        "n_unique_tiles": n_tiles,
        "n_unique_files_found": n_files,
        "n_already_present": n_already,
        "n_to_download": n_to_download,
        "estimated_size_mb_all_found": size_total,
        "target_dir": str(product["target_dir"]),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "Table_PRODUCT02s_bbox_search_download_summary.csv", index=False)

download_rows = []
if DO_DOWNLOAD:
    for product in PRODUCTS:
        pg = product["product_group"]
        results = download_results_by_product.get(pg, [])
        target_dir = str(product["target_dir"])

        if not results:
            download_rows.append({
                "product_group": pg,
                "target_dir": target_dir,
                "status": "NOTHING_TO_DOWNLOAD",
                "n_requested": 0,
            })
            continue

        print("")
        print(f"Downloading {len(results)} files for {pg} into {target_dir}")

        try:
            earthaccess.download(results, local_path=target_dir)
            status = "DOWNLOAD_CALLED"
        except Exception as e:
            status = "DOWNLOAD_ERROR: " + str(e)[:500]

        download_rows.append({
            "product_group": pg,
            "target_dir": target_dir,
            "status": status,
            "n_requested": len(results),
        })
else:
    for product in PRODUCTS:
        pg = product["product_group"]
        download_rows.append({
            "product_group": pg,
            "target_dir": str(product["target_dir"]),
            "status": "DRY_RUN_NO_DOWNLOAD",
            "n_requested": len(download_results_by_product.get(pg, [])),
        })

download_log = pd.DataFrame(download_rows)
download_log.to_csv(TAB / "Table_PRODUCT02t_bbox_download_log.csv", index=False)

post_rows = []
for product in PRODUCTS:
    target_dir = product["target_dir"]
    files = sorted(target_dir.rglob("*.hdf"))
    post_rows.append({
        "product_group": product["product_group"],
        "short_name": product["short_name"],
        "target_dir": str(target_dir),
        "n_hdf_files_present_after_run": len(files),
        "example_files": "; ".join(f.name for f in files[:10]),
    })

post = pd.DataFrame(post_rows)
post.to_csv(TAB / "Table_PRODUCT02u_bbox_post_download_file_counts.csv", index=False)

report = []
report.append("# Stage 1B.6b coordinate-bbox MODIS/MCD search/download")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append(f"Target scope: {TARGET_SCOPE}")
report.append(f"Coordinate file: {coord_path}")
report.append(f"Download mode: {'DOWNLOAD ENABLED' if DO_DOWNLOAD else 'DRY RUN ONLY'}")
report.append("")
report.append("## Search/download summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Download log")
report.append("")
report.append("```text")
report.append(download_log.to_string(index=False))
report.append("```")
report.append("")
report.append("## Post-download file counts")
report.append("")
report.append("```text")
report.append(post.to_string(index=False))
report.append("```")
report.append("")
report.append("## Rule")
report.append("")
report.append("If n_sites_zero_results is 0 for each product, the coordinate search is working. If DO_DOWNLOAD=0, rerun with DO_DOWNLOAD=1 after reviewing the estimated download size.")
report.append("")

(TXT / "STAGE1B6B_BBOX_MODIS_SEARCH_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6b_bbox_modis_search_download",
    "status": "download_called" if DO_DOWNLOAD else "dry_run_complete",
    "target_scope": TARGET_SCOPE,
    "do_download": DO_DOWNLOAD,
    "coordinate_file": str(coord_path),
    "outputs": {
        "site_search": str(TAB / "Table_PRODUCT02q_bbox_search_by_site_product.csv"),
        "manifest": str(TAB / "Table_PRODUCT02r_bbox_unique_download_manifest.csv"),
        "summary": str(TAB / "Table_PRODUCT02s_bbox_search_download_summary.csv"),
        "download_log": str(TAB / "Table_PRODUCT02t_bbox_download_log.csv"),
        "post_counts": str(TAB / "Table_PRODUCT02u_bbox_post_download_file_counts.csv"),
        "report": str(TXT / "STAGE1B6B_BBOX_MODIS_SEARCH_REPORT.md"),
    }
}
(TAB / "STAGE1B6B_BBOX_MODIS_SEARCH_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02q_bbox_search_by_site_product.csv")
print("WROTE", TAB / "Table_PRODUCT02r_bbox_unique_download_manifest.csv")
print("WROTE", TAB / "Table_PRODUCT02s_bbox_search_download_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02t_bbox_download_log.csv")
print("WROTE", TAB / "Table_PRODUCT02u_bbox_post_download_file_counts.csv")
print("WROTE", TXT / "STAGE1B6B_BBOX_MODIS_SEARCH_REPORT.md")
