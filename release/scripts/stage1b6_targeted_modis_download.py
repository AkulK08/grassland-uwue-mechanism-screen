from pathlib import Path
from datetime import datetime
import os
import re
import json
import math
import pandas as pd
import earthaccess

OUT = Path("results/stage1b6_targeted_modis_download")
TAB = OUT / "tables"
TXT = OUT / "text"
BASE_DL = Path("data/raw_local/no_gee_downloads")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
BASE_DL.mkdir(parents=True, exist_ok=True)

DO_DOWNLOAD = os.environ.get("DO_DOWNLOAD", "1").strip() == "1"
TARGET_SCOPE = os.environ.get("TARGET_SCOPE", "all49").strip()
START_YEAR = int(os.environ.get("START_YEAR", "2001"))
END_YEAR = int(os.environ.get("END_YEAR", "2024"))
COUNT_PER_TILE_PRODUCT = int(os.environ.get("COUNT_PER_TILE_PRODUCT", "3000"))

COORD_CANDIDATES = [
    Path("results/tower_satellite_extraction_targets_FINAL/CONTRAST_all_49_tower_coordinates.csv"),
    Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv"),
    Path("results/tower_satellite_extraction_targets_FINAL/SENSITIVITY_strict_GRA_coordinates.csv"),
    Path("results/tower_satellite_extraction_targets_FINAL/main_13_coordinates.csv"),
]

coord_path = None
for p in COORD_CANDIDATES:
    if p.exists():
        if TARGET_SCOPE == "main13" and "MAIN" in str(p):
            coord_path = p
            break
        if TARGET_SCOPE == "strict5" and "SENSITIVITY" in str(p):
            coord_path = p
            break
        if TARGET_SCOPE == "all49" and "CONTRAST" in str(p):
            coord_path = p
            break

if coord_path is None:
    for p in COORD_CANDIDATES:
        if p.exists():
            coord_path = p
            break

if coord_path is None:
    raise FileNotFoundError("Could not find tower coordinate CSV in results/tower_satellite_extraction_targets_FINAL")

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
    raise ValueError(f"Could not identify lat/lon columns in {coord_path}. Columns: {list(coords.columns)}")

if id_col is None:
    coords["target_id"] = [f"target_{i:03d}" for i in range(len(coords))]
    id_col = "target_id"

coords = coords[[id_col, lat_col, lon_col]].rename(columns={id_col: "target_id", lat_col: "lat", lon_col: "lon"})
coords["lat"] = pd.to_numeric(coords["lat"], errors="coerce")
coords["lon"] = pd.to_numeric(coords["lon"], errors="coerce")
coords = coords.dropna(subset=["lat", "lon"]).drop_duplicates()

def modis_tile_from_latlon(lat, lon):
    R = 6371007.181
    tile_size = 1111950.5196666666
    xmin = -20015109.354
    ymax = 10007554.677

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    x = R * lon_rad * math.cos(lat_rad)
    y = R * lat_rad

    h = int(math.floor((x - xmin) / tile_size))
    v = int(math.floor((ymax - y) / tile_size))
    return h, v

tile_rows = []
for _, r in coords.iterrows():
    h, v = modis_tile_from_latlon(float(r["lat"]), float(r["lon"]))
    tile_rows.append({
        "target_id": r["target_id"],
        "lat": r["lat"],
        "lon": r["lon"],
        "h": h,
        "v": v,
        "tile": f"h{h:02d}v{v:02d}",
    })

tiles = pd.DataFrame(tile_rows).sort_values(["h", "v", "target_id"])
tiles.to_csv(TAB / "Table_PRODUCT02j_target_modis_tiles_by_site.csv", index=False)

unique_tiles = tiles[["h", "v", "tile"]].drop_duplicates().sort_values(["h", "v"])
unique_tiles.to_csv(TAB / "Table_PRODUCT02k_unique_target_modis_tiles.csv", index=False)

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

def data_links(result):
    try:
        links = result.data_links()
        if links:
            return links
    except Exception:
        pass
    s = str(result)
    return re.findall(r"https?://[^'\"]+?\.hdf", s)

def hdf_filename_from_result(result):
    links = data_links(result)
    for link in links:
        m = re.search(r"/([^/]+\.hdf)", link)
        if m:
            return m.group(1)
    return ""

def result_size_mb(result):
    try:
        s = result.size()
        return float(s) if s is not None else None
    except Exception:
        try:
            s = result.size
            return float(s) if s is not None else None
        except Exception:
            return None

auth = earthaccess.login(strategy="netrc")

manifest_rows = []
download_plan = {}

for product in PRODUCTS:
    short_name = product["short_name"]
    version = product["version"]
    temporal = product["temporal"]
    product_group = product["product_group"]
    target_dir = product["target_dir"]

    product_results = []

    for _, tile_row in unique_tiles.iterrows():
        tile = tile_row["tile"]

        patterns = [
            f"{short_name}.A*.{tile}.*.hdf",
            f"*{tile}*.hdf",
        ]

        found = []
        used_pattern = ""

        for pattern in patterns:
            try:
                results = earthaccess.search_data(
                    short_name=short_name,
                    version=version,
                    temporal=temporal,
                    granule_name=pattern,
                    count=COUNT_PER_TILE_PRODUCT,
                )
                if results:
                    found = list(results)
                    used_pattern = pattern
                    break
            except TypeError:
                results = earthaccess.search_data(
                    short_name=short_name,
                    version=version,
                    temporal=temporal,
                    count=COUNT_PER_TILE_PRODUCT,
                )
                all_results = list(results)
                found = [r for r in all_results if tile in str(r)]
                used_pattern = "fallback_filter_tile_from_result_string"
                break
            except Exception as e:
                manifest_rows.append({
                    "product_group": product_group,
                    "short_name": short_name,
                    "version": version,
                    "tile": tile,
                    "temporal_start": temporal[0],
                    "temporal_end": temporal[1],
                    "search_pattern": pattern,
                    "search_status": "ERROR",
                    "error": str(e)[:1000],
                    "n_results_for_tile": 0,
                    "filename": "",
                    "size_mb": None,
                    "url": "",
                    "target_dir": str(target_dir),
                    "already_present": False,
                })

        print(f"{product_group} {short_name} {tile}: {len(found)} granules")

        for r in found:
            fn = hdf_filename_from_result(r)
            links = data_links(r)
            url = links[0] if links else ""
            size_mb = result_size_mb(r)
            already = bool(fn and (target_dir / fn).exists())

            manifest_rows.append({
                "product_group": product_group,
                "short_name": short_name,
                "version": version,
                "tile": tile,
                "temporal_start": temporal[0],
                "temporal_end": temporal[1],
                "search_pattern": used_pattern,
                "search_status": "FOUND",
                "error": "",
                "n_results_for_tile": len(found),
                "filename": fn,
                "size_mb": size_mb,
                "url": url,
                "target_dir": str(target_dir),
                "already_present": already,
            })

            if not already:
                product_results.append(r)

    download_plan[product_group] = {
        "target_dir": str(target_dir),
        "n_to_download": len(product_results),
        "results": product_results,
    }

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(TAB / "Table_PRODUCT02l_targeted_modis_download_manifest.csv", index=False)

summary_rows = []
for product in PRODUCTS:
    pg = product["product_group"]
    sub = manifest[manifest["product_group"].eq(pg)]
    found = sub[sub["search_status"].eq("FOUND")]
    n_unique_files = found["filename"].dropna().replace("", pd.NA).dropna().nunique() if len(found) else 0
    n_already = int(found["already_present"].sum()) if len(found) else 0
    size_total = pd.to_numeric(found["size_mb"], errors="coerce").sum() if len(found) else 0
    n_to_download = download_plan[pg]["n_to_download"]

    summary_rows.append({
        "product_group": pg,
        "short_name": product["short_name"],
        "version": product["version"],
        "temporal_start": product["temporal"][0],
        "temporal_end": product["temporal"][1],
        "n_target_tiles": int(len(unique_tiles)),
        "n_manifest_rows": int(len(sub)),
        "n_unique_files_found": int(n_unique_files),
        "n_already_present": int(n_already),
        "n_to_download": int(n_to_download),
        "estimated_total_size_mb": float(size_total),
        "target_dir": str(product["target_dir"]),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(TAB / "Table_PRODUCT02m_targeted_modis_download_summary.csv", index=False)

download_log_rows = []
if DO_DOWNLOAD:
    for product in PRODUCTS:
        pg = product["product_group"]
        target_dir = download_plan[pg]["target_dir"]
        results = download_plan[pg]["results"]

        if not results:
            download_log_rows.append({
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
            status = f"DOWNLOAD_ERROR: {str(e)[:500]}"

        download_log_rows.append({
            "product_group": pg,
            "target_dir": target_dir,
            "status": status,
            "n_requested": len(results),
        })
else:
    for product in PRODUCTS:
        pg = product["product_group"]
        download_log_rows.append({
            "product_group": pg,
            "target_dir": download_plan[pg]["target_dir"],
            "status": "DRY_RUN_NO_DOWNLOAD",
            "n_requested": download_plan[pg]["n_to_download"],
        })

download_log = pd.DataFrame(download_log_rows)
download_log.to_csv(TAB / "Table_PRODUCT02n_targeted_modis_download_log.csv", index=False)

post_rows = []
for product in PRODUCTS:
    target_dir = product["target_dir"]
    files = sorted(target_dir.rglob("*.hdf"))
    post_rows.append({
        "product_group": product["product_group"],
        "short_name": product["short_name"],
        "target_dir": str(target_dir),
        "n_hdf_files_present_after_run": len(files),
        "example_files": "; ".join([f.name for f in files[:10]]),
    })

post = pd.DataFrame(post_rows)
post.to_csv(TAB / "Table_PRODUCT02o_post_download_file_counts.csv", index=False)

report = []
report.append("# Stage 1B.6 targeted MODIS/MCD raw download")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append(f"Coordinate source: {coord_path}")
report.append(f"Target scope: {TARGET_SCOPE}")
report.append(f"Download mode: {'DOWNLOAD ENABLED' if DO_DOWNLOAD else 'DRY RUN ONLY'}")
report.append("")
report.append("## Unique target MODIS tiles")
report.append("")
report.append("```text")
report.append(unique_tiles.to_string(index=False))
report.append("```")
report.append("")
report.append("## Download summary")
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
report.append("## Strict completion rule")
report.append("")
report.append("Stage 1B.6 is complete when MOD17A2HGF, MOD16A2GF, MCD15A2H, and MCD64A1 HDF files exist locally for every target MODIS tile and year/date window needed for the all49 tower target scope.")
report.append("")

(TXT / "STAGE1B6_TARGETED_MODIS_DOWNLOAD_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6_targeted_modis_mcd_download",
    "status": "complete" if DO_DOWNLOAD else "dry_run_complete",
    "do_download": DO_DOWNLOAD,
    "target_scope": TARGET_SCOPE,
    "coordinate_source": str(coord_path),
    "n_target_sites": int(len(coords)),
    "n_unique_modis_tiles": int(len(unique_tiles)),
    "outputs": {
        "tiles_by_site": str(TAB / "Table_PRODUCT02j_target_modis_tiles_by_site.csv"),
        "unique_tiles": str(TAB / "Table_PRODUCT02k_unique_target_modis_tiles.csv"),
        "manifest": str(TAB / "Table_PRODUCT02l_targeted_modis_download_manifest.csv"),
        "summary": str(TAB / "Table_PRODUCT02m_targeted_modis_download_summary.csv"),
        "download_log": str(TAB / "Table_PRODUCT02n_targeted_modis_download_log.csv"),
        "post_counts": str(TAB / "Table_PRODUCT02o_post_download_file_counts.csv"),
        "report": str(TXT / "STAGE1B6_TARGETED_MODIS_DOWNLOAD_REPORT.md"),
    }
}
(TAB / "STAGE1B6_TARGETED_MODIS_DOWNLOAD_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02j_target_modis_tiles_by_site.csv")
print("WROTE", TAB / "Table_PRODUCT02k_unique_target_modis_tiles.csv")
print("WROTE", TAB / "Table_PRODUCT02l_targeted_modis_download_manifest.csv")
print("WROTE", TAB / "Table_PRODUCT02m_targeted_modis_download_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02n_targeted_modis_download_log.csv")
print("WROTE", TAB / "Table_PRODUCT02o_post_download_file_counts.csv")
print("WROTE", TXT / "STAGE1B6_TARGETED_MODIS_DOWNLOAD_REPORT.md")
