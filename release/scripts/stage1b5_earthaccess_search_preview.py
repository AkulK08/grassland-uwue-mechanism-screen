#!/usr/bin/env python
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
