from pathlib import Path
from datetime import datetime
import json
import math
import re
import pandas as pd
import numpy as np

OUT = Path("results/stage1b6l_georef_sanity_check")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

SHARDS = Path("data/raw_local/no_gee_direct_point_extract_full/_shards")

# MODIS sinusoidal constants.
R = 6371007.181
XMIN = -20015109.354
YMAX = 10007554.677
TILE_SIZE = 1111950.519667
PIXEL_SIZE_500M = TILE_SIZE / 2400.0

def expected_modis_row_col(lat, lon, tile):
    m = re.match(r"h(\d{2})v(\d{2})", str(tile))
    if not m:
        return np.nan, np.nan
    h = int(m.group(1))
    v = int(m.group(2))

    lat_rad = math.radians(float(lat))
    lon_rad = math.radians(float(lon))

    x = R * lon_rad * math.cos(lat_rad)
    y = R * lat_rad

    tile_x0 = XMIN + h * TILE_SIZE
    tile_y0 = YMAX - v * TILE_SIZE

    col = math.floor((x - tile_x0) / PIXEL_SIZE_500M)
    row = math.floor((tile_y0 - y) / PIXEL_SIZE_500M)

    return row, col

rows = []
files = sorted(SHARDS.glob("*/*.csv"))

for p in files[:5000]:
    try:
        df = pd.read_csv(p)
    except Exception as e:
        rows.append({
            "shard": str(p),
            "status": "READ_ERROR",
            "error": str(e)[:300],
        })
        continue

    need = {"id", "latitude", "longitude", "row", "col", "tile", "layer", "product_group", "filename"}
    if not need.issubset(set(df.columns)):
        rows.append({
            "shard": str(p),
            "status": "MISSING_COLUMNS",
            "columns": ";".join(df.columns),
        })
        continue

    sample = df[df["sample_status"].astype(str).eq("OK")].copy()
    if len(sample) == 0:
        sample = df.head(5).copy()
    else:
        sample = sample.head(10).copy()

    for _, r in sample.iterrows():
        try:
            exp_row, exp_col = expected_modis_row_col(r["latitude"], r["longitude"], r["tile"])
            got_row = int(float(r["row"]))
            got_col = int(float(r["col"]))
            drow = abs(got_row - exp_row)
            dcol = abs(got_col - exp_col)

            rows.append({
                "shard": str(p),
                "product_group": r["product_group"],
                "filename": r["filename"],
                "tile": r["tile"],
                "layer": r["layer"],
                "id": r["id"],
                "lat": r["latitude"],
                "lon": r["longitude"],
                "got_row": got_row,
                "got_col": got_col,
                "expected_row": exp_row,
                "expected_col": exp_col,
                "abs_row_diff": drow,
                "abs_col_diff": dcol,
                "status": "CHECKED",
            })
        except Exception as e:
            rows.append({
                "shard": str(p),
                "status": "CHECK_ERROR",
                "error": str(e)[:300],
            })

qa = pd.DataFrame(rows)
qa.to_csv(TAB / "Table_PRODUCT02bn_georef_rowcol_sanity_rows.csv", index=False)

checked = qa[qa["status"].eq("CHECKED")].copy() if len(qa) else pd.DataFrame()

if len(checked):
    checked["pass_rowcol"] = (checked["abs_row_diff"] <= 1) & (checked["abs_col_diff"] <= 1)
    pass_frac = float(checked["pass_rowcol"].mean())
    max_row_diff = float(checked["abs_row_diff"].max())
    max_col_diff = float(checked["abs_col_diff"].max())
    n_checked = int(len(checked))
    n_fail = int((~checked["pass_rowcol"]).sum())
else:
    pass_frac = 0.0
    max_row_diff = None
    max_col_diff = None
    n_checked = 0
    n_fail = 0

if n_checked > 0 and pass_frac >= 0.99:
    verdict = "PASS_ROWCOL_MATCHES_MODIS_SINUSOIDAL_GRID"
    next_action = "Let Stage 1B.6J continue."
else:
    verdict = "FAIL_OR_INSUFFICIENT_ROWCOL_MATCH"
    next_action = "Pause Stage 1B.6J and rerun extraction with manual MODIS sinusoidal row/col indexing."

summary = pd.DataFrame([{
    "n_shards_seen": len(files),
    "n_rows_checked": n_checked,
    "n_failed_rowcol": n_fail,
    "pass_fraction": pass_frac,
    "max_row_diff": max_row_diff,
    "max_col_diff": max_col_diff,
    "verdict": verdict,
    "next_action": next_action,
}])
summary.to_csv(TAB / "Table_PRODUCT02bo_georef_sanity_summary.csv", index=False)

report = []
report.append("# Stage 1B.6L georeference sanity check")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## First checked rows")
report.append("")
report.append("```text")
report.append(checked.head(40).to_string(index=False) if len(checked) else "No checked rows.")
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("If row/col values do not match the MODIS sinusoidal tile formula, the current shards are not spatially trustworthy and the extraction must be rerun with manual tile indexing.")
report.append("")

(TXT / "STAGE1B6L_GEOREF_SANITY_CHECK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6L_georef_sanity_check",
    "status": verdict,
    "outputs": {
        "rows": str(TAB / "Table_PRODUCT02bn_georef_rowcol_sanity_rows.csv"),
        "summary": str(TAB / "Table_PRODUCT02bo_georef_sanity_summary.csv"),
        "report": str(TXT / "STAGE1B6L_GEOREF_SANITY_CHECK_REPORT.md"),
    }
}
(TAB / "STAGE1B6L_GEOREF_SANITY_CHECK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02bn_georef_rowcol_sanity_rows.csv")
print("WROTE", TAB / "Table_PRODUCT02bo_georef_sanity_summary.csv")
print("WROTE", TXT / "STAGE1B6L_GEOREF_SANITY_CHECK_REPORT.md")
