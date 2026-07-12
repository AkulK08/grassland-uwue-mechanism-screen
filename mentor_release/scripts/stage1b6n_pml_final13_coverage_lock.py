from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import xarray as xr

OUT = Path("results/stage1b6n_pml_final13_coverage_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/raw_local/pml_final13_point_extract")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

TARGET_CANDIDATES = [
    Path("data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv"),
    Path("results/stage1b6g_scientific_target_lock/tables/Table_PRODUCT02aq_FINAL_no_gee_product_point_request.csv"),
]

GPP_FILE = Path("data/processed/gpp_PML.nc")
ET_FILE = Path("data/processed/et_PML.nc")

def find_target_file():
    for p in TARGET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find final 13 target file.")

def normalize_targets(df):
    cols = {c.lower(): c for c in df.columns}

    id_col = None
    for k in ["id", "point_id", "site_id", "site", "name"]:
        if k in cols:
            id_col = cols[k]
            break

    lat_col = None
    for k in ["lat", "latitude"]:
        if k in cols:
            lat_col = cols[k]
            break

    lon_col = None
    for k in ["lon", "longitude"]:
        if k in cols:
            lon_col = cols[k]
            break

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not identify lat/lon columns in target file. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "point_id": df[id_col].astype(str) if id_col else [f"P{i+1:02d}" for i in range(len(df))],
        "lat": pd.to_numeric(df[lat_col]),
        "lon": pd.to_numeric(df[lon_col]),
    })

    out = out.drop_duplicates(["point_id", "lat", "lon"]).reset_index(drop=True)
    return out

def coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
        if c in ds.dims:
            return c
    raise ValueError(f"Could not find coord among {candidates}. Coords={list(ds.coords)}, dims={list(ds.dims)}")

def data_var_name(ds, preferred):
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    pref_low = preferred.lower()
    for v in ds.data_vars:
        if pref_low in v.lower():
            return v
    raise ValueError(f"Could not find variable {preferred}. Data vars={list(ds.data_vars)}")

def sample_nc(nc_path, preferred_var, out_name, targets):
    ds = xr.open_dataset(nc_path)

    lat_name = coord_name(ds, ["lat", "latitude", "y"])
    lon_name = coord_name(ds, ["lon", "longitude", "x"])
    time_name = coord_name(ds, ["time", "date"])
    var = data_var_name(ds, preferred_var)

    rows = []
    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    for _, site in targets.iterrows():
        point_id = site["point_id"]
        lat = float(site["lat"])
        lon = float(site["lon"])

        # Nearest-neighbor extraction from local PML grid.
        sub = ds[var].sel({lat_name: lat, lon_name: lon}, method="nearest")
        nearest_lat = float(sub[lat_name].values)
        nearest_lon = float(sub[lon_name].values)

        tmp = sub.to_dataframe(name=preferred_var).reset_index()
        tmp["point_id"] = point_id
        tmp["target_lat"] = lat
        tmp["target_lon"] = lon
        tmp["nearest_lat"] = nearest_lat
        tmp["nearest_lon"] = nearest_lon
        tmp["abs_lat_diff"] = abs(nearest_lat - lat)
        tmp["abs_lon_diff"] = abs(nearest_lon - lon)

        # Normalize time column.
        if time_name in tmp.columns:
            tmp = tmp.rename(columns={time_name: "date"})
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")

        rows.append(tmp[[
            "point_id", "target_lat", "target_lon",
            "nearest_lat", "nearest_lon", "abs_lat_diff", "abs_lon_diff",
            "date", preferred_var
        ]])

    out = pd.concat(rows, ignore_index=True)
    out_csv = DATA / out_name
    out.to_csv(out_csv, index=False)

    site_summary = (
        out.groupby("point_id")
        .agg(
            n_rows=("date", "size"),
            n_nonmissing=(preferred_var, lambda s: int(pd.notna(s).sum())),
            date_min=("date", "min"),
            date_max=("date", "max"),
            max_abs_lat_diff=("abs_lat_diff", "max"),
            max_abs_lon_diff=("abs_lon_diff", "max"),
        )
        .reset_index()
    )
    site_summary["product"] = preferred_var

    ds.close()

    meta = {
        "nc_path": str(nc_path),
        "variable": var,
        "lat_coord": lat_name,
        "lon_coord": lon_name,
        "time_coord": time_name,
        "n_times": int(out["date"].nunique()),
        "date_min": str(out["date"].min()),
        "date_max": str(out["date"].max()),
        "n_sites": int(out["point_id"].nunique()),
        "out_csv": str(out_csv),
    }

    return out, site_summary, meta

target_file = find_target_file()
targets = normalize_targets(pd.read_csv(target_file))

targets.to_csv(TAB / "Table_PRODUCT02bt_pml_final13_targets_used.csv", index=False)

errors = []
outputs = []
site_summaries = []
metas = []

for nc_path, preferred_var, out_name in [
    (GPP_FILE, "gpp", "PML_GPP_FINAL13_point_samples.csv"),
    (ET_FILE, "et", "PML_ET_FINAL13_point_samples.csv"),
]:
    try:
        if not nc_path.exists():
            raise FileNotFoundError(str(nc_path))
        out, site_summary, meta = sample_nc(nc_path, preferred_var, out_name, targets)
        outputs.append(out)
        site_summaries.append(site_summary)
        metas.append(meta)
    except Exception as e:
        errors.append({
            "product": preferred_var,
            "path": str(nc_path),
            "error": repr(e),
        })

if outputs:
    merged = None
    for out in outputs:
        keep_cols = ["point_id", "date"] + [c for c in out.columns if c in ["gpp", "et"]]
        if merged is None:
            merged = out[keep_cols].copy()
        else:
            merged = merged.merge(out[keep_cols], on=["point_id", "date"], how="outer")

    merged.to_csv(DATA / "PML_GPP_ET_FINAL13_point_samples_merged.csv", index=False)

if site_summaries:
    site_summary_all = pd.concat(site_summaries, ignore_index=True)
else:
    site_summary_all = pd.DataFrame()

site_summary_all.to_csv(TAB / "Table_PRODUCT02bu_pml_final13_site_coverage.csv", index=False)

meta_df = pd.DataFrame(metas)
meta_df.to_csv(TAB / "Table_PRODUCT02bv_pml_file_time_metadata.csv", index=False)

err_df = pd.DataFrame(errors)
err_df.to_csv(TAB / "Table_PRODUCT02bw_pml_coverage_errors.csv", index=False)

if len(meta_df):
    n_sites_min = int(meta_df["n_sites"].min())
    n_times_min = int(meta_df["n_times"].min())
    date_min_overall = str(meta_df["date_min"].min())
    date_max_overall = str(meta_df["date_max"].max())
else:
    n_sites_min = 0
    n_times_min = 0
    date_min_overall = ""
    date_max_overall = ""

if len(site_summary_all):
    max_lat_diff = float(site_summary_all["max_abs_lat_diff"].max())
    max_lon_diff = float(site_summary_all["max_abs_lon_diff"].max())
    min_nonmissing = int(site_summary_all["n_nonmissing"].min())
else:
    max_lat_diff = np.nan
    max_lon_diff = np.nan
    min_nonmissing = 0

# PML grid is coarse/local here, so do not require tiny nearest-pixel distance.
# The strict minimum is: both products open, all 13 sites sampled, nonmissing time series exist.
has_both = len(meta_df) == 2 and set(meta_df["variable"].str.lower()).issuperset({"gpp", "et"})
all_sites = n_sites_min == len(targets) == 13
enough_time = n_times_min >= 300
nonmissing_ok = min_nonmissing >= 300
no_errors = len(err_df) == 0

if has_both and all_sites and enough_time and nonmissing_ok and no_errors:
    verdict = "PML_FINAL13_COVERAGE_LOCK_PASS"
    blocking_for_3x3 = False
else:
    verdict = "PML_FINAL13_COVERAGE_LOCK_FAIL_OR_WEAK"
    blocking_for_3x3 = True

summary = pd.DataFrame([{
    "target_file": str(target_file),
    "n_targets": len(targets),
    "has_both_pml_gpp_and_et": has_both,
    "min_sites_across_products": n_sites_min,
    "min_times_across_products": n_times_min,
    "date_min": date_min_overall,
    "date_max": date_max_overall,
    "min_nonmissing_site_product_rows": min_nonmissing,
    "max_abs_lat_diff": max_lat_diff,
    "max_abs_lon_diff": max_lon_diff,
    "n_errors": len(err_df),
    "verdict": verdict,
    "blocking_for_3x3": blocking_for_3x3,
}])
summary.to_csv(TAB / "Table_PRODUCT02bx_pml_final13_coverage_summary.csv", index=False)

report = []
report.append("# Stage 1B.6N PML final-13 coverage lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## PML metadata")
report.append("")
report.append("```text")
report.append(meta_df.to_string(index=False) if len(meta_df) else "No metadata.")
report.append("```")
report.append("")
report.append("## Site coverage")
report.append("")
report.append("```text")
report.append(site_summary_all.to_string(index=False) if len(site_summary_all) else "No site coverage.")
report.append("```")
report.append("")
report.append("## Errors")
report.append("")
report.append("```text")
report.append(err_df.to_string(index=False) if len(err_df) else "No errors.")
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("PML can enter the 3x3 matrix only if both GPP and ET sample all final 13 target points with usable time series. This stage checks coverage; later product-matrix code handles temporal intersection/resampling.")
report.append("")

(TXT / "STAGE1B6N_PML_FINAL13_COVERAGE_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6N_pml_final13_coverage_lock",
    "status": verdict,
    "blocking_for_3x3": bool(blocking_for_3x3),
    "outputs": {
        "pml_gpp": str(DATA / "PML_GPP_FINAL13_point_samples.csv"),
        "pml_et": str(DATA / "PML_ET_FINAL13_point_samples.csv"),
        "pml_merged": str(DATA / "PML_GPP_ET_FINAL13_point_samples_merged.csv"),
        "summary": str(TAB / "Table_PRODUCT02bx_pml_final13_coverage_summary.csv"),
        "report": str(TXT / "STAGE1B6N_PML_FINAL13_COVERAGE_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6N_PML_FINAL13_COVERAGE_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "PML_GPP_FINAL13_point_samples.csv")
print("WROTE", DATA / "PML_ET_FINAL13_point_samples.csv")
print("WROTE", DATA / "PML_GPP_ET_FINAL13_point_samples_merged.csv")
print("WROTE", TAB / "Table_PRODUCT02bx_pml_final13_coverage_summary.csv")
print("WROTE", TXT / "STAGE1B6N_PML_FINAL13_COVERAGE_LOCK_REPORT.md")
