from pathlib import Path
from glob import glob
import pandas as pd
import sys

gee_files = sorted(glob("data/raw/gee/wue_timeseries_*.csv"))
if not gee_files:
    raise SystemExit("No Earth Engine files found in data/raw/gee.")

print("Loading Earth Engine files:", len(gee_files))
base = pd.concat([pd.read_csv(f) for f in gee_files], ignore_index=True)

for c in [".geo", "system:index"]:
    if c in base.columns:
        base = base.drop(columns=[c])

if "lat" in base.columns and "latitude" not in base.columns:
    base["latitude"] = base["lat"]
if "lon" in base.columns and "longitude" not in base.columns:
    base["longitude"] = base["lon"]

required_base = [
    "point_id", "date",
    "gpp_modis", "gpp_pml",
    "et_modis", "et_pml",
    "vpd", "soil_moisture",
    "temperature", "precipitation",
    "lai", "burned"
]

missing_base = [c for c in required_base if c not in base.columns]
if missing_base:
    raise SystemExit(f"Earth Engine base table is missing columns: {missing_base}")

base["date"] = pd.to_datetime(base["date"])

gosif_path = Path("data/raw/agents/gosif_point_timeseries.csv")
gleam_path = Path("data/raw/agents/gleam_point_timeseries.csv")

if not gosif_path.exists():
    raise SystemExit("Missing data/raw/agents/gosif_point_timeseries.csv with columns point_id,date,gpp_gosif")

if not gleam_path.exists():
    raise SystemExit("Missing data/raw/agents/gleam_point_timeseries.csv with columns point_id,date,et_gleam")

gosif = pd.read_csv(gosif_path)
gleam = pd.read_csv(gleam_path)

for name, df, needed in [
    ("GOSIF", gosif, ["point_id", "date", "gpp_gosif"]),
    ("GLEAM", gleam, ["point_id", "date", "et_gleam"]),
]:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} file missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])

merged = base.merge(gosif[["point_id", "date", "gpp_gosif"]], on=["point_id", "date"], how="left")
merged = merged.merge(gleam[["point_id", "date", "et_gleam"]], on=["point_id", "date"], how="left")

print("Merged shape before drop:", merged.shape)
print("Missing gpp_gosif:", merged["gpp_gosif"].isna().mean())
print("Missing et_gleam:", merged["et_gleam"].isna().mean())

merged = merged.dropna(subset=[
    "gpp_modis", "gpp_gosif", "gpp_pml",
    "et_modis", "et_gleam", "et_pml",
    "vpd", "soil_moisture"
])

print("Merged shape after core drop:", merged.shape)

if merged.empty:
    raise SystemExit("Merged full matrix is empty. Dates/point_id likely do not align.")

out_raw = Path("data/raw/agents/merged_full_matrix_raw.csv")
out_raw.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_raw, index=False)
print("Wrote", out_raw)

co2_path = Path("data/external/noaa_co2_8day.csv")
if co2_path.exists():
    co2 = pd.read_csv(co2_path)
    if not {"date", "co2_ppm"}.issubset(co2.columns):
        raise SystemExit("data/external/noaa_co2_8day.csv must have columns date,co2_ppm")
    co2["date"] = pd.to_datetime(co2["date"])
    corrected = merged.merge(co2[["date", "co2_ppm"]], on="date", how="left")
    if corrected["co2_ppm"].isna().any():
        raise SystemExit("CO2 file does not cover all merged dates.")

    ref = corrected["co2_ppm"].median()
    corrected["co2_correction_factor"] = ref / corrected["co2_ppm"]

    # WUE correction is equivalent to scaling GPP columns when ET is unchanged.
    for col in ["gpp_modis", "gpp_gosif", "gpp_pml"]:
        corrected[col] = corrected[col] * corrected["co2_correction_factor"]

    out_corr = Path("data/raw/agents/merged_full_matrix_co2corrected.csv")
    corrected.to_csv(out_corr, index=False)
    print("Wrote", out_corr)
else:
    print("No CO2 file found. Wrote raw merged matrix only.")
