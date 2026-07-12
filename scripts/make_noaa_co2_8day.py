from pathlib import Path
from io import StringIO
import requests
import pandas as pd
import numpy as np

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
OUT = ROOT / "data/external/noaa_co2_8day.csv"
RAW = ROOT / "data/external/noaa_co2_monthly_mlo_raw.csv"

URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"

START_YEAR = 2001
END_YEAR = 2024

print("Downloading NOAA Mauna Loa monthly CO2:")
print(URL)

r = requests.get(URL, timeout=60)
r.raise_for_status()
text = r.text

# NOAA text file has comment lines beginning with #.
rows = []
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) < 4:
        continue

    year = int(parts[0])
    month = int(parts[1])
    decimal_date = float(parts[2])
    average = float(parts[3])

    # NOAA missing values are negative placeholders such as -99.99.
    if average < 0:
        continue

    rows.append(
        {
            "year": year,
            "month": month,
            "decimal_date": decimal_date,
            "co2_ppm": average,
        }
    )

monthly = pd.DataFrame(rows)
if monthly.empty:
    raise RuntimeError("No NOAA monthly CO2 rows parsed.")

# Use middle of each month for interpolation.
monthly["date"] = pd.to_datetime(
    monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-15",
    errors="coerce",
)

monthly = monthly.dropna(subset=["date", "co2_ppm"])
monthly = monthly.sort_values("date")

RAW.parent.mkdir(parents=True, exist_ok=True)
monthly.to_csv(RAW, index=False)

print("Raw monthly rows:", len(monthly))
print("Raw monthly date range:", monthly["date"].min().date(), "to", monthly["date"].max().date())

# Build GOSIF-style 8-day dates: DOY 001, 009, 017, ..., 361.
dates = []
for year in range(START_YEAR, END_YEAR + 1):
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    for doy in range(1, 362, 8):
        dates.append(jan1 + pd.Timedelta(days=doy - 1))

target = pd.DataFrame({"date": pd.to_datetime(dates)})

# Interpolate monthly CO2 to target 8-day dates using time interpolation.
series = monthly[["date", "co2_ppm"]].set_index("date").sort_index()
combined_index = series.index.union(pd.DatetimeIndex(target["date"]))
interp = (
    series.reindex(combined_index)
    .sort_index()
    .interpolate(method="time")
    .reindex(pd.DatetimeIndex(target["date"]))
)

target["co2_ppm"] = interp["co2_ppm"].values
target = target.dropna(subset=["co2_ppm"]).copy()
target["date"] = target["date"].dt.strftime("%Y-%m-%d")
target["co2_ppm"] = target["co2_ppm"].round(4)

expected_rows = (END_YEAR - START_YEAR + 1) * 46
print("Expected rows:", expected_rows)
print("Actual rows:", len(target))

if len(target) != expected_rows:
    raise RuntimeError(f"Expected {expected_rows} rows but got {len(target)}.")

OUT.parent.mkdir(parents=True, exist_ok=True)
target.to_csv(OUT, index=False)

print("\nWrote:", OUT)
print(target.head())
print(target.tail())
print("\nCO2 summary:")
print(target["co2_ppm"].describe())
