from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    import ee
except Exception as e:
    print("ERROR: earthengine-api is not installed.")
    print(e)
    sys.exit(1)

ROOT = Path("/Users/me/Downloads/grassland_wue_nature_repo")
POINTS_PATH = ROOT / "data/raw/gee/stable_grassland_points.csv"
OUT_PATH = ROOT / "data/external/aridity_by_point.csv"

if not POINTS_PATH.exists():
    raise FileNotFoundError(f"Missing points file: {POINTS_PATH}")

try:
    ee.Initialize()
except Exception:
    print("Earth Engine is not authenticated/initialized.")
    print("Run this command, finish browser login, then rerun this script:")
    print("earthengine authenticate")
    raise

points = pd.read_csv(POINTS_PATH)

print("Loaded points:", POINTS_PATH)
print("Rows:", len(points))
print("Columns:", list(points.columns))

def find_col(candidates):
    lower_map = {c.lower(): c for c in points.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

lat_col = find_col(["lat", "latitude", "y"])
lon_col = find_col(["lon", "longitude", "x", "lng"])
id_col = find_col(["point_id", "id", "site_id", "pixel_id"])

if lat_col is None or lon_col is None:
    raise ValueError(
        "Could not detect latitude/longitude columns. "
        f"Columns are: {list(points.columns)}"
    )

if id_col is None:
    points = points.copy()
    points["point_id"] = np.arange(len(points))
    id_col = "point_id"

points = points[[id_col, lat_col, lon_col]].copy()
points = points.rename(columns={id_col: "point_id", lat_col: "lat", lon_col: "lon"})
points = points.dropna(subset=["lat", "lon"])
points["point_id"] = points["point_id"].astype(str)

print("Using columns: point_id, lat, lon")
print(points.head())

# Global Aridity Index yearly asset from SAT-IO community catalog.
# Values are stored as integers scaled by 10000.
img = ee.Image("projects/sat-io/open-datasets/global_ai/global_ai_yearly")

features = []
for _, row in points.iterrows():
    geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
    features.append(
        ee.Feature(
            geom,
            {
                "point_id": str(row["point_id"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
            },
        )
    )

fc = ee.FeatureCollection(features)

sampled = img.sampleRegions(
    collection=fc,
    scale=1000,
    geometries=False,
    tileScale=4,
)

records = sampled.getInfo()["features"]

out_rows = []
for feat in records:
    props = feat["properties"]
    point_id = props.get("point_id")
    lat = props.get("lat")
    lon = props.get("lon")

    # Find the aridity band automatically.
    value_keys = [k for k in props.keys() if k not in {"point_id", "lat", "lon"}]
    if not value_keys:
        ai = np.nan
    else:
        raw = props[value_keys[0]]
        ai = np.nan if raw is None else float(raw) / 10000.0

    out_rows.append(
        {
            "point_id": point_id,
            "lat": lat,
            "lon": lon,
            "aridity_index": ai,
        }
    )

out = pd.DataFrame(out_rows)
out = out.dropna(subset=["aridity_index"]).copy()

# Quartiles: Q1 = most arid among your sampled grassland points, Q4 = most humid.
out["aridity_quartile"] = pd.qcut(
    out["aridity_index"],
    q=4,
    labels=["Q1", "Q2", "Q3", "Q4"],
    duplicates="drop",
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print("\nWrote:", OUT_PATH)
print("Rows written:", len(out))
print(out.head())
print("\nAridity summary:")
print(out["aridity_index"].describe())
print("\nQuartile counts:")
print(out["aridity_quartile"].value_counts(dropna=False).sort_index())
