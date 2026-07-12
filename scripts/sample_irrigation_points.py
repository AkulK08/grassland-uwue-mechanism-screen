from pathlib import Path
import pandas as pd
import rasterio

pts = pd.read_csv("data/raw/gee/stable_grassland_points.csv")
lat_col = "lat" if "lat" in pts.columns else "latitude"
lon_col = "lon" if "lon" in pts.columns else "longitude"

tifs = list(Path("data/external/irrigation").rglob("*.tif"))
print("TIFS FOUND:", len(tifs))

if not tifs:
    raise SystemExit("No .tif irrigation files found.")

vals = []
for _, row in pts.iterrows():
    lon, lat = float(row[lon_col]), float(row[lat_col])
    val = 0
    for tif in tifs:
        with rasterio.open(tif) as src:
            if src.bounds.left <= lon <= src.bounds.right and src.bounds.bottom <= lat <= src.bounds.top:
                val = list(src.sample([(lon, lat)]))[0][0]
                break
    vals.append(val)

out = pts.copy()
out["irrigation_or_agri_mask"] = vals
out["exclude_irrigated"] = pd.to_numeric(out["irrigation_or_agri_mask"], errors="coerce").fillna(0).astype(int) > 0

out.to_csv("data/external/irrigation_by_point.csv", index=False)

bad = set(out.loc[out["exclude_irrigated"], "point_id"].astype(str))
srcdir = Path("data/raw/gee")
dstdir = Path("data/raw/gee_final_filtered_no_irrigation")
dstdir.mkdir(parents=True, exist_ok=True)

for p in srcdir.glob("*.csv"):
    d = pd.read_csv(p)
    if "point_id" in d.columns:
        d = d[~d["point_id"].astype(str).isin(bad)]
    d.to_csv(dstdir / p.name, index=False)

summary = pd.DataFrame([{
    "points_before": len(out),
    "points_excluded": int(out["exclude_irrigated"].sum()),
    "exclusion_fraction": float(out["exclude_irrigated"].mean()),
    "points_after": int((~out["exclude_irrigated"]).sum())
}])

summary.to_csv("results/qc/irrigation_exclusion_summary.csv", index=False)
print(summary)
