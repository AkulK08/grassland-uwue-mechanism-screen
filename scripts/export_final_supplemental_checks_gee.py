#!/usr/bin/env python
from __future__ import annotations

import argparse
import ee
import pandas as pd


def init(project: str):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def local_csv_to_fc(path: str) -> ee.FeatureCollection:
    df = pd.read_csv(path)

    lat_col = next((c for c in ["lat", "latitude", "LAT", "Latitude", "y", "Y"] if c in df.columns), None)
    lon_col = next((c for c in ["lon", "longitude", "LON", "Longitude", "x", "X"] if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise SystemExit(f"Could not find lat/lon columns in {path}. Columns={list(df.columns)}")

    if "point_id" not in df.columns:
        df["point_id"] = range(len(df))

    df = df.dropna(subset=[lat_col, lon_col]).copy()

    feats = []
    for _, r in df.iterrows():
        lat = float(r[lat_col])
        lon = float(r[lon_col])
        feats.append(
            ee.Feature(
                ee.Geometry.Point([lon, lat]),
                {
                    "point_id": str(r["point_id"]),
                    "lat": lat,
                    "lon": lon,
                },
            )
        )

    print(f"Built local point FeatureCollection: {len(feats)} points")
    return ee.FeatureCollection(feats)


def choose_smap_collection() -> str:
    candidates = [
        "NASA/SMAP/SPL4SMGP/008",
        "NASA/SMAP/SPL4SMGP/007",
        "NASA/SMAP/SPL4SMGP/006",
    ]
    for cid in candidates:
        try:
            first = ee.Image(ee.ImageCollection(cid).first())
            names = first.bandNames().getInfo()
            if "sm_rootzone" in names:
                print("Using SMAP collection:", cid)
                return cid
        except Exception as e:
            print("SMAP candidate failed:", cid, e)
    raise SystemExit("No accessible SMAP SPL4SMGP collection with sm_rootzone found.")


def irrigation_image(user_asset: str = "") -> ee.Image:
    if user_asset:
        img = ee.Image(user_asset)
        band = ee.String(img.bandNames().get(0))
        print("Using user irrigation asset:", user_asset)
        return img.select([band]).gt(0).rename("irrigation_or_agri_mask")

    # Explicit agriculture/irrigation screen. In GFSAD1000, class 2 is irrigated major cropland.
    img = ee.Image("USGS/GFSAD1000_V1")
    band = ee.String(img.bandNames().get(0))
    print("Using irrigation/agriculture mask: USGS/GFSAD1000_V1, class == 2")
    return img.select([band]).eq(2).rename("irrigation_or_agri_mask")


def export_fc(fc: ee.FeatureCollection, folder: str, desc: str):
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        fileFormat="CSV",
    )
    task.start()
    print("Started export:", desc, task.id)


def sample_img(img: ee.Image, points: ee.FeatureCollection, scale: int, date_str, year, doy) -> ee.FeatureCollection:
    out = img.sampleRegions(
        collection=points,
        properties=["point_id", "lat", "lon"],
        scale=scale,
        geometries=False,
        tileScale=8,
    )
    return out.map(lambda f: f.set({"date": date_str, "year": year, "doy": doy}))


def smap_year_fc(collection_id: str, points: ee.FeatureCollection, year: int, scale: int) -> ee.FeatureCollection:
    doys = ee.List.sequence(1, 361, 8)

    def one_doy(doy):
        doy = ee.Number(doy)
        start = ee.Date.fromYMD(year, 1, 1).advance(doy.subtract(1), "day")
        end = start.advance(8, "day")
        img = (
            ee.ImageCollection(collection_id)
            .filterDate(start, end)
            .select(["sm_rootzone"])
            .mean()
            .rename("smap_sm_rootzone")
        )
        return sample_img(img, points, scale, start.format("YYYY-MM-dd"), year, doy)

    return ee.FeatureCollection(doys.map(one_doy)).flatten()


def qa_year_fc(points: ee.FeatureCollection, year: int, scale: int) -> ee.FeatureCollection:
    doys = ee.List.sequence(1, 361, 8)

    def one_doy(doy):
        doy = ee.Number(doy)
        start = ee.Date.fromYMD(year, 1, 1).advance(doy.subtract(1), "day")
        end = start.advance(8, "day")

        gpp_qc = (
            ee.ImageCollection("MODIS/061/MOD17A2HGF")
            .filterDate(start, end)
            .select("Psn_QC")
            .mode()
            .bitwiseAnd(3)
            .lte(1)
            .rename("modis_gpp_qc_good")
        )

        et_qc = (
            ee.ImageCollection("MODIS/061/MOD16A2GF")
            .filterDate(start, end)
            .select("ET_QC")
            .mode()
            .bitwiseAnd(3)
            .lte(1)
            .rename("modis_et_qc_good")
        )

        img = gpp_qc.addBands(et_qc)
        return sample_img(img, points, scale, start.format("YYYY-MM-dd"), year, doy)

    return ee.FeatureCollection(doys.map(one_doy)).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--points-csv", required=True)
    ap.add_argument("--folder", default="grassland_wue_supplemental_FINAL")
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--scale", type=int, default=10000)
    ap.add_argument("--irrigation-asset", default="")
    args = ap.parse_args()

    init(args.project)
    points = local_csv_to_fc(args.points_csv)

    # Static irrigation/agriculture mask.
    irr = irrigation_image(args.irrigation_asset)
    irr_fc = irr.sampleRegions(
        collection=points,
        properties=["point_id", "lat", "lon"],
        scale=args.scale,
        geometries=False,
        tileScale=8,
    )
    export_fc(irr_fc, args.folder, "final_irrigation_mask_by_point")

    # Yearly SMAP exports, 2015–end year.
    smap_collection = choose_smap_collection()
    for year in range(max(args.start_year, 2015), args.end_year + 1):
        fc = smap_year_fc(smap_collection, points, year, args.scale)
        export_fc(fc, args.folder, f"final_smap_l4_by_point_8day_{year}")

    # Yearly MODIS QA exports, start year–end year.
    for year in range(args.start_year, args.end_year + 1):
        fc = qa_year_fc(points, year, args.scale)
        export_fc(fc, args.folder, f"final_modis_qa_by_point_8day_{year}")


if __name__ == "__main__":
    main()
