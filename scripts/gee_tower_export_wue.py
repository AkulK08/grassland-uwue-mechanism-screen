#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import ee


def init_ee(project):
    ee.Initialize(project=project)


def read_points(path):
    df = pd.read_csv(path)
    required = ["point_id", "lat", "lon"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column {c} in {path}. Columns: {list(df.columns)}")

    feats = []
    for _, r in df.iterrows():
        props = {}
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                continue
            if c in ["lat", "lon"]:
                continue
            props[c] = str(v)
        geom = ee.Geometry.Point([float(r["lon"]), float(r["lat"])])
        feats.append(ee.Feature(geom, props))
    return ee.FeatureCollection(feats)


def esat_kpa(temp_c):
    return ee.Image(0.6108).multiply(
        ee.Image(17.27).multiply(temp_c).divide(temp_c.add(237.3)).exp()
    )


def first_or_constant(collection, band_names, fill_values):
    band_names = list(band_names)
    fill_values = list(fill_values)
    fallback = ee.Image.constant(fill_values).rename(band_names)
    return ee.Image(
        ee.Algorithms.If(
            collection.size().gt(0),
            ee.Image(collection.first()).select(band_names),
            fallback
        )
    )


def image_for_window(start, end):
    start = ee.Date(start)
    end = ee.Date(end)

    # MODIS GPP: keep same scaling convention as the main project exporter.
    mod17_col = ee.ImageCollection("MODIS/061/MOD17A2HGF").filterDate(start, end)
    mod17 = first_or_constant(mod17_col, ["Gpp"], [-9999])
    gpp_modis = mod17.select("Gpp").multiply(0.0001 * 1000.0 / 8.0).rename("gpp_modis")

    # MODIS ET: keep same scaling convention as the main project exporter.
    mod16_col = ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(start, end)
    mod16 = first_or_constant(mod16_col, ["ET"], [-9999])
    et_modis = mod16.select("ET").multiply(0.1 / 8.0).rename("et_modis")

    # MODIS QA bands for later QA filtering.
    psn_qc = first_or_constant(mod17_col, ["Psn_QC"], [-9999]).select("Psn_QC").rename("Psn_QC_500m")
    et_qc = first_or_constant(mod16_col, ["ET_QC"], [-9999]).select("ET_QC").rename("ET_QC_500m")

    # PML-V2 GPP/ET.
    pml_col = ee.ImageCollection("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a").filterDate(start, end)
    pml_img = first_or_constant(pml_col, ["GPP", "ET"], [-9999, -9999])
    gpp_pml = pml_img.select("GPP").multiply(0.01).rename("gpp_pml")
    et_pml = pml_img.select("ET").multiply(0.01).rename("et_pml")

    # ERA5-Land hourly meteorology.
    era = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end)

    temp_k = era.select("temperature_2m").mean()
    dew_k = era.select("dewpoint_temperature_2m").mean()
    temp_c = temp_k.subtract(273.15).rename("temp_c")
    dew_c = dew_k.subtract(273.15)

    vpd = esat_kpa(temp_c).subtract(esat_kpa(dew_c)).max(0).rename("vpd")

    swvl1 = era.select("volumetric_soil_water_layer_1").mean()
    swvl2 = era.select("volumetric_soil_water_layer_2").mean()
    soil_moisture = swvl1.multiply(0.25).add(swvl2.multiply(0.75)).rename("soil_moisture")

    precip = era.select("total_precipitation").sum().multiply(1000.0).rename("precip_mm")

    # MODIS LAI.
    lai_col = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start, end)
    lai_img = first_or_constant(lai_col, ["Lai_500m"], [-9999])
    lai = lai_img.select("Lai_500m").multiply(0.1).rename("lai")

    # Burned area flag.
    burn_col = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start, end)
    burned = ee.Image(
        ee.Algorithms.If(
            burn_col.size().gt(0),
            burn_col.select("BurnDate").max().gt(0).rename("burned"),
            ee.Image.constant(0).rename("burned")
        )
    )

    return ee.Image.cat([
        gpp_modis,
        et_modis,
        psn_qc,
        et_qc,
        gpp_pml,
        et_pml,
        vpd,
        soil_moisture,
        temp_c.rename("temp_c"),
        precip,
        lai,
        burned
    ])


def sample_one_date(points, date_string, scale):
    start = ee.Date(date_string)
    end = start.advance(8, "day")
    doy = start.getRelative("day", "year").add(1)
    year = start.get("year")

    img = image_for_window(start, end)

    sampled = img.sampleRegions(
        collection=points,
        properties=["point_id", "tower_id", "source_network", "igbp"],
        scale=scale,
        geometries=True
    )

    return sampled.map(
        lambda f: f.set({
            "date": start.format("YYYY-MM-dd"),
            "year": year,
            "doy": doy
        })
    )


def export_year(points, year, folder, scale):
    dates = []
    for doy in range(1, 362, 8):
        dates.append(f"{year}-01-01")

    fc_list = []
    for doy in range(1, 362, 8):
        date = ee.Date.fromYMD(year, 1, 1).advance(doy - 1, "day")
        fc_list.append(sample_one_date(points, date, scale))

    out_fc = ee.FeatureCollection(fc_list).flatten()

    desc = f"tower_wue_timeseries_{year}"
    task = ee.batch.Export.table.toDrive(
        collection=out_fc,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        fileFormat="CSV"
    )
    task.start()
    print("Started export:", desc, task.id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--points", default="data/raw/towers/tower_validation_points_agent.csv")
    ap.add_argument("--folder", default="grassland_wue_tower_validation_core")
    ap.add_argument("--start-year", type=int, default=2001)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--scale", type=int, default=500)
    args = ap.parse_args()

    init_ee(args.project)
    points = read_points(args.points)

    for year in range(args.start_year, args.end_year + 1):
        export_year(points, year, args.folder, args.scale)


if __name__ == "__main__":
    main()
