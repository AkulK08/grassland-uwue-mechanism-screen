import argparse
import time
import ee


def init(project):
    ee.Initialize(project=project)


def esat_kpa(temp_k):
    temp_c = temp_k.subtract(273.15)
    return ee.Image(0.6108).multiply(
        temp_c.multiply(17.27).divide(temp_c.add(237.3)).exp()
    )


def get_stable_grassland_points(start_year, end_year, n_points, seed, scale):
    years = ee.List.sequence(start_year, end_year)

    def lc_for_year(y):
        y = ee.Number(y)
        img = (
            ee.ImageCollection("MODIS/061/MCD12Q1")
            .filter(ee.Filter.calendarRange(y, y, "year"))
            .first()
            .select("LC_Type1")
        )
        return img.eq(10).rename("grass")

    lc_stack = ee.ImageCollection.fromImages(years.map(lc_for_year))
    stable = lc_stack.sum().eq(end_year - start_year + 1).rename("stable")

    points = stable.selfMask().sample(
        region=ee.Geometry.Rectangle([-180, -60, 180, 85], geodesic=False),
        scale=scale,
        numPixels=n_points,
        seed=seed,
        geometries=True,
        tileScale=8,
    )

    points = points.randomColumn("rand", seed)

    def add_id(f):
        coords = f.geometry().coordinates()
        lon = ee.Number(coords.get(0))
        lat = ee.Number(coords.get(1))
        pid = lon.format("%.5f").cat("_").cat(lat.format("%.5f"))
        return f.set({
            "point_id": pid,
            "lon": lon,
            "lat": lat,
        })

    return points.map(add_id)


def date_list_for_year(year):
    start = ee.Date.fromYMD(year, 1, 1)
    offsets = ee.List.sequence(0, 360, 8)
    return offsets.map(lambda d: start.advance(ee.Number(d), "day"))


def image_for_period(start_date):
    start = ee.Date(start_date)
    end = start.advance(8, "day")

    mod17_col = ee.ImageCollection("MODIS/061/MOD17A2HGF").filterDate(start, end)

    modis_gpp = ee.Image(
        ee.Algorithms.If(
            mod17_col.size().gt(0),
            mod17_col.first().select("Gpp").multiply(0.0001 * 1000.0 / 8.0).rename("gpp_modis"),
            ee.Image.constant(-9999).rename("gpp_modis"),
        )
    )

    mod16_col = ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(start, end)

    modis_et = ee.Image(
        ee.Algorithms.If(
            mod16_col.size().gt(0),
            mod16_col.first().select("ET").multiply(0.1 / 8.0).rename("et_modis"),
            ee.Image.constant(-9999).rename("et_modis"),
        )
    )

    pml_col = ee.ImageCollection("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a").filterDate(start, end)

    pml_img = ee.Image(
        ee.Algorithms.If(
            pml_col.size().gt(0),
            pml_col.first(),
            ee.Image.constant([-9999, -9999]).rename(["GPP", "ET"]),
        )
    )

    pml_gpp = pml_img.select("GPP").multiply(0.01).rename("gpp_pml")
    pml_et = pml_img.select("ET").multiply(0.01).rename("et_pml")

    era = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start, end)

    def hourly_vpd(img):
        t = img.select("temperature_2m")
        td = img.select("dewpoint_temperature_2m")
        return esat_kpa(t).subtract(esat_kpa(td)).rename("vpd")

    vpd = era.map(hourly_vpd).mean().rename("vpd")
    temperature = era.select("temperature_2m").mean().subtract(273.15).rename("temperature")
    precipitation = era.select("total_precipitation").sum().multiply(1000.0).rename("precipitation")

    swvl1 = era.select("volumetric_soil_water_layer_1").mean()
    swvl2 = era.select("volumetric_soil_water_layer_2").mean()
    soil_moisture = swvl1.multiply(0.25).add(swvl2.multiply(0.75)).rename("soil_moisture")

    lai_col = ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start, end)

    lai = ee.Image(
        ee.Algorithms.If(
            lai_col.size().gt(0),
            lai_col.first().select("Lai_500m").multiply(0.1).rename("lai"),
            ee.Image.constant(-9999).rename("lai"),
        )
    )

    burn_col = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start, end)

    burned = ee.Image(
        ee.Algorithms.If(
            burn_col.size().gt(0),
            burn_col.select("BurnDate").max().gt(0).rename("burned"),
            ee.Image.constant(0).rename("burned"),
        )
    )

    return ee.Image.cat([
        modis_gpp,
        modis_et,
        pml_gpp,
        pml_et,
        vpd,
        soil_moisture,
        temperature,
        precipitation,
        lai,
        burned,
    ])


def export_points_to_drive(points, folder):
    task = ee.batch.Export.table.toDrive(
        collection=points,
        description="stable_grassland_points",
        folder=folder,
        fileNamePrefix="stable_grassland_points",
        fileFormat="CSV",
        selectors=["point_id", "lon", "lat", "rand"],
    )
    task.start()
    print("Started points export:", task.id)


def export_year_shard_to_drive(points, year, shard, n_shards, folder, scale):
    lower = shard / n_shards
    upper = (shard + 1) / n_shards

    shard_points = points.filter(
        ee.Filter.And(
            ee.Filter.gte("rand", lower),
            ee.Filter.lt("rand", upper),
        )
    )

    dates = date_list_for_year(year)

    def sample_date(d):
        d = ee.Date(d)
        img = image_for_period(d)

        sampled = img.sampleRegions(
            collection=shard_points,
            properties=["point_id", "lon", "lat", "rand"],
            scale=scale,
            geometries=False,
            tileScale=8,
        )

        return sampled.map(lambda f: f.set({
            "date": d.format("YYYY-MM-dd"),
            "year": year,
            "doy": d.getRelative("day", "year").add(1),
            "shard": shard,
        }))

    fc = ee.FeatureCollection(dates.map(sample_date)).flatten()

    desc = f"wue_timeseries_{year}_shard{shard:02d}"

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        fileFormat="CSV",
    )
    task.start()
    print("Started timeseries export:", desc, task.id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--folder", default="grassland_wue_exports")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--n-shards", type=int, default=5)
    parser.add_argument("--scale", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-points", action="store_true")
    parser.add_argument("--export-years", action="store_true")
    args = parser.parse_args()

    init(args.project)

    points = get_stable_grassland_points(
        start_year=args.start_year,
        end_year=args.end_year,
        n_points=args.n_points,
        seed=args.seed,
        scale=args.scale,
    )

    if args.export_points:
        export_points_to_drive(points, args.folder)

    if args.export_years:
        for year in range(args.start_year, args.end_year + 1):
            for shard in range(args.n_shards):
                export_year_shard_to_drive(
                    points=points,
                    year=year,
                    shard=shard,
                    n_shards=args.n_shards,
                    folder=args.folder,
                    scale=args.scale,
                )
                time.sleep(1)


if __name__ == "__main__":
    main()
