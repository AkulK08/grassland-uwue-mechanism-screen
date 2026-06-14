"""Command-line Earth Engine extraction agents.

These functions avoid local downloads of global raster stacks. They submit Earth
Engine batch exports that sample remote gridded products at stable grassland
points and write compact CSV tables to Google Cloud Storage. The local point
backend can then analyze those CSVs with the same Gate 1/Gate 2 response-shape
criteria used by the gridded backend.

The main idea is:
    global rasters stay in Earth Engine -> CSV point-time tables go to GCS.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import time
from typing import Iterable, List, Optional

try:
    import ee
except Exception:  # pragma: no cover - ee is optional in CI
    ee = None


WORLD = [-180, -60, 180, 85]
MODIS_SCALE = 500
ERA_SCALE = 11132
EXPORT_SELECTORS = [
    "point_id", "date", "year", "doy", "longitude", "latitude",
    "gpp_modis", "et_modis", "gpp_pml", "et_pml",
    "vpd", "soil_moisture", "temperature", "precipitation", "lai", "burned",
]


@dataclass
class ExportTaskInfo:
    description: str
    task_id: str
    state: str


def initialize_ee(project: str) -> None:
    if ee is None:
        raise RuntimeError("earthengine-api is not installed. Run: pip install earthengine-api")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def _safe_first(collection, fallback_image):
    """Return first image from collection if present, else a fallback image.

    Earth Engine has no simple Python-side 'if collection empty' without getInfo,
    so use ee.Algorithms.If server-side.
    """
    return ee.Image(ee.Algorithms.If(collection.size().gt(0), collection.first(), fallback_image))


def stable_grassland_points(start_year: int, end_year: int, n_points: int, seed: int = 42,
                            include_savanna: bool = False):
    """Create a stable grassland FeatureCollection entirely inside Earth Engine.

    Stable means the IGBP LC_Type1 class is grassland (10) in every available
    MCD12Q1 year in the requested period. If include_savanna=True, classes 8 and
    9 are also allowed.
    """
    years = list(range(start_year, min(end_year, 2024) + 1))
    land = ee.ImageCollection("MODIS/061/MCD12Q1")
    allowed = [10, 8, 9] if include_savanna else [10]

    masks = []
    for y in years:
        img = land.filterDate(f"{y}-01-01", f"{y+1}-01-01").first().select("LC_Type1")
        m = img.remap(allowed, [1] * len(allowed), 0).rename("stable")
        masks.append(m)
    stable = ee.ImageCollection(masks).min().selfMask().rename("stable")
    # Sampling at ~0.1 degree scale prevents dense neighboring 500 m pixels from dominating.
    pts = stable.addBands(ee.Image.pixelLonLat()).sample(
        region=ee.Geometry.Rectangle(WORLD, geodesic=False),
        scale=10000,
        numPixels=n_points,
        seed=seed,
        geometries=True,
        tileScale=8,
    )

    def add_id(f):
        lon = ee.Number(f.get("longitude"))
        lat = ee.Number(f.get("latitude"))
        pid = ee.String("p_").cat(lon.format("%.4f")).cat("_").cat(lat.format("%.4f"))
        return f.set({"point_id": pid})

    return pts.map(add_id).select(["point_id", "longitude", "latitude"])


def submit_points_export(project: str, bucket: str, prefix: str, start_year: int, end_year: int,
                         n_points: int, seed: int = 42, include_savanna: bool = False) -> ExportTaskInfo:
    """Submit a GCS export task for the stable grassland point sample."""
    initialize_ee(project)
    points = stable_grassland_points(start_year, end_year, n_points, seed, include_savanna)
    desc = f"wue_points_{start_year}_{end_year}_{n_points}"
    task = ee.batch.Export.table.toCloudStorage(
        collection=points,
        description=desc,
        bucket=bucket,
        fileNamePrefix=f"{prefix.rstrip('/')}/points/stable_grassland_points",
        fileFormat="CSV",
        selectors=["point_id", "longitude", "latitude"],
    )
    task.start()
    status = task.status()
    return ExportTaskInfo(description=desc, task_id=status.get("id", ""), state=status.get("state", "UNKNOWN"))


def _eight_day_starts(year: int) -> List[date]:
    d = date(year, 1, 1)
    out = []
    while d.year == year:
        out.append(d)
        d += timedelta(days=8)
    return out


def _vpd_kpa_from_t_dew(temp_k: ee.Image, dew_k: ee.Image) -> ee.Image:
    """Tetens VPD in kPa from temperature and dewpoint in Kelvin."""
    t = temp_k.subtract(273.15)
    td = dew_k.subtract(273.15)
    es = t.multiply(17.27).divide(t.add(237.3)).exp().multiply(0.6108)
    ea = td.multiply(17.27).divide(td.add(237.3)).exp().multiply(0.6108)
    return es.subtract(ea).max(0).rename("vpd")


def composite_for_8day(start: date) -> ee.Image:
    """Return one multi-band 8-day composite image with all Earth Engine products."""
    end = start + timedelta(days=8)
    start_s, end_s = start.isoformat(), end.isoformat()
    year, doy = start.year, int(start.strftime("%j"))

    pml = _safe_first(
        ee.ImageCollection("projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a").filterDate(start_s, end_s),
        ee.Image.constant([float("nan"), float("nan")]).rename(["GPP", "ET"]),
    ).select(["GPP", "ET"]).multiply(0.01).rename(["gpp_pml", "et_pml"])

    mod16 = _safe_first(
        ee.ImageCollection("MODIS/061/MOD16A2GF").filterDate(start_s, end_s),
        ee.Image.constant(float("nan")).rename("ET"),
    ).select("ET").multiply(0.1 / 8.0).rename("et_modis")

    # Earth Engine currently lists MOD17 GPP from 2021 onward. For earlier years,
    # export NaN so the downstream pipeline can use PML/GOSIF or AppEEARS MOD17.
    mod17_fallback = ee.Image.constant(float("nan")).rename("Gpp")
    mod17 = _safe_first(
        ee.ImageCollection("MODIS/061/MOD17A2HGF").filterDate(start_s, end_s),
        mod17_fallback,
    ).select("Gpp").multiply(0.0001 * 1000.0 / 8.0).rename("gpp_modis")

    era = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(start_s, end_s)
    t_mean = era.select("temperature_2m").mean()
    td_mean = era.select("dewpoint_temperature_2m").mean()
    vpd = _vpd_kpa_from_t_dew(t_mean, td_mean)
    temp_c = t_mean.subtract(273.15).rename("temperature")
    sm = era.select(["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2"]).mean()
    root_sm = sm.select("volumetric_soil_water_layer_1").multiply(0.25).add(
        sm.select("volumetric_soil_water_layer_2").multiply(0.75)
    ).rename("soil_moisture")
    precip = era.select("total_precipitation_hourly").sum().multiply(1000.0).rename("precipitation")

    lai = _safe_first(
        ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(start_s, end_s),
        ee.Image.constant(float("nan")).rename("Lai_500m"),
    ).select("Lai_500m").multiply(0.1).rename("lai")

    burned = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start_s, end_s).select("BurnDate").max().gt(0).rename("burned")

    return ee.Image.cat([mod17, mod16, pml, vpd, root_sm, temp_c, precip, lai, burned]).set({
        "date": start_s,
        "year": year,
        "doy": doy,
    })


def _sample_one_date(img: ee.Image, points: ee.FeatureCollection, scale: int):
    date_s = ee.String(img.get("date"))
    year = img.get("year")
    doy = img.get("doy")
    fc = img.sampleRegions(
        collection=points,
        properties=["point_id", "longitude", "latitude"],
        scale=scale,
        geometries=False,
        tileScale=8,
    )
    return fc.map(lambda f: f.set({"date": date_s, "year": year, "doy": doy}))


def submit_year_export(project: str, bucket: str, prefix: str, points_asset: str, year: int,
                       scale: int = 10000, priority: int = 100) -> ExportTaskInfo:
    """Submit one year of point-time table extraction to Cloud Storage."""
    initialize_ee(project)
    points = ee.FeatureCollection(points_asset)
    collections = []
    for d in _eight_day_starts(year):
        img = composite_for_8day(d)
        collections.append(_sample_one_date(img, points, scale=scale))
    out = ee.FeatureCollection(collections).flatten()
    desc = f"wue_timeseries_{year}"
    task = ee.batch.Export.table.toCloudStorage(
        collection=out,
        description=desc,
        bucket=bucket,
        fileNamePrefix=f"{prefix.rstrip('/')}/timeseries/wue_timeseries_{year}",
        fileFormat="CSV",
        selectors=EXPORT_SELECTORS,
        priority=priority,
    )
    task.start()
    status = task.status()
    return ExportTaskInfo(description=desc, task_id=status.get("id", ""), state=status.get("state", "UNKNOWN"))


def submit_year_exports(project: str, bucket: str, prefix: str, points_asset: str,
                        start_year: int, end_year: int, scale: int = 10000) -> List[ExportTaskInfo]:
    infos = []
    for y in range(start_year, end_year + 1):
        infos.append(submit_year_export(project, bucket, prefix, points_asset, y, scale=scale))
        time.sleep(1)
    return infos


def list_tasks(project: str) -> list[dict]:
    initialize_ee(project)
    return [t.status() for t in ee.batch.Task.list()]
