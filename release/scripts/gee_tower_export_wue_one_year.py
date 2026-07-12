#!/usr/bin/env python
import argparse
import sys
import ee

sys.path.insert(0, "scripts")
from gee_tower_export_wue import init_ee, read_points, sample_one_date

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--points", default="data/raw/towers/tower_validation_points_agent.csv")
    ap.add_argument("--folder", default="grassland_wue_tower_validation_core")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--scale", type=int, default=500)
    args = ap.parse_args()

    init_ee(args.project)
    points = read_points(args.points)

    fcs = []
    for doy in range(1, 362, 8):
        date = ee.Date.fromYMD(args.year, 1, 1).advance(doy - 1, "day")
        fcs.append(sample_one_date(points, date, args.scale))

    out_fc = ee.FeatureCollection(fcs).flatten()
    desc = f"tower_wue_timeseries_{args.year}"

    task = ee.batch.Export.table.toDrive(
        collection=out_fc,
        description=desc,
        folder=args.folder,
        fileNamePrefix=desc,
        fileFormat="CSV"
    )
    task.start()

    print("STARTED_TOWER_YEAR_EXPORT")
    print("year:", args.year)
    print("description:", desc)
    print("task_id:", task.id)
    print("drive_folder:", args.folder)

if __name__ == "__main__":
    main()
