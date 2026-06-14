# Command-line-only autonomous workflow

This workflow avoids manual data downloads of 23 years of global rasters. It uses remote agents:

1. Google Earth Engine keeps MODIS/PML/ERA5/MCD12Q1/MCD64A1/MOD15 rasters server-side.
2. Earth Engine exports only point-time CSVs to Google Cloud Storage.
3. `gcloud storage cp` downloads only compact CSV tables.
4. The local point backend runs the same response-shape classification and robustness matrix by `point_id`.
5. GOSIF/GLEAM can be added by separate command-line agents because they are not standard Earth Engine catalog datasets.

## One-time command-line setup

```bash
pip install -e .
pip install earthengine-api google-cloud-storage

gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable earthengine.googleapis.com storage.googleapis.com

earthengine authenticate

gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=US
```

## Export stable grassland points

```bash
wue remote gee-submit-points \
  --project YOUR_GCP_PROJECT_ID \
  --bucket YOUR_BUCKET_NAME \
  --prefix wue_remote \
  --start-year 2001 \
  --end-year 2024 \
  --n-points 50000
```

Download the exported points:

```bash
mkdir -p data/raw/gee
mkdir -p data/raw/gee/points
gcloud storage cp gs://YOUR_BUCKET_NAME/wue_remote/points/*.csv data/raw/gee/points/
```

Upload that CSV as an Earth Engine asset using the Earth Engine CLI:

```bash
earthengine upload table \
  --asset_id=projects/YOUR_GCP_PROJECT_ID/assets/wue/stable_grassland_points \
  gs://YOUR_BUCKET_NAME/wue_remote/points/stable_grassland_points.csv
```

Wait until ingestion finishes:

```bash
earthengine task list
```

## Export 8-day point-time tables

```bash
wue remote gee-submit-years \
  --project YOUR_GCP_PROJECT_ID \
  --bucket YOUR_BUCKET_NAME \
  --prefix wue_remote \
  --points-asset projects/YOUR_GCP_PROJECT_ID/assets/wue/stable_grassland_points \
  --start-year 2001 \
  --end-year 2024 \
  --scale 10000
```

Monitor tasks:

```bash
wue remote gee-tasks --project YOUR_GCP_PROJECT_ID
```

Download the CSV exports:

```bash
mkdir -p data/raw/gee
gcloud storage cp 'gs://YOUR_BUCKET_NAME/wue_remote/timeseries/*.csv' data/raw/gee/
```

## Run point-table Gate 1/Gate 2 analysis

```bash
wue points run-all \
  --input-glob 'data/raw/gee/wue_timeseries_*.csv' \
  --gpp-products MODIS,PML \
  --et-products MODIS,PML \
  --min-obs 50 \
  --n-boot 1000
```

Outputs:

```text
results/tables/point_gate2_pixel_results.csv
results/tables/point_gate2_robustness_matrix.csv
```

## Full-original matrix caveat

The original plan requires MODIS/GOSIF/PML × MODIS/GLEAM/PML. Earth Engine can cover PML, MODIS ET, ERA5-Land, land cover, burned area, and LAI. Current Earth Engine MODIS GPP availability starts in 2021, so full 2001-2024 MODIS GPP must be handled via an external command-line agent such as AppEEARS/LP DAAC. GOSIF can be scripted from the UNH public repository. GLEAM requires SFTP credentials from the GLEAM site; once credentials exist, an autonomous cloud VM can download/scratch-sample/delete raw rasters and keep only point-time CSVs.

This is still command-line only; the limitation is data access policy/availability, not the statistical workflow.
