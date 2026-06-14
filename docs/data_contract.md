# Data Contract

Production mode expects all gridded products to share the following dimensions or be convertible to them:

- `time`
- `lat`
- `lon`

The pipeline accepts NetCDF, Zarr, or GeoTIFF inputs. If a file contains a single variable, it is renamed to the required variable name.

## Required gridded variables

- `gpp`: gross primary productivity.
- `et`: evapotranspiration.
- `vpd`: vapor pressure deficit.
- `soil_moisture`: root-zone soil moisture.
- `temperature`: near-surface temperature.
- `precipitation`: precipitation.
- `landcover`: IGBP land-cover class.
- `burned`: Boolean burned-area disturbance flag.
- `irrigated_fraction`: irrigated-area fraction.
- `aridity_index`: aridity index.
- `lai`: leaf area index.
- `psi50`: xylem vulnerability.
- `isohydricity`: ecosystem-scale isohydricity.
- `rooting_depth`: effective rooting depth or rooting-zone water storage proxy.

## Required CO2 table

CSV columns:

- `time`
- `co2_ppm`

## Required tower table

CSV columns:

- `site_id`
- `time`
- `latitude`
- `longitude`
- `igbp`
- `GPP_NT_VUT_REF`
- `LE_F_MDS`
- `LE_F_MDS_QC`
- `NEE_VUT_REF_QC`
- `H_F_MDS`
- `NETRAD`
- `G`
- `VPD_F_MDS`
- `SWC_F_MDS`
- `TA_F_MDS`

The pipeline can run without optional QC columns, but manuscript-quality validation should provide them.
