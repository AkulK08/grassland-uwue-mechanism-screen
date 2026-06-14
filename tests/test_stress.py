import numpy as np
import pandas as pd
import xarray as xr
from wue_pipeline.processing.stress import csi_zscore


def test_csi_zscore_shape():
    time = pd.date_range('2020-01-01', periods=10)
    v = xr.DataArray(np.random.randn(10, 2, 3), coords={'time': time, 'lat': [0,1], 'lon': [0,1,2]}, dims=('time','lat','lon'))
    sm = xr.DataArray(np.random.randn(10, 2, 3), coords=v.coords, dims=v.dims)
    out = csi_zscore(v, sm)
    assert out.shape == v.shape
