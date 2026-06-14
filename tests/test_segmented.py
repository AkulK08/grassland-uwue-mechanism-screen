import numpy as np
from wue_pipeline.models.segmented import segmented_with_uncertainty


def test_segmented_recovers_reversal():
    rng = np.random.default_rng(1)
    x = np.linspace(-2, 4, 120)
    y = 0.2 * x - 0.55 * np.maximum(x - 1.0, 0) + rng.normal(0, 0.03, size=x.size)
    fit = segmented_with_uncertainty(x, y, min_obs=40, n_boot=20, seed=1)
    assert fit.converged
    assert np.isfinite(fit.breakpoint)
    assert fit.pre_slope > 0
    assert fit.post_slope < 0
