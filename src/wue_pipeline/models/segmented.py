"""Segmented regression, bootstrap uncertainty, and response classification."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


@dataclass
class SegmentedFit:
    n: int
    breakpoint: float
    pre_slope: float
    post_slope: float
    slope_change: float
    intercept: float
    rss: float
    aic: float
    bic: float
    pre_slope_low: float = np.nan
    pre_slope_high: float = np.nan
    post_slope_low: float = np.nan
    post_slope_high: float = np.nan
    slope_change_low: float = np.nan
    slope_change_high: float = np.nan
    breakpoint_low: float = np.nan
    breakpoint_high: float = np.nan
    bayes_breakpoint_low: float = np.nan
    bayes_breakpoint_high: float = np.nan
    bayes_overlap: bool = False
    response_class: str = "inconclusive"
    converged: bool = True
    reason: str = "ok"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _piecewise_design(x: np.ndarray, tau: float) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, np.maximum(x - tau, 0.0)])


def _fit_for_tau(x: np.ndarray, y: np.ndarray, tau: float):
    X = _piecewise_design(x, tau)
    model = OLS(y, X).fit()
    b0, b1, b2 = model.params
    pre = b1
    post = b1 + b2
    return model, float(pre), float(post), float(b2)


def fit_segmented(x, y, min_obs: int = 50, n_grid: int = 80) -> SegmentedFit:
    x, y = _clean_xy(x, y)
    n = len(x)
    if n < min_obs:
        return SegmentedFit(n=n, breakpoint=np.nan, pre_slope=np.nan, post_slope=np.nan, slope_change=np.nan, intercept=np.nan, rss=np.nan, aic=np.nan, bic=np.nan, converged=False, reason="insufficient_data")
    # Candidate taus inside central data range to keep both segments supported.
    low, high = np.quantile(x, [0.15, 0.85])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return SegmentedFit(n=n, breakpoint=np.nan, pre_slope=np.nan, post_slope=np.nan, slope_change=np.nan, intercept=np.nan, rss=np.nan, aic=np.nan, bic=np.nan, converged=False, reason="invalid_x_range")
    taus = np.linspace(low, high, n_grid)
    best = None
    for tau in taus:
        if (x <= tau).sum() < max(8, min_obs // 5) or (x > tau).sum() < max(8, min_obs // 5):
            continue
        model, pre, post, change = _fit_for_tau(x, y, tau)
        rss = float(np.sum(model.resid ** 2))
        if best is None or rss < best[0]:
            best = (rss, tau, model, pre, post, change)
    if best is None:
        lin = OLS(y, add_constant(x)).fit()
        slope = float(lin.params[1])
        return SegmentedFit(n=n, breakpoint=np.nan, pre_slope=slope, post_slope=slope, slope_change=0.0, intercept=float(lin.params[0]), rss=float(lin.ssr), aic=float(lin.aic), bic=float(lin.bic), reason="linear_only")
    rss, tau, model, pre, post, change = best
    return SegmentedFit(n=n, breakpoint=float(tau), pre_slope=pre, post_slope=post, slope_change=change, intercept=float(model.params[0]), rss=rss, aic=float(model.aic), bic=float(model.bic))


def bootstrap_segmented(x, y, min_obs: int, n_boot: int, seed: int = 42, n_grid: int = 80) -> dict:
    x, y = _clean_xy(x, y)
    n = len(x)
    rng = np.random.default_rng(seed)
    rows = []
    if n < min_obs:
        return {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fit = fit_segmented(x[idx], y[idx], min_obs=min_obs, n_grid=n_grid)
        if fit.converged:
            rows.append({
                "breakpoint": fit.breakpoint,
                "pre_slope": fit.pre_slope,
                "post_slope": fit.post_slope,
                "slope_change": fit.slope_change,
            })
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        arr = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(arr) == 0:
            lo, hi = np.nan, np.nan
        else:
            lo, hi = np.quantile(arr, [0.025, 0.975])
        out[f"{col}_low"] = float(lo)
        out[f"{col}_high"] = float(hi)
    return out


def classify_response(pre_low: float, pre_high: float, post_low: float, post_high: float, slope_change_low: float, slope_change_high: float) -> str:
    # Enhancement: positive across full range, no meaningful negative post-transition.
    if np.isfinite(pre_low) and np.isfinite(post_low) and pre_low > 0 and post_low > 0:
        if not (slope_change_high < 0):
            return "enhancement"
        return "enhancement"
    # Reversal: pre positive and post significantly negative.
    if np.isfinite(pre_low) and np.isfinite(post_high) and pre_low > 0 and post_high < 0:
        return "reversal"
    # Saturation: pre positive, post interval includes zero, and not significantly negative.
    if np.isfinite(pre_low) and np.isfinite(post_low) and np.isfinite(post_high):
        if pre_low > 0 and post_low <= 0 <= post_high:
            return "saturation"
    return "inconclusive"


def segmented_with_uncertainty(x, y, min_obs: int = 50, n_boot: int = 1000, seed: int = 42, n_grid: int = 80) -> SegmentedFit:
    fit = fit_segmented(x, y, min_obs=min_obs, n_grid=n_grid)
    if not fit.converged:
        fit.response_class = fit.reason
        return fit
    ci = bootstrap_segmented(x, y, min_obs=min_obs, n_boot=n_boot, seed=seed, n_grid=n_grid)
    for k, v in ci.items():
        setattr(fit, k, v)
    fit.response_class = classify_response(
        fit.pre_slope_low, fit.pre_slope_high,
        fit.post_slope_low, fit.post_slope_high,
        fit.slope_change_low, fit.slope_change_high,
    )
    return fit


def fit_interaction_model(df: pd.DataFrame, y_col="log_wue", vpd_col="vpd_z", sm_col="sm_z") -> dict:
    d = df[[y_col, vpd_col, sm_col]].dropna().copy()
    if len(d) < 30:
        return {"n": len(d), "interaction_coef": np.nan, "interaction_p": np.nan, "response_class": "insufficient_data"}
    d["interaction"] = d[vpd_col] * d[sm_col]
    X = add_constant(d[[vpd_col, sm_col, "interaction"]])
    model = OLS(d[y_col], X).fit()
    coef = float(model.params["interaction"])
    p = float(model.pvalues["interaction"])
    return {"n": len(d), "interaction_coef": coef, "interaction_p": p, "interaction_r2": float(model.rsquared), "response_class": "interaction_negative" if coef < 0 and p < 0.05 else "interaction_not_negative"}
