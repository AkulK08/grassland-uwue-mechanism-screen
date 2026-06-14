"""Bayesian change-point and trait models.

The functions are deliberately optional. They run when PyMC is installed and
`bayesian_enabled: true` in the configuration. For large global analyses, use
Bayesian fits on stratified samples or aggregated response curves rather than
every pixel.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def bayesian_change_point(x, y, draws: int = 1000, tune: int = 1000, seed: int = 42) -> dict:
    try:
        import pymc as pm
        import arviz as az
    except Exception as exc:
        return {"available": False, "reason": f"pymc/arviz unavailable: {exc}", "breakpoint_low": np.nan, "breakpoint_high": np.nan}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 40:
        return {"available": False, "reason": "insufficient_data", "breakpoint_low": np.nan, "breakpoint_high": np.nan}
    x_min, x_max = np.quantile(x, [0.10, 0.90])
    with pm.Model() as model:
        tau = pm.Uniform("tau", lower=float(x_min), upper=float(x_max))
        alpha = pm.Normal("alpha", 0, 2)
        beta1 = pm.Normal("beta1", 0, 1)
        beta2 = pm.Normal("beta2", 0, 1)
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha + beta1 * x + beta2 * pm.math.maximum(x - tau, 0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=2, random_seed=seed, progressbar=False, target_accept=0.9)
    q = az.summary(trace, var_names=["tau", "beta1", "beta2"], hdi_prob=0.95)
    tau_low = float(q.loc["tau", "hdi_2.5%"] if "hdi_2.5%" in q.columns else q.loc["tau", "hdi_3%"])
    tau_high = float(q.loc["tau", "hdi_97.5%"] if "hdi_97.5%" in q.columns else q.loc["tau", "hdi_97%"])
    return {"available": True, "breakpoint_low": tau_low, "breakpoint_high": tau_high, "summary": q.reset_index().to_dict("records")}


def intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    if not all(np.isfinite(v) for v in [a_low, a_high, b_low, b_high]):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def bayesian_trait_regression(df: pd.DataFrame, y_col: str, predictors: list[str], draws: int = 2000, tune: int = 1000, seed: int = 42) -> pd.DataFrame:
    try:
        import pymc as pm
        import arviz as az
    except Exception as exc:
        return pd.DataFrame({"term": predictors, "mean": np.nan, "hdi_low": np.nan, "hdi_high": np.nan, "available": False, "reason": str(exc)})
    d = df[[y_col] + predictors].dropna().copy()
    X = d[predictors].to_numpy(dtype=float)
    y = d[y_col].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 1)
        beta = pm.Normal("beta", 0, 1, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", 1)
        mu = alpha + pm.math.dot(X, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=2, random_seed=seed, progressbar=False, target_accept=0.9)
    summ = az.summary(trace, var_names=["beta"], hdi_prob=0.95).reset_index()
    rows = []
    for i, p in enumerate(predictors):
        row = summ.iloc[i]
        low_col = "hdi_2.5%" if "hdi_2.5%" in summ.columns else "hdi_3%"
        high_col = "hdi_97.5%" if "hdi_97.5%" in summ.columns else "hdi_97%"
        rows.append({"term": p, "mean": float(row["mean"]), "hdi_low": float(row[low_col]), "hdi_high": float(row[high_col]), "available": True})
    return pd.DataFrame(rows)
