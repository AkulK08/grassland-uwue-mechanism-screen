#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


GPP_PRODUCTS = ["modis", "gosif", "pml"]
ET_PRODUCTS = ["modis", "gleam", "pml"]
METRICS = ["uwue"]


def ols_fit(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    n, k = X.shape
    sigma2 = rss / max(1, n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)
    return beta, se, rss


def bic(rss, n, k):
    rss = max(float(rss), 1e-12)
    return n * math.log(rss / n) + k * math.log(max(n, 2))


def fit_linear(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, se, rss = ols_fit(X, y)
    return {
        "linear_slope": float(beta[1]),
        "linear_slope_se": float(se[1]),
        "linear_rss": float(rss),
        "linear_bic": float(bic(rss, len(x), 2)),
    }


def fit_segmented(x, y, min_side=8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    uniq = np.unique(x)
    if len(uniq) < 2 * min_side + 1:
        return None

    candidates = uniq[min_side:-min_side]
    best = None

    for tau in candidates:
        left = np.minimum(x, tau)
        hinge = np.maximum(0, x - tau)
        X = np.column_stack([np.ones(len(x)), left, hinge])
        beta, se, rss = ols_fit(X, y)
        pre = beta[1]
        change = beta[2]
        post = pre + change
        out = {
            "breakpoint": float(tau),
            "pre_slope": float(pre),
            "post_slope": float(post),
            "slope_change": float(change),
            "segmented_rss": float(rss),
            "segmented_bic": float(bic(rss, len(x), 3)),
        }
        if best is None or out["segmented_rss"] < best["segmented_rss"]:
            best = out

    return best


def segmented_rss_at_tau(x, y, tau):
    left = np.minimum(x, tau)
    hinge = np.maximum(0, x - tau)
    X = np.column_stack([np.ones(len(x)), left, hinge])
    beta, se, rss = ols_fit(X, y)
    return rss


def profile_tau_interval(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uniq = np.unique(x)

    if len(uniq) < 20:
        return np.nan, np.nan

    taus = uniq[6:-6]
    if len(taus) < 4:
        return np.nan, np.nan

    rss = np.array([segmented_rss_at_tau(x, y, t) for t in taus])
    if not np.isfinite(rss).all():
        return np.nan, np.nan

    sigma2 = max(float(np.nanmin(rss) / max(1, len(x) - 3)), 1e-12)
    w = np.exp(-0.5 * (rss - np.nanmin(rss)) / sigma2)

    if not np.isfinite(w).any() or w.sum() <= 0:
        return np.nan, np.nan

    w = w / w.sum()
    order = np.argsort(taus)
    taus = taus[order]
    w = w[order]
    cdf = np.cumsum(w)

    lo = float(np.interp(0.025, cdf, taus))
    hi = float(np.interp(0.975, cdf, taus))
    return lo, hi


def supf_permutation_p(x, y, n_perm=9, seed=1):
    rng = np.random.default_rng(seed)

    lin = fit_linear(x, y)
    seg = fit_segmented(x, y)
    if seg is None:
        return np.nan

    obs = max(0.0, lin["linear_rss"] - seg["segmented_rss"])
    if obs <= 0:
        return 1.0

    count = 1
    y = np.asarray(y, dtype=float)

    for _ in range(n_perm):
        yp = rng.permutation(y)
        linp = fit_linear(x, yp)
        segp = fit_segmented(x, yp)
        stat = 0.0 if segp is None else max(0.0, linp["linear_rss"] - segp["segmented_rss"])
        if stat >= obs:
            count += 1

    return float(count / (n_perm + 1))


def intervals_overlap(a_low, a_high, b_low, b_high):
    vals = [a_low, a_high, b_low, b_high]
    if not all(np.isfinite(vals)):
        return False
    return max(a_low, b_low) <= min(a_high, b_high)


def year_block_sample(sub, rng):
    years = pd.Series(sub["year"]).dropna().astype(int).unique()
    if len(years) == 0:
        return sub.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1_000_000)))
    chosen = rng.choice(years, size=len(years), replace=True)
    parts = [sub[sub["year"].astype("Int64") == int(y)] for y in chosen]
    return pd.concat(parts, ignore_index=True)


def block_bootstrap_ci(sub, ycol, n_boot=10, seed=1):
    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_boot):
        bs = year_block_sample(sub, rng)
        x = bs["compound_z"].to_numpy(float)
        y = bs[ycol].to_numpy(float)

        if len(bs) < 20 or len(np.unique(x)) < 15:
            continue

        seg = fit_segmented(x, y)
        if seg is not None:
            vals.append(seg)

    out = {"n_block_boot_success": len(vals)}

    for key in ["breakpoint", "pre_slope", "post_slope", "slope_change"]:
        arr = np.array([v[key] for v in vals if np.isfinite(v[key])], dtype=float)
        if len(arr) >= 3:
            out[f"{key}_block_ci_low"] = float(np.nanpercentile(arr, 2.5))
            out[f"{key}_block_ci_high"] = float(np.nanpercentile(arr, 97.5))
        else:
            out[f"{key}_block_ci_low"] = np.nan
            out[f"{key}_block_ci_high"] = np.nan

    return out


def classify_strict(row):
    accepted = bool(row.get("accepted_transition", False))

    pre_lo = row.get("pre_slope_block_ci_low", np.nan)
    post_lo = row.get("post_slope_block_ci_low", np.nan)
    post_hi = row.get("post_slope_block_ci_high", np.nan)
    change_hi = row.get("slope_change_block_ci_high", np.nan)
    linear_slope = row.get("linear_slope", np.nan)

    if not accepted:
        if np.isfinite(linear_slope) and linear_slope > 0:
            return "enhancement_no_accepted_breakpoint"
        return "inconclusive"

    if np.isfinite(pre_lo) and np.isfinite(post_lo) and np.isfinite(post_hi) and np.isfinite(change_hi):
        if pre_lo > 0 and post_hi < 0 and change_hi < 0:
            return "breakdown"
        if pre_lo > 0 and post_lo <= 0 <= post_hi and change_hi < 0:
            return "saturation"
        if pre_lo > 0 and post_lo > 0:
            return "enhancement"

    return "inconclusive"


def fit_surface(sub, ycol):
    use = sub[["vpd_z", "sm_z", "vpd_x_sm", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 20:
        return {}

    X = np.column_stack([
        np.ones(len(use)),
        use["vpd_z"].to_numpy(float),
        use["sm_z"].to_numpy(float),
        use["vpd_x_sm"].to_numpy(float),
    ])
    y = use[ycol].to_numpy(float)
    beta, se, rss = ols_fit(X, y)

    return {
        "surface_n": int(len(use)),
        "surface_intercept": float(beta[0]),
        "vpd_partial_effect": float(beta[1]),
        "sm_partial_effect": float(beta[2]),
        "vpd_sm_interaction": float(beta[3]),
        "vpd_partial_se": float(se[1]),
        "sm_partial_se": float(se[2]),
        "vpd_sm_interaction_se": float(se[3]),
        "surface_rss": float(rss),
    }


def analyze_one(path, version, n_boot, n_perm, min_obs, max_points):
    df = pd.read_csv(path)
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    points = sorted(df["point_id"].dropna().unique())
    if max_points is not None:
        points = points[:max_points]

    rows = []
    surf_rows = []

    for i, point_id in enumerate(points, start=1):
        print(f"[{version}] point {i}/{len(points)} {point_id}", flush=True)
        d0 = df[df["point_id"] == point_id].copy()

        for gpp in GPP_PRODUCTS:
            for et in ET_PRODUCTS:
                combo = f"{gpp}_{et}"

                for metric in METRICS:
                    ycol = f"log_{metric}_{combo}"
                    if ycol not in d0.columns:
                        continue

                    sub = d0[["point_id", "date", "year", "compound_z", "vpd_z", "sm_z", "vpd_x_sm", ycol]].copy()
                    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(sub) < min_obs:
                        continue

                    x = sub["compound_z"].to_numpy(float)
                    y = sub[ycol].to_numpy(float)

                    if len(np.unique(x)) < 15:
                        continue

                    lin = fit_linear(x, y)
                    seg = fit_segmented(x, y)

                    if seg is None:
                        continue

                    delta_bic = lin["linear_bic"] - seg["segmented_bic"]
                    supf_p = supf_permutation_p(
                        x,
                        y,
                        n_perm=n_perm,
                        seed=abs(hash((version, point_id, combo, metric))) % (2**32)
                    )
                    tau_lo, tau_hi = profile_tau_interval(x, y)
                    boot = block_bootstrap_ci(
                        sub,
                        ycol,
                        n_boot=n_boot,
                        seed=abs(hash(("boot", version, point_id, combo, metric))) % (2**32)
                    )

                    bayes_overlap = intervals_overlap(
                        boot.get("breakpoint_block_ci_low", np.nan),
                        boot.get("breakpoint_block_ci_high", np.nan),
                        tau_lo,
                        tau_hi,
                    )

                    accepted_transition = (
                        np.isfinite(delta_bic) and delta_bic >= 6 and
                        np.isfinite(supf_p) and supf_p <= 0.05 and
                        bayes_overlap
                    )

                    row = {
                        "version": version,
                        "point_id": point_id,
                        "gpp_product": gpp.upper(),
                        "et_product": et.upper(),
                        "combo": combo,
                        "metric": metric,
                        "primary_metric": metric == "uwue",
                        "n": int(len(sub)),
                        **lin,
                        **seg,
                        "delta_bic_linear_minus_segmented": float(delta_bic),
                        "supf_permutation_p": float(supf_p) if np.isfinite(supf_p) else np.nan,
                        "bayes_tau_ci_low": tau_lo,
                        "bayes_tau_ci_high": tau_hi,
                        "bayes_overlap": bool(bayes_overlap),
                        "accepted_transition": bool(accepted_transition),
                        **boot,
                    }
                    row["response_class_strict"] = classify_strict(row)
                    rows.append(row)

                    surf = fit_surface(sub, ycol)
                    if surf:
                        surf_rows.append({
                            "version": version,
                            "point_id": point_id,
                            "gpp_product": gpp.upper(),
                            "et_product": et.upper(),
                            "combo": combo,
                            "metric": metric,
                            "primary_metric": metric == "uwue",
                            **surf,
                        })

    outdir = Path("results/project_strict_boot100_primary")
    outdir.mkdir(parents=True, exist_ok=True)

    res = pd.DataFrame(rows)
    surf = pd.DataFrame(surf_rows)

    res_path = outdir / f"strict_response_results_{version}.csv"
    surf_path = outdir / f"vpd_sm_surface_partial_effects_{version}.csv"
    summary_path = outdir / f"strict_response_summary_{version}.csv"

    res.to_csv(res_path, index=False)
    surf.to_csv(surf_path, index=False)

    if len(res):
        summary = (
            res.groupby(["version", "metric", "gpp_product", "et_product", "response_class_strict"], dropna=False)
            .size()
            .reset_index(name="n_fits")
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(summary_path, index=False)

    print("WROTE", res_path, res.shape)
    print("WROTE", surf_path, surf.shape)
    print("WROTE", summary_path, summary.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=9)
    ap.add_argument("--min-obs", type=int, default=50)
    ap.add_argument("--max-points", type=int, default=None)
    args = ap.parse_args()

    analyze_one("data/processed/project_metric_matrix_raw.csv", "raw", args.n_boot, args.n_perm, args.min_obs, args.max_points)
    analyze_one("data/processed/project_metric_matrix_co2corrected.csv", "co2corrected", args.n_boot, args.n_perm, args.min_obs, args.max_points)


if __name__ == "__main__":
    main()
