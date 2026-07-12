#!/usr/bin/env python
from __future__ import annotations

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd

GPP_PRODUCTS = ["modis", "gosif", "pml"]
ET_PRODUCTS = ["modis", "gleam", "pml"]

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    return (s - s.mean(skipna=True)) / sd if pd.notna(sd) and sd != 0 else s * np.nan

def ols(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid ** 2))
    return beta, rss

def bic(rss, n, k):
    return n * math.log(max(rss, 1e-12) / max(n, 1)) + k * math.log(max(n, 2))

def fit_linear(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b, rss = ols(X, y)
    return {"linear_slope": float(b[1]), "linear_rss": rss, "linear_bic": bic(rss, len(x), 2)}

def fit_segmented(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x = x[o]
    y = y[o]
    u = np.unique(x)
    if len(u) < 18:
        return None
    best = None
    for tau in u[8:-8]:
        X = np.column_stack([np.ones(len(x)), np.minimum(x, tau), np.maximum(0, x - tau)])
        b, rss = ols(X, y)
        pre = b[1]
        change = b[2]
        post = pre + change
        out = {
            "breakpoint": float(tau),
            "pre_slope": float(pre),
            "post_slope": float(post),
            "slope_change": float(change),
            "segmented_rss": rss,
            "segmented_bic": bic(rss, len(x), 3),
        }
        if best is None or out["segmented_rss"] < best["segmented_rss"]:
            best = out
    return best

def profile_ci(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u = np.unique(x)
    if len(u) < 18:
        return np.nan, np.nan
    taus = u[8:-8]
    vals = []
    for tau in taus:
        X = np.column_stack([np.ones(len(x)), np.minimum(x, tau), np.maximum(0, x - tau)])
        _, rss = ols(X, y)
        vals.append(rss)
    vals = np.asarray(vals)
    sig2 = max(float(np.nanmin(vals) / max(1, len(x)-3)), 1e-12)
    w = np.exp(-0.5 * (vals - np.nanmin(vals)) / sig2)
    if not np.isfinite(w).any() or w.sum() == 0:
        return np.nan, np.nan
    w = w / w.sum()
    c = np.cumsum(w)
    return float(np.interp(0.025, c, taus)), float(np.interp(0.975, c, taus))

def supf_p(x, y, n_perm, seed):
    rng = np.random.default_rng(seed)
    lin = fit_linear(x, y)
    seg = fit_segmented(x, y)
    if seg is None:
        return np.nan
    obs = max(0, lin["linear_rss"] - seg["segmented_rss"])
    count = 1
    for _ in range(n_perm):
        yp = rng.permutation(y)
        lp = fit_linear(x, yp)
        sp = fit_segmented(x, yp)
        stat = 0 if sp is None else max(0, lp["linear_rss"] - sp["segmented_rss"])
        if stat >= obs:
            count += 1
    return count / (n_perm + 1)

def year_block_boot(sub, ycol, xcol, n_boot, seed):
    rng = np.random.default_rng(seed)
    years = pd.Series(sub["year"]).dropna().astype(int).unique()
    vals = []
    if len(years) == 0:
        return {}
    for _ in range(n_boot):
        chosen = rng.choice(years, size=len(years), replace=True)
        bs = pd.concat([sub[sub["year"].astype("Int64") == int(y)] for y in chosen], ignore_index=True)
        if len(bs) < 20 or bs[xcol].nunique() < 18:
            continue
        seg = fit_segmented(bs[xcol].to_numpy(float), bs[ycol].to_numpy(float))
        if seg:
            vals.append(seg)
    out = {"n_block_boot_success": len(vals)}
    for k in ["breakpoint", "pre_slope", "post_slope", "slope_change"]:
        a = np.asarray([v[k] for v in vals if np.isfinite(v[k])], float)
        if len(a) >= 3:
            out[f"{k}_block_ci_low"] = float(np.nanpercentile(a, 2.5))
            out[f"{k}_block_ci_high"] = float(np.nanpercentile(a, 97.5))
        else:
            out[f"{k}_block_ci_low"] = np.nan
            out[f"{k}_block_ci_high"] = np.nan
    return out

def overlap(a, b, c, d):
    return all(np.isfinite([a,b,c,d])) and max(a,c) <= min(b,d)

def classify(row):
    if not row["accepted_transition"]:
        return "enhancement_no_accepted_breakpoint" if row["linear_slope"] > 0 else "inconclusive"
    pre_lo = row.get("pre_slope_block_ci_low", np.nan)
    post_lo = row.get("post_slope_block_ci_low", np.nan)
    post_hi = row.get("post_slope_block_ci_high", np.nan)
    ch_hi = row.get("slope_change_block_ci_high", np.nan)
    if np.isfinite(pre_lo) and np.isfinite(post_lo) and np.isfinite(post_hi) and np.isfinite(ch_hi):
        if pre_lo > 0 and post_hi < 0 and ch_hi < 0:
            return "breakdown"
        if pre_lo > 0 and post_lo <= 0 <= post_hi and ch_hi < 0:
            return "saturation"
        if pre_lo > 0 and post_lo > 0:
            return "enhancement"
    return "inconclusive"

def first_col(df, choices):
    low = {c.lower(): c for c in df.columns}
    for x in choices:
        if x.lower() in low:
            return low[x.lower()]
    for c in df.columns:
        for x in choices:
            if x.lower() in c.lower():
                return c
    return None

def prep(df):
    df = df.copy()
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    if "vpd_z" not in df.columns:
        v = first_col(df, ["vpd", "vpd_kpa"])
        df["vpd_z"] = df.groupby("point_id")[v].transform(zscore)
    if "sm_z" not in df.columns:
        s = first_col(df, ["soil_moisture", "sm", "rootzone"])
        df["sm_z"] = df.groupby("point_id")[s].transform(zscore)
    if "compound_z" not in df.columns:
        df["compound_z"] = 0.5 * df["vpd_z"] - 0.5 * df["sm_z"]

    df["vpd_pct"] = df.groupby("point_id")["vpd_z"].rank(pct=True)
    df["sm_pct"] = df.groupby("point_id")["sm_z"].rank(pct=True)
    df["stress_zscore"] = df["compound_z"]
    df["stress_percentile_joint"] = 0.5 * df["vpd_pct"] + 0.5 * (1 - df["sm_pct"])
    df["stress_copula_joint"] = df["vpd_pct"] * (1 - df["sm_pct"])
    df["vpd_x_sm"] = df["vpd_z"] * df["sm_z"]

    t = first_col(df, ["temperature", "temp", "tmean"])
    p = first_col(df, ["precipitation", "precip", "ppt"])
    if t:
        ok = pd.to_numeric(df[t], errors="coerce") > 5
        if p:
            ok &= pd.to_numeric(df[p], errors="coerce").fillna(0) >= 1
        df["gs_climate_common"] = ok
    else:
        df["gs_climate_common"] = True

    for g in GPP_PRODUCTS:
        c = f"gpp_{g}"
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            peak = x.groupby([df["point_id"], df["year"]]).transform("max")
            df[f"gs_gpp_threshold_{g}"] = x >= 0.2 * peak
        else:
            df[f"gs_gpp_threshold_{g}"] = False
    return df

def residualize_month(sub, ycol):
    tmp = sub.copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=[ycol, "month"])
    if len(tmp) == 0:
        tmp[ycol + "_month_resid"] = np.nan
        return tmp, ycol + "_month_resid"
    d = pd.get_dummies(tmp["month"], prefix="m", drop_first=True)
    X = np.column_stack([np.ones(len(tmp)), d.to_numpy(float)])
    y = tmp[ycol].to_numpy(float)
    if len(y) == 0 or not np.isfinite(y).any():
        tmp[ycol + "_month_resid"] = np.nan
        return tmp, ycol + "_month_resid"
    b, _ = ols(X, y)
    tmp[ycol + "_month_resid"] = y - X @ b + np.nanmean(y)
    return tmp, ycol + "_month_resid"

def surface(sub, ycol):
    u = sub[["vpd_z", "sm_z", "vpd_x_sm", ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(u) < 20:
        return {}
    X = np.column_stack([np.ones(len(u)), u["vpd_z"], u["sm_z"], u["vpd_x_sm"]])
    y = u[ycol].to_numpy(float)
    b, rss = ols(X, y)
    return {"surface_n": len(u), "vpd_partial_effect": b[1], "sm_partial_effect": b[2], "vpd_sm_interaction": b[3], "surface_rss": rss}

def analyze(path, version, args):
    df = prep(pd.read_csv(path))
    outdir = Path("results/project_final_nature")
    outdir.mkdir(parents=True, exist_ok=True)

    qc = {"rows_before": int(len(df))}
    keep = pd.Series(True, index=df.index)
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["burn", "disturb"]):
            bad = pd.to_numeric(df[c], errors="coerce").fillna(0) > 0
            qc[f"excluded_{c}"] = int(bad.sum())
            keep &= ~bad
        if any(k in cl for k in ["cropland_fraction", "crop_fraction"]):
            bad = pd.to_numeric(df[c], errors="coerce").fillna(0) > 0
            qc[f"excluded_{c}"] = int(bad.sum())
            keep &= ~bad
    df = df[keep].copy()
    qc["rows_after"] = int(len(df))
    Path(outdir / f"qc_audit_{version}.json").write_text(pd.Series(qc).to_json(indent=2))

    pts = sorted(df["point_id"].dropna().unique())
    if args.max_points:
        pts = pts[:args.max_points]

    metrics = args.metrics.split(",")
    stresses = args.stress_defs.split(",")
    seasons = args.growing_seasons.split(",")

    rows, surfs = [], []
    for i, pid in enumerate(pts, 1):
        print(f"[{version}] point {i}/{len(pts)} {pid}", flush=True)
        d0 = df[df["point_id"] == pid].copy()
        for g in GPP_PRODUCTS:
            for e in ET_PRODUCTS:
                combo = f"{g}_{e}"
                for metric in metrics:
                    y0 = f"log_{metric}_{combo}"
                    if y0 not in d0.columns:
                        continue
                    for season in seasons:
                        if season == "gpp_threshold":
                            d1 = d0[d0[f"gs_gpp_threshold_{g}"] == True].copy()
                            ycol = y0
                        elif season == "climate_common":
                            d1 = d0[d0["gs_climate_common"] == True].copy()
                            ycol = y0
                        else:
                            d1, ycol = residualize_month(d0.replace([np.inf, -np.inf], np.nan).dropna(subset=[y0]), y0)
                        for stress in stresses:
                            if stress == "zscore":
                                xcol = "stress_zscore"
                            elif stress == "percentile_joint":
                                xcol = "stress_percentile_joint"
                            elif stress == "copula_joint":
                                xcol = "stress_copula_joint"
                            else:
                                xcol = "stress_zscore"

                            sub = d1[["point_id","date","year","month","vpd_z","sm_z","vpd_x_sm",xcol,ycol]].replace([np.inf,-np.inf], np.nan).dropna()
                            if len(sub) < args.min_obs or sub[xcol].nunique() < 18:
                                continue
                            x = sub[xcol].to_numpy(float)
                            y = sub[ycol].to_numpy(float)
                            lin = fit_linear(x,y)
                            seg = fit_segmented(x,y)
                            if seg is None:
                                continue
                            pval = supf_p(x,y,args.n_perm,abs(hash((pid,version,combo,metric,season,stress)))%(2**32))
                            lo, hi = profile_ci(x,y)
                            boot = year_block_boot(sub,ycol,xcol,args.n_boot,abs(hash(("b",pid,version,combo,metric,season,stress)))%(2**32))
                            bo = overlap(boot.get("breakpoint_block_ci_low",np.nan), boot.get("breakpoint_block_ci_high",np.nan), lo, hi)
                            dbic = lin["linear_bic"] - seg["segmented_bic"]
                            row = {
                                "version": version, "point_id": pid, "gpp_product": g.upper(), "et_product": e.upper(),
                                "combo": combo, "metric": metric, "primary_metric": metric == "uwue",
                                "stress_definition": stress, "growing_season": season, "n": len(sub),
                                **lin, **seg, "delta_bic_linear_minus_segmented": dbic,
                                "supf_permutation_p": pval, "bayes_tau_ci_low": lo, "bayes_tau_ci_high": hi,
                                "bayes_overlap": bo, "accepted_transition": bool(dbic >= 6 and pval <= 0.05 and bo), **boot
                            }
                            row["response_class_strict"] = classify(row)
                            rows.append(row)
                            sf = surface(sub, ycol)
                            if sf:
                                surfs.append({k: row[k] for k in ["version","point_id","gpp_product","et_product","combo","metric","primary_metric","stress_definition","growing_season"]} | sf)

    res = pd.DataFrame(rows)
    surf = pd.DataFrame(surfs)
    res.to_csv(outdir / f"fullspec_response_results_{version}.csv", index=False)
    surf.to_csv(outdir / f"fullspec_vpd_sm_surface_{version}.csv", index=False)
    if len(res):
        sm = res.groupby(["metric","stress_definition","growing_season","gpp_product","et_product","response_class_strict"]).size().reset_index(name="n")
    else:
        sm = pd.DataFrame()
    sm.to_csv(outdir / f"fullspec_summary_{version}.csv", index=False)
    print("WROTE", outdir / f"fullspec_response_results_{version}.csv", res.shape)
    print("WROTE", outdir / f"fullspec_vpd_sm_surface_{version}.csv", surf.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=9)
    ap.add_argument("--min-obs", type=int, default=50)
    ap.add_argument("--max-points", type=int, default=3)
    ap.add_argument("--metrics", default="uwue,iwue,raw_wue")
    ap.add_argument("--stress-defs", default="zscore,percentile_joint,copula_joint,interaction_surface")
    ap.add_argument("--growing-seasons", default="gpp_threshold,climate_common,month_fixed")
    args = ap.parse_args()
    analyze("data/processed/final_nature/project_metric_matrix_raw_FINAL3x3.csv", "raw", args)
    analyze("data/processed/final_nature/project_metric_matrix_co2corrected_FINAL3x3.csv", "co2corrected", args)

if __name__ == "__main__":
    main()
