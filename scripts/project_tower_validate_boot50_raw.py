#!/usr/bin/env python
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

OUT = Path("results/project_tower_validation_boot50_raw")
OUT.mkdir(parents=True, exist_ok=True)

SAT_RESULTS = Path("results/project_final_nature_boot50/fullspec_response_results_raw.csv")
SAT_MATRIX = Path("data/processed/final_nature/project_metric_matrix_raw_FINAL3x3.csv")
TOWER_DIR = Path("data/raw/towers")

MAX_MATCH_KM = 50.0
MIN_YEARS = 3
MIN_8DAY_OBS = 50

GPP_PRODUCTS = ["GOSIF", "MODIS", "PML"]
ET_PRODUCTS = ["GLEAM", "MODIS", "PML"]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def parse_point_id(pid):
    vals = str(pid).replace(",", "_").split("_")
    if len(vals) < 2:
        return np.nan, np.nan
    try:
        lon = float(vals[0])
        lat = float(vals[1])
        return lat, lon
    except Exception:
        return np.nan, np.nan

def load_towers():
    files = sorted(TOWER_DIR.glob("*grassland_sites.csv"))
    if not files:
        raise SystemExit(f"No tower grassland CSVs found in {TOWER_DIR}")

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print("SKIP unreadable", f, e)
            continue
        d["__source_file"] = str(f)
        frames.append(d)

    if not frames:
        raise SystemExit("No readable tower files.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["tower_id", "date", "lat", "lon", "GPP_NT_VUT_REF", "VPD"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Tower data missing required column {c}. Columns: {list(df.columns)}")

    if "source_network" not in df.columns:
        df["source_network"] = "unknown"
    if "igbp" not in df.columns:
        df["igbp"] = "unknown"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["lat", "lon", "GPP_NT_VUT_REF", "LE_F_MDS", "ET", "VPD", "soil_moisture", "energy_balance_closure", "gapfill_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer ET if present. Otherwise approximate ET from latent heat if units are compatible.
    # In the local tower harmonization files ET appears already present, so use it.
    if "ET" not in df.columns:
        if "LE_F_MDS" in df.columns:
            df["ET"] = df["LE_F_MDS"] / 2.45
        else:
            raise SystemExit("No ET or LE_F_MDS available in tower data.")

    # Keep grassland/savanna/shrubland-ish sites. Exclude CRO for main validation.
    keep_igbp = {"GRA", "SAV", "WSA", "OSH", "CSH"}
    df = df[df["igbp"].astype(str).str.upper().isin(keep_igbp)].copy()

    # Row-level quality screens where available.
    if "gapfill_fraction" in df.columns:
        df = df[(df["gapfill_fraction"].isna()) | (df["gapfill_fraction"] <= 0.30)].copy()
    if "energy_balance_closure" in df.columns:
        df = df[
            (df["energy_balance_closure"].isna()) |
            ((df["energy_balance_closure"] >= 0.70) & (df["energy_balance_closure"] <= 1.30))
        ].copy()

    df = df.dropna(subset=["tower_id", "date", "lat", "lon", "GPP_NT_VUT_REF", "ET", "VPD"])
    df = df[(df["GPP_NT_VUT_REF"] > 0) & (df["ET"] > 0.1) & (df["VPD"] > 0)].copy()

    return df

def aggregate_8day(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    df["doy8"] = ((df["doy"] - 1) // 8) * 8 + 1
    df["period_start"] = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(df["doy8"] - 1, unit="D")

    agg_dict = {
        "GPP_NT_VUT_REF": "sum",
        "ET": "sum",
        "VPD": "mean",
        "lat": "first",
        "lon": "first",
        "source_network": "first",
        "igbp": "first",
        "date": "count",
    }
    if "soil_moisture" in df.columns:
        agg_dict["soil_moisture"] = "mean"

    g = (
        df.groupby(["tower_id", "period_start"], as_index=False)
        .agg(agg_dict)
        .rename(columns={"date": "n_days"})
    )

    g = g[g["n_days"] >= 4].copy()
    g["raw_wue"] = g["GPP_NT_VUT_REF"] / g["ET"]
    g["uwue"] = g["GPP_NT_VUT_REF"] * np.sqrt(g["VPD"]) / g["ET"]
    g["iwue"] = g["GPP_NT_VUT_REF"] * g["VPD"] / g["ET"]

    for m in ["raw_wue", "uwue", "iwue"]:
        g[f"log_{m}"] = np.log(g[m].where(g[m] > 0))

    return g

def add_stress(g):
    parts = []
    for tid, d in g.groupby("tower_id"):
        d = d.sort_values("period_start").copy()
        d["vpd_z"] = (d["VPD"] - d["VPD"].mean()) / d["VPD"].std(ddof=0)
        if "soil_moisture" in d.columns and d["soil_moisture"].notna().sum() >= 30:
            sm = d["soil_moisture"]
            d["sm_z"] = (sm - sm.mean()) / sm.std(ddof=0)
            d["stress_zscore"] = d["vpd_z"] - d["sm_z"]
            d["stress_percentile_joint"] = d["VPD"].rank(pct=True) + (1 - d["soil_moisture"].rank(pct=True))
            d["stress_copula_joint"] = d["stress_percentile_joint"]
        else:
            d["sm_z"] = np.nan
            # Fallback when tower soil moisture is missing: VPD-only atmospheric stress.
            d["stress_zscore"] = d["vpd_z"]
            d["stress_percentile_joint"] = d["VPD"].rank(pct=True)
            d["stress_copula_joint"] = d["stress_percentile_joint"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)

def fit_linear(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    sse = float(np.sum((y - pred) ** 2))
    return beta, pred, sse

def fit_segmented(x, y):
    qs = np.quantile(x, np.linspace(0.20, 0.80, 31))
    best = None
    for tau in np.unique(qs):
        h = np.maximum(0, x - tau)
        X = np.column_stack([np.ones(len(x)), x, h])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            continue
        pred = X @ beta
        sse = float(np.sum((y - pred) ** 2))
        pre = beta[1]
        post = beta[1] + beta[2]
        if best is None or sse < best["sse"]:
            best = {"tau": float(tau), "beta": beta, "pred": pred, "sse": sse, "pre_slope": float(pre), "post_slope": float(post)}
    return best

def bic(n, sse, k):
    sse = max(sse, 1e-12)
    return n * math.log(sse / n) + k * math.log(n)

def bootstrap_segmented(x, y, n_boot=100, seed=123):
    rng = np.random.default_rng(seed)
    rows = []
    years = None
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if len(np.unique(xb)) < 10:
            continue
        fit = fit_segmented(xb, yb)
        if fit is None:
            continue
        rows.append([fit["tau"], fit["pre_slope"], fit["post_slope"]])
    if not rows:
        return None
    arr = np.array(rows)
    return {
        "tau_lo": float(np.nanpercentile(arr[:,0], 2.5)),
        "tau_hi": float(np.nanpercentile(arr[:,0], 97.5)),
        "pre_lo": float(np.nanpercentile(arr[:,1], 2.5)),
        "pre_hi": float(np.nanpercentile(arr[:,1], 97.5)),
        "post_lo": float(np.nanpercentile(arr[:,2], 2.5)),
        "post_hi": float(np.nanpercentile(arr[:,2], 97.5)),
    }

def classify_response(pre, post, pre_lo, pre_hi, post_lo, post_hi, accepted):
    if not accepted:
        if pre > 0:
            return "enhancement_no_accepted_breakpoint"
        return "inconclusive"
    if pre > 0 and pre_lo > 0 and post_hi < 0:
        return "breakdown"
    if pre > 0 and pre_lo > 0 and post_lo <= 0 <= post_hi:
        return "saturation"
    if pre > 0 and post > 0:
        return "enhancement"
    return "inconclusive"

def classify_towers(g):
    rows = []
    stress_cols = ["stress_zscore", "stress_percentile_joint", "stress_copula_joint"]
    metrics = ["uwue", "iwue", "raw_wue"]

    for tid, d0 in g.groupby("tower_id"):
        years = d0["period_start"].dt.year.nunique()
        if years < MIN_YEARS:
            continue

        for metric in metrics:
            ycol = f"log_{metric}"
            for stress in stress_cols:
                d = d0[[stress, ycol, "period_start", "lat", "lon", "source_network", "igbp"]].dropna().copy()
                if len(d) < MIN_8DAY_OBS:
                    continue
                x = d[stress].to_numpy(float)
                y = d[ycol].to_numpy(float)
                if len(np.unique(x)) < 10:
                    continue

                lin_beta, lin_pred, lin_sse = fit_linear(x, y)
                seg = fit_segmented(x, y)
                if seg is None:
                    continue

                n = len(d)
                bic_lin = bic(n, lin_sse, 2)
                bic_seg = bic(n, seg["sse"], 3)
                delta_bic = bic_seg - bic_lin
                accepted = delta_bic < -2

                boot = bootstrap_segmented(x, y, n_boot=100)
                if boot is None:
                    continue

                cls = classify_response(
                    seg["pre_slope"], seg["post_slope"],
                    boot["pre_lo"], boot["pre_hi"],
                    boot["post_lo"], boot["post_hi"],
                    accepted
                )

                rows.append({
                    "tower_id": tid,
                    "source_network": d["source_network"].iloc[0],
                    "igbp": d["igbp"].iloc[0],
                    "lat": float(d["lat"].iloc[0]),
                    "lon": float(d["lon"].iloc[0]),
                    "years": int(years),
                    "n_8day": int(n),
                    "metric": metric,
                    "stress_definition": stress.replace("stress_", ""),
                    "tau": seg["tau"],
                    "pre_slope": seg["pre_slope"],
                    "post_slope": seg["post_slope"],
                    "tau_lo": boot["tau_lo"],
                    "tau_hi": boot["tau_hi"],
                    "pre_lo": boot["pre_lo"],
                    "pre_hi": boot["pre_hi"],
                    "post_lo": boot["post_lo"],
                    "post_hi": boot["post_hi"],
                    "bic_linear": bic_lin,
                    "bic_segmented": bic_seg,
                    "delta_bic_seg_minus_linear": delta_bic,
                    "accepted_transition": bool(accepted),
                    "response_class": cls,
                })

    return pd.DataFrame(rows)

def load_sat_points():
    if not SAT_MATRIX.exists():
        raise SystemExit(f"Missing satellite matrix {SAT_MATRIX}")
    m = pd.read_csv(SAT_MATRIX, usecols=["point_id"], low_memory=False)
    pts = pd.DataFrame({"point_id": sorted(m["point_id"].astype(str).unique())})
    coords = pts["point_id"].apply(parse_point_id)
    pts["lat"] = [x[0] for x in coords]
    pts["lon"] = [x[1] for x in coords]
    pts = pts.dropna(subset=["lat","lon"])
    return pts

def nearest_satellite_points(towers, sat_pts):
    tower_sites = towers[["tower_id","source_network","igbp","lat","lon"]].drop_duplicates("tower_id").copy()
    rows = []
    for _, t in tower_sites.iterrows():
        best = None
        for _, p in sat_pts.iterrows():
            d = haversine_km(t["lat"], t["lon"], p["lat"], p["lon"])
            if best is None or d < best[0]:
                best = (d, p["point_id"], p["lat"], p["lon"])
        rows.append({
            "tower_id": t["tower_id"],
            "source_network": t["source_network"],
            "igbp": t["igbp"],
            "tower_lat": t["lat"],
            "tower_lon": t["lon"],
            "nearest_point_id": best[1],
            "nearest_point_lat": best[2],
            "nearest_point_lon": best[3],
            "distance_km": best[0],
            "match_within_50km": best[0] <= MAX_MATCH_KM,
        })
    return pd.DataFrame(rows)

def compare_to_satellite(tower_fit, matches):
    if not SAT_RESULTS.exists():
        raise SystemExit(f"Missing satellite results {SAT_RESULTS}")
    sat = pd.read_csv(SAT_RESULTS, low_memory=False)
    sat["gpp_product"] = sat["gpp_product"].astype(str).str.upper()
    sat["et_product"] = sat["et_product"].astype(str).str.upper()

    matches_ok = matches[matches["match_within_50km"]].copy()
    if matches_ok.empty:
        return pd.DataFrame()

    t = tower_fit.copy()
    t = t[t["metric"].eq("uwue")].copy()
    t = t.merge(matches_ok[["tower_id", "nearest_point_id", "distance_km"]], on="tower_id", how="inner")
    if t.empty:
        return pd.DataFrame()

    # Satellite stress naming in fullspec may be zscore, percentile_joint, copula_joint.
    t["sat_stress_definition"] = t["stress_definition"].replace({
        "zscore": "zscore",
        "percentile_joint": "percentile_joint",
        "copula_joint": "copula_joint",
    })

    sat_u = sat[sat["metric"].eq("uwue")].copy()

    rows = []
    for _, tr in t.iterrows():
        s = sat_u[
            (sat_u["point_id"].astype(str) == str(tr["nearest_point_id"])) &
            (sat_u["stress_definition"].astype(str) == str(tr["sat_stress_definition"]))
        ].copy()

        for _, sr in s.iterrows():
            rows.append({
                "tower_id": tr["tower_id"],
                "tower_class": tr["response_class"],
                "tower_accepted_transition": tr["accepted_transition"],
                "tower_stress_definition": tr["stress_definition"],
                "nearest_point_id": tr["nearest_point_id"],
                "distance_km": tr["distance_km"],
                "gpp_product": sr["gpp_product"],
                "et_product": sr["et_product"],
                "satellite_class": sr["response_class_strict"],
                "satellite_accepted_transition": sr["accepted_transition"],
                "class_match": tr["response_class"] == sr["response_class_strict"],
                "both_nonmonotonic": (tr["response_class"] in ["saturation","breakdown"]) and (sr["response_class_strict"] in ["saturation","breakdown"]),
            })

    return pd.DataFrame(rows)

def main():
    print("Loading tower data...")
    daily = load_towers()
    daily.to_csv(OUT / "tower_daily_screened.csv", index=False)
    print("daily screened:", daily.shape)
    print("towers:", daily["tower_id"].nunique())

    print("Aggregating to 8-day tower data...")
    eight = aggregate_8day(daily)
    eight = add_stress(eight)
    eight.to_csv(OUT / "tower_8day_metrics.csv", index=False)
    print("8-day:", eight.shape)
    print("8-day towers:", eight["tower_id"].nunique())

    site_summary = (
        eight.groupby(["tower_id","source_network","igbp"])
        .agg(
            lat=("lat","first"),
            lon=("lon","first"),
            years=("period_start", lambda s: s.dt.year.nunique()),
            n_8day=("period_start","size"),
            vpd_mean=("VPD","mean"),
            et_sum=("ET","sum"),
            gpp_sum=("GPP_NT_VUT_REF","sum"),
        )
        .reset_index()
        .sort_values(["years","n_8day"], ascending=False)
    )
    site_summary.to_csv(OUT / "tower_site_screening_summary.csv", index=False)
    print("site summary:")
    print(site_summary.to_string(index=False))

    print("Classifying tower response shapes...")
    tower_fit = classify_towers(eight)
    tower_fit.to_csv(OUT / "tower_response_shape_fits.csv", index=False)
    print("tower fits:", tower_fit.shape)
    if not tower_fit.empty:
        print(tower_fit["response_class"].value_counts(dropna=False).to_string())

    print("Matching towers to nearest satellite sample points...")
    sat_pts = load_sat_points()
    matches = nearest_satellite_points(site_summary, sat_pts)
    matches.to_csv(OUT / "tower_to_nearest_satellite_point.csv", index=False)
    print(matches.sort_values("distance_km").to_string(index=False))

    print("Comparing tower classes to satellite product-combo classes where spatial match is close enough...")
    comp = compare_to_satellite(tower_fit, matches)
    comp.to_csv(OUT / "tower_satellite_class_comparison.csv", index=False)
    print("comparison:", comp.shape)
    if not comp.empty:
        prod = (
            comp.groupby(["gpp_product","et_product"])
            .agg(
                n=("class_match","size"),
                tower_sites=("tower_id","nunique"),
                median_distance_km=("distance_km","median"),
                class_match_frac=("class_match","mean"),
                both_nonmonotonic_frac=("both_nonmonotonic","mean"),
            )
            .reset_index()
            .sort_values("class_match_frac", ascending=False)
        )
        prod.to_csv(OUT / "tower_product_family_scores.csv", index=False)
        print(prod.to_string(index=False))
    else:
        prod = pd.DataFrame()

    verdict = {
        "daily_screened_rows": int(len(daily)),
        "tower_sites_daily": int(daily["tower_id"].nunique()),
        "eight_day_rows": int(len(eight)),
        "tower_sites_8day": int(eight["tower_id"].nunique()),
        "tower_fit_rows": int(len(tower_fit)),
        "tower_fit_sites": int(tower_fit["tower_id"].nunique()) if not tower_fit.empty else 0,
        "satellite_points": int(len(sat_pts)),
        "towers_with_satellite_match_within_50km": int(matches["match_within_50km"].sum()),
        "tower_satellite_comparison_rows": int(len(comp)),
        "can_arbitrate_products_now": bool(len(comp) > 0 and matches["match_within_50km"].sum() >= 3),
    }

    with open(OUT / "tower_validation_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2)

    md = []
    md.append("# Tower validation verdict")
    md.append("")
    md.append(f"- Daily screened tower rows: `{verdict['daily_screened_rows']}`")
    md.append(f"- Tower sites after daily screening: `{verdict['tower_sites_daily']}`")
    md.append(f"- 8-day tower rows: `{verdict['eight_day_rows']}`")
    md.append(f"- Tower sites after 8-day aggregation: `{verdict['tower_sites_8day']}`")
    md.append(f"- Tower response-shape fit rows: `{verdict['tower_fit_rows']}`")
    md.append(f"- Tower sites with response-shape fits: `{verdict['tower_fit_sites']}`")
    md.append(f"- Satellite sample points: `{verdict['satellite_points']}`")
    md.append(f"- Towers with satellite sample point within 50 km: `{verdict['towers_with_satellite_match_within_50km']}`")
    md.append(f"- Tower-satellite comparison rows: `{verdict['tower_satellite_comparison_rows']}`")
    md.append("")
    if verdict["can_arbitrate_products_now"]:
        md.append("## Conclusion")
        md.append("The existing sampled satellite points are close enough to at least some towers to perform a provisional tower arbitration.")
    else:
        md.append("## Conclusion")
        md.append("The existing 199-point satellite sample is not sufficient for rigorous tower arbitration. The next required step is tower-centered satellite extraction, not trait attribution.")
    (OUT / "README_tower_validation_verdict.md").write_text("\n".join(md))

    print()
    print("==============================")
    print("TOWER VALIDATION VERDICT")
    print("==============================")
    print(json.dumps(verdict, indent=2))
    print("WROTE", OUT / "README_tower_validation_verdict.md")

if __name__ == "__main__":
    main()
