#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/paper_ready_now")
OUT.mkdir(parents=True, exist_ok=True)
PAPER = Path("paper")
PAPER.mkdir(exist_ok=True)

RAW = Path("results/reza_final_nature_boot50/fullspec_response_results_raw.csv")
TOWER_VERDICT = Path("results/reza_tower_validation_boot50_raw/README_tower_validation_verdict.md")

if not RAW.exists():
    raise SystemExit(f"Missing {RAW}")

df = pd.read_csv(RAW, low_memory=False)
df.columns = [str(c).strip() for c in df.columns]

# Normalize product labels for reliable grouping.
df["gpp_product_norm"] = df["gpp_product"].astype(str).str.upper()
df["et_product_norm"] = df["et_product"].astype(str).str.upper()

# Main primary metric.
u = df[df["metric"].astype(str).str.lower().eq("uwue")].copy()
u["sat_or_breakdown"] = u["response_class_strict"].isin(["saturation", "breakdown"])
u["breakdown_only"] = u["response_class_strict"].eq("breakdown")
u["saturation_only"] = u["response_class_strict"].eq("saturation")

# ----------------------------
# Gate 1: response-shape summary
# ----------------------------
gate1_counts = (
    u["response_class_strict"]
    .value_counts(dropna=False)
    .rename_axis("response_class")
    .reset_index(name="n")
)
gate1_counts["fraction"] = gate1_counts["n"] / gate1_counts["n"].sum()
gate1_counts.to_csv(OUT / "table_gate1_uwue_class_counts.csv", index=False)

# ----------------------------
# Gate 2: product matrix summary
# ----------------------------
prod = (
    u.groupby(["gpp_product_norm", "et_product_norm"])
    .agg(
        n=("point_id", "size"),
        points=("point_id", "nunique"),
        sat_or_breakdown_frac=("sat_or_breakdown", "mean"),
        breakdown_frac=("breakdown_only", "mean"),
        saturation_frac=("saturation_only", "mean"),
    )
    .reset_index()
    .sort_values("sat_or_breakdown_frac", ascending=False)
)
prod.to_csv(OUT / "table_gate2_uwue_product_signal.csv", index=False)

# Product × stress × season robustness.
robust = (
    u.groupby(["gpp_product_norm", "et_product_norm", "stress_definition", "growing_season"])
    .agg(
        n=("point_id", "size"),
        points=("point_id", "nunique"),
        sat_or_breakdown_frac=("sat_or_breakdown", "mean"),
        breakdown_frac=("breakdown_only", "mean"),
        saturation_frac=("saturation_only", "mean"),
    )
    .reset_index()
    .sort_values(["gpp_product_norm", "et_product_norm", "sat_or_breakdown_frac"], ascending=[True, True, False])
)
robust.to_csv(OUT / "table_gate2_product_stress_season_robustness.csv", index=False)

stress = (
    u.groupby(["stress_definition"])
    .agg(
        n=("point_id", "size"),
        points=("point_id", "nunique"),
        sat_or_breakdown_frac=("sat_or_breakdown", "mean"),
        breakdown_frac=("breakdown_only", "mean"),
        saturation_frac=("saturation_only", "mean"),
    )
    .reset_index()
    .sort_values("sat_or_breakdown_frac", ascending=False)
)
stress.to_csv(OUT / "table_gate2_by_stress_definition.csv", index=False)

season = (
    u.groupby(["growing_season"])
    .agg(
        n=("point_id", "size"),
        points=("point_id", "nunique"),
        sat_or_breakdown_frac=("sat_or_breakdown", "mean"),
        breakdown_frac=("breakdown_only", "mean"),
        saturation_frac=("saturation_only", "mean"),
    )
    .reset_index()
    .sort_values("sat_or_breakdown_frac", ascending=False)
)
season.to_csv(OUT / "table_gate2_by_growing_season.csv", index=False)

# ----------------------------
# Trait coverage / preliminary trait scaffold
# ----------------------------
pts = pd.DataFrame({"point_id": sorted(u["point_id"].astype(str).unique())})

def parse_pid(pid):
    s = str(pid).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        return float(s[0]), float(s[1])
    except Exception:
        return np.nan, np.nan

pts[["lon", "lat"]] = pts["point_id"].apply(lambda x: pd.Series(parse_pid(x)))
pts = pts.dropna(subset=["lon", "lat"]).copy()

trait_status = {}
try:
    import xarray as xr

    trait_paths = {
        "psi50": Path("data/external/liu_2021_psi50_0p1deg.nc"),
        "isohydricity": Path("data/external/konings_gentine_isohydricity_0p1deg.nc"),
        "rooting_depth": Path("data/external/stocker_2023_rooting_depth_0p1deg.nc"),
    }

    for name, path in trait_paths.items():
        if not path.exists():
            pts[name] = np.nan
            trait_status[name] = "missing"
            continue

        ds = xr.open_dataset(path)
        lat_name = next((c for c in ds.coords if c.lower() in ["lat", "latitude", "y"]), None)
        lon_name = next((c for c in ds.coords if c.lower() in ["lon", "longitude", "x"]), None)
        var_name = next(iter(ds.data_vars))

        vals = []
        for _, r in pts.iterrows():
            try:
                val = ds[var_name].sel({lat_name: r["lat"], lon_name: r["lon"]}, method="nearest").values
                vals.append(float(np.asarray(val).squeeze()))
            except Exception:
                vals.append(np.nan)

        pts[name] = vals
        trait_status[name] = {
            "file": str(path),
            "variable": var_name,
            "finite_points": int(np.isfinite(pts[name]).sum()),
            "total_points": int(len(pts)),
        }

except Exception as e:
    trait_status["trait_sampling_error"] = f"{type(e).__name__}: {e}"

# Merge aridity if available.
aridity_path = Path("data/external/aridity_by_point.csv")
if aridity_path.exists():
    ar = pd.read_csv(aridity_path, low_memory=False)
    ar.columns = [str(c).strip() for c in ar.columns]
    if "point_id" in ar.columns:
        ar["point_id"] = ar["point_id"].astype(str)
        pts = pts.merge(ar, on="point_id", how="left", suffixes=("", "_aridityfile"))

pts.to_csv(OUT / "table_trait_covariates_sampled_to_global_points.csv", index=False)

trait_cov = []
for c in ["psi50", "isohydricity", "rooting_depth"]:
    if c in pts.columns:
        trait_cov.append({
            "covariate": c,
            "finite_points": int(np.isfinite(pd.to_numeric(pts[c], errors="coerce")).sum()),
            "total_points": int(len(pts)),
            "coverage_frac": float(np.isfinite(pd.to_numeric(pts[c], errors="coerce")).mean()),
        })

# detect aridity columns
for c in pts.columns:
    if "aridity" in c.lower() or c.lower() in ["ai", "arid_index"]:
        vals = pd.to_numeric(pts[c], errors="coerce")
        trait_cov.append({
            "covariate": c,
            "finite_points": int(vals.notna().sum()),
            "total_points": int(len(pts)),
            "coverage_frac": float(vals.notna().mean()),
        })

trait_cov_df = pd.DataFrame(trait_cov)
trait_cov_df.to_csv(OUT / "table_trait_covariate_coverage.csv", index=False)

# ----------------------------
# Preliminary trait outcome scaffold
# ----------------------------
# Aggregate point-level response. This is NOT the final causal model; it is the model-ready scaffold.
point_response = (
    u.groupby(["point_id", "gpp_product_norm", "et_product_norm"])
    .agg(
        n_fits=("point_id", "size"),
        sat_or_breakdown_rate=("sat_or_breakdown", "mean"),
        breakdown_rate=("breakdown_only", "mean"),
        saturation_rate=("saturation_only", "mean"),
        inconclusive_rate=("response_class_strict", lambda s: (s == "inconclusive").mean()),
    )
    .reset_index()
)

# Try to create slope-change outcome if slope columns exist.
cols_lower = {c.lower(): c for c in u.columns}
pre_candidates = [c for c in u.columns if "pre" in c.lower() and "slope" in c.lower()]
post_candidates = [c for c in u.columns if "post" in c.lower() and "slope" in c.lower()]
pre_col = pre_candidates[0] if pre_candidates else None
post_col = post_candidates[0] if post_candidates else None

if pre_col and post_col:
    tmp = u[["point_id", "gpp_product_norm", "et_product_norm", pre_col, post_col]].copy()
    tmp[pre_col] = pd.to_numeric(tmp[pre_col], errors="coerce")
    tmp[post_col] = pd.to_numeric(tmp[post_col], errors="coerce")
    tmp["slope_change"] = tmp[post_col] - tmp[pre_col]
    slope_agg = (
        tmp.groupby(["point_id", "gpp_product_norm", "et_product_norm"])
        .agg(
            median_pre_slope=(pre_col, "median"),
            median_post_slope=(post_col, "median"),
            median_slope_change=("slope_change", "median"),
        )
        .reset_index()
    )
    point_response = point_response.merge(slope_agg, on=["point_id", "gpp_product_norm", "et_product_norm"], how="left")

trait_ready = point_response.merge(pts, on="point_id", how="left")
trait_ready.to_csv(OUT / "table_trait_model_ready_point_product_response.csv", index=False)

# Simple non-final model screen if possible.
model_rows = []
candidate_outcomes = ["sat_or_breakdown_rate", "breakdown_rate", "saturation_rate"]
if "median_slope_change" in trait_ready.columns:
    candidate_outcomes.append("median_slope_change")

predictors = [c for c in ["psi50", "isohydricity", "rooting_depth"] if c in trait_ready.columns]

# Add one aridity-like predictor if present.
for c in trait_ready.columns:
    if "aridity" in c.lower() or c.lower() in ["ai", "arid_index"]:
        predictors.append(c)
        break

def fit_ols(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    return beta, r2

for combo, d0 in trait_ready.groupby(["gpp_product_norm", "et_product_norm"]):
    for outcome in candidate_outcomes:
        use_cols = [outcome] + predictors
        d = d0[use_cols].copy()
        for c in use_cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 20 or len(predictors) == 0:
            continue
        try:
            beta, r2 = fit_ols(d[outcome], d[predictors])
            row = {
                "gpp_product": combo[0],
                "et_product": combo[1],
                "outcome": outcome,
                "n": len(d),
                "r2_screen": r2,
                "intercept": beta[0],
            }
            for p, b in zip(predictors, beta[1:]):
                row[f"beta_{p}"] = b
            model_rows.append(row)
        except Exception:
            pass

model_df = pd.DataFrame(model_rows)
model_df.to_csv(OUT / "table_trait_preliminary_model_screen_NOT_FINAL.csv", index=False)

# ----------------------------
# Draft paper outline / to-do file
# ----------------------------
summary = []
summary.append("# Paper-ready status and next steps")
summary.append("")
summary.append("## What is already paper-usable")
summary.append("")
summary.append("- The raw BOOT50 global matrix has all 9 GPP × ET product combinations.")
summary.append("- It includes all 3 WUE metrics: raw WUE, uWUE, and iWUE.")
summary.append("- It includes all 4 stress definitions and all 3 growing-season definitions.")
summary.append("- The primary uWUE response classification is complete for preliminary Gate 1/Gate 2 tables.")
summary.append("- Local tower-side GOSIF and GLEAM time series exist for 7 tower points.")
summary.append("- Trait/covariate files exist for P50, isohydricity, rooting depth, and aridity.")
summary.append("")
summary.append("## What is not yet unlocked")
summary.append("")
summary.append("- The mentor-level trait thesis is not unlocked until tower-centered MODIS/PML/ERA5 extraction completes.")
summary.append("- Current tower validation cannot be done with the 199 random/global points because no tower had a satellite point within 50 km.")
summary.append("- The pending task `tower_wue_timeseries_2001` is the correct bottleneck to watch.")
summary.append("")
summary.append("## Results statements allowed now")
summary.append("")
summary.append("- We can say the computational response-shape framework is implemented.")
summary.append("- We can say the full 3×3 raw product matrix exists for BOOT50.")
summary.append("- We can say the strict nonmonotonic response is a minority of uWUE fits.")
summary.append("- We can say product families differ and require tower arbitration before trait attribution.")
summary.append("")
summary.append("## Results statements not allowed yet")
summary.append("")
summary.append("- Do not say traits explain the response.")
summary.append("- Do not say any satellite product family is tower-validated.")
summary.append("- Do not claim a final biological breakdown threshold.")
summary.append("- Do not claim the mentor-level causal trait result until the tower-centered matrix and trait model are complete.")
summary.append("")
summary.append("## Next run order")
summary.append("")
summary.append("1. Let the pending one-year tower-centered GEE export finish.")
summary.append("2. Download `tower_wue_timeseries_2001.csv` from Drive.")
summary.append("3. Build `data/processed/tower_validation/tower_metric_matrix_raw.csv`.")
summary.append("4. Run tower-centered response classification.")
summary.append("5. If the one-year pilot works, queue remaining high-coverage tower years: 2003, 2006, 2002, 2005.")
summary.append("6. Once tower arbitration picks a credible product/metric family, run the final trait model.")
summary.append("")
summary.append("## Files generated by this pre-paper package")
summary.append("")
for p in sorted(OUT.glob("*")):
    summary.append(f"- `{p}`")

(PAPER / "paper_status_and_next_steps.md").write_text("\n".join(summary))

verdict = {
    "global_raw_3x3_complete": True,
    "tower_gosif_gleam_ready": True,
    "tower_gee_core_ready": len(list(Path("data/raw/towers/tower_gee").glob("tower_wue_timeseries_*.csv"))) > 0 if Path("data/raw/towers/tower_gee").exists() else False,
    "trait_files_present": bool(len(trait_cov_df) > 0),
    "mentor_trait_thesis_unlocked": False,
    "blocker": "tower-centered MODIS/PML/ERA5 export"
}
with open(OUT / "paper_ready_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)

print("\n===== PAPER READY NOW SUMMARY =====")
print((PAPER / "paper_status_and_next_steps.md").read_text())
