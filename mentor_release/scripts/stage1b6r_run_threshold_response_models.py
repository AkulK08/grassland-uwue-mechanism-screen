from pathlib import Path
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6r_threshold_response_models")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6r")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

DESIGN = Path("data/processed/stage1b6q2/analysis_design_strict_2x2_with_tower_stress_and_gs.csv")

BOOT_N = int(os.environ.get("BOOT_N", "50"))
MIN_N = int(os.environ.get("MIN_N", "60"))
SEED = int(os.environ.get("SEED", "20260629"))

rng = np.random.default_rng(SEED)

METRICS = [
    ("log_wue", "log_wue"),
    ("log_uwue", "log_uwue"),
]

STRESS_DEFS = [
    ("equal_weight_z", "stress_equal_weight_z"),
    ("joint_percentile", "stress_joint_percentile"),
    ("copula_product", "stress_copula_product"),
    ("vpd_x_dryness_z", "stress_vpd_x_dryness_z"),
]

GROWING_SEASONS = [
    ("gpp20_peak", "gs_gpp20_peak"),
    ("fixed_climate", "gs_fixed_climate"),
    ("month_fe_all", None),
]

def bic(n, rss, k):
    if n <= 0 or not np.isfinite(rss) or rss <= 0:
        return np.nan
    return n * np.log(rss / n) + k * np.log(n)

def aic(n, rss, k):
    if n <= 0 or not np.isfinite(rss) or rss <= 0:
        return np.nan
    return n * np.log(rss / n) + 2 * k

def ols_fit(X, y):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rss = float(np.sum(resid ** 2))
    return coef, rss

def fit_segmented_core(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = len(x)

    if n < MIN_N or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {
            "n_fit": n,
            "tau": np.nan,
            "pre_slope": np.nan,
            "post_slope": np.nan,
            "slope_change": np.nan,
            "intercept": np.nan,
            "rss_segmented": np.nan,
            "rss_linear": np.nan,
            "bic_segmented": np.nan,
            "bic_linear": np.nan,
            "delta_bic_seg_minus_linear": np.nan,
            "fit_status": "INSUFFICIENT_OR_CONSTANT",
        }

    Xlin = np.column_stack([np.ones(n), x])
    lin_coef, rss_lin = ols_fit(Xlin, y)

    qs = np.linspace(0.15, 0.85, 35)
    taus = np.unique(np.quantile(x, qs))
    best = None

    for tau in taus:
        h = np.maximum(0, x - tau)
        X = np.column_stack([np.ones(n), x, h])
        try:
            coef, rss = ols_fit(X, y)
        except Exception:
            continue

        if best is None or rss < best["rss_segmented"]:
            pre = float(coef[1])
            change = float(coef[2])
            best = {
                "n_fit": n,
                "tau": float(tau),
                "pre_slope": pre,
                "post_slope": pre + change,
                "slope_change": change,
                "intercept": float(coef[0]),
                "rss_segmented": float(rss),
                "rss_linear": float(rss_lin),
                "bic_segmented": bic(n, rss, 3),
                "bic_linear": bic(n, rss_lin, 2),
                "aic_segmented": aic(n, rss, 3),
                "aic_linear": aic(n, rss_lin, 2),
                "linear_slope": float(lin_coef[1]),
                "fit_status": "OK",
            }

    if best is None:
        return {
            "n_fit": n,
            "tau": np.nan,
            "pre_slope": np.nan,
            "post_slope": np.nan,
            "slope_change": np.nan,
            "intercept": np.nan,
            "rss_segmented": np.nan,
            "rss_linear": float(rss_lin),
            "bic_segmented": np.nan,
            "bic_linear": bic(n, rss_lin, 2),
            "delta_bic_seg_minus_linear": np.nan,
            "fit_status": "NO_SEGMENTED_FIT",
        }

    best["delta_bic_seg_minus_linear"] = best["bic_segmented"] - best["bic_linear"]
    return best

def boot_ci(x, y, boot_n):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = len(x)

    if n < MIN_N or boot_n <= 0:
        return {}

    vals = []
    for _ in range(boot_n):
        idx = rng.integers(0, n, size=n)
        fit = fit_segmented_core(x[idx], y[idx])
        if fit.get("fit_status") == "OK":
            vals.append({
                "tau": fit["tau"],
                "pre_slope": fit["pre_slope"],
                "post_slope": fit["post_slope"],
                "slope_change": fit["slope_change"],
            })

    if not vals:
        return {}

    b = pd.DataFrame(vals)
    out = {}
    for c in ["tau", "pre_slope", "post_slope", "slope_change"]:
        out[f"{c}_boot_median"] = float(b[c].median())
        out[f"{c}_ci_low"] = float(b[c].quantile(0.025))
        out[f"{c}_ci_high"] = float(b[c].quantile(0.975))
    out["n_boot_success"] = int(len(b))
    return out

def classify(row):
    if row.get("fit_status") != "OK" or row.get("n_fit", 0) < MIN_N:
        return "insufficient"

    pre = row.get("pre_slope_boot_median", row.get("pre_slope", np.nan))
    post = row.get("post_slope_boot_median", row.get("post_slope", np.nan))
    change = row.get("slope_change_boot_median", row.get("slope_change", np.nan))

    post_lo = row.get("post_slope_ci_low", np.nan)
    post_hi = row.get("post_slope_ci_high", np.nan)
    change_hi = row.get("slope_change_ci_high", np.nan)

    if np.isfinite(post_hi) and post_hi < 0:
        return "breakdown"

    if (
        np.isfinite(pre)
        and np.isfinite(post_lo)
        and np.isfinite(post_hi)
        and np.isfinite(change_hi)
        and pre > 0
        and post_lo <= 0 <= post_hi
        and change_hi < 0
    ):
        return "saturation"

    if np.isfinite(post_lo) and post_lo > 0:
        return "enhancement"

    if np.isfinite(change_hi) and change_hi < 0:
        return "weakening"

    return "inconclusive"

def apply_growing_season(sub, gs_name, gs_col):
    if gs_name == "month_fe_all":
        return sub.copy()

    if gs_col not in sub.columns:
        return sub.iloc[0:0].copy()

    return sub[sub[gs_col].astype(bool)].copy()

def metric_series(sub, metric_col, gs_name):
    y = pd.to_numeric(sub[metric_col], errors="coerce")

    if gs_name == "month_fe_all":
        month_mean = y.groupby(sub["month"]).transform("mean")
        overall = y.mean(skipna=True)
        return y - month_mean + overall

    return y

if not DESIGN.exists():
    raise FileNotFoundError(f"Missing design table: {DESIGN}")

df = pd.read_csv(DESIGN)
df = df[df["matrix_role"].eq("STRICT_PRIMARY")].copy()
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df["month"] = pd.to_datetime(df["date"]).dt.month
df["year"] = pd.to_datetime(df["date"]).dt.year
df["combo"] = df["gpp_product"].astype(str) + " x " + df["et_product"].astype(str)

fit_rows = []
error_rows = []

groups = list(df.groupby(["point_id", "gpp_product", "et_product"], dropna=False))
total_jobs = len(groups) * len(METRICS) * len(STRESS_DEFS) * len(GROWING_SEASONS)
job_i = 0

print(f"Running threshold models: {total_jobs} fits, BOOT_N={BOOT_N}, MIN_N={MIN_N}")


for (point_id, gpp_product, et_product), base in groups:
    for metric_name, metric_col in METRICS:
        for stress_name, stress_col in STRESS_DEFS:
            for gs_name, gs_col in GROWING_SEASONS:
                job_i += 1
                if job_i % 100 == 0:
                    print(f"[{job_i}/{total_jobs}] {point_id} {gpp_product} x {et_product} {metric_name} {stress_name} {gs_name}", flush=True)

                try:
                    sub = apply_growing_season(base, gs_name, gs_col)

                    if stress_col not in sub.columns or metric_col not in sub.columns:
                        fit_rows.append({
                            "point_id": point_id,
                            "gpp_product": gpp_product,
                            "et_product": et_product,
                            "combo": f"{gpp_product} x {et_product}",
                            "metric": metric_name,
                            "stress_definition": stress_name,
                            "stress_col": stress_col,
                            "growing_season": gs_name,
                            "fit_status": "MISSING_COLUMN",
                            "n_fit": 0,
                        })
                        continue

                    x = pd.to_numeric(sub[stress_col], errors="coerce")
                    y = metric_series(sub, metric_col, gs_name)

                    ok = np.isfinite(x) & np.isfinite(y)
                    x2 = x[ok].to_numpy(dtype=float)
                    y2 = y[ok].to_numpy(dtype=float)

                    fit = fit_segmented_core(x2, y2)
                    boot = boot_ci(x2, y2, BOOT_N)

                    row = {
                        "point_id": point_id,
                        "gpp_product": gpp_product,
                        "et_product": et_product,
                        "combo": f"{gpp_product} x {et_product}",
                        "metric": metric_name,
                        "stress_definition": stress_name,
                        "stress_col": stress_col,
                        "growing_season": gs_name,
                        "n_rows_before_dropna": int(len(sub)),
                        "n_rows_after_dropna": int(ok.sum()),
                    }
                    row.update(fit)
                    row.update(boot)
                    row["segmented_preferred_delta_bic_lt_minus2"] = bool(
                        np.isfinite(row.get("delta_bic_seg_minus_linear", np.nan))
                        and row.get("delta_bic_seg_minus_linear", np.nan) < -2
                    )
                    row["response_class"] = classify(row)
                    row["sat_or_breakdown"] = row["response_class"] in ["saturation", "breakdown", "weakening"]
                    fit_rows.append(row)

                except Exception as e:
                    error_rows.append({
                        "point_id": point_id,
                        "gpp_product": gpp_product,
                        "et_product": et_product,
                        "metric": metric_name,
                        "stress_definition": stress_name,
                        "growing_season": gs_name,
                        "error": repr(e),
                    })

fits = pd.DataFrame(fit_rows)
errors = pd.DataFrame(error_rows)

fits_out = DATA / "threshold_response_fits_strict_2x2.csv"
fits.to_csv(fits_out, index=False)
errors.to_csv(TAB / "Table_PRODUCT02cs_threshold_model_errors.csv", index=False)

summary = (
    fits.groupby(["metric", "stress_definition", "growing_season", "response_class"], dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values(["metric", "stress_definition", "growing_season", "response_class"])
)
summary.to_csv(TAB / "Table_PRODUCT02ct_response_class_summary.csv", index=False)

combo_summary = (
    fits.groupby(["gpp_product", "et_product", "metric", "stress_definition", "growing_season"], dropna=False)
    .agg(
        n_fits=("response_class", "size"),
        n_ok=("fit_status", lambda s: int((s == "OK").sum())),
        n_breakdown=("response_class", lambda s: int((s == "breakdown").sum())),
        n_saturation=("response_class", lambda s: int((s == "saturation").sum())),
        n_weakening=("response_class", lambda s: int((s == "weakening").sum())),
        n_enhancement=("response_class", lambda s: int((s == "enhancement").sum())),
        n_inconclusive=("response_class", lambda s: int((s == "inconclusive").sum())),
        median_tau=("tau", "median"),
        median_pre_slope=("pre_slope", "median"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
    )
    .reset_index()
)
combo_summary["sat_break_weak_frac"] = (
    combo_summary["n_breakdown"] + combo_summary["n_saturation"] + combo_summary["n_weakening"]
) / combo_summary["n_fits"].replace(0, np.nan)
combo_summary.to_csv(TAB / "Table_PRODUCT02cu_threshold_summary_by_combo_metric_stress_gs.csv", index=False)

point_robust = (
    fits.groupby(["point_id", "metric"], dropna=False)
    .agg(
        n_fits=("response_class", "size"),
        n_breakdown=("response_class", lambda s: int((s == "breakdown").sum())),
        n_saturation=("response_class", lambda s: int((s == "saturation").sum())),
        n_weakening=("response_class", lambda s: int((s == "weakening").sum())),
        n_enhancement=("response_class", lambda s: int((s == "enhancement").sum())),
        n_inconclusive=("response_class", lambda s: int((s == "inconclusive").sum())),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
    )
    .reset_index()
)
point_robust["sat_break_weak_frac"] = (
    point_robust["n_breakdown"] + point_robust["n_saturation"] + point_robust["n_weakening"]
) / point_robust["n_fits"].replace(0, np.nan)
point_robust.to_csv(TAB / "Table_PRODUCT02cv_point_level_response_robustness.csv", index=False)

expected_fit_count = 13 * 4 * len(METRICS) * len(STRESS_DEFS) * len(GROWING_SEASONS)
actual_fit_count = int(len(fits))
n_ok = int((fits["fit_status"] == "OK").sum()) if len(fits) else 0

decision_status = (
    actual_fit_count == expected_fit_count
    and n_ok > 0
    and len(errors) == 0
)

if decision_status:
    verdict = "THRESHOLD_RESPONSE_MODELS_COMPLETE_STRICT_2X2"
    blocking_next = False
else:
    verdict = "THRESHOLD_RESPONSE_MODELS_INCOMPLETE_OR_REVIEW_NEEDED"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "boot_n": BOOT_N,
    "min_n": MIN_N,
    "expected_fit_count": expected_fit_count,
    "actual_fit_count": actual_fit_count,
    "n_ok_fits": n_ok,
    "n_error_rows": int(len(errors)),
    "n_points": int(fits["point_id"].nunique()) if len(fits) else 0,
    "n_product_combos": int(fits[["gpp_product", "et_product"]].drop_duplicates().shape[0]) if len(fits) else 0,
    "n_metrics": int(fits["metric"].nunique()) if len(fits) else 0,
    "n_stress_definitions": int(fits["stress_definition"].nunique()) if len(fits) else 0,
    "n_growing_seasons": int(fits["growing_season"].nunique()) if len(fits) else 0,
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "TOWER_ARBITRATION_AND_TRAIT_MODEL_PREP" if not blocking_next else "REVIEW_THRESHOLD_MODEL_ERRORS",
}])
decision.to_csv(TAB / "Table_PRODUCT02cw_threshold_model_decision.csv", index=False)

report = []
report.append("# Stage 1B.6R threshold response models")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Response-class summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False) if len(summary) else "No summary.")
report.append("```")
report.append("")
report.append("## Combo/metric/stress/growing-season summary")
report.append("")
report.append("```text")
report.append(combo_summary.head(100).to_string(index=False) if len(combo_summary) else "No combo summary.")
report.append("```")
report.append("")
report.append("## Point-level robustness")
report.append("")
report.append("```text")
report.append(point_robust.to_string(index=False) if len(point_robust) else "No point summary.")
report.append("```")
report.append("")
report.append("## Errors")
report.append("")
report.append("```text")
report.append(errors.to_string(index=False) if len(errors) else "No errors.")
report.append("```")
report.append("")
report.append("## Output")
report.append("")
report.append(f"- Fit table: `{fits_out}`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("These are strict 2x2 tower-centered response fits. Do not describe them as a PML-inclusive strict 3x3 result. PML remains coarse sensitivity only.")
report.append("")

(TXT / "STAGE1B6R_THRESHOLD_RESPONSE_MODELS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6R_threshold_response_models",
    "status": verdict,
    "blocking_next_stage": bool(blocking_next),
    "outputs": {
        "fit_table": str(fits_out),
        "class_summary": str(TAB / "Table_PRODUCT02ct_response_class_summary.csv"),
        "combo_summary": str(TAB / "Table_PRODUCT02cu_threshold_summary_by_combo_metric_stress_gs.csv"),
        "point_robustness": str(TAB / "Table_PRODUCT02cv_point_level_response_robustness.csv"),
        "decision": str(TAB / "Table_PRODUCT02cw_threshold_model_decision.csv"),
        "report": str(TXT / "STAGE1B6R_THRESHOLD_RESPONSE_MODELS_REPORT.md"),
    }
}
(TAB / "STAGE1B6R_THRESHOLD_RESPONSE_MODELS_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", fits_out)
print("WROTE", TAB / "Table_PRODUCT02ct_response_class_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02cu_threshold_summary_by_combo_metric_stress_gs.csv")
print("WROTE", TAB / "Table_PRODUCT02cv_point_level_response_robustness.csv")
print("WROTE", TAB / "Table_PRODUCT02cw_threshold_model_decision.csv")
print("WROTE", TXT / "STAGE1B6R_THRESHOLD_RESPONSE_MODELS_REPORT.md")
