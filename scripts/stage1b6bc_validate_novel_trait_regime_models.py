from pathlib import Path
import re, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bc_validate_novel_trait_regime_models"
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"
for p in [TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]

INPUT = next((p for p in CANDIDATES if p.exists()), None)
if INPUT is None:
    raise SystemExit("No suitable input table found.")

raw = pd.read_csv(INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()

# ------------------------------------------------------------
# Canonical variable selection
# ------------------------------------------------------------

def first_existing(*names):
    for n in names:
        if n in raw.columns:
            return n
    return None

sources = {
    "y": first_existing("latent_slope_change"),
    "post": first_existing("latent_post_slope"),
    "sat": first_existing("latent_satbreak_probability", "p_satbreak", "p_threshold_like"),
    "vpd": first_existing("mean_vpd", "mean_obs_vpd"),
    "lai": first_existing("growing_season_mean_lai", "mean_lai"),
    "mat": first_existing("mean_annual_temperature", "mean_temperature"),
    "map": first_existing("mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing("aridity"),
    "sm": first_existing("mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing("lat"),
    "lon": first_existing("lon"),
    "root": first_existing("rooting_depth"),
    "sand": first_existing("soil_sand"),
    "clay": first_existing("soil_clay"),
    "silt": first_existing("soil_silt"),
    "p50": first_existing("p50", "psi50"),
    "unc_slope": first_existing("product_uncertainty_slope_change_range"),
    "unc_sat": first_existing("product_uncertainty_satbreak_range"),
}

d = pd.DataFrame(index=raw.index)
for canon, src in sources.items():
    if src is not None:
        d[canon] = pd.to_numeric(raw[src], errors="coerce")

if "lat" in d.columns:
    d["abs_lat"] = d["lat"].abs()

required = ["y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"]
missing_req = [c for c in required if c not in d.columns]
if missing_req:
    raise SystemExit(f"Missing required canonical variables: {missing_req}")

# spatial region blocks
d["region"] = (
    pd.cut(d["lat"], [-90, -30, 0, 30, 60, 90], labels=False).astype(str)
    + "_"
    + pd.cut(d["lon"], [-180, -90, 0, 90, 180], labels=False).astype(str)
)

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

for c in list(d.columns):
    if c != "region":
        d[c + "_z"] = zscore(d[c])

# hinge variables using thresholds discovered in the previous scan
if "mat" in d.columns:
    d["mat_hinge_2c"] = np.maximum(d["mat"] - 2.08, 0)
    d["mat_hinge_2c_z"] = zscore(d["mat_hinge_2c"])

if "abs_lat" in d.columns:
    d["abs_lat_hinge_31"] = np.maximum(d["abs_lat"] - 30.8, 0)
    d["abs_lat_hinge_31_z"] = zscore(d["abs_lat_hinge_31"])
    d["abs_lat_hinge_48"] = np.maximum(d["abs_lat"] - 47.9, 0)
    d["abs_lat_hinge_48_z"] = zscore(d["abs_lat_hinge_48"])

# ------------------------------------------------------------
# Modeling helpers
# ------------------------------------------------------------

def formula_vars(formula):
    toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
    ignore = {"C", "I", "Q"}
    return sorted(set(t for t in toks if t not in ignore))

def can_fit(formula):
    return all(v in d.columns for v in formula_vars(formula))

def fit_model(formula, data=None, cov_type="HC3"):
    if data is None:
        data = d
    vars_needed = formula_vars(formula)
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 50:
        return None, use
    fit = smf.ols(formula, data=use).fit(cov_type=cov_type)
    return fit, use

def bh_q(pvals):
    p = np.asarray([np.nan if pd.isna(x) else x for x in pvals], dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.minimum(vals, 1.0)
    return q

def bootstrap_terms(formula, terms, n_boot=1000, seed=123):
    fit, use = fit_model(formula)
    if fit is None:
        return {}

    rng = np.random.default_rng(seed)
    out = {}
    for term in terms:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(use), len(use))
            bs = use.iloc[idx].copy()
            try:
                f = smf.ols(formula, data=bs).fit()
                vals.append(f.params.get(term, np.nan))
            except Exception:
                pass
        vals = np.asarray([v for v in vals if np.isfinite(v)])
        if len(vals) >= 100:
            out[term] = {
                "boot_n": len(vals),
                "boot_median": float(np.median(vals)),
                "boot_ci_low": float(np.quantile(vals, 0.025)),
                "boot_ci_high": float(np.quantile(vals, 0.975)),
                "boot_ci_excludes_zero": bool(np.quantile(vals, 0.025) * np.quantile(vals, 0.975) > 0),
            }
        else:
            out[term] = {
                "boot_n": len(vals),
                "boot_median": np.nan,
                "boot_ci_low": np.nan,
                "boot_ci_high": np.nan,
                "boot_ci_excludes_zero": False,
            }
    return out

def leave_region_out_terms(formula, terms):
    vars_needed = formula_vars(formula) + ["region"]
    use = d[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    out = {}
    if len(use) < 60 or "region" not in use.columns:
        return {t: {"lro_n": 0, "lro_median": np.nan, "lro_sign_stability": np.nan} for t in terms}

    regions = sorted(use["region"].dropna().unique())
    for term in terms:
        vals = []
        for r in regions:
            train = use[use["region"] != r].copy()
            if len(train) < 50:
                continue
            try:
                f = smf.ols(formula, data=train).fit()
                vals.append(f.params.get(term, np.nan))
            except Exception:
                pass
        vals = np.asarray([v for v in vals if np.isfinite(v)])
        if len(vals) == 0:
            out[term] = {"lro_n": 0, "lro_median": np.nan, "lro_sign_stability": np.nan}
        else:
            med = np.median(vals)
            out[term] = {
                "lro_n": len(vals),
                "lro_median": float(med),
                "lro_sign_stability": float(np.mean(np.sign(vals) == np.sign(med))) if med != 0 else np.nan,
            }
    return out

def block_cv_r2(formula):
    vars_needed = formula_vars(formula) + ["region"]
    use = d[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 60:
        return np.nan, 0

    yvar = formula.split("~")[0].strip()
    regions = sorted(use["region"].dropna().unique())

    y_true_all = []
    y_pred_all = []

    for r in regions:
        train = use[use["region"] != r].copy()
        test = use[use["region"] == r].copy()
        if len(train) < 50 or len(test) < 3:
            continue
        try:
            f = smf.ols(formula, data=train).fit()
            pred = f.predict(test)
            y_true_all.extend(test[yvar].values)
            y_pred_all.extend(pred.values)
        except Exception:
            pass

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    if len(y_true_all) < 20:
        return np.nan, len(y_true_all)

    sse = np.sum((y_true_all - y_pred_all) ** 2)
    sst = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
    if sst == 0:
        return np.nan, len(y_true_all)
    return float(1 - sse / sst), len(y_true_all)

def compare_full_reduced(full_formula, reduced_formula):
    full, use_full = fit_model(full_formula)
    red, use_red = fit_model(reduced_formula)

    if full is None or red is None:
        return None

    # Fit reduced on same full complete-case set when possible.
    try:
        red_same = smf.ols(reduced_formula, data=use_full).fit()
        full_nonrobust = smf.ols(full_formula, data=use_full).fit()
        ftest = full_nonrobust.compare_f_test(red_same)
        nested_f_p = float(ftest[1])
    except Exception:
        nested_f_p = np.nan

    cv_full, cv_n = block_cv_r2(full_formula)
    cv_red, _ = block_cv_r2(reduced_formula)

    return {
        "n": int(full.nobs),
        "full_r2": full.rsquared,
        "reduced_r2": red.rsquared,
        "delta_r2": full.rsquared - red.rsquared,
        "full_adj_r2": full.rsquared_adj,
        "reduced_adj_r2": red.rsquared_adj,
        "full_aic": full.aic,
        "reduced_aic": red.aic,
        "delta_aic_full_minus_reduced": full.aic - red.aic,
        "full_bic": full.bic,
        "reduced_bic": red.bic,
        "nested_f_p": nested_f_p,
        "block_cv_r2_full": cv_full,
        "block_cv_r2_reduced": cv_red,
        "block_cv_delta_r2": cv_full - cv_red if np.isfinite(cv_full) and np.isfinite(cv_red) else np.nan,
        "block_cv_n_pred": cv_n,
        "fit": full,
        "use": use_full,
    }

# ------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------

base_controls_no_lai = "vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z"
base_controls_no_vpd = "arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z"
base_controls_full = "vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z"

candidates = [
    {
        "id": "KNOWN_BASELINE_vpd_not_novel",
        "claim_type": "known_baseline",
        "interpretation": "VPD is expected to matter; this is a benchmark, not the novel claim.",
        "full": "y_z ~ " + base_controls_full,
        "reduced": "y_z ~ arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["vpd_z"],
    },
    {
        "id": "NOVEL_canopy_structure_LAI_beyond_VPD",
        "claim_type": "primary_trait_structure",
        "interpretation": "Canopy structure / growing-season LAI predicts latent slope-change beyond VPD, hydroclimate, and geography.",
        "full": "y_z ~ " + base_controls_full,
        "reduced": "y_z ~ vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z",
        "focal_terms": ["lai_z"],
    },
    {
        "id": "NOVEL_abs_latitude_modifies_VPD_response",
        "claim_type": "primary_regime_interaction",
        "interpretation": "The VPD effect is regime-dependent across latitude; this is more novel than a VPD main effect.",
        "full": "y_z ~ vpd_z * abs_lat_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ vpd_z + abs_lat_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["vpd_z:abs_lat_z"],
    },
    {
        "id": "NOVEL_temperature_modifies_VPD_response",
        "claim_type": "primary_regime_interaction",
        "interpretation": "The VPD effect depends on mean annual temperature regime.",
        "full": "y_z ~ vpd_z * mat_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ vpd_z + mat_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["vpd_z:mat_z"],
    },
    {
        "id": "NOVEL_LAI_temperature_interaction",
        "claim_type": "primary_trait_regime_interaction",
        "interpretation": "Canopy structure matters differently across temperature regimes.",
        "full": "y_z ~ lai_z * mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ lai_z + mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z",
        "focal_terms": ["lai_z:mat_z"],
    },
    {
        "id": "NOVEL_temperature_threshold_around_2C",
        "claim_type": "primary_threshold",
        "interpretation": "Latent slope-change has a nonlinear climate-regime transition around mean annual temperature ≈ 2°C.",
        "full": "y_z ~ mat_z + mat_hinge_2c_z + vpd_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ mat_z + vpd_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["mat_hinge_2c_z"],
    },
    {
        "id": "NOVEL_abs_latitude_threshold_around_31deg",
        "claim_type": "primary_threshold",
        "interpretation": "Latent slope-change has a nonlinear latitude-regime transition around |latitude| ≈ 31°.",
        "full": "y_z ~ abs_lat_z + abs_lat_hinge_31_z + vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ abs_lat_z + vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["abs_lat_hinge_31_z"],
    },
    {
        "id": "NOVEL_abs_latitude_threshold_around_48deg",
        "claim_type": "primary_threshold",
        "interpretation": "Latent slope-change has a nonlinear high-latitude transition around |latitude| ≈ 48°.",
        "full": "y_z ~ abs_lat_z + abs_lat_hinge_48_z + vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "y_z ~ abs_lat_z + vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["abs_lat_hinge_48_z"],
    },
]

if "sat_z" in d.columns and "clay_z" in d.columns:
    candidates.append({
        "id": "SECONDARY_soil_clay_moisture_satbreak_modifier",
        "claim_type": "secondary_soil_modifier",
        "interpretation": "Soil texture may modulate threshold-like response through soil moisture.",
        "full": "sat_z ~ clay_z * sm_z + vpd_z + arid_z + mat_z + map_z + lai_z + lat_z + lon_z",
        "reduced": "sat_z ~ clay_z + sm_z + vpd_z + arid_z + mat_z + map_z + lai_z + lat_z + lon_z",
        "focal_terms": ["clay_z:sm_z"],
    })

if "post_z" in d.columns and "sand_z" in d.columns:
    candidates.append({
        "id": "SECONDARY_soil_sand_temperature_post_slope_modifier",
        "claim_type": "secondary_soil_modifier",
        "interpretation": "Soil sand fraction may modulate post-stress slope across temperature regimes.",
        "full": "post_z ~ sand_z * mat_z + vpd_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "post_z ~ sand_z + mat_z + vpd_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["sand_z:mat_z"],
    })

if "unc_slope_z" in d.columns and "root_z" in d.columns:
    candidates.append({
        "id": "SECONDARY_rooting_depth_VPD_product_uncertainty_modifier",
        "claim_type": "secondary_uncertainty_not_primary_flux",
        "interpretation": "Rooting depth modifies product uncertainty under VPD; useful as uncertainty analysis, not main flux biology.",
        "full": "unc_slope_z ~ root_z * vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "reduced": "unc_slope_z ~ root_z + vpd_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
        "focal_terms": ["root_z:vpd_z"],
    })

# remove candidates that cannot be fit
candidates = [c for c in candidates if can_fit(c["full"]) and can_fit(c["reduced"])]

# ------------------------------------------------------------
# Run validation
# ------------------------------------------------------------

summary_rows = []
coef_rows = []

for cand in candidates:
    cmp = compare_full_reduced(cand["full"], cand["reduced"])
    if cmp is None:
        continue

    fit = cmp["fit"]
    ci = fit.conf_int()
    boot = bootstrap_terms(cand["full"], cand["focal_terms"], n_boot=1000)
    lro = leave_region_out_terms(cand["full"], cand["focal_terms"])

    for term in cand["focal_terms"]:
        coef = fit.params.get(term, np.nan)
        p = fit.pvalues.get(term, np.nan)

        b = boot.get(term, {})
        lo = lro.get(term, {})

        summary_rows.append({
            "candidate_id": cand["id"],
            "claim_type": cand["claim_type"],
            "interpretation": cand["interpretation"],
            "focal_term": term,
            "n": cmp["n"],
            "coef_hc3": coef,
            "se_hc3": fit.bse.get(term, np.nan),
            "p_hc3": p,
            "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
            "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
            "ci_excludes_zero": bool((ci.loc[term, 0] * ci.loc[term, 1] > 0)) if term in ci.index else False,
            "full_r2": cmp["full_r2"],
            "reduced_r2": cmp["reduced_r2"],
            "delta_r2": cmp["delta_r2"],
            "delta_aic_full_minus_reduced": cmp["delta_aic_full_minus_reduced"],
            "nested_f_p": cmp["nested_f_p"],
            "block_cv_r2_full": cmp["block_cv_r2_full"],
            "block_cv_r2_reduced": cmp["block_cv_r2_reduced"],
            "block_cv_delta_r2": cmp["block_cv_delta_r2"],
            "block_cv_n_pred": cmp["block_cv_n_pred"],
            "boot_median": b.get("boot_median", np.nan),
            "boot_ci_low": b.get("boot_ci_low", np.nan),
            "boot_ci_high": b.get("boot_ci_high", np.nan),
            "boot_n": b.get("boot_n", 0),
            "boot_ci_excludes_zero": b.get("boot_ci_excludes_zero", False),
            "lro_median": lo.get("lro_median", np.nan),
            "lro_sign_stability": lo.get("lro_sign_stability", np.nan),
            "lro_n": lo.get("lro_n", 0),
            "full_formula": cand["full"],
            "reduced_formula": cand["reduced"],
        })

    for term in fit.params.index:
        coef_rows.append({
            "candidate_id": cand["id"],
            "term": term,
            "coef_hc3": fit.params[term],
            "se_hc3": fit.bse[term],
            "p_hc3": fit.pvalues[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
        })

summary = pd.DataFrame(summary_rows)
if len(summary):
    summary["q_bh_across_candidates"] = bh_q(summary["p_hc3"])
    summary["robust_score"] = (
        summary["ci_excludes_zero"].astype(int)
        + summary["boot_ci_excludes_zero"].astype(int)
        + (summary["lro_sign_stability"].fillna(0) >= 0.85).astype(int)
        + (summary["q_bh_across_candidates"] < 0.05).astype(int)
        + (summary["delta_aic_full_minus_reduced"] < -2).astype(int)
    )
    summary = summary.sort_values(
        ["claim_type", "robust_score", "q_bh_across_candidates", "delta_aic_full_minus_reduced"],
        ascending=[True, False, True, True],
    )

summary.to_csv(TAB / "FINAL_VALIDATED_NOVEL_CANDIDATES.csv", index=False)
pd.DataFrame(coef_rows).to_csv(TAB / "FINAL_VALIDATED_MODEL_COEFFICIENTS.csv", index=False)

# ------------------------------------------------------------
# Partial residual plots and interaction plots
# ------------------------------------------------------------

def residualize(target, controls):
    cols = [target] + controls
    use = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 50:
        return None
    f = smf.ols(target + " ~ " + " + ".join(controls), data=use).fit()
    return use.index, f.resid

def save_partial_residual_plot(y, x, controls, title, filename):
    out_y = residualize(y, controls)
    out_x = residualize(x, controls)
    if out_y is None or out_x is None:
        return
    iy, ry = out_y
    ix, rx = out_x
    idx = iy.intersection(ix)
    if len(idx) < 50:
        return

    xx = rx.loc[idx]
    yy = ry.loc[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xx, yy, alpha=0.75)
    slope, intercept, r, p, se = stats.linregress(xx, yy)
    grid = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(grid, intercept + slope * grid)
    ax.set_title(title)
    ax.set_xlabel(f"{x} residualized")
    ax.set_ylabel(f"{y} residualized")
    ax.text(0.02, 0.98, f"partial r={r:.3f}, p={p:.3g}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)

save_partial_residual_plot(
    "y_z",
    "lai_z",
    ["vpd_z", "arid_z", "mat_z", "map_z", "sm_z", "lat_z", "lon_z"],
    "Canopy structure / LAI effect beyond VPD and controls",
    "partial_residual_LAI_beyond_VPD_controls.png",
)

save_partial_residual_plot(
    "y_z",
    "mat_hinge_2c_z",
    ["mat_z", "vpd_z", "arid_z", "map_z", "lai_z", "sm_z", "lat_z", "lon_z"],
    "Temperature threshold effect beyond VPD and controls",
    "partial_residual_temperature_threshold_2C.png",
)

save_partial_residual_plot(
    "y_z",
    "abs_lat_hinge_31_z",
    ["abs_lat_z", "vpd_z", "arid_z", "mat_z", "map_z", "lai_z", "sm_z", "lat_z", "lon_z"],
    "Latitude threshold effect beyond VPD and controls",
    "partial_residual_abs_lat_threshold_31deg.png",
)

def save_interaction_plot(formula, moderator, title, filename):
    fit, use = fit_model(formula)
    if fit is None:
        return

    # Plot predicted y over VPD at low/median/high moderator, holding controls at 0.
    grid_vpd = np.linspace(use["vpd_z"].quantile(0.05), use["vpd_z"].quantile(0.95), 100)
    mod_vals = [
        use[moderator].quantile(0.15),
        use[moderator].quantile(0.50),
        use[moderator].quantile(0.85),
    ]

    pred_rows = []
    all_vars = [v for v in formula_vars(formula) if v != "y_z"]
    for mv in mod_vals:
        for vv in grid_vpd:
            row = {v: 0.0 for v in all_vars}
            row["vpd_z"] = vv
            row[moderator] = mv
            pred_rows.append(row)
    pred = pd.DataFrame(pred_rows)
    yhat = fit.predict(pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    start = 0
    labels = ["low", "median", "high"]
    for label, mv in zip(labels, mod_vals):
        end = start + len(grid_vpd)
        ax.plot(grid_vpd, yhat.iloc[start:end], label=f"{label} {moderator}")
        start = end

    ax.set_title(title)
    ax.set_xlabel("Baseline VPD, standardized")
    ax.set_ylabel("Predicted latent slope-change, standardized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)

save_interaction_plot(
    "y_z ~ vpd_z * abs_lat_z + arid_z + mat_z + map_z + lai_z + sm_z + lat_z + lon_z",
    "abs_lat_z",
    "Latitude modifies VPD response",
    "interaction_abs_lat_x_vpd.png",
)

save_interaction_plot(
    "y_z ~ vpd_z * mat_z + arid_z + map_z + lai_z + sm_z + lat_z + lon_z",
    "mat_z",
    "Temperature modifies VPD response",
    "interaction_temperature_x_vpd.png",
)

# ------------------------------------------------------------
# Manuscript-style interpretation table
# ------------------------------------------------------------

interpret_rows = []

if len(summary):
    for _, r in summary.iterrows():
        if r["claim_type"] == "known_baseline":
            priority = "background_covariate_not_main_claim"
        elif r["robust_score"] >= 4 and r["q_bh_across_candidates"] < 0.05:
            priority = "strong_candidate_main_or_secondary_result"
        elif r["robust_score"] >= 3:
            priority = "moderate_candidate_needs_manual_review"
        else:
            priority = "weak_or_sensitivity_only"

        interpret_rows.append({
            "candidate_id": r["candidate_id"],
            "priority": priority,
            "claim_type": r["claim_type"],
            "focal_term": r["focal_term"],
            "direction": "positive" if r["coef_hc3"] > 0 else "negative",
            "coef_hc3": r["coef_hc3"],
            "p_hc3": r["p_hc3"],
            "q_bh": r["q_bh_across_candidates"],
            "delta_r2": r["delta_r2"],
            "delta_aic": r["delta_aic_full_minus_reduced"],
            "bootstrap_ci": f"[{r['boot_ci_low']:.3f}, {r['boot_ci_high']:.3f}]",
            "leave_region_sign_stability": r["lro_sign_stability"],
            "interpretation": r["interpretation"],
        })

pd.DataFrame(interpret_rows).to_csv(TAB / "MANUSCRIPT_CANDIDATE_INTERPRETATION_TABLE.csv", index=False)

# ------------------------------------------------------------
# Memo
# ------------------------------------------------------------

def show(path, n=50):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BC validation of novel trait/regime ecosystem-flux candidates")
memo.append("=" * 90)
memo.append("")
memo.append(f"Input: {INPUT}")
memo.append(f"Rows: {len(raw)}")
memo.append("")
memo.append("Core framing:")
memo.append("- VPD alone is treated as a known baseline driver, not a novel result.")
memo.append("- The novel targets are trait/structure and regime-dependence beyond VPD.")
memo.append("")
memo.append("Canonical source columns:")
for k, v in sources.items():
    memo.append(f"- {k}: {v}")
memo.append("")
memo.append("Final validated candidates:")
memo.append(show("FINAL_VALIDATED_NOVEL_CANDIDATES.csv", 80))
memo.append("")
memo.append("Manuscript interpretation table:")
memo.append(show("MANUSCRIPT_CANDIDATE_INTERPRETATION_TABLE.csv", 80))
memo.append("")
memo.append("Important figures:")
for f in sorted(FIG.glob("*.png")):
    memo.append(f"- {f}")
memo.append("")
memo.append("Important tables:")
for f in [
    "FINAL_VALIDATED_NOVEL_CANDIDATES.csv",
    "FINAL_VALIDATED_MODEL_COEFFICIENTS.csv",
    "MANUSCRIPT_CANDIDATE_INTERPRETATION_TABLE.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_validate_novel_trait_regime_models.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_validate_novel_trait_regime_models.txt'}")
