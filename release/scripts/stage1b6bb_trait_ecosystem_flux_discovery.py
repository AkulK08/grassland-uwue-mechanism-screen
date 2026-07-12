from pathlib import Path
import itertools, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bb_trait_ecosystem_flux_discovery"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]

INPUT = next((p for p in CANDIDATES if p.exists()), None)
if INPUT is None:
    raise SystemExit("No suitable input table found.")

df = pd.read_csv(INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)

# Remove duplicate column names, which can make df[c] return a DataFrame instead of a Series.
df = df.loc[:, ~df.columns.duplicated()].copy()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def num(s):
    # If duplicate column names make df[c] return a DataFrame, keep the first column.
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

def zcol(d, c):
    x = num(d[c])
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=d.index)
    return (x - x.mean()) / sd

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

def safe_fit(d, y, xs):
    cols = [y] + xs
    use = d[cols].copy()
    for c in cols:
        use[c] = num(use[c])
    use = use.dropna()
    if len(use) < max(40, len(xs) + 10):
        return None, use

    work = pd.DataFrame(index=use.index)
    work["y_z"] = zcol(use, y)
    zxs = []
    for x in xs:
        zx = f"x_{len(zxs)}"
        work[zx] = zcol(use, x)
        zxs.append(zx)

    work = work.dropna()
    if len(work) < max(40, len(xs) + 10):
        return None, work

    formula = "y_z ~ " + " + ".join(zxs)
    fit = smf.ols(formula, data=work).fit(cov_type="HC3")
    return fit, work

def leave_region_out(df0, y, x, controls):
    if "lat" not in df0.columns or "lon" not in df0.columns:
        return np.nan, np.nan, 0

    d = df0[[y, x] + controls + ["lat", "lon"]].copy()
    for c in [y, x] + controls + ["lat", "lon"]:
        d[c] = num(d[c])
    d = d.dropna()
    if len(d) < 60:
        return np.nan, np.nan, 0

    lat_s = num(d["lat"])
    lon_s = num(d["lon"])

    d["region"] = (
        pd.cut(lat_s, [-90, -30, 0, 30, 60, 90], labels=False).astype(str)
        + "_"
        + pd.cut(lon_s, [-180, -90, 0, 90, 180], labels=False).astype(str)
    )

    coefs = []
    for region in sorted(d["region"].dropna().unique()):
        train = d[d["region"] != region]
        if len(train) < max(40, len(controls) + 15):
            continue
        fit, _ = safe_fit(train, y, [x] + controls)
        if fit is None:
            continue
        coefs.append(fit.params.get("x_0", np.nan))

    coefs = np.asarray([c for c in coefs if np.isfinite(c)])
    if len(coefs) == 0:
        return np.nan, np.nan, 0
    return float(np.median(coefs)), float(np.mean(np.sign(coefs) == np.sign(np.nanmedian(coefs)))), len(coefs)

def bootstrap_coef(df0, y, x, controls, n_boot=500, seed=123):
    cols = [y, x] + controls
    d = df0[cols].copy()
    for c in cols:
        d[c] = num(d[c])
    d = d.dropna()
    if len(d) < max(40, len(controls) + 10):
        return np.nan, np.nan, np.nan, 0

    rng = np.random.default_rng(seed)
    coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        bs = d.iloc[idx]
        fit, _ = safe_fit(bs, y, [x] + controls)
        if fit is not None:
            coefs.append(fit.params.get("x_0", np.nan))
    coefs = np.asarray([c for c in coefs if np.isfinite(c)])
    if len(coefs) < 50:
        return np.nan, np.nan, np.nan, len(coefs)
    return float(np.median(coefs)), float(np.quantile(coefs, 0.025)), float(np.quantile(coefs, 0.975)), len(coefs)

# ------------------------------------------------------------
# Outcomes and predictors
# ------------------------------------------------------------

outcome_keywords = [
    "latent_slope_change",
    "latent_post_slope",
    "latent_satbreak_probability",
    "p_satbreak",
    "p_threshold_like",
    "product_uncertainty_slope_change_range",
    "product_uncertainty_post_slope_range",
    "product_uncertainty_satbreak_range",
]

outcomes = [c for c in outcome_keywords if c in df.columns]

bad_predictor_substrings = [
    "ci_low", "ci_high", "posterior_sd", "uncertainty",
    "response", "latent_", "p_satbreak", "p_threshold",
    "event_", "threshold_", "product_uncertainty",
    "row_index", "has_", "missing_", "duplicate",
]

candidate_predictors = []
for c in df.columns:
    lc = c.lower()
    if c in outcomes:
        continue
    if any(b in lc for b in bad_predictor_substrings):
        continue
    x = num(df[c])
    if x.notna().sum() < 40:
        continue
    if x.nunique(dropna=True) < 5:
        continue
    candidate_predictors.append(c)

# Prioritize trait/ecosystem/physical covariates, but keep geography/climate too.
trait_terms = [
    "root", "rooting", "soil", "lai", "aridity", "vpd",
    "temperature", "precip", "c4", "lat", "lon",
    "moisture", "sand", "clay", "silt", "texture",
    "p50", "psi", "isohydric", "trait", "wood", "height",
]

predictors = [
    c for c in candidate_predictors
    if any(t in c.lower() for t in trait_terms)
]

# Controls should be confounders, not the tested predictor itself.
base_controls = [
    c for c in [
        "mean_vpd",
        "aridity",
        "mean_annual_temperature",
        "mean_annual_precipitation",
        "growing_season_mean_lai",
        "mean_soil_moisture",
        "lat",
        "lon",
    ]
    if c in df.columns
]

# Create zero-imputed C4 sensitivity.
if "c4_fraction" in df.columns and "c4_zero_if_missing" not in df.columns:
    df["c4_zero_if_missing"] = num(df["c4_fraction"]).fillna(0.0)
    predictors.append("c4_zero_if_missing")

predictors = list(dict.fromkeys(predictors))

pd.DataFrame({"outcome": outcomes}).to_csv(TAB / "outcomes_scanned.csv", index=False)
pd.DataFrame({"predictor": predictors}).to_csv(TAB / "predictors_scanned.csv", index=False)
pd.DataFrame({"base_control": base_controls}).to_csv(TAB / "base_controls.csv", index=False)

# ------------------------------------------------------------
# 1. Main controlled scan
# ------------------------------------------------------------

rows = []

for y, x in itertools.product(outcomes, predictors):
    if x == y:
        continue

    control_sets = {
        "uncontrolled": [],
        "climate_core": [c for c in base_controls if c != x and c not in ["lat", "lon"]],
        "climate_plus_geo": [c for c in base_controls if c != x],
        "vpd_only_control": ["mean_vpd"] if "mean_vpd" in base_controls and x != "mean_vpd" else [],
    }

    for control_name, controls in control_sets.items():
        xs = [x] + controls
        fit, use = safe_fit(df, y, xs)
        if fit is None:
            continue

        term = "x_0"
        coef = fit.params.get(term, np.nan)
        p = fit.pvalues.get(term, np.nan)
        ci = fit.conf_int()
        ci_low = ci.loc[term, 0] if term in ci.index else np.nan
        ci_high = ci.loc[term, 1] if term in ci.index else np.nan

        rows.append({
            "outcome": y,
            "predictor": x,
            "control_set": control_name,
            "controls": "; ".join(controls),
            "n": int(fit.nobs),
            "coef_z": coef,
            "se_hc3": fit.bse.get(term, np.nan),
            "p": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "r2": fit.rsquared,
            "adj_r2": fit.rsquared_adj,
            "aic": fit.aic,
            "bic": fit.bic,
            "sign": np.sign(coef) if np.isfinite(coef) else np.nan,
        })

scan = pd.DataFrame(rows)
scan["q_bh"] = bh_q(scan["p"])
scan["ci_excludes_zero"] = (scan["ci_low"] * scan["ci_high"] > 0)
scan = scan.sort_values(["q_bh", "p", "r2"], ascending=[True, True, False])
scan.to_csv(TAB / "controlled_trait_flux_scan_all_results.csv", index=False)

# ------------------------------------------------------------
# 2. Robust candidates: survive climate+geo controls
# ------------------------------------------------------------

robust = scan[
    (scan["control_set"] == "climate_plus_geo")
    & (scan["q_bh"] < 0.10)
    & (scan["ci_excludes_zero"])
    & (scan["n"] >= 80)
].copy()

# Add bootstrap and leave-region-out stability.
robust_rows = []
for _, r in robust.iterrows():
    y = r["outcome"]
    x = r["predictor"]
    controls = [c for c in base_controls if c != x]

    boot_med, boot_low, boot_high, n_boot = bootstrap_coef(df, y, x, controls)
    lro_med, lro_sign_stab, n_regions = leave_region_out(df, y, x, controls)

    robust_rows.append({
        **r.to_dict(),
        "boot_median_coef": boot_med,
        "boot_ci_low": boot_low,
        "boot_ci_high": boot_high,
        "boot_n": n_boot,
        "boot_ci_excludes_zero": boot_low * boot_high > 0 if np.isfinite(boot_low) and np.isfinite(boot_high) else False,
        "leave_region_median_coef": lro_med,
        "leave_region_sign_stability": lro_sign_stab,
        "leave_region_n": n_regions,
    })

robust2 = pd.DataFrame(robust_rows)
if len(robust2):
    robust2["robust_score"] = (
        robust2["ci_excludes_zero"].astype(int)
        + robust2["boot_ci_excludes_zero"].astype(int)
        + (robust2["leave_region_sign_stability"].fillna(0) >= 0.75).astype(int)
        + (robust2["q_bh"] < 0.05).astype(int)
    )
    robust2 = robust2.sort_values(
        ["robust_score", "q_bh", "r2"], ascending=[False, True, False]
    )
robust2.to_csv(TAB / "ROBUST_CANDIDATE_RELATIONSHIPS.csv", index=False)

# ------------------------------------------------------------
# 3. Interaction discovery: trait × VPD, trait × aridity, trait × temperature
# ------------------------------------------------------------

moderators = [c for c in ["mean_vpd", "aridity", "mean_annual_temperature", "mean_soil_moisture"] if c in df.columns]
int_rows = []

for y, x, m in itertools.product(outcomes, predictors, moderators):
    if x == m or x == y or m == y:
        continue

    controls = [c for c in base_controls if c not in [x, m]]
    cols = [y, x, m] + controls
    use = df[cols].copy()
    for c in cols:
        use[c] = num(use[c])
    use = use.dropna()
    if len(use) < max(60, len(controls) + 15):
        continue

    work = pd.DataFrame(index=use.index)
    work["y_z"] = zcol(use, y)
    work["x_z"] = zcol(use, x)
    work["m_z"] = zcol(use, m)
    zcs = []
    for i, c in enumerate(controls):
        name = f"c{i}_z"
        work[name] = zcol(use, c)
        zcs.append(name)
    work = work.dropna()

    formula = "y_z ~ x_z * m_z"
    if zcs:
        formula += " + " + " + ".join(zcs)

    fit = smf.ols(formula, data=work).fit(cov_type="HC3")
    term = "x_z:m_z"
    ci = fit.conf_int()

    int_rows.append({
        "outcome": y,
        "predictor": x,
        "moderator": m,
        "controls": "; ".join(controls),
        "n": int(fit.nobs),
        "interaction_coef": fit.params.get(term, np.nan),
        "interaction_p": fit.pvalues.get(term, np.nan),
        "interaction_ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
        "interaction_ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
        "r2": fit.rsquared,
        "aic": fit.aic,
    })

ints = pd.DataFrame(int_rows)
if len(ints):
    ints["q_bh"] = bh_q(ints["interaction_p"])
    ints["ci_excludes_zero"] = ints["interaction_ci_low"] * ints["interaction_ci_high"] > 0
    ints = ints.sort_values(["q_bh", "interaction_p", "r2"], ascending=[True, True, False])
ints.to_csv(TAB / "TRAIT_ENVIRONMENT_INTERACTION_DISCOVERY.csv", index=False)

# ------------------------------------------------------------
# 4. Nonlinear/piecewise scan for thresholds
# ------------------------------------------------------------

piece_rows = []

for y, x in itertools.product(outcomes, predictors):
    controls = [c for c in base_controls if c != x]
    cols = [y, x] + controls
    use = df[cols].copy()
    for c in cols:
        use[c] = num(use[c])
    use = use.dropna()

    if len(use) < max(80, len(controls) + 20):
        continue

    qs = [0.25, 0.5, 0.75]
    for q in qs:
        knot = use[x].quantile(q)
        work = pd.DataFrame(index=use.index)
        work["y_z"] = zcol(use, y)
        work["x_z"] = zcol(use, x)
        work["hinge_z"] = zcol(pd.DataFrame({"h": np.maximum(use[x] - knot, 0)}), "h")
        zcs = []
        for i, c in enumerate(controls):
            name = f"c{i}_z"
            work[name] = zcol(use, c)
            zcs.append(name)
        work = work.dropna()

        linear_formula = "y_z ~ x_z" + (" + " + " + ".join(zcs) if zcs else "")
        hinge_formula = "y_z ~ x_z + hinge_z" + (" + " + " + ".join(zcs) if zcs else "")

        fit_lin = smf.ols(linear_formula, data=work).fit()
        fit_hinge = smf.ols(hinge_formula, data=work).fit(cov_type="HC3")

        piece_rows.append({
            "outcome": y,
            "predictor": x,
            "knot_quantile": q,
            "knot_value": knot,
            "n": int(fit_hinge.nobs),
            "linear_r2": fit_lin.rsquared,
            "hinge_r2": fit_hinge.rsquared,
            "delta_r2": fit_hinge.rsquared - fit_lin.rsquared,
            "linear_aic": fit_lin.aic,
            "hinge_aic": fit_hinge.aic,
            "delta_aic_hinge_minus_linear": fit_hinge.aic - fit_lin.aic,
            "hinge_coef": fit_hinge.params.get("hinge_z", np.nan),
            "hinge_p": fit_hinge.pvalues.get("hinge_z", np.nan),
        })

piece = pd.DataFrame(piece_rows)
if len(piece):
    piece["q_bh"] = bh_q(piece["hinge_p"])
    piece = piece.sort_values(["q_bh", "delta_aic_hinge_minus_linear"], ascending=[True, True])
piece.to_csv(TAB / "NONLINEAR_THRESHOLD_DISCOVERY.csv", index=False)

# ------------------------------------------------------------
# 5. Optional ML importance if sklearn is available
# ------------------------------------------------------------

ml_rows = []
try:
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    ml_predictors = list(dict.fromkeys([p for p in predictors + base_controls if p in df.columns]))
    for y in outcomes:
        use = df[[y] + ml_predictors].copy()
        for c in use.columns:
            use[c] = num(use[c])
        use = use.dropna(subset=[y])
        if len(use) < 80:
            continue

        X = use[ml_predictors]
        Y = use[y]

        models = {
            "random_forest": RandomForestRegressor(n_estimators=300, random_state=123, min_samples_leaf=8),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=123, max_iter=300, l2_regularization=0.1),
        }

        for mname, model in models.items():
            pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
            cv = KFold(n_splits=5, shuffle=True, random_state=123)
            scores = cross_val_score(pipe, X, Y, cv=cv, scoring="r2")
            pipe.fit(X, Y)
            perm = permutation_importance(pipe, X, Y, n_repeats=30, random_state=123, scoring="r2")
            for pred, imp_mean, imp_sd in zip(ml_predictors, perm.importances_mean, perm.importances_std):
                ml_rows.append({
                    "outcome": y,
                    "model": mname,
                    "n": len(use),
                    "cv_r2_mean": scores.mean(),
                    "cv_r2_sd": scores.std(),
                    "predictor": pred,
                    "perm_importance_mean": imp_mean,
                    "perm_importance_sd": imp_sd,
                })
except Exception as e:
    (TXT / "ML_IMPORTANCE_ERROR.txt").write_text(str(e))

ml = pd.DataFrame(ml_rows)
if len(ml):
    ml = ml.sort_values(["outcome", "model", "perm_importance_mean"], ascending=[True, True, False])
ml.to_csv(TAB / "ML_PERMUTATION_IMPORTANCE.csv", index=False)

# ------------------------------------------------------------
# Memo
# ------------------------------------------------------------

def show(path, n=40):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    d = pd.read_csv(p)
    if len(d) == 0:
        return "EMPTY"
    return d.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BB trait-based ecosystem-flux discovery")
memo.append("=" * 80)
memo.append("")
memo.append(f"Input: {INPUT}")
memo.append(f"Rows: {len(df)}")
memo.append("")
memo.append("Outcomes scanned:")
memo.append(", ".join(outcomes))
memo.append("")
memo.append("Predictors scanned:")
memo.append(", ".join(predictors))
memo.append("")
memo.append("Top controlled scan results:")
memo.append(show("controlled_trait_flux_scan_all_results.csv", 60))
memo.append("")
memo.append("Robust candidate relationships:")
memo.append(show("ROBUST_CANDIDATE_RELATIONSHIPS.csv", 60))
memo.append("")
memo.append("Trait/environment interactions:")
memo.append(show("TRAIT_ENVIRONMENT_INTERACTION_DISCOVERY.csv", 60))
memo.append("")
memo.append("Nonlinear / threshold candidates:")
memo.append(show("NONLINEAR_THRESHOLD_DISCOVERY.csv", 60))
memo.append("")
memo.append("ML permutation importance:")
memo.append(show("ML_PERMUTATION_IMPORTANCE.csv", 80))
memo.append("")
memo.append("Important files:")
for f in [
    "controlled_trait_flux_scan_all_results.csv",
    "ROBUST_CANDIDATE_RELATIONSHIPS.csv",
    "TRAIT_ENVIRONMENT_INTERACTION_DISCOVERY.csv",
    "NONLINEAR_THRESHOLD_DISCOVERY.csv",
    "ML_PERMUTATION_IMPORTANCE.csv",
    "outcomes_scanned.csv",
    "predictors_scanned.csv",
    "base_controls.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_trait_ecosystem_flux_discovery.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_trait_ecosystem_flux_discovery.txt'}")
