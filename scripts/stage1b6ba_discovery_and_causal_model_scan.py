from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6ba_discovery_and_causal_model_scan"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

KEY = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv"
if not KEY.exists():
    KEY = ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

df = pd.read_csv(KEY, low_memory=False).replace([np.inf, -np.inf], np.nan)

# -----------------------------
# Core columns
# -----------------------------

outcomes = [
    "latent_slope_change",
    "latent_post_slope",
    "latent_satbreak_probability",
    "p_satbreak",
    "p_threshold_like",
    "product_uncertainty_slope_change_range",
    "product_uncertainty_post_slope_range",
    "product_uncertainty_satbreak_range",
]

predictors = [
    "c4_fraction",
    "c4_zero_if_missing",
    "mean_vpd",
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "growing_season_mean_lai",
    "mean_soil_moisture",
    "rooting_depth",
    "lat",
    "lon",
]

# Add zero-imputed C4.
if "c4_fraction" in df.columns:
    df["c4_zero_if_missing"] = pd.to_numeric(df["c4_fraction"], errors="coerce").fillna(0.0)

outcomes = [c for c in outcomes if c in df.columns]
predictors = [c for c in predictors if c in df.columns]

core_controls = [
    c for c in [
        "mean_vpd",
        "aridity",
        "mean_annual_temperature",
        "mean_annual_precipitation",
        "growing_season_mean_lai",
        "mean_soil_moisture",
    ]
    if c in df.columns
]

# -----------------------------
# Helpers
# -----------------------------

def num(s):
    return pd.to_numeric(s, errors="coerce")

def z_inplace(d, col):
    x = num(d[col])
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        d[col + "_z"] = np.nan
    else:
        d[col + "_z"] = (x - x.mean()) / sd

def fit_ols(d, y, xs, robust=True):
    cols = [y] + xs
    use = d[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(use) < max(20, len(xs) + 8):
        return None, use
    for c in cols:
        z_inplace(use, c)
    zy = y + "_z"
    zxs = [x + "_z" for x in xs]
    formula = zy + " ~ " + " + ".join(zxs)
    fit = smf.ols(formula, data=use).fit(cov_type="HC3" if robust else "nonrobust")
    return fit, use

def partial_corr(d, x, y, controls):
    cols = [x, y] + controls
    use = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(20, len(controls) + 8):
        return np.nan, np.nan, len(use)
    X = sm.add_constant(use[controls], has_constant="add") if controls else np.ones((len(use), 1))
    rx = sm.OLS(use[x], X).fit().resid
    ry = sm.OLS(use[y], X).fit().resid
    r = pd.Series(rx).corr(pd.Series(ry))
    # approximate p-value
    dfree = len(use) - len(controls) - 2
    if dfree > 0 and abs(r) < 1:
        t = r * np.sqrt(dfree / (1 - r*r))
        p = 2 * stats.t.sf(abs(t), dfree)
    else:
        p = np.nan
    return r, p, len(use)

def bh_q(pvals):
    p = np.array([np.nan if pd.isna(x) else x for x in pvals], dtype=float)
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

# -----------------------------
# 1. Broad bivariate relationship scan
# -----------------------------

rows = []
for y, x in itertools.product(outcomes, predictors):
    if y == x:
        continue
    use = df[[y, x]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 20:
        continue
    xr = num(use[x])
    yr = num(use[y])
    if xr.nunique(dropna=True) <= 1 or yr.nunique(dropna=True) <= 1:
        continue
    pearson_r, pearson_p = stats.pearsonr(xr, yr)
    spearman_r, spearman_p = stats.spearmanr(xr, yr)
    fit, use2 = fit_ols(df, y, [x])
    coef = fit.params.get(x + "_z", np.nan) if fit is not None else np.nan
    p = fit.pvalues.get(x + "_z", np.nan) if fit is not None else np.nan
    r2 = fit.rsquared if fit is not None else np.nan
    rows.append({
        "outcome": y,
        "predictor": x,
        "n": len(use),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "ols_z_coef": coef,
        "ols_p": p,
        "r2": r2,
    })

bivar = pd.DataFrame(rows)
if len(bivar):
    bivar["ols_q_bh"] = bh_q(bivar["ols_p"])
    bivar = bivar.sort_values(["ols_q_bh", "r2"], ascending=[True, False])
bivar.to_csv(TAB / "broad_bivariate_relationship_scan.csv", index=False)

# -----------------------------
# 2. Partial correlation scan controlling VPD/climate
# -----------------------------

partial_rows = []
for y, x in itertools.product(outcomes, predictors):
    if y == x:
        continue
    if x in ["mean_vpd"]:
        controls = [c for c in core_controls if c != x]
    else:
        controls = [c for c in core_controls if c != x]
    r, p, n = partial_corr(df, x, y, controls)
    partial_rows.append({
        "outcome": y,
        "predictor": x,
        "controls": "; ".join(controls),
        "n": n,
        "partial_r": r,
        "p": p,
    })

partial = pd.DataFrame(partial_rows)
partial["q_bh"] = bh_q(partial["p"])
partial = partial.sort_values(["q_bh", "partial_r"], ascending=[True, False])
partial.to_csv(TAB / "partial_correlation_scan_core_controls.csv", index=False)

# -----------------------------
# 3. Interaction scan: predictor × VPD
# -----------------------------

interaction_rows = []
for y in outcomes:
    for x in predictors:
        if x in ["mean_vpd"] or x == y:
            continue
        cols = [y, x, "mean_vpd"]
        if not all(c in df.columns for c in cols):
            continue
        use = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if len(use) < 30:
            continue
        for c in cols:
            z_inplace(use, c)
        formula = f"{y}_z ~ {x}_z * mean_vpd_z"
        fit = smf.ols(formula, data=use).fit(cov_type="HC3")
        term = f"{x}_z:mean_vpd_z"
        interaction_rows.append({
            "outcome": y,
            "modifier": x,
            "n": int(fit.nobs),
            "interaction_term": term,
            "interaction_coef": fit.params.get(term, np.nan),
            "interaction_p": fit.pvalues.get(term, np.nan),
            "interaction_ci_low": fit.conf_int().loc[term, 0] if term in fit.params.index else np.nan,
            "interaction_ci_high": fit.conf_int().loc[term, 1] if term in fit.params.index else np.nan,
            "r2": fit.rsquared,
            "aic": fit.aic,
        })

inter = pd.DataFrame(interaction_rows)
if len(inter):
    inter["interaction_q_bh"] = bh_q(inter["interaction_p"])
    inter = inter.sort_values(["interaction_q_bh", "interaction_p"])
inter.to_csv(TAB / "interaction_scan_predictor_x_vpd.csv", index=False)

# -----------------------------
# 4. Causal model families
# -----------------------------

causal_rows = []
coef_rows = []

def add_fit(label, sample_label, d, formula):
    use = d.copy()
    fit = smf.ols(formula, data=use).fit(cov_type="HC3")
    causal_rows.append({
        "causal_family": label,
        "sample": sample_label,
        "n": int(fit.nobs),
        "r2": fit.rsquared,
        "adj_r2": fit.rsquared_adj,
        "aic": fit.aic,
        "bic": fit.bic,
        "formula": formula,
    })
    ci = fit.conf_int()
    for term in fit.params.index:
        coef_rows.append({
            "causal_family": label,
            "sample": sample_label,
            "term": term,
            "coef": fit.params[term],
            "se_hc3": fit.bse[term],
            "p": fit.pvalues[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
        })

# Use latent_slope_change as primary outcome.
Y = "latent_slope_change"
if Y in df.columns and "mean_vpd" in df.columns:
    causal_vars = [Y, "mean_vpd"] + [c for c in [
        "c4_fraction", "c4_zero_if_missing",
        "aridity", "mean_annual_temperature", "mean_annual_precipitation",
        "growing_season_mean_lai", "mean_soil_moisture"
    ] if c in df.columns]

    d = df[causal_vars].replace([np.inf, -np.inf], np.nan).copy()

    # z-score original columns into simple names.
    for c in causal_vars:
        z_inplace(d, c)

    # current C4-covered sample
    d142 = d.dropna(subset=[
        Y + "_z", "mean_vpd_z", "c4_fraction_z",
        "aridity_z", "mean_annual_temperature_z", "mean_annual_precipitation_z",
        "growing_season_mean_lai_z", "mean_soil_moisture_z"
    ]).copy()

    # global zero-impute sample
    d199 = d.dropna(subset=[
        Y + "_z", "mean_vpd_z", "c4_zero_if_missing_z",
        "aridity_z", "mean_annual_temperature_z", "mean_annual_precipitation_z",
        "growing_season_mean_lai_z", "mean_soil_moisture_z"
    ]).copy()

    control_str = "aridity_z + mean_annual_temperature_z + mean_annual_precipitation_z + growing_season_mean_lai_z + mean_soil_moisture_z"

    if len(d142) >= 30:
        add_fit("A_raw_C4_total_effect", "c4_covered_n142", d142, f"{Y}_z ~ c4_fraction_z")
        add_fit("B_climate_confounder_model", "c4_covered_n142", d142, f"{Y}_z ~ c4_fraction_z + mean_vpd_z")
        add_fit("C_direct_functional_interaction", "c4_covered_n142", d142, f"{Y}_z ~ c4_fraction_z * mean_vpd_z")
        add_fit("D_full_confounder_adjusted_interaction", "c4_covered_n142", d142, f"{Y}_z ~ c4_fraction_z * mean_vpd_z + {control_str}")
        add_fit("E_climate_only_full_controls", "c4_covered_n142", d142, f"{Y}_z ~ mean_vpd_z + {control_str}")

        # Reverse direction / climate cause of C4.
        add_fit("F_C4_as_climate_outcome", "c4_covered_n142", d142, f"c4_fraction_z ~ mean_vpd_z + {control_str}")

    if len(d199) >= 30:
        add_fit("A_raw_C4_total_effect", "zero_imputed_global_n199", d199, f"{Y}_z ~ c4_zero_if_missing_z")
        add_fit("B_climate_confounder_model", "zero_imputed_global_n199", d199, f"{Y}_z ~ c4_zero_if_missing_z + mean_vpd_z")
        add_fit("C_direct_functional_interaction", "zero_imputed_global_n199", d199, f"{Y}_z ~ c4_zero_if_missing_z * mean_vpd_z")
        add_fit("D_full_confounder_adjusted_interaction", "zero_imputed_global_n199", d199, f"{Y}_z ~ c4_zero_if_missing_z * mean_vpd_z + {control_str}")
        add_fit("E_climate_only_full_controls", "zero_imputed_global_n199", d199, f"{Y}_z ~ mean_vpd_z + {control_str}")
        add_fit("F_C4_as_climate_outcome", "zero_imputed_global_n199", d199, f"c4_zero_if_missing_z ~ mean_vpd_z + {control_str}")

causal = pd.DataFrame(causal_rows)
coefs = pd.DataFrame(coef_rows)

causal.to_csv(TAB / "causal_model_family_fit_comparison.csv", index=False)
coefs.to_csv(TAB / "causal_model_family_coefficients.csv", index=False)

# -----------------------------
# 5. Mediation-style decomposition, but labeled as statistical only
# -----------------------------

med_rows = []

def mediation_decomp(d, c4col, sample_label):
    cols = [Y, c4col, "mean_vpd"] + [c for c in [
        "aridity", "mean_annual_temperature", "mean_annual_precipitation",
        "growing_season_mean_lai", "mean_soil_moisture"
    ] if c in d.columns]
    use = d[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(use) < 40:
        return
    for c in cols:
        z_inplace(use, c)

    y_z = Y + "_z"
    c4_z = c4col + "_z"
    vpd_z = "mean_vpd_z"
    controls_z = [c + "_z" for c in cols if c not in [Y, c4col, "mean_vpd"]]

    # a path: VPD ~ C4 + controls
    f_a = f"{vpd_z} ~ {c4_z}" + (" + " + " + ".join(controls_z) if controls_z else "")
    fit_a = smf.ols(f_a, data=use).fit(cov_type="HC3")
    a = fit_a.params.get(c4_z, np.nan)
    p_a = fit_a.pvalues.get(c4_z, np.nan)

    # total: Y ~ C4 + controls
    f_total = f"{y_z} ~ {c4_z}" + (" + " + " + ".join(controls_z) if controls_z else "")
    fit_total = smf.ols(f_total, data=use).fit(cov_type="HC3")
    total = fit_total.params.get(c4_z, np.nan)
    p_total = fit_total.pvalues.get(c4_z, np.nan)

    # direct and b path: Y ~ C4 + VPD + controls
    f_direct = f"{y_z} ~ {c4_z} + {vpd_z}" + (" + " + " + ".join(controls_z) if controls_z else "")
    fit_direct = smf.ols(f_direct, data=use).fit(cov_type="HC3")
    direct = fit_direct.params.get(c4_z, np.nan)
    p_direct = fit_direct.pvalues.get(c4_z, np.nan)
    b = fit_direct.params.get(vpd_z, np.nan)
    p_b = fit_direct.pvalues.get(vpd_z, np.nan)

    indirect = a * b

    # Bootstrap indirect.
    rng = np.random.default_rng(123)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(use), len(use))
        bs = use.iloc[idx].copy()
        try:
            fa = smf.ols(f_a, data=bs).fit()
            fd = smf.ols(f_direct, data=bs).fit()
            boots.append(fa.params.get(c4_z, np.nan) * fd.params.get(vpd_z, np.nan))
        except Exception:
            pass
    boots = np.array([x for x in boots if np.isfinite(x)])

    med_rows.append({
        "sample": sample_label,
        "c4_variable": c4col,
        "n": len(use),
        "a_C4_to_VPD_coef": a,
        "a_p": p_a,
        "b_VPD_to_Y_given_C4_coef": b,
        "b_p": p_b,
        "total_C4_to_Y_coef": total,
        "total_p": p_total,
        "direct_C4_to_Y_given_VPD_coef": direct,
        "direct_p": p_direct,
        "indirect_a_times_b": indirect,
        "boot_ci_low": np.quantile(boots, 0.025) if len(boots) else np.nan,
        "boot_ci_high": np.quantile(boots, 0.975) if len(boots) else np.nan,
        "note": "Statistical decomposition only; not causal mediation unless DAG assumptions hold.",
    })

if Y in df.columns and "mean_vpd" in df.columns:
    if "c4_fraction" in df.columns:
        mediation_decomp(df, "c4_fraction", "c4_covered_n142")
    if "c4_zero_if_missing" in df.columns:
        mediation_decomp(df, "c4_zero_if_missing", "zero_imputed_global_n199")

pd.DataFrame(med_rows).to_csv(TAB / "statistical_direct_indirect_decomposition.csv", index=False)

# -----------------------------
# Memo
# -----------------------------

def show(path, n=40):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    d = pd.read_csv(p)
    return d.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BA discovery and causal-model scan")
memo.append("=" * 80)
memo.append("")
memo.append("Top broad bivariate relationships:")
memo.append(show("broad_bivariate_relationship_scan.csv", 40))
memo.append("")
memo.append("Top partial correlations after core controls:")
memo.append(show("partial_correlation_scan_core_controls.csv", 40))
memo.append("")
memo.append("Top VPD interaction scan:")
memo.append(show("interaction_scan_predictor_x_vpd.csv", 40))
memo.append("")
memo.append("Causal model family fit comparison:")
memo.append(show("causal_model_family_fit_comparison.csv", 40))
memo.append("")
memo.append("Key causal-model coefficients:")
if (TAB / "causal_model_family_coefficients.csv").exists():
    cc = pd.read_csv(TAB / "causal_model_family_coefficients.csv")
    key = cc[cc["term"].str.contains("c4|vpd|:", case=False, na=False)]
    memo.append(key.to_string(index=False))
else:
    memo.append("MISSING")
memo.append("")
memo.append("Statistical direct/indirect decomposition:")
memo.append(show("statistical_direct_indirect_decomposition.csv", 20))
memo.append("")
memo.append("Important files:")
for f in [
    "broad_bivariate_relationship_scan.csv",
    "partial_correlation_scan_core_controls.csv",
    "interaction_scan_predictor_x_vpd.csv",
    "causal_model_family_fit_comparison.csv",
    "causal_model_family_coefficients.csv",
    "statistical_direct_indirect_decomposition.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_discovery_and_causal_model_scan.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_discovery_and_causal_model_scan.txt'}")
