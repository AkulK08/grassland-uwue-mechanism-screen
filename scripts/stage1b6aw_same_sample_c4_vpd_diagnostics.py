from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6aw_same_sample_c4_vpd_diagnostics"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

POINTS = ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

df = pd.read_csv(POINTS, low_memory=False).replace([np.inf, -np.inf], np.nan)

y = "latent_slope_change"
c4 = "c4_fraction"
vpd = "mean_vpd"

core_controls = [
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "growing_season_mean_lai",
    "mean_soil_moisture",
]

rooting = "rooting_depth"

needed = [y, c4, vpd] + core_controls
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

# ============================================================
# 1. Why are 57 points missing C4?
# ============================================================

missing_c4 = df[df[c4].isna()].copy()
has_c4 = df[df[c4].notna()].copy()

for d in [missing_c4, has_c4]:
    if "lat" in d.columns and "lon" in d.columns:
        d["sahel_broad_lat10_20_lon-20_40"] = d["lat"].between(10, 20) & d["lon"].between(-20, 40)
        d["sahel_core_lat12_18_lon-17_35"] = d["lat"].between(12, 18) & d["lon"].between(-17, 35)

id_cols = [c for c in ["point_id", "lat", "lon"] if c in df.columns]
missing_c4.to_csv(TAB / "POINTS_missing_c4_fraction_n57_expected.csv", index=False)
has_c4.to_csv(TAB / "POINTS_with_c4_fraction.csv", index=False)

missingness_compare = []
for col in df.columns:
    missingness_compare.append({
        "column": col,
        "missing_rate_among_missing_c4_points": missing_c4[col].isna().mean() if len(missing_c4) else np.nan,
        "missing_rate_among_has_c4_points": has_c4[col].isna().mean() if len(has_c4) else np.nan,
        "unique_missing_c4": missing_c4[col].nunique(dropna=True) if len(missing_c4) else 0,
        "unique_has_c4": has_c4[col].nunique(dropna=True) if len(has_c4) else 0,
    })

pd.DataFrame(missingness_compare).to_csv(
    TAB / "missingness_compare_missing_c4_vs_has_c4.csv",
    index=False
)

geo_rows = []
if "lat" in df.columns and "lon" in df.columns:
    for name, d in [
        ("raw_all", df),
        ("missing_c4", missing_c4),
        ("has_c4", has_c4),
    ]:
        geo_rows.append({
            "group": name,
            "n": len(d),
            "lat_min": d["lat"].min(),
            "lat_max": d["lat"].max(),
            "lon_min": d["lon"].min(),
            "lon_max": d["lon"].max(),
            "n_sahel_broad": int((d["lat"].between(10, 20) & d["lon"].between(-20, 40)).sum()),
            "rate_sahel_broad": float((d["lat"].between(10, 20) & d["lon"].between(-20, 40)).mean()) if len(d) else np.nan,
            "n_sahel_core": int((d["lat"].between(12, 18) & d["lon"].between(-17, 35)).sum()),
            "rate_sahel_core": float((d["lat"].between(12, 18) & d["lon"].between(-17, 35)).mean()) if len(d) else np.nan,
        })

pd.DataFrame(geo_rows).to_csv(TAB / "missing_c4_geographic_summary.csv", index=False)

# Search likely C4-related columns.
c4_like_cols = [col for col in df.columns if "c4" in col.lower() or "photosyn" in col.lower() or "grass" in col.lower()]
pd.DataFrame({
    "c4_like_column": c4_like_cols,
    "nonmissing_n": [df[col].notna().sum() for col in c4_like_cols],
    "missing_n": [df[col].isna().sum() for col in c4_like_cols],
    "unique_n": [df[col].nunique(dropna=True) for col in c4_like_cols],
}).to_csv(TAB / "c4_like_columns_inventory.csv", index=False)

# ============================================================
# 2. Force every model onto same 142-point sample
# ============================================================

same = df.dropna(subset=[y, c4, vpd] + core_controls).copy()
same["_analysis_sample"] = "same_142_y_c4_vpd_core_controls"

def zscore(d, col):
    sd = d[col].std(ddof=0)
    if pd.isna(sd) or sd == 0:
        d[col + "_z"] = np.nan
    else:
        d[col + "_z"] = (d[col] - d[col].mean()) / sd

for col in [y, c4, vpd] + core_controls + ([rooting] if rooting in df.columns else []):
    if col in same.columns:
        zscore(same, col)

zy = y + "_z"
zc4 = c4 + "_z"
zvpd = vpd + "_z"
zcontrols = [c + "_z" for c in core_controls]

same.to_csv(TAB / "SAME_SAMPLE_142_USED_FOR_ALL_MAIN_MODELS.csv", index=False)

specs = [
    ("M0_C4_only_same142", f"{zy} ~ {zc4}"),
    ("M1_VPD_only_same142", f"{zy} ~ {zvpd}"),
    ("M2_C4_plus_VPD_same142", f"{zy} ~ {zc4} + {zvpd}"),
    ("M3_C4xVPD_same142", f"{zy} ~ {zc4} * {zvpd}"),
    ("M4_C4xVPD_plus_core_controls_same142", f"{zy} ~ {zc4} * {zvpd} + " + " + ".join(zcontrols)),
]

fit_rows = []
coef_rows = []

for name, formula in specs:
    fit = smf.ols(formula, data=same).fit(cov_type="HC3")
    fit_rows.append({
        "model": name,
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
            "model": name,
            "n": int(fit.nobs),
            "term": term,
            "coef": fit.params[term],
            "se_hc3": fit.bse[term],
            "t": fit.tvalues[term],
            "p": fit.pvalues[term],
            "ci_low": ci.loc[term, 0],
            "ci_high": ci.loc[term, 1],
            "r2": fit.rsquared,
            "aic": fit.aic,
            "bic": fit.bic,
        })

fits = pd.DataFrame(fit_rows)
coefs = pd.DataFrame(coef_rows)

r2_m1 = float(fits.loc[fits.model == "M1_VPD_only_same142", "r2"].iloc[0])
r2_m2 = float(fits.loc[fits.model == "M2_C4_plus_VPD_same142", "r2"].iloc[0])
r2_m3 = float(fits.loc[fits.model == "M3_C4xVPD_same142", "r2"].iloc[0])

fits["delta_r2_vs_VPD_only_same142"] = fits["r2"] - r2_m1
fits["delta_r2_vs_C4_plus_VPD_same142"] = fits["r2"] - r2_m2
fits["delta_r2_vs_interaction_same142"] = fits["r2"] - r2_m3

fits.to_csv(TAB / "SAME_SAMPLE_142_model_fit_comparison.csv", index=False)
coefs.to_csv(TAB / "SAME_SAMPLE_142_model_coefficients_hc3.csv", index=False)

# ============================================================
# 3. Correlation, partial correlation, VIF
# ============================================================

corr_rows = []
for a, b in [
    (c4, vpd),
    (c4, y),
    (vpd, y),
]:
    corr_rows.append({
        "x": a,
        "y": b,
        "pearson_r": same[a].corr(same[b], method="pearson"),
        "spearman_r": same[a].corr(same[b], method="spearman"),
        "n": len(same[[a, b]].dropna()),
    })

pd.DataFrame(corr_rows).to_csv(TAB / "correlations_same142.csv", index=False)

# Partial correlation helper: residualize x and y on covariates, then correlate residuals.
def partial_corr(d, x, ycol, covars):
    d2 = d[[x, ycol] + covars].dropna()
    if len(d2) < len(covars) + 5:
        return np.nan, len(d2)
    X = sm.add_constant(d2[covars], has_constant="add")
    rx = sm.OLS(d2[x], X).fit().resid
    ry = sm.OLS(d2[ycol], X).fit().resid
    return pd.Series(rx).corr(pd.Series(ry)), len(d2)

partial_rows = []

r, n = partial_corr(same, c4, y, [vpd])
partial_rows.append({
    "partial_correlation": "C4_vs_response_controlling_VPD",
    "r_partial": r,
    "n": n,
    "controls": vpd,
})

r, n = partial_corr(same, c4, y, [vpd] + core_controls)
partial_rows.append({
    "partial_correlation": "C4_vs_response_controlling_VPD_and_core_climate",
    "r_partial": r,
    "n": n,
    "controls": "; ".join([vpd] + core_controls),
})

r, n = partial_corr(same, vpd, y, [c4])
partial_rows.append({
    "partial_correlation": "VPD_vs_response_controlling_C4",
    "r_partial": r,
    "n": n,
    "controls": c4,
})

r, n = partial_corr(same, c4, vpd, core_controls)
partial_rows.append({
    "partial_correlation": "C4_vs_VPD_controlling_core_climate",
    "r_partial": r,
    "n": n,
    "controls": "; ".join(core_controls),
})

pd.DataFrame(partial_rows).to_csv(TAB / "partial_correlations_same142.csv", index=False)

# VIF for main predictor set.
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_cols = [zc4, zvpd] + zcontrols
X = same[vif_cols].dropna()
X_const = sm.add_constant(X, has_constant="add")

vif_rows = []
for i, col in enumerate(X_const.columns):
    if col == "const":
        continue
    vif_rows.append({
        "variable": col,
        "vif": variance_inflation_factor(X_const.values, i),
        "n": len(X_const),
    })

pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
    TAB / "VIF_same142_core_controls.csv",
    index=False
)

# ============================================================
# 4. Likelihood-ratio nested comparisons on same sample
#    Note: regular OLS fit, not robust covariance.
# ============================================================

lr_specs = {
    "M1_VPD_only": f"{zy} ~ {zvpd}",
    "M2_C4_plus_VPD": f"{zy} ~ {zvpd} + {zc4}",
    "M3_interaction": f"{zy} ~ {zvpd} * {zc4}",
    "M4_core_controls": f"{zy} ~ {zvpd} * {zc4} + " + " + ".join(zcontrols),
}

ols_fits = {name: smf.ols(formula, data=same).fit() for name, formula in lr_specs.items()}

compare_rows = []

def add_compare(reduced, full):
    lr_stat, pval, df_diff = ols_fits[full].compare_lr_test(ols_fits[reduced])
    compare_rows.append({
        "reduced_model": reduced,
        "full_model": full,
        "lr_stat": lr_stat,
        "df_diff": df_diff,
        "p": pval,
        "delta_r2": ols_fits[full].rsquared - ols_fits[reduced].rsquared,
        "delta_aic": ols_fits[full].aic - ols_fits[reduced].aic,
        "delta_bic": ols_fits[full].bic - ols_fits[reduced].bic,
    })

add_compare("M1_VPD_only", "M2_C4_plus_VPD")
add_compare("M2_C4_plus_VPD", "M3_interaction")
add_compare("M3_interaction", "M4_core_controls")

pd.DataFrame(compare_rows).to_csv(TAB / "nested_model_comparisons_same142.csv", index=False)

# ============================================================
# 5. Memo
# ============================================================

memo = []
memo.append("Same-sample C4/VPD diagnostic memo")
memo.append("=" * 70)
memo.append("")
memo.append(f"Input table: {POINTS}")
memo.append(f"Raw n: {len(df)}")
memo.append(f"Missing C4 fraction n: {len(missing_c4)}")
memo.append(f"Same-sample main analysis n: {len(same)}")
memo.append("")
memo.append("Same-sample model fits:")
memo.append(fits.to_string(index=False))
memo.append("")
memo.append("Key C4/VPD coefficients:")
key = coefs[coefs["term"].str.contains("c4|vpd|:", case=False, na=False)]
memo.append(key.to_string(index=False))
memo.append("")
memo.append("Correlations:")
memo.append(pd.read_csv(TAB / "correlations_same142.csv").to_string(index=False))
memo.append("")
memo.append("Partial correlations:")
memo.append(pd.read_csv(TAB / "partial_correlations_same142.csv").to_string(index=False))
memo.append("")
memo.append("VIF:")
memo.append(pd.read_csv(TAB / "VIF_same142_core_controls.csv").to_string(index=False))
memo.append("")
memo.append("Nested comparisons:")
memo.append(pd.read_csv(TAB / "nested_model_comparisons_same142.csv").to_string(index=False))
memo.append("")
memo.append("Files to inspect:")
memo.append("- POINTS_missing_c4_fraction_n57_expected.csv")
memo.append("- SAME_SAMPLE_142_USED_FOR_ALL_MAIN_MODELS.csv")
memo.append("- SAME_SAMPLE_142_model_fit_comparison.csv")
memo.append("- SAME_SAMPLE_142_model_coefficients_hc3.csv")
memo.append("- correlations_same142.csv")
memo.append("- partial_correlations_same142.csv")
memo.append("- VIF_same142_core_controls.csv")
memo.append("- nested_model_comparisons_same142.csv")

(TXT / "READ_ME_same_sample_c4_vpd_diagnostics.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste-back command:")
print(f"cat {TXT / 'READ_ME_same_sample_c4_vpd_diagnostics.txt'}")
