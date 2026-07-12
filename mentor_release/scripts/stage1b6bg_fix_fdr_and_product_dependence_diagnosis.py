from pathlib import Path
import re, json, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
PREV = ROOT / "results/stage1b6bf_lai_reza_lock_package"
PREV_TAB = PREV / "tables"

OUT = ROOT / "results/stage1b6bg_fix_fdr_and_product_dependence_diagnosis"
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT_INPUT = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv"
if not POINT_INPUT.exists():
    POINT_INPUT = ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"
if not POINT_INPUT.exists():
    POINT_INPUT = ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv"
if not POINT_INPUT.exists():
    raise SystemExit("No point table found.")

raw = pd.read_csv(POINT_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def first_existing(cols, *names):
    cols = list(cols)
    for n in names:
        if n in cols:
            return n
    low = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in low:
            return low[norm(n)]
    return None

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def formula_vars(formula):
    toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
    return sorted(set(t for t in toks if t not in {"C", "I", "Q"}))

def fit_ols(data, formula):
    vars_needed = formula_vars(formula)
    missing = [v for v in vars_needed if v not in data.columns]
    if missing:
        return None, pd.DataFrame(), f"MISSING: {missing}"
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(40, len(vars_needed) + 8):
        return None, use, "N_TOO_SMALL"
    try:
        fit = smf.ols(formula, data=use).fit(cov_type="HC3")
        return fit, use, "FIT_OK"
    except Exception as e:
        return None, use, f"FIT_FAIL: {e}"

def compare(data, label, family, full, reduced, focal):
    full_fit, use, status = fit_ols(data, full)
    red_fit, _, red_status = fit_ols(data, reduced)

    if full_fit is None or red_fit is None:
        return {
            "test_label": label,
            "family": family,
            "status": status,
            "reduced_status": red_status,
            "focal_term": focal,
            "n": len(use),
            "full_formula": full,
            "reduced_formula": reduced,
        }

    try:
        full_nr = smf.ols(full, data=use).fit()
        red_same = smf.ols(reduced, data=use).fit()
        nested_p = float(full_nr.compare_f_test(red_same)[1])
    except Exception:
        nested_p = np.nan

    ci = full_fit.conf_int()

    return {
        "test_label": label,
        "family": family,
        "status": "FIT_OK",
        "focal_term": focal,
        "n": int(full_fit.nobs),
        "coef": full_fit.params.get(focal, np.nan),
        "se_hc3": full_fit.bse.get(focal, np.nan),
        "p": full_fit.pvalues.get(focal, np.nan),
        "ci_low": ci.loc[focal, 0] if focal in ci.index else np.nan,
        "ci_high": ci.loc[focal, 1] if focal in ci.index else np.nan,
        "ci_excludes_zero": bool(ci.loc[focal, 0] * ci.loc[focal, 1] > 0) if focal in ci.index else False,
        "full_r2": full_fit.rsquared,
        "reduced_r2": red_fit.rsquared,
        "delta_r2": full_fit.rsquared - red_fit.rsquared,
        "delta_aic_full_minus_reduced": full_fit.aic - red_fit.aic,
        "nested_f_p": nested_p,
        "full_formula": full,
        "reduced_formula": reduced,
    }

def p_adjust_bh(pvals):
    p = np.asarray(pvals, dtype=float)
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
    q[order] = np.minimum(vals, 1)
    return q

def p_adjust_by(pvals):
    p = np.asarray(pvals, dtype=float)
    q = p_adjust_bh(p)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    harmonic = np.sum(1 / np.arange(1, m + 1))
    return np.minimum(q * harmonic, 1)

def p_adjust_holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return adj
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = (m - np.arange(0, m)) * ranked
    vals = np.maximum.accumulate(vals)
    vals = np.minimum(vals, 1)
    adj[order] = vals
    return adj

# ---------------------------------------------------------------------
# Build corrected canonical dataframe
# ---------------------------------------------------------------------

cols = list(raw.columns)

src = {
    "point_id": first_existing(cols, "point_id"),
    "y": first_existing(cols, "latent_slope_change"),
    "vpd": first_existing(cols, "mean_vpd", "mean_obs_vpd"),
    "lai": first_existing(cols, "growing_season_mean_lai", "mean_lai"),
    "mat": first_existing(cols, "mean_annual_temperature", "mean_temperature"),
    "map": first_existing(cols, "mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing(cols, "aridity"),
    "sm": first_existing(cols, "mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing(cols, "lat", "latitude"),
    "lon": first_existing(cols, "lon", "longitude"),
    "c4": first_existing(cols, "c4_fraction", "c4_fraction_raw"),
    "rooting_depth": first_existing(cols, "rooting_depth"),
    "p50": first_existing(cols, "p50", "psi50"),
    "sand": first_existing(cols, "soil_sand", "sand", "sand_fraction"),
    "clay": first_existing(cols, "soil_clay", "clay", "clay_fraction"),
    "silt": first_existing(cols, "soil_silt", "silt", "silt_fraction"),
}

need = ["y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"]
missing = [k for k in need if src[k] is None]
if missing:
    raise SystemExit(f"Missing required variables: {missing}")

d = pd.DataFrame(index=raw.index)
for k, c in src.items():
    if c is None:
        continue
    if k == "point_id":
        d[k] = raw[c].astype(str)
    else:
        d[k] = pd.to_numeric(raw[c], errors="coerce")

# Soil texture PC1.
if all(c in d.columns for c in ["sand", "clay", "silt"]):
    soil = d[["sand", "clay", "silt"]].apply(zscore)
    use = soil.dropna()
    pc1 = pd.Series(np.nan, index=d.index)
    if len(use) >= 40:
        X = use.values
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        scores = X @ vt[0, :]
        pc1.loc[use.index] = scores
        if pc1.corr(d["clay"]) < 0:
            pc1 = -pc1
        d["soil_texture_pc1"] = pc1

for c in list(d.columns):
    if c != "point_id" and pd.api.types.is_numeric_dtype(d[c]):
        d[c + "_z"] = zscore(d[c])

controls = [
    "vpd_z",
    "arid_z",
    "mat_z",
    "map_z",
    "sm_z",
    "soil_texture_pc1_z",
    "lat_z",
    "lon_z",
]
controls = [c for c in controls if c in d.columns and d[c].notna().sum() >= 40]
controls_no_mat = [c for c in controls if c != "mat_z"]

# ---------------------------------------------------------------------
# Corrected FDR families
# ---------------------------------------------------------------------

candidate_map = [
    ("growing_season_mean_lai", "lai_z", "canopy_structure"),
    ("c4_fraction", "c4_z", "photosynthetic_pathway_fraction"),
    ("rooting_depth", "rooting_depth_z", "root_trait"),
    ("p50_or_psi50", "p50_z", "hydraulic_trait"),
    ("soil_texture_pc1", "soil_texture_pc1_z", "soil_structure"),
    ("soil_sand", "sand_z", "soil_structure"),
    ("soil_clay", "clay_z", "soil_structure"),
    ("soil_silt", "silt_z", "soil_structure"),
]

rows = []

for canonical, zcol, trait_class in candidate_map:
    if zcol not in d.columns or d[zcol].notna().sum() < 40:
        continue

    ctrl = [c for c in controls if c != zcol]

    if len(ctrl) == 0:
        continue

    full = "y_z ~ " + zcol + " + " + " + ".join(ctrl)
    reduced = "y_z ~ " + " + ".join(ctrl)

    row = compare(
        d,
        f"main_effect__{canonical}",
        "A_trait_structure_main_effects_on_latent_slope_change",
        full,
        reduced,
        zcol,
    )
    row["canonical_predictor"] = canonical
    row["trait_class"] = trait_class
    row["original_source_column"] = src.get(canonical, canonical)
    rows.append(row)

for canonical, zcol, trait_class in candidate_map:
    if zcol not in d.columns or d[zcol].notna().sum() < 40:
        continue

    ctrl = [c for c in controls_no_mat if c != zcol]

    if len(ctrl) == 0:
        continue

    full = "y_z ~ " + zcol + " * mat_z + " + " + ".join(ctrl)
    reduced = "y_z ~ " + zcol + " + mat_z + " + " + ".join(ctrl)
    focal = zcol + ":mat_z"

    row = compare(
        d,
        f"interaction_with_MAT__{canonical}",
        "B_trait_structure_x_MAT_interactions_on_latent_slope_change",
        full,
        reduced,
        focal,
    )
    row["canonical_predictor"] = canonical
    row["trait_class"] = trait_class
    row["original_source_column"] = src.get(canonical, canonical)
    rows.append(row)

fdr = pd.DataFrame(rows)

for family, idx in fdr.groupby("family").groups.items():
    pvals = fdr.loc[idx, "p"].values
    fdr.loc[idx, "bh_q"] = p_adjust_bh(pvals)
    fdr.loc[idx, "by_q"] = p_adjust_by(pvals)
    fdr.loc[idx, "holm_p"] = p_adjust_holm(pvals)

fdr = fdr.sort_values(["family", "bh_q", "p"])
fdr.to_csv(TAB / "CORRECTED_FULL_DISCOVERY_FDR_TABLE.csv", index=False)

lai_extract = fdr[fdr["canonical_predictor"].eq("growing_season_mean_lai")].copy()
lai_extract.to_csv(TAB / "CORRECTED_LAI_DISCOVERY_Q_VALUE_EXTRACT.csv", index=False)

# ---------------------------------------------------------------------
# Product-dependence diagnosis from previous exact 3×3 matrix
# ---------------------------------------------------------------------

pm_path = PREV_TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv"
if not pm_path.exists():
    raise SystemExit("Previous EXACT_3x3_PRODUCT_MATRIX_LAI.csv not found.")

pm = pd.read_csv(pm_path)

fit = pm[pm["status"].eq("FIT_OK")].copy()
fit["is_least_dependent_pair"] = fit["least_directly_LAI_dependent_overall_pair"].fillna(False).astype(bool)

# Main diagnosis rows by sample mode.
diag_rows = []

for sample_mode, g in fit.groupby("sample_mode"):
    least = g[g["is_least_dependent_pair"]].copy()
    dependent = g[~g["is_least_dependent_pair"]].copy()

    row = {
        "sample_mode": sample_mode,
        "n_product_rows": len(g),
        "least_dependent_rows": len(least),
        "least_dependent_any_p_lt_0p05": bool((least["p"] < 0.05).any()) if len(least) else False,
        "least_dependent_min_p": float(least["p"].min()) if len(least) else np.nan,
        "least_dependent_median_coef": float(least["coef"].median()) if len(least) else np.nan,
        "least_dependent_direction_negative": bool((least["coef"] < 0).all()) if len(least) else False,
        "dependent_any_p_lt_0p05": bool((dependent["p"] < 0.05).any()) if len(dependent) else False,
        "dependent_min_p": float(dependent["p"].min()) if len(dependent) else np.nan,
        "dependent_median_coef": float(dependent["coef"].median()) if len(dependent) else np.nan,
    }

    # Correlation between dependency rank and absolute/significant effect.
    if "combo_dependency_rank_sum" in g.columns and len(g.dropna(subset=["combo_dependency_rank_sum", "coef"])) >= 4:
        row["spearman_dependency_rank_vs_abs_coef"] = stats.spearmanr(g["combo_dependency_rank_sum"], g["coef"].abs(), nan_policy="omit").correlation
        row["spearman_dependency_rank_vs_negative_coef_strength"] = stats.spearmanr(g["combo_dependency_rank_sum"], -g["coef"], nan_policy="omit").correlation
    else:
        row["spearman_dependency_rank_vs_abs_coef"] = np.nan
        row["spearman_dependency_rank_vs_negative_coef_strength"] = np.nan

    # Bottom-line circularity read.
    if len(least) and not row["least_dependent_any_p_lt_0p05"]:
        row["circularity_defense_status"] = "FAILS_STRONG_GOSIF_GLEAM_INDEPENDENCE_DEFENSE"
    elif len(least) and row["least_dependent_any_p_lt_0p05"]:
        row["circularity_defense_status"] = "SUPPORTS_INDEPENDENCE_DEFENSE"
    else:
        row["circularity_defense_status"] = "NO_LEAST_DEPENDENT_PAIR_IDENTIFIED"

    diag_rows.append(row)

diag = pd.DataFrame(diag_rows)
diag.to_csv(TAB / "PRODUCT_DEPENDENCE_CIRCULARITY_DIAGNOSIS.csv", index=False)

least_rows = fit[fit["is_least_dependent_pair"]].copy()
least_rows.to_csv(TAB / "GOSIF_GLEAM_EXACT_ROWS.csv", index=False)

# ---------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------

def show(path, n=80):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BG corrected FDR and product-dependence diagnosis")
memo.append("=" * 90)
memo.append("")
memo.append("Why this was needed:")
memo.append("- Previous stage1b6bf produced an EMPTY LAI discovery q-value extract because the FDR reconstruction used raw-source names after canonical renaming.")
memo.append("- Previous exact 3x3 product matrix showed that GOSIF × GLEAM, the least directly LAI-dependent pair, does NOT reproduce the LAI signal.")
memo.append("")
memo.append("Corrected LAI discovery q-value extract:")
memo.append(show("CORRECTED_LAI_DISCOVERY_Q_VALUE_EXTRACT.csv", 20))
memo.append("")
memo.append("Full corrected FDR table:")
memo.append(show("CORRECTED_FULL_DISCOVERY_FDR_TABLE.csv", 80))
memo.append("")
memo.append("GOSIF × GLEAM exact rows:")
memo.append(show("GOSIF_GLEAM_EXACT_ROWS.csv", 20))
memo.append("")
memo.append("Product-dependence / circularity diagnosis:")
memo.append(show("PRODUCT_DEPENDENCE_CIRCULARITY_DIAGNOSIS.csv", 20))
memo.append("")
memo.append("Interpretation:")
memo.append("- If the GOSIF × GLEAM rows remain null/positive, we cannot claim the least LAI-dependent product pair supports LAI.")
memo.append("- In that case, LAI can still be a conditional/global regime signal, but algorithmic circularity remains a serious limitation.")
memo.append("- The honest message to Reza should distinguish the latent/global LAI result from the exact product-pair failure.")
memo.append("")
memo.append("Important files:")
for f in [
    "CORRECTED_LAI_DISCOVERY_Q_VALUE_EXTRACT.csv",
    "CORRECTED_FULL_DISCOVERY_FDR_TABLE.csv",
    "GOSIF_GLEAM_EXACT_ROWS.csv",
    "PRODUCT_DEPENDENCE_CIRCULARITY_DIAGNOSIS.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_corrected_fdr_and_product_dependence.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_corrected_fdr_and_product_dependence.txt'}")
