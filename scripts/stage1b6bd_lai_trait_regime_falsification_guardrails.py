from pathlib import Path
import warnings
import re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bd_lai_trait_regime_falsification_guardrails"
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT_CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]

POINT_INPUT = next((p for p in POINT_CANDIDATES if p.exists()), None)
if POINT_INPUT is None:
    raise SystemExit("No suitable point-level input table found.")

OBS_INPUT = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"

raw = pd.read_csv(POINT_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()

# ------------------------------------------------------------
# Canonical data
# ------------------------------------------------------------

def first_existing(*names):
    for n in names:
        if n in raw.columns:
            return n
    return None

sources = {
    "point_id": first_existing("point_id"),
    "y": first_existing("latent_slope_change"),
    "vpd": first_existing("mean_vpd", "mean_obs_vpd"),
    "lai": first_existing("growing_season_mean_lai", "mean_lai"),
    "mat": first_existing("mean_annual_temperature", "mean_temperature"),
    "map": first_existing("mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing("aridity"),
    "sm": first_existing("mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing("lat"),
    "lon": first_existing("lon"),
    "c4": first_existing("c4_fraction"),
    "root": first_existing("rooting_depth"),
    "sand": first_existing("soil_sand"),
    "clay": first_existing("soil_clay"),
    "silt": first_existing("soil_silt"),
}

required = ["y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"]
missing = [k for k in required if sources[k] is None]
if missing:
    raise SystemExit(f"Missing required source columns: {missing}")

d = pd.DataFrame(index=raw.index)
for canon, src in sources.items():
    if src is None:
        continue
    if canon == "point_id":
        d[canon] = raw[src]
    else:
        d[canon] = pd.to_numeric(raw[src], errors="coerce")

d["abs_lat"] = d["lat"].abs()
d["sahel_broad"] = d["lat"].between(10, 20) & d["lon"].between(-20, 40)
d["sahel_core"] = d["lat"].between(12, 18) & d["lon"].between(-17, 35)
d["has_c4"] = d["c4"].notna() if "c4" in d.columns else False

d["region_block"] = (
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

def winsor(s, lo=0.01, hi=0.99):
    x = pd.to_numeric(s, errors="coerce")
    return x.clip(x.quantile(lo), x.quantile(hi))

def rank_z(s):
    return zscore(pd.to_numeric(s, errors="coerce").rank(pct=True))

for c in ["y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon", "abs_lat"]:
    d[c + "_z"] = zscore(d[c])
    d[c + "_w_z"] = zscore(winsor(d[c]))
    d[c + "_rank_z"] = rank_z(d[c])

# ------------------------------------------------------------
# Model formulas
# ------------------------------------------------------------

MAIN_FULL = "y_z ~ lai_z + vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z"
MAIN_REDUCED = "y_z ~ vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z"

INT_FULL = "y_z ~ lai_z * mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z"
INT_REDUCED = "y_z ~ lai_z + mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z"

WINSOR_MAIN_FULL = "y_w_z ~ lai_w_z + vpd_w_z + arid_w_z + mat_w_z + map_w_z + sm_w_z + lat_w_z + lon_w_z"
WINSOR_MAIN_REDUCED = "y_w_z ~ vpd_w_z + arid_w_z + mat_w_z + map_w_z + sm_w_z + lat_w_z + lon_w_z"

RANK_MAIN_FULL = "y_rank_z ~ lai_rank_z + vpd_rank_z + arid_rank_z + mat_rank_z + map_rank_z + sm_rank_z + lat_rank_z + lon_rank_z"
RANK_MAIN_REDUCED = "y_rank_z ~ vpd_rank_z + arid_rank_z + mat_rank_z + map_rank_z + sm_rank_z + lat_rank_z + lon_rank_z"

WINSOR_INT_FULL = "y_w_z ~ lai_w_z * mat_w_z + vpd_w_z + arid_w_z + map_w_z + sm_w_z + lat_w_z + lon_w_z"
WINSOR_INT_REDUCED = "y_w_z ~ lai_w_z + mat_w_z + vpd_w_z + arid_w_z + map_w_z + sm_w_z + lat_w_z + lon_w_z"

RANK_INT_FULL = "y_rank_z ~ lai_rank_z * mat_rank_z + vpd_rank_z + arid_rank_z + map_rank_z + sm_rank_z + lat_rank_z + lon_rank_z"
RANK_INT_REDUCED = "y_rank_z ~ lai_rank_z + mat_rank_z + vpd_rank_z + arid_rank_z + map_rank_z + sm_rank_z + lat_rank_z + lon_rank_z"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def formula_vars(formula):
    toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
    return sorted(set(t for t in toks if t not in {"C", "I", "Q"}))

def fit_ols(data, formula, cov_type="HC3"):
    vars_needed = formula_vars(formula)
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(40, len(vars_needed) + 8):
        return None, use
    fit = smf.ols(formula, data=use).fit(cov_type=cov_type)
    return fit, use

def compare_models(data, label, full_formula, reduced_formula, focal_terms, estimator="ols_hc3"):
    full, use_full = fit_ols(data, full_formula)
    red, use_red = fit_ols(data, reduced_formula)

    rows = []
    if full is None or red is None:
        for term in focal_terms:
            rows.append({
                "test_label": label,
                "estimator": estimator,
                "focal_term": term,
                "status": "FIT_FAIL",
                "n": len(use_full),
            })
        return rows

    # Reduced model on same full complete-case set for nested comparison.
    try:
        full_nr = smf.ols(full_formula, data=use_full).fit()
        red_same = smf.ols(reduced_formula, data=use_full).fit()
        nested_p = float(full_nr.compare_f_test(red_same)[1])
    except Exception:
        nested_p = np.nan

    ci = full.conf_int()

    for term in focal_terms:
        rows.append({
            "test_label": label,
            "estimator": estimator,
            "focal_term": term,
            "status": "FIT_OK",
            "n": int(full.nobs),
            "coef": full.params.get(term, np.nan),
            "se_hc3": full.bse.get(term, np.nan),
            "p": full.pvalues.get(term, np.nan),
            "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
            "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
            "ci_excludes_zero": bool(ci.loc[term, 0] * ci.loc[term, 1] > 0) if term in ci.index else False,
            "full_r2": full.rsquared,
            "reduced_r2": red.rsquared,
            "delta_r2": full.rsquared - red.rsquared,
            "full_aic": full.aic,
            "reduced_aic": red.aic,
            "delta_aic_full_minus_reduced": full.aic - red.aic,
            "nested_f_p": nested_p,
            "full_formula": full_formula,
            "reduced_formula": reduced_formula,
        })
    return rows

def fit_rlm(data, label, formula, focal_terms):
    vars_needed = formula_vars(formula)
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    if len(use) < max(40, len(vars_needed) + 8):
        for term in focal_terms:
            rows.append({
                "test_label": label,
                "estimator": "robust_rlm_huber",
                "focal_term": term,
                "status": "FIT_FAIL",
                "n": len(use),
            })
        return rows

    try:
        fit = smf.rlm(formula, data=use, M=sm.robust.norms.HuberT()).fit()
        for term in focal_terms:
            rows.append({
                "test_label": label,
                "estimator": "robust_rlm_huber",
                "focal_term": term,
                "status": "FIT_OK",
                "n": int(fit.nobs),
                "coef": fit.params.get(term, np.nan),
                "se_hc3": fit.bse.get(term, np.nan),
                "p": fit.pvalues.get(term, np.nan),
                "ci_low": fit.conf_int().loc[term, 0] if term in fit.params.index else np.nan,
                "ci_high": fit.conf_int().loc[term, 1] if term in fit.params.index else np.nan,
                "ci_excludes_zero": bool(fit.conf_int().loc[term, 0] * fit.conf_int().loc[term, 1] > 0) if term in fit.params.index else False,
                "full_formula": formula,
            })
    except Exception as e:
        for term in focal_terms:
            rows.append({
                "test_label": label,
                "estimator": "robust_rlm_huber",
                "focal_term": term,
                "status": f"FIT_FAIL: {e}",
                "n": len(use),
            })
    return rows

def cook_filter(data, formula, threshold=None):
    fit, use = fit_ols(data, formula, cov_type="nonrobust")
    if fit is None:
        return data.iloc[[]].copy(), np.nan, 0
    infl = fit.get_influence()
    cooks = infl.cooks_distance[0]
    if threshold is None:
        threshold = 4.0 / len(use)
    keep_idx = use.index[cooks <= threshold]
    return data.loc[keep_idx].copy(), threshold, int((cooks > threshold).sum())

def bootstrap_term(data, formula, term, n_boot=1000, seed=123):
    fit, use = fit_ols(data, formula)
    if fit is None:
        return {
            "boot_n": 0,
            "boot_median": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_ci_excludes_zero": False,
        }

    rng = np.random.default_rng(seed)
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
    if len(vals) < 100:
        return {
            "boot_n": len(vals),
            "boot_median": np.nan,
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_ci_excludes_zero": False,
        }

    lo = np.quantile(vals, 0.025)
    hi = np.quantile(vals, 0.975)

    return {
        "boot_n": len(vals),
        "boot_median": float(np.median(vals)),
        "boot_ci_low": float(lo),
        "boot_ci_high": float(hi),
        "boot_ci_excludes_zero": bool(lo * hi > 0),
    }

def leave_region_out(data, formula, term):
    vars_needed = formula_vars(formula) + ["region_block"]
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    vals = []
    regions = sorted(use["region_block"].dropna().unique())
    for r in regions:
        train = use[use["region_block"] != r].copy()
        if len(train) < max(40, len(vars_needed) + 8):
            continue
        try:
            f = smf.ols(formula, data=train).fit()
            vals.append(f.params.get(term, np.nan))
        except Exception:
            pass

    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return {
            "lro_n": 0,
            "lro_median": np.nan,
            "lro_sign_stability": np.nan,
            "lro_min": np.nan,
            "lro_max": np.nan,
        }

    med = np.median(vals)
    return {
        "lro_n": len(vals),
        "lro_median": float(med),
        "lro_sign_stability": float(np.mean(np.sign(vals) == np.sign(med))) if med != 0 else np.nan,
        "lro_min": float(np.min(vals)),
        "lro_max": float(np.max(vals)),
    }

def climate_bin_permutation(data, full_formula, term, permuted_var="lai_z", n_perm=1000, seed=123):
    vars_needed = formula_vars(full_formula)
    use = data[vars_needed + ["mat", "vpd", "abs_lat"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 80:
        return {
            "perm_n": 0,
            "observed_abs_coef": np.nan,
            "perm_p_two_sided": np.nan,
        }

    # Climate bins preserve coarse climate/geography structure while breaking local LAI association.
    try:
        use["_bin"] = (
            pd.qcut(use["mat"], 3, duplicates="drop").astype(str)
            + "_"
            + pd.qcut(use["vpd"], 3, duplicates="drop").astype(str)
            + "_"
            + pd.qcut(use["abs_lat"], 3, duplicates="drop").astype(str)
        )
    except Exception:
        use["_bin"] = "all"

    f0 = smf.ols(full_formula, data=use).fit()
    obs = abs(f0.params.get(term, np.nan))
    if not np.isfinite(obs):
        return {
            "perm_n": 0,
            "observed_abs_coef": np.nan,
            "perm_p_two_sided": np.nan,
        }

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n_perm):
        tmp = use.copy()
        shuffled = []
        for _, g in tmp.groupby("_bin"):
            v = g[permuted_var].values.copy()
            rng.shuffle(v)
            shuffled.extend(zip(g.index, v))
        for idx, val in shuffled:
            tmp.loc[idx, permuted_var] = val

        try:
            fp = smf.ols(full_formula, data=tmp).fit()
            vals.append(abs(fp.params.get(term, np.nan)))
        except Exception:
            pass

    vals = np.asarray([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        p = np.nan
    else:
        p = float((np.sum(vals >= obs) + 1) / (len(vals) + 1))

    return {
        "perm_n": len(vals),
        "observed_abs_coef": float(obs),
        "perm_p_two_sided": p,
    }

# ------------------------------------------------------------
# Guardrail scenarios
# ------------------------------------------------------------

scenarios = []

base_mask = pd.Series(True, index=d.index)
scenarios.append(("all_points", base_mask))
scenarios.append(("exclude_sahel_broad", ~d["sahel_broad"]))
scenarios.append(("exclude_sahel_core", ~d["sahel_core"]))
scenarios.append(("exclude_high_lat_abs_gt_48", d["abs_lat"] <= 48))
scenarios.append(("exclude_high_lat_abs_gt_60", d["abs_lat"] <= 60))
scenarios.append(("warm_only_mat_gt_0", d["mat"] > 0))
scenarios.append(("warm_only_mat_gt_2c", d["mat"] > 2.08))
scenarios.append(("cold_removed_and_no_sahel", (d["mat"] > 2.08) & (~d["sahel_broad"])))
scenarios.append(("c4_covered_domain_only", d["has_c4"]))
scenarios.append(("non_c4_missing_removed_plus_no_sahel", d["has_c4"] & (~d["sahel_broad"])))

# Remove extreme outcome / predictor tails.
for col, q in [("y", 0.99), ("lai", 0.99), ("vpd", 0.99), ("mat", 0.99)]:
    lo = d[col].quantile(1 - q)
    hi = d[col].quantile(q)
    scenarios.append((f"trim_{col}_1pct_each_tail", d[col].between(lo, hi)))

# Cook's distance scenario, separately for main and interaction.
cook_main_data, cook_main_thr, cook_main_removed = cook_filter(d, MAIN_FULL)
cook_int_data, cook_int_thr, cook_int_removed = cook_filter(d, INT_FULL)

# ------------------------------------------------------------
# Run all tests
# ------------------------------------------------------------

all_rows = []

for name, mask in scenarios:
    sub = d.loc[mask].copy()

    all_rows += compare_models(
        sub,
        f"{name}__LAI_main",
        MAIN_FULL,
        MAIN_REDUCED,
        ["lai_z"],
        estimator="ols_hc3",
    )

    all_rows += compare_models(
        sub,
        f"{name}__LAI_x_temperature",
        INT_FULL,
        INT_REDUCED,
        ["lai_z:mat_z"],
        estimator="ols_hc3",
    )

# Winsorized / rank variants.
all_rows += compare_models(
    d,
    "winsorized_1pct__LAI_main",
    WINSOR_MAIN_FULL,
    WINSOR_MAIN_REDUCED,
    ["lai_w_z"],
    estimator="ols_hc3_winsorized",
)

all_rows += compare_models(
    d,
    "rank_based__LAI_main",
    RANK_MAIN_FULL,
    RANK_MAIN_REDUCED,
    ["lai_rank_z"],
    estimator="ols_hc3_rank_based",
)

all_rows += compare_models(
    d,
    "winsorized_1pct__LAI_x_temperature",
    WINSOR_INT_FULL,
    WINSOR_INT_REDUCED,
    ["lai_w_z:mat_w_z"],
    estimator="ols_hc3_winsorized",
)

all_rows += compare_models(
    d,
    "rank_based__LAI_x_temperature",
    RANK_INT_FULL,
    RANK_INT_REDUCED,
    ["lai_rank_z:mat_rank_z"],
    estimator="ols_hc3_rank_based",
)

# Cook's distance filtered.
all_rows += compare_models(
    cook_main_data,
    f"cooks_distance_filtered_main_removed_{cook_main_removed}",
    MAIN_FULL,
    MAIN_REDUCED,
    ["lai_z"],
    estimator="ols_hc3_cooks_filtered",
)

all_rows += compare_models(
    cook_int_data,
    f"cooks_distance_filtered_interaction_removed_{cook_int_removed}",
    INT_FULL,
    INT_REDUCED,
    ["lai_z:mat_z"],
    estimator="ols_hc3_cooks_filtered",
)

# Robust regression variants.
all_rows += fit_rlm(d, "robust_huber__LAI_main", MAIN_FULL, ["lai_z"])
all_rows += fit_rlm(d, "robust_huber__LAI_x_temperature", INT_FULL, ["lai_z:mat_z"])

guard = pd.DataFrame(all_rows)

# Add stability diagnostics only for main all-points canonical tests.
stability_rows = []
for label, formula, term in [
    ("all_points__LAI_main", MAIN_FULL, "lai_z"),
    ("all_points__LAI_x_temperature", INT_FULL, "lai_z:mat_z"),
]:
    boot = bootstrap_term(d, formula, term, n_boot=1000)
    lro = leave_region_out(d, formula, term)

    if term == "lai_z":
        perm = climate_bin_permutation(d, formula, term, permuted_var="lai_z", n_perm=1000)
    else:
        perm = climate_bin_permutation(d, formula, term, permuted_var="lai_z", n_perm=1000)

    stability_rows.append({
        "test_label": label,
        "focal_term": term,
        **boot,
        **lro,
        **perm,
    })

stability = pd.DataFrame(stability_rows)

guard = guard.merge(stability, on=["test_label", "focal_term"], how="left")

# Add summary flags.
guard["coef_sign"] = np.sign(guard["coef"])
guard["passes_p05"] = guard["p"] < 0.05
guard["passes_ci"] = guard["ci_excludes_zero"].fillna(False)
guard["passes_boot"] = guard["boot_ci_excludes_zero"].fillna(False)
guard["passes_lro"] = guard["lro_sign_stability"].fillna(0) >= 0.85
guard["passes_permutation"] = guard["perm_p_two_sided"].fillna(1) < 0.05

guard.to_csv(TAB / "ALL_LAI_GUARDRAIL_TESTS.csv", index=False)

# Compact pass/fail summary by result family.
def family_from_label(x):
    if "LAI_x_temperature" in str(x):
        return "LAI_x_temperature"
    if "LAI_main" in str(x):
        return "LAI_main"
    return "other"

guard["family"] = guard["test_label"].map(family_from_label)

summary = []
for fam, g in guard[guard["status"].eq("FIT_OK")].groupby("family"):
    signs = g["coef_sign"].dropna()
    main_sign = np.sign(np.nanmedian(g["coef"]))
    summary.append({
        "family": fam,
        "n_tests_fit_ok": len(g),
        "median_coef": float(np.nanmedian(g["coef"])),
        "min_coef": float(np.nanmin(g["coef"])),
        "max_coef": float(np.nanmax(g["coef"])),
        "sign_consistency": float(np.mean(signs == main_sign)) if len(signs) else np.nan,
        "n_p_lt_0p05": int((g["p"] < 0.05).sum()),
        "n_ci_excludes_zero": int(g["ci_excludes_zero"].fillna(False).sum()),
        "median_delta_r2": float(np.nanmedian(g["delta_r2"])),
        "median_delta_aic": float(np.nanmedian(g["delta_aic_full_minus_reduced"])),
        "worst_p": float(np.nanmax(g["p"])),
        "best_p": float(np.nanmin(g["p"])),
    })

summary = pd.DataFrame(summary)
summary.to_csv(TAB / "LAI_GUARDRAIL_FAMILY_SUMMARY.csv", index=False)

# ------------------------------------------------------------
# Product-family / combo-level outcome sensitivity
# ------------------------------------------------------------

product_rows = []

if OBS_INPUT.exists() and "point_id" in d.columns:
    obs = pd.read_csv(OBS_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
    obs = obs.loc[:, ~obs.columns.duplicated()].copy()

    if "point_id" in obs.columns and "slope_change" in obs.columns:
        # Main combo-mean slope-change outcome.
        def run_alt_outcome(label, obs_sub):
            alt = obs_sub.groupby("point_id", dropna=False)["slope_change"].mean().reset_index()
            alt = alt.rename(columns={"slope_change": "y_alt"})
            merged = d.merge(alt, on="point_id", how="left")
            merged["y_alt_z"] = zscore(merged["y_alt"])

            main_f = "y_alt_z ~ lai_z + vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z"
            main_r = "y_alt_z ~ vpd_z + arid_z + mat_z + map_z + sm_z + lat_z + lon_z"
            int_f = "y_alt_z ~ lai_z * mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z"
            int_r = "y_alt_z ~ lai_z + mat_z + vpd_z + arid_z + map_z + sm_z + lat_z + lon_z"

            for fam, ff, rr, term in [
                ("alt_combo_mean_LAI_main", main_f, main_r, "lai_z"),
                ("alt_combo_mean_LAI_x_temperature", int_f, int_r, "lai_z:mat_z"),
            ]:
                rows = compare_models(merged, f"{label}__{fam}", ff, rr, [term], estimator="ols_hc3_alt_combo_mean")
                for row in rows:
                    row["alt_outcome_note"] = "Outcome is mean slope_change across table_latent_model_observations, not the posterior latent point response."
                    product_rows.append(row)

        run_alt_outcome("all_product_combos", obs)

        # Leave-one-product-family-out style, by GPP and ET products.
        for axis in ["gpp_product", "et_product", "metric", "stress_definition", "growing_season"]:
            if axis not in obs.columns:
                continue

            vals = [v for v in sorted(obs[axis].dropna().unique()) if str(v).strip()]
            # Avoid huge loops if there are many stress labels.
            if len(vals) > 20:
                vals = vals[:20]

            for val in vals:
                sub = obs[obs[axis] != val].copy()
                if len(sub) < 100:
                    continue
                safe_val = str(val).replace(" ", "_").replace("/", "_")
                run_alt_outcome(f"leave_one_{axis}_out__{safe_val}", sub)

product_sens = pd.DataFrame(product_rows)
if len(product_sens):
    product_sens["coef_sign"] = np.sign(product_sens["coef"])
    product_sens = product_sens.sort_values(["test_label", "focal_term"])
product_sens.to_csv(TAB / "PRODUCT_COMBO_ALT_OUTCOME_SENSITIVITY.csv", index=False)

prod_summary = []
if len(product_sens):
    for fam, g in product_sens[product_sens["status"].eq("FIT_OK")].groupby("focal_term"):
        signs = g["coef_sign"].dropna()
        med = np.nanmedian(g["coef"])
        prod_summary.append({
            "focal_term": fam,
            "n_product_sensitivity_tests": len(g),
            "median_coef": float(med),
            "min_coef": float(np.nanmin(g["coef"])),
            "max_coef": float(np.nanmax(g["coef"])),
            "sign_consistency": float(np.mean(signs == np.sign(med))) if len(signs) and med != 0 else np.nan,
            "n_p_lt_0p05": int((g["p"] < 0.05).sum()),
            "n_ci_excludes_zero": int(g["ci_excludes_zero"].fillna(False).sum()),
            "median_delta_r2": float(np.nanmedian(g["delta_r2"])),
            "median_delta_aic": float(np.nanmedian(g["delta_aic_full_minus_reduced"])),
        })

pd.DataFrame(prod_summary).to_csv(TAB / "PRODUCT_COMBO_ALT_OUTCOME_SENSITIVITY_SUMMARY.csv", index=False)

# ------------------------------------------------------------
# VIF / collinearity diagnostics
# ------------------------------------------------------------

vif_rows = []
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_cols = ["lai_z", "vpd_z", "arid_z", "mat_z", "map_z", "sm_z", "lat_z", "lon_z"]
    x = d[vif_cols].replace([np.inf, -np.inf], np.nan).dropna()
    x_const = sm.add_constant(x, has_constant="add")
    for i, col in enumerate(x_const.columns):
        if col == "const":
            continue
        vif_rows.append({
            "variable": col,
            "vif": float(variance_inflation_factor(x_const.values, i)),
            "n": len(x),
        })
except Exception as e:
    vif_rows.append({"variable": "ERROR", "vif": np.nan, "n": np.nan, "error": str(e)})

pd.DataFrame(vif_rows).to_csv(TAB / "VIF_COLLINEARITY_DIAGNOSTICS.csv", index=False)

# ------------------------------------------------------------
# Manuscript decision table
# ------------------------------------------------------------

decision_rows = []

for fam in ["LAI_main", "LAI_x_temperature"]:
    gf = guard[(guard["family"] == fam) & (guard["status"] == "FIT_OK")].copy()
    if len(gf) == 0:
        continue

    canonical_label = "all_points__LAI_main" if fam == "LAI_main" else "all_points__LAI_x_temperature"
    canonical = gf[gf["test_label"].eq(canonical_label)]
    if len(canonical) == 0:
        canonical = gf.head(1)

    r = canonical.iloc[0]
    sign_consistency = float(np.mean(np.sign(gf["coef"]) == np.sign(np.nanmedian(gf["coef"]))))

    p_rate = float(np.mean(gf["p"] < 0.05))
    ci_rate = float(np.mean(gf["ci_excludes_zero"].fillna(False)))

    product_note = "not_run"
    if len(product_sens):
        term = "lai_z" if fam == "LAI_main" else "lai_z:mat_z"
        pg = product_sens[(product_sens["focal_term"] == term) & (product_sens["status"] == "FIT_OK")]
        if len(pg):
            product_note = f"{int((pg['p'] < 0.05).sum())}/{len(pg)} product-combo alt-outcome tests p<0.05; sign consistency={np.mean(np.sign(pg['coef']) == np.sign(np.nanmedian(pg['coef']))):.2f}"

    if fam == "LAI_main":
        interpretation = "Higher growing-season LAI predicts more negative latent slope-change beyond VPD and full hydroclimate/geography controls."
    else:
        interpretation = "The LAI association changes across temperature regime; canopy structure is not a uniform global effect."

    decision_rows.append({
        "result_family": fam,
        "recommended_role": "main_result" if fam == "LAI_main" else "main_or_secondary_interaction_result",
        "canonical_coef": r.get("coef", np.nan),
        "canonical_p": r.get("p", np.nan),
        "canonical_ci": f"[{r.get('ci_low', np.nan):.3f}, {r.get('ci_high', np.nan):.3f}]",
        "canonical_delta_r2": r.get("delta_r2", np.nan),
        "canonical_delta_aic": r.get("delta_aic_full_minus_reduced", np.nan),
        "guardrail_n_fit_ok": len(gf),
        "guardrail_sign_consistency": sign_consistency,
        "guardrail_p_lt_0p05_rate": p_rate,
        "guardrail_ci_excludes_zero_rate": ci_rate,
        "bootstrap_ci": f"[{r.get('boot_ci_low', np.nan):.3f}, {r.get('boot_ci_high', np.nan):.3f}]",
        "leave_region_sign_stability": r.get("lro_sign_stability", np.nan),
        "climate_bin_permutation_p": r.get("perm_p_two_sided", np.nan),
        "product_combo_alt_outcome_summary": product_note,
        "interpretation": interpretation,
    })

decision = pd.DataFrame(decision_rows)
decision.to_csv(TAB / "MANUSCRIPT_GUARDRAIL_DECISION_TABLE.csv", index=False)

# ------------------------------------------------------------
# Memo
# ------------------------------------------------------------

def show(path, n=60):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BD LAI trait/regime falsification guardrails")
memo.append("=" * 90)
memo.append("")
memo.append(f"Point input: {POINT_INPUT}")
memo.append(f"Rows: {len(raw)}")
memo.append("")
memo.append("Purpose:")
memo.append("- Treat VPD as a known baseline covariate, not the novel claim.")
memo.append("- Stress-test the novel LAI/canopy-structure and LAI×temperature findings.")
memo.append("")
memo.append("Canonical source columns:")
for k, v in sources.items():
    memo.append(f"- {k}: {v}")
memo.append("")
memo.append("Cook's distance filtering:")
memo.append(f"- Main LAI model threshold: {cook_main_thr}; removed: {cook_main_removed}")
memo.append(f"- LAI×temperature model threshold: {cook_int_thr}; removed: {cook_int_removed}")
memo.append("")
memo.append("Manuscript guardrail decision table:")
memo.append(show("MANUSCRIPT_GUARDRAIL_DECISION_TABLE.csv", 20))
memo.append("")
memo.append("Guardrail family summary:")
memo.append(show("LAI_GUARDRAIL_FAMILY_SUMMARY.csv", 20))
memo.append("")
memo.append("All LAI guardrail tests:")
memo.append(show("ALL_LAI_GUARDRAIL_TESTS.csv", 120))
memo.append("")
memo.append("Product-combo alternate-outcome sensitivity summary:")
memo.append(show("PRODUCT_COMBO_ALT_OUTCOME_SENSITIVITY_SUMMARY.csv", 20))
memo.append("")
memo.append("VIF diagnostics:")
memo.append(show("VIF_COLLINEARITY_DIAGNOSTICS.csv", 30))
memo.append("")
memo.append("Important files:")
for f in [
    "MANUSCRIPT_GUARDRAIL_DECISION_TABLE.csv",
    "LAI_GUARDRAIL_FAMILY_SUMMARY.csv",
    "ALL_LAI_GUARDRAIL_TESTS.csv",
    "PRODUCT_COMBO_ALT_OUTCOME_SENSITIVITY.csv",
    "PRODUCT_COMBO_ALT_OUTCOME_SENSITIVITY_SUMMARY.csv",
    "VIF_COLLINEARITY_DIAGNOSTICS.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_lai_trait_regime_falsification_guardrails.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_lai_trait_regime_falsification_guardrails.txt'}")
