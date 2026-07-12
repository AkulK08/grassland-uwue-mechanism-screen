from pathlib import Path
from datetime import datetime
import json
import math
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise ImportError("Need scikit-learn. Run: pip install scikit-learn") from e

OUT = Path("results/stage1b6aj_clean_c4_model_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260704
rng = np.random.default_rng(SEED)
N_BOOT = 500

JOINED = Path("results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv")
if not JOINED.exists():
    raise FileNotFoundError(f"Missing C4 joined point table: {JOINED}. Run stage1b6ai first.")

df = pd.read_csv(JOINED)

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def first_existing(cols, candidates):
    lut = {norm(c): c for c in cols}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None

def z(s):
    s = to_num(s)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def p_norm_from_t(t):
    if not np.isfinite(t):
        return np.nan
    return float(math.erfc(abs(t) / math.sqrt(2)))

def bh_q(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order]
    m = len(ranked)
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[order] = np.minimum(qv, 1)
    return q

def ols(df, y, xs):
    d = df[[y] + xs].copy()
    for c in d.columns:
        d[c] = to_num(d[c])
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < max(25, len(xs) + 6):
        return None

    yv = d[y].to_numpy(float)
    X_parts = [np.ones(len(d))]
    kept = []
    for c in xs:
        zv = z(d[c]).to_numpy(float)
        if np.isfinite(zv).all() and np.nanstd(zv) > 0:
            X_parts.append(zv)
            kept.append(c)

    if "c4_fraction" not in kept:
        return None

    X = np.column_stack(X_parts)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ beta
    resid = yv - pred
    n = len(yv)
    k = X.shape[1]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1-r2)*(n-1)/max(1, n-k) if np.isfinite(r2) else np.nan

    sigma2 = rss / max(1, n-k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)

    rows = []
    for term, b, s in zip(["intercept"] + kept, beta, se):
        t = b / s if np.isfinite(s) and s != 0 else np.nan
        rows.append({
            "term": term,
            "coef_standardized": float(b),
            "se": float(s) if np.isfinite(s) else np.nan,
            "t_normal_approx": float(t) if np.isfinite(t) else np.nan,
            "p_normal_approx": p_norm_from_t(t),
        })

    return {
        "n": n,
        "k": k,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coef_table": pd.DataFrame(rows),
        "fit_index": d.index,
        "resid": resid,
        "pred": pred,
        "predictors_used": kept,
        "fit_data": d,
    }

def make_pc(data, cols, name):
    use = [c for c in cols if c in data.columns and to_num(data[c]).notna().sum() >= 40 and to_num(data[c]).nunique(dropna=True) > 2]
    if len(use) == 0:
        return data, None, []
    d = data[use].apply(to_num)
    med = d.median()
    d = d.fillna(med)
    X = StandardScaler().fit_transform(d)
    pc = PCA(n_components=1).fit_transform(X)[:, 0]
    data[name] = pc
    return data, name, use

# ---------------------------------------------------------------------
# Clean covariates: remove duplicates and collapse collinear blocks.
# ---------------------------------------------------------------------
if "c4_fraction" not in df.columns:
    raise ValueError("Joined C4 table lacks c4_fraction.")

df["c4_fraction"] = to_num(df["c4_fraction"])

# Climate dryness axis: use aridity, mean VPD, precip/temp if present, but collapse to PC1.
climate_cols = []
for candidates in [
    ["aridity", "aridity_index"],
    ["mean_vpd", "baseline_vpd"],
    ["mean_annual_temperature", "mean_temperature"],
    ["mean_annual_precipitation", "mean_precipitation"],
]:
    c = first_existing(df.columns, candidates)
    if c and c not in climate_cols:
        climate_cols.append(c)

# Soil texture axis: use sand/clay only if present; including all three causes perfect/compositional collinearity.
soil_cols = []
for candidates in [
    ["soil_sand", "sand"],
    ["soil_clay", "clay"],
]:
    c = first_existing(df.columns, candidates)
    if c and c not in soil_cols:
        soil_cols.append(c)

# Vegetation structure: choose one LAI variable.
lai_col = first_existing(df.columns, ["growing_season_mean_lai", "mean_lai", "lai", "lai_max"])

# Observed water: choose one soil moisture variable.
sm_col = first_existing(df.columns, ["mean_soil_moisture", "baseline_soil_moisture", "soil_moisture", "rootzone_soil_moisture"])

# Rooting depth is a sensitivity covariate, not primary C4 control.
root_col = first_existing(df.columns, ["rooting_depth", "root_depth", "rooting_zone_storage"])

df, climate_pc, climate_used = make_pc(df, climate_cols, "climate_dryness_PC1")
df, soil_pc, soil_used = make_pc(df, soil_cols, "soil_texture_PC1")

response_candidates = [
    "latent_post_slope",
    "p_threshold_like",
    "latent_slope_change",
    "p_satbreak",
    "latent_satbreak_probability",
]
responses = [r for r in response_candidates if r in df.columns]

if not responses:
    raise ValueError("No response metrics found in joined C4 table.")

base_controls = []
if climate_pc:
    base_controls.append(climate_pc)
if soil_pc:
    base_controls.append(soil_pc)
if lai_col:
    base_controls.append(lai_col)
if sm_col:
    base_controls.append(sm_col)

root_controls = base_controls.copy()
if root_col:
    root_controls.append(root_col)

model_specs = {
    "minimal_c4_only": ["c4_fraction"],
    "clean_controls_no_rooting": ["c4_fraction"] + base_controls,
    "clean_controls_with_rooting_sensitivity": ["c4_fraction"] + root_controls,
}

# Optional: only grassland/savanna-like rows as sensitivity.
if "eco_biome" in df.columns:
    grass_mask = df["eco_biome"].astype(str).str.contains("Grassland|Savanna|Shrubland", case=False, na=False)
else:
    grass_mask = pd.Series(True, index=df.index)

sample_specs = {
    "all_available_points": pd.Series(True, index=df.index),
    "grassland_savanna_shrubland_only": grass_mask,
}

# If tower-ranked ET/product indicator exists, use it; usually point table is product-collapsed, so this may not apply.
if "uses_tower_ranked_et" in df.columns:
    sample_specs["tower_ranked_et_only"] = df["uses_tower_ranked_et"].astype(bool)

rows = []
boot_rows = []
loo_rows = []
vif_rows = []
ridge_rows = []

def calc_vif(data, predictors):
    dd = data[predictors].copy()
    for c in predictors:
        dd[c] = to_num(dd[c])
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < len(predictors) + 5:
        return []
    out = []
    for c in predictors:
        others = [x for x in predictors if x != c]
        if not others:
            continue
        f = ols(dd, c, others)
        if f is None or not np.isfinite(f["r2"]):
            vif = np.nan
        else:
            vif = 1 / max(1e-9, 1 - f["r2"])
        out.append({"term": c, "vif": vif})
    return out

total = len(responses) * len(sample_specs) * len(model_specs)
iterator = []
for response in responses:
    for sample_name, mask in sample_specs.items():
        for model_name, predictors in model_specs.items():
            iterator.append((response, sample_name, mask, model_name, predictors))

for response, sample_name, mask, model_name, predictors in tqdm(iterator, desc="Clean C4 model grid", unit="model") if tqdm else iterator:
    sub = df[mask].copy()
    predictors = [p for p in predictors if p in sub.columns and to_num(sub[p]).notna().sum() >= 25 and to_num(sub[p]).nunique(dropna=True) > 2]
    if "c4_fraction" not in predictors:
        continue

    fit = ols(sub, response, predictors)
    if fit is None:
        continue

    coef = fit["coef_table"].copy()
    coef["response"] = response
    coef["sample"] = sample_name
    coef["model"] = model_name
    coef["n"] = fit["n"]
    coef["r2"] = fit["r2"]
    coef["adj_r2"] = fit["adj_r2"]
    coef["rmse"] = fit["rmse"]
    coef["predictors_used"] = ";".join(fit["predictors_used"])
    rows.append(coef)

    # VIF audit for predictors.
    for vr in calc_vif(fit["fit_data"], fit["predictors_used"]):
        vr.update({"response": response, "sample": sample_name, "model": model_name, "n": fit["n"]})
        vif_rows.append(vr)

    # Ridge stability: c4 coefficient under ridge CV.
    fit_data = fit["fit_data"].copy()
    yv = fit_data[response].to_numpy(float)
    X = fit_data[fit["predictors_used"]].apply(to_num)
    Xz = StandardScaler().fit_transform(X)
    alphas = np.logspace(-3, 3, 25)
    ridge = RidgeCV(alphas=alphas).fit(Xz, yv)
    c4_idx = fit["predictors_used"].index("c4_fraction")
    ridge_rows.append({
        "response": response,
        "sample": sample_name,
        "model": model_name,
        "n": fit["n"],
        "alpha": float(ridge.alpha_),
        "ridge_c4_coef": float(ridge.coef_[c4_idx]),
        "ols_c4_coef": float(coef.loc[coef["term"] == "c4_fraction", "coef_standardized"].iloc[0]),
    })

    # Spatial/block structure.
    fit_df = sub.loc[fit["fit_index"]].copy()
    if "eco_biome" in fit_df.columns:
        block_col = "eco_biome"
    elif "eco_realm" in fit_df.columns:
        block_col = "eco_realm"
    elif "lat" in fit_df.columns and "lon" in fit_df.columns:
        fit_df["spatial_block_10deg"] = (
            np.floor(to_num(fit_df["lat"]) / 10).astype("Int64").astype(str)
            + "_"
            + np.floor(to_num(fit_df["lon"]) / 10).astype("Int64").astype(str)
        )
        block_col = "spatial_block_10deg"
    else:
        fit_df["all_block"] = "all"
        block_col = "all_block"

    blocks = list(fit_df[block_col].dropna().unique())
    c4_coef = float(coef.loc[coef["term"] == "c4_fraction", "coef_standardized"].iloc[0])
    c4_p = float(coef.loc[coef["term"] == "c4_fraction", "p_normal_approx"].iloc[0])

    # Leave-one-block-out.
    loo_coefs = []
    for b in blocks:
        train = fit_df[fit_df[block_col] != b].copy()
        if len(train) < 25:
            continue
        lf = ols(train, response, fit["predictors_used"])
        if lf is None:
            continue
        lt = lf["coef_table"]
        if (lt["term"] == "c4_fraction").any():
            val = float(lt.loc[lt["term"] == "c4_fraction", "coef_standardized"].iloc[0])
            loo_coefs.append(val)
            loo_rows.append({
                "response": response,
                "sample": sample_name,
                "model": model_name,
                "left_out_block": str(b),
                "n_train": lf["n"],
                "c4_coef_standardized": val,
            })

    # Block bootstrap.
    boot_coefs = []
    if len(blocks) >= 3:
        for _ in range(N_BOOT):
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            bd = pd.concat([fit_df[fit_df[block_col] == b] for b in sampled], ignore_index=True)
            bf = ols(bd, response, fit["predictors_used"])
            if bf is not None:
                bt = bf["coef_table"]
                if (bt["term"] == "c4_fraction").any():
                    boot_coefs.append(float(bt.loc[bt["term"] == "c4_fraction", "coef_standardized"].iloc[0]))

    boot_rows.append({
        "response": response,
        "sample": sample_name,
        "model": model_name,
        "n": fit["n"],
        "block_col": block_col,
        "n_blocks": len(blocks),
        "c4_coef_standardized": c4_coef,
        "c4_p_normal_approx": c4_p,
        "bootstrap_n": len(boot_coefs),
        "bootstrap_median": float(np.median(boot_coefs)) if boot_coefs else np.nan,
        "bootstrap_p025": float(np.quantile(boot_coefs, 0.025)) if boot_coefs else np.nan,
        "bootstrap_p975": float(np.quantile(boot_coefs, 0.975)) if boot_coefs else np.nan,
        "loo_n": len(loo_coefs),
        "loo_sign_stability": float(np.mean(np.sign(loo_coefs) == np.sign(c4_coef))) if loo_coefs else np.nan,
    })

coef_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
if len(coef_table):
    coef_table["bh_q_normal_approx"] = bh_q(coef_table["p_normal_approx"].to_numpy(float))

boot_table = pd.DataFrame(boot_rows)
loo_table = pd.DataFrame(loo_rows)
vif_table = pd.DataFrame(vif_rows)
ridge_table = pd.DataFrame(ridge_rows)

coef_table.to_csv(TAB / "Table_PRODUCT03aw_clean_c4_coefficients.csv", index=False)
boot_table.to_csv(TAB / "Table_PRODUCT03ax_clean_c4_block_bootstrap.csv", index=False)
loo_table.to_csv(TAB / "Table_PRODUCT03ay_clean_c4_leave_one_block_out.csv", index=False)
vif_table.to_csv(TAB / "Table_PRODUCT03az_clean_c4_vif_audit.csv", index=False)
ridge_table.to_csv(TAB / "Table_PRODUCT03ba_clean_c4_ridge_stability.csv", index=False)

# Decision table.
if len(coef_table):
    focal = coef_table[coef_table["term"] == "c4_fraction"].copy()
    focal = focal.merge(
        boot_table[["response", "sample", "model", "bootstrap_p025", "bootstrap_p975", "loo_sign_stability", "bootstrap_n", "n_blocks"]],
        on=["response", "sample", "model"],
        how="left"
    )
    focal = focal.merge(
        ridge_table[["response", "sample", "model", "ridge_c4_coef", "alpha"]],
        on=["response", "sample", "model"],
        how="left"
    )

    focal["ci_excludes_zero"] = (
        focal["bootstrap_p025"].notna()
        & focal["bootstrap_p975"].notna()
        & (focal["bootstrap_p025"] * focal["bootstrap_p975"] > 0)
    )
    focal["ridge_same_sign"] = (
        focal["ridge_c4_coef"].notna()
        & (np.sign(focal["ridge_c4_coef"]) == np.sign(focal["coef_standardized"]))
    )
    focal["passes_clean_screen"] = (
        (focal["p_normal_approx"] <= 0.05)
        & (focal["loo_sign_stability"].fillna(0) >= 0.80)
        & (focal["ci_excludes_zero"])
        & (focal["ridge_same_sign"])
    )
    focal = focal.sort_values(
        ["passes_clean_screen", "p_normal_approx", "ci_excludes_zero", "loo_sign_stability"],
        ascending=[False, True, False, False]
    )
else:
    focal = pd.DataFrame()

focal.to_csv(TAB / "Table_PRODUCT03bb_clean_c4_decision_by_response.csv", index=False)

if len(focal) and focal["passes_clean_screen"].any():
    best = focal[focal["passes_clean_screen"]].iloc[0]
    verdict = "C4_PASSES_CLEAN_PRE_SPECIFIED_SCREEN"
    paper_fork = "ecological_C3C4_mechanism_paper_candidate"
elif len(focal):
    best = focal.iloc[0]
    verdict = "C4_DOES_NOT_PASS_CLEAN_PRE_SPECIFIED_SCREEN"
    paper_fork = "methods_identifiability_paper_preferred"
else:
    best = None
    verdict = "NO_CLEAN_C4_MODELS_RAN"
    paper_fork = "blocked"

if best is not None:
    best_claim = (
        f"Best C4 result: response={best['response']}, sample={best['sample']}, model={best['model']}, "
        f"n={int(best['n'])}, standardized C4 coef={best['coef_standardized']:.3f}, "
        f"p={best['p_normal_approx']:.4g}, q={best['bh_q_normal_approx']:.4g}, "
        f"bootstrap CI=[{best['bootstrap_p025']:.3f}, {best['bootstrap_p975']:.3f}], "
        f"LOO sign stability={best['loo_sign_stability']:.3f}, ridge_same_sign={bool(best['ridge_same_sign'])}."
    )
else:
    best_claim = "No clean C4 model ran."

decision = {
    "generated": now(),
    "stage": "1B.6AJ_clean_C4_model_lock",
    "verdict": verdict,
    "paper_fork": paper_fork,
    "best_claim": best_claim,
    "c4_points_total": int(len(df)),
    "c4_points_nonmissing": int(df["c4_fraction"].notna().sum()),
    "responses_tested": responses,
    "climate_pc_inputs": climate_used,
    "soil_pc_inputs": soil_used,
    "lai_control": lai_col,
    "soil_moisture_control": sm_col,
    "rooting_depth_sensitivity_control": root_col,
    "interpretation": (
        "If C4 fails this clean screen, do not keep hunting C4 variants. "
        "Use C4 as a pre-specified failed mechanism test and pivot to product-identifiability/tower-ranking paper."
    ),
}
Path(TAB / "STAGE1B6AJ_CLEAN_C4_MODEL_LOCK_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

# Figure.
fig_status = []
if plt is not None and len(focal):
    try:
        plot = focal.head(20).copy()
        labels = plot["response"].astype(str) + " | " + plot["sample"].astype(str) + " | " + plot["model"].astype(str)
        plt.figure(figsize=(12, max(6, len(plot)*0.4)))
        plt.barh(labels[::-1], plot["coef_standardized"][::-1])
        plt.axvline(0, linestyle="--")
        plt.xlabel("Standardized C4 coefficient")
        plt.ylabel("Response | sample | model")
        plt.title("Clean pre-specified C4 effect screen")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT03e_clean_c4_effect_screen.png", dpi=220)
        plt.close()
        fig_status.append("clean_c4_effect_screen")
    except Exception as e:
        fig_status.append(f"figure_failed:{e}")

report = []
report.append("# Stage 1B.6AJ clean C4 model lock")
report.append("")
report.append(f"Generated: {decision['generated']}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```json")
report.append(json.dumps(decision, indent=2))
report.append("```")
report.append("")
report.append("## Why this stage matters")
report.append("")
report.append("The previous C4 model included duplicated climate variables and compositional soil fractions together, producing extreme standard errors and an ill-conditioned control set. This stage uses a cleaner pre-specified control design: climate PC1, soil-texture PC1, one LAI variable, one observed soil-moisture variable, and rooting depth only as a sensitivity control.")
report.append("")
report.append("## C4 decision rows")
report.append("")
report.append("```text")
report.append(focal.head(80).to_string(index=False) if len(focal) else "No clean C4 decision rows.")
report.append("```")
report.append("")
report.append("## VIF audit")
report.append("")
report.append("```text")
report.append(vif_table.head(100).to_string(index=False) if len(vif_table) else "No VIF rows.")
report.append("```")
report.append("")
report.append("## Ridge stability")
report.append("")
report.append("```text")
report.append(ridge_table.head(100).to_string(index=False) if len(ridge_table) else "No ridge rows.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
if verdict == "C4_PASSES_CLEAN_PRE_SPECIFIED_SCREEN":
    report.append("C4 survives the clean pre-specified screen. Next step: upgrade to a full mixed/hierarchical model with ecoregion random effects and write the ecology-mechanism paper.")
elif verdict == "C4_DOES_NOT_PASS_CLEAN_PRE_SPECIFIED_SCREEN":
    report.append("C4 does not survive the clean pre-specified screen. The honest next paper is product identifiability / satellite WUE uncertainty, with C4 reported as a pre-specified failed mechanism test.")
else:
    report.append("Clean C4 model did not run; inspect missing data and joined C4 table.")
report.append("")
report.append("## Figures")
report.append("; ".join(fig_status) if fig_status else "No figures written.")

report_text = "\n".join(report)
(TXT / "STAGE1B6AJ_CLEAN_C4_MODEL_LOCK_REPORT.md").write_text(report_text, encoding="utf-8")

print(report_text)
print("")
print("WROTE", TAB / "Table_PRODUCT03aw_clean_c4_coefficients.csv")
print("WROTE", TAB / "Table_PRODUCT03ax_clean_c4_block_bootstrap.csv")
print("WROTE", TAB / "Table_PRODUCT03ay_clean_c4_leave_one_block_out.csv")
print("WROTE", TAB / "Table_PRODUCT03az_clean_c4_vif_audit.csv")
print("WROTE", TAB / "Table_PRODUCT03ba_clean_c4_ridge_stability.csv")
print("WROTE", TAB / "Table_PRODUCT03bb_clean_c4_decision_by_response.csv")
print("WROTE", TAB / "STAGE1B6AJ_CLEAN_C4_MODEL_LOCK_DECISION.json")
print("WROTE", TXT / "STAGE1B6AJ_CLEAN_C4_MODEL_LOCK_REPORT.md")
