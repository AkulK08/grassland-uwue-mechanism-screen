from pathlib import Path
from datetime import datetime
import json
import itertools
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6u_trait_climate_soil_spatial_model")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6u")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

INPUT = Path("data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv")
GROUP_SCAN = Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02da_spatial_biome_group_signal_scan.csv")
CONT_SCAN = Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv")

OUTCOME = "satellite_limitation_mean_fraction"
SEED = 20260629
rng = np.random.default_rng(SEED)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def z(s):
    s = to_num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def ols_fit(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    n, k = X.shape
    aicc = np.nan
    if n > k + 1 and rss > 0:
        aic = n * np.log(rss / n) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return beta, pred, rss, r2, aicc

def loocv_rmse(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    preds = []
    actuals = []
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        try:
            beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
            preds.append(float(X[i] @ beta))
            actuals.append(float(y[i]))
        except Exception:
            pass
    if not preds:
        return np.nan
    preds = np.array(preds)
    actuals = np.array(actuals)
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))

def spearman_perm_p(x, y, n_perm=5000):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    ok = x.notna() & y.notna()
    x = x[ok].to_numpy()
    y = y[ok].to_numpy()
    n = len(x)
    if n < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    obs = pd.Series(x).corr(pd.Series(y), method="spearman")
    vals = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        vals.append(pd.Series(x).corr(pd.Series(yp), method="spearman"))
    vals = np.array(vals, dtype=float)
    p = float((np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1))
    return float(obs), p

def add_numeric_candidate(df, col, family, label=None):
    if col not in df.columns:
        return None
    x = z(df[col])
    if x.notna().sum() < 5 or x.nunique(dropna=True) < 2:
        return None
    return {
        "name": label or col,
        "source_col": col,
        "family": family,
        "type": "numeric_z",
        "x": x,
    }

def add_binary_level_candidate(df, col, level, family):
    x = (df[col].astype(str) == str(level)).astype(float)
    if x.sum() < 2 or (len(x) - x.sum()) < 2:
        return None
    return {
        "name": f"{col}=={level}",
        "source_col": col,
        "family": family,
        "type": "binary_level",
        "x": x,
    }

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input table: {INPUT}")

df = pd.read_csv(INPUT)
df["point_id"] = df["point_id"].astype(str)
df[OUTCOME] = to_num(df[OUTCOME])

# Keep one row per site.
df = df.drop_duplicates("point_id").copy()
df = df.dropna(subset=[OUTCOME]).copy()

# Save model input.
df.to_csv(DATA / "trait_spatial_model_input_13_sites.csv", index=False)

candidates = []

numeric_specs = [
    ("lat", "spatial"),
    ("lon", "spatial"),
    ("abs_lat", "spatial"),
    ("mean_vpd", "climate"),
    ("mean_soil_moisture", "climate"),
    ("mean_annual_temperature", "climate"),
    ("mean_annual_precipitation", "climate"),
    ("mean_lai", "canopy"),
    ("growing_season_mean_lai", "canopy"),
    ("aridity_index", "climate"),
    ("aridity", "climate"),
    ("soil_sand", "soil"),
    ("soil_clay", "soil"),
    ("soil_silt", "soil"),
    ("soil_sand_mean", "soil"),
    ("soil_clay_mean", "soil"),
    ("soil_silt_mean", "soil"),
    ("rooting_depth", "trait"),
    ("p50", "trait"),
    ("isohydricity", "trait"),
]

for col, family in numeric_specs:
    cand = add_numeric_candidate(df, col, family)
    if cand is not None:
        candidates.append(cand)

group_specs = [
    ("us_vs_nonus", "spatial_group"),
    ("latitude_band_handbuilt", "spatial_group"),
    ("longitude_sector_handbuilt", "spatial_group"),
    ("broad_region_handbuilt", "spatial_group"),
    ("eco_biome", "biome"),
    ("eco_realm", "biome"),
    ("hydroclimatic_regime", "climate_group"),
    ("aridity_quartile", "climate_group"),
    ("mean_vpd_quartile", "climate_group"),
]

for col, family in group_specs:
    if col not in df.columns:
        continue
    vals = df[col].dropna().astype(str)
    for level, n in vals.value_counts().items():
        if n >= 2:
            cand = add_binary_level_candidate(df, col, level, family)
            if cand is not None:
                candidates.append(cand)

rows = []
y = df[OUTCOME].to_numpy(dtype=float)

# Intercept-only baseline.
X0 = np.ones((len(y), 1))
_, pred0, rss0, r20, aicc0 = ols_fit(X0, y)
baseline_rmse = float(np.sqrt(np.mean((y - pred0) ** 2)))
baseline_loocv = loocv_rmse(X0, y)

baseline = {
    "model_name": "intercept_only",
    "model_family": "baseline",
    "n": int(len(y)),
    "k": 1,
    "predictors": "",
    "rss": rss0,
    "r2": r20,
    "aicc": aicc0,
    "rmse": baseline_rmse,
    "loocv_rmse": baseline_loocv,
    "delta_loocv_vs_intercept": 0.0,
}

rows.append(baseline)

# Univariate models.
for cand in candidates:
    sub = pd.DataFrame({
        "y": df[OUTCOME],
        "x": cand["x"],
    }).dropna()
    if len(sub) < 6 or sub["x"].nunique() < 2:
        continue

    X = np.column_stack([np.ones(len(sub)), sub["x"].to_numpy(dtype=float)])
    yy = sub["y"].to_numpy(dtype=float)
    beta, pred, rss, r2, aicc = ols_fit(X, yy)
    rmse = float(np.sqrt(np.mean((yy - pred) ** 2)))
    lrmse = loocv_rmse(X, yy)
    rho, p_perm = spearman_perm_p(sub["x"], sub["y"], n_perm=2000)

    rows.append({
        "model_name": cand["name"],
        "model_family": cand["family"],
        "type": cand["type"],
        "n": int(len(sub)),
        "k": 2,
        "predictors": cand["name"],
        "coef_intercept": float(beta[0]),
        "coef_1": float(beta[1]),
        "rss": rss,
        "r2": r2,
        "aicc": aicc,
        "rmse": rmse,
        "loocv_rmse": lrmse,
        "delta_loocv_vs_intercept": float(lrmse - baseline_loocv) if np.isfinite(lrmse) and np.isfinite(baseline_loocv) else np.nan,
        "spearman_r": rho,
        "spearman_perm_p": p_perm,
    })

# Tiny-n-safe two-predictor models: only from different families, max k=3.
pair_rows = []
for c1, c2 in itertools.combinations(candidates, 2):
    if c1["family"] == c2["family"]:
        continue

    tmp = pd.DataFrame({
        "y": df[OUTCOME],
        "x1": c1["x"],
        "x2": c2["x"],
    }).dropna()
    if len(tmp) < 8:
        continue
    if tmp[["x1", "x2"]].corr().abs().iloc[0, 1] > 0.90:
        continue

    X = np.column_stack([
        np.ones(len(tmp)),
        tmp["x1"].to_numpy(dtype=float),
        tmp["x2"].to_numpy(dtype=float),
    ])
    yy = tmp["y"].to_numpy(dtype=float)
    beta, pred, rss, r2, aicc = ols_fit(X, yy)
    rmse = float(np.sqrt(np.mean((yy - pred) ** 2)))
    lrmse = loocv_rmse(X, yy)

    pair_rows.append({
        "model_name": f"{c1['name']} + {c2['name']}",
        "model_family": f"{c1['family']}+{c2['family']}",
        "type": "two_predictor_tiny_n_screen",
        "n": int(len(tmp)),
        "k": 3,
        "predictors": f"{c1['name']};{c2['name']}",
        "coef_intercept": float(beta[0]),
        "coef_1": float(beta[1]),
        "coef_2": float(beta[2]),
        "rss": rss,
        "r2": r2,
        "aicc": aicc,
        "rmse": rmse,
        "loocv_rmse": lrmse,
        "delta_loocv_vs_intercept": float(lrmse - baseline_loocv) if np.isfinite(lrmse) and np.isfinite(baseline_loocv) else np.nan,
    })

rows.extend(pair_rows)

model_df = pd.DataFrame(rows)
model_df["delta_aicc_vs_best"] = model_df["aicc"] - model_df["aicc"].min(skipna=True)
model_df = model_df.sort_values(["loocv_rmse", "aicc"], ascending=[True, True])
model_df.to_csv(TAB / "Table_PRODUCT02df_trait_spatial_model_screen.csv", index=False)

# Family summary.
fam = (
    model_df[model_df["model_family"].ne("baseline")]
    .groupby("model_family", dropna=False)
    .agg(
        n_models=("model_name", "size"),
        best_loocv_rmse=("loocv_rmse", "min"),
        best_r2=("r2", "max"),
        best_aicc=("aicc", "min"),
    )
    .reset_index()
    .sort_values("best_loocv_rmse")
)
fam.to_csv(TAB / "Table_PRODUCT02dg_trait_spatial_family_summary.csv", index=False)

# Best model predictions.
best = model_df.iloc[0].to_dict()
best_name = best["model_name"]

if best_name == "intercept_only":
    pred_df = df[["point_id", OUTCOME]].copy()
    pred_df["best_model_prediction"] = float(df[OUTCOME].mean())
else:
    preds = None
    # Reconstruct best model from predictors.
    pred_names = str(best["predictors"]).split(";") if pd.notna(best.get("predictors")) and str(best.get("predictors")) else []
    chosen = [c for c in candidates if c["name"] in pred_names]

    tmp = pd.DataFrame({"point_id": df["point_id"], "y": df[OUTCOME]})
    for i, c in enumerate(chosen):
        tmp[f"x{i+1}"] = c["x"]

    tmp2 = tmp.dropna().copy()
    X = np.column_stack([np.ones(len(tmp2))] + [tmp2[f"x{i+1}"].to_numpy(dtype=float) for i in range(len(chosen))])
    yy = tmp2["y"].to_numpy(dtype=float)
    beta, pred, *_ = ols_fit(X, yy)

    pred_df = tmp2[["point_id", "y"]].copy().rename(columns={"y": OUTCOME})
    pred_df["best_model_prediction"] = pred
    pred_df["best_model_residual"] = pred_df[OUTCOME] - pred_df["best_model_prediction"]

pred_df.to_csv(TAB / "Table_PRODUCT02dh_best_model_site_predictions.csv", index=False)

# Interpretation lock.
top_uni = model_df[(model_df["k"] == 2) & (model_df["model_family"] != "baseline")].head(10).copy()
top_all = model_df.head(15).copy()

best_family = str(model_df.iloc[0]["model_family"])
best_loocv = float(model_df.iloc[0]["loocv_rmse"])
loocv_gain = float(baseline_loocv - best_loocv) if np.isfinite(best_loocv) and np.isfinite(baseline_loocv) else np.nan

if best_family == "baseline" or (np.isfinite(loocv_gain) and loocv_gain <= 0):
    verdict = "NO_PREDICTOR_BEATS_INTERCEPT_IN_LOOCV"
    claim = "Do not claim a mechanism; report heterogeneity descriptively."
elif any(x in best_family for x in ["spatial", "biome", "climate_group"]):
    verdict = "SPATIAL_OR_BIOME_TERMS_EXPLAIN_HETEROGENEITY_BEST_EXPLORATORY"
    claim = "Report exploratory spatial/biome heterogeneity; avoid causal mechanism language."
elif any(x in best_family for x in ["trait", "climate", "soil", "canopy"]):
    verdict = "TRAIT_CLIMATE_SOIL_COVARIATE_SIGNAL_PRESENT_EXPLORATORY"
    claim = "Report exploratory covariate association with uncertainty; avoid causal proof."
else:
    verdict = "MIXED_EXPLORATORY_SIGNAL"
    claim = "Report mixed exploratory heterogeneity."

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_sites": int(len(df)),
    "n_candidate_predictors": int(len(candidates)),
    "n_models_screened": int(len(model_df)),
    "baseline_loocv_rmse": baseline_loocv,
    "best_model": str(model_df.iloc[0]["model_name"]),
    "best_model_family": best_family,
    "best_model_loocv_rmse": best_loocv,
    "loocv_gain_vs_intercept": loocv_gain,
    "verdict": verdict,
    "recommended_claim": claim,
    "blocking_next_stage": False,
    "next_stage": "FINAL_CLAIM_LOCK_AND_FIGURE_TABLE_EXPORT",
}])
decision.to_csv(TAB / "Table_PRODUCT02di_trait_spatial_model_decision.csv", index=False)

report = []
report.append("# Stage 1B.6U trait/climate/soil/spatial model screen")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Top models")
report.append("")
report.append("```text")
report.append(top_all.to_string(index=False))
report.append("```")
report.append("")
report.append("## Top univariate models")
report.append("")
report.append("```text")
report.append(top_uni.to_string(index=False))
report.append("```")
report.append("")
report.append("## Family summary")
report.append("")
report.append("```text")
report.append(fam.to_string(index=False) if len(fam) else "No family summary.")
report.append("```")
report.append("")
report.append("## Best model site predictions")
report.append("")
report.append("```text")
report.append(pred_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("This is a tiny-n mechanism screen. Use it to support heterogeneity and model-prioritization language, not causal proof.")
report.append("")

(TXT / "STAGE1B6U_TRAIT_CLIMATE_SOIL_SPATIAL_MODEL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6U_trait_climate_soil_spatial_model",
    "status": str(decision["verdict"].iloc[0]),
    "outputs": {
        "input": str(DATA / "trait_spatial_model_input_13_sites.csv"),
        "model_screen": str(TAB / "Table_PRODUCT02df_trait_spatial_model_screen.csv"),
        "family_summary": str(TAB / "Table_PRODUCT02dg_trait_spatial_family_summary.csv"),
        "site_predictions": str(TAB / "Table_PRODUCT02dh_best_model_site_predictions.csv"),
        "decision": str(TAB / "Table_PRODUCT02di_trait_spatial_model_decision.csv"),
        "report": str(TXT / "STAGE1B6U_TRAIT_CLIMATE_SOIL_SPATIAL_MODEL_REPORT.md"),
    }
}
(TAB / "STAGE1B6U_TRAIT_CLIMATE_SOIL_SPATIAL_MODEL_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "trait_spatial_model_input_13_sites.csv")
print("WROTE", TAB / "Table_PRODUCT02df_trait_spatial_model_screen.csv")
print("WROTE", TAB / "Table_PRODUCT02dg_trait_spatial_family_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02dh_best_model_site_predictions.csv")
print("WROTE", TAB / "Table_PRODUCT02di_trait_spatial_model_decision.csv")
print("WROTE", TXT / "STAGE1B6U_TRAIT_CLIMATE_SOIL_SPATIAL_MODEL_REPORT.md")
