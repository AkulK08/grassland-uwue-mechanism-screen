from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6ac_environment_locked_trait_model")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6ac")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 20260630
rng = np.random.default_rng(SEED)
N_PERM = 10000

SRC = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")

if not SRC.exists():
    raise FileNotFoundError(f"Missing required expanded trait-covered point table: {SRC}")

df = pd.read_csv(SRC)

def num(s):
    return pd.to_numeric(s, errors="coerce")

def z(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def fit_ols(data, y_col, x_cols):
    cols = [y_col] + x_cols
    d = data[cols].copy()
    for c in cols:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < max(6, len(x_cols) + 3):
        return None

    y = d[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d))] + [z(d[c]).to_numpy(dtype=float) for c in x_cols])

    # Drop any columns that became all-NaN or constant.
    keep = [True]
    kept_x = []
    for j, c in enumerate(x_cols, start=1):
        col = X[:, j]
        if np.isfinite(col).all() and np.nanstd(col) > 0:
            keep.append(True)
            kept_x.append(c)
        else:
            keep.append(False)
    X = X[:, keep]

    if X.shape[1] < 2:
        return None

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / max(1, len(y) - X.shape[1]) if np.isfinite(r2) else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    coefs = {"intercept": float(beta[0])}
    for c, b in zip(kept_x, beta[1:]):
        coefs[c] = float(b)

    return {
        "n": int(len(y)),
        "predictors_requested": ";".join(x_cols),
        "predictors_used": ";".join(kept_x),
        "n_predictors_used": int(len(kept_x)),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": rmse,
        "coefs": coefs,
        "residuals": resid,
        "pred": pred,
        "y": y,
        "data": d,
    }

def loocv_rmse(data, y_col, x_cols):
    cols = [y_col] + x_cols
    d = data[cols].copy()
    for c in cols:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < max(8, len(x_cols) + 4):
        return np.nan, np.nan, ""

    preds = []
    obs = []
    coef_signs = []

    for i in range(len(d)):
        train = d.drop(d.index[i])
        test = d.iloc[[i]]
        fit = fit_ols(train, y_col, x_cols)
        if fit is None:
            continue

        used = fit["predictors_used"].split(";") if fit["predictors_used"] else []
        Xtest = [1.0]
        train_used = fit["data"]

        for c in used:
            mu = num(train_used[c]).mean()
            sd = num(train_used[c]).std()
            val = num(test[c]).iloc[0]
            if pd.isna(val) or pd.isna(sd) or sd == 0:
                Xtest.append(np.nan)
            else:
                Xtest.append((val - mu) / sd)

        if not np.all(np.isfinite(Xtest)):
            continue

        beta = [fit["coefs"]["intercept"]] + [fit["coefs"][c] for c in used]
        pred = float(np.dot(np.asarray(Xtest), np.asarray(beta)))
        preds.append(pred)
        obs.append(float(test[y_col].iloc[0]))

        if "rooting_depth" in fit["coefs"]:
            coef_signs.append(np.sign(fit["coefs"]["rooting_depth"]))

    if not preds:
        return np.nan, np.nan, ""

    preds = np.asarray(preds)
    obs = np.asarray(obs)
    rmse = float(np.sqrt(np.mean((obs - preds) ** 2)))

    if coef_signs:
        full_fit = fit_ols(d, y_col, x_cols)
        full_sign = np.sign(full_fit["coefs"].get("rooting_depth", np.nan)) if full_fit else np.nan
        sign_stability = float(np.mean(np.asarray(coef_signs) == full_sign)) if np.isfinite(full_sign) else np.nan
    else:
        sign_stability = np.nan

    return rmse, sign_stability, ";".join(str(round(v, 4)) for v in preds)

def residualize(data, y_col, control_cols):
    fit = fit_ols(data, y_col, control_cols)
    if fit is None:
        return None
    d = fit["data"].copy()
    d["climate_control_residual"] = fit["residuals"]
    return d, fit

def spearman_perm(x, y):
    d = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(d) < 8 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan, np.nan

    obs = float(d["x"].corr(d["y"], method="spearman"))
    vals = []
    yy = d["y"].to_numpy()
    xx = d["x"].to_numpy()

    for _ in range(N_PERM):
        vals.append(pd.Series(xx).corr(pd.Series(rng.permutation(yy)), method="spearman"))

    vals = np.asarray(vals)
    p = float((np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1))
    return obs, p

# Normalize important columns.
if "lat" not in df.columns and "latitude" in df.columns:
    df["lat"] = df["latitude"]
if "lon" not in df.columns and "longitude" in df.columns:
    df["lon"] = df["longitude"]

for c in [
    "latent_post_slope", "latent_slope_change",
    "rooting_depth", "p50", "psi50", "isohydricity",
    "aridity", "aridity_index",
    "mean_annual_precipitation", "mean_precipitation",
    "mean_annual_temperature", "mean_temperature",
    "mean_lai", "growing_season_mean_lai",
    "soil_sand", "soil_silt", "soil_clay",
    "lat", "lon"
]:
    if c in df.columns:
        df[c] = num(df[c])

# Primary environment discovered in 1B.6Z.
df["env_primary_north_midlatitude_30N_45N"] = df["lat"].between(30, 45, inclusive="both")

# Sensitivity environment: temperate grassland/savanna/shrubland.
if "eco_biome" in df.columns:
    df["env_sensitivity_temperate_grassland_savanna_shrubland"] = (
        df["eco_biome"].astype(str).str.contains("Temperate Grasslands", case=False, na=False)
    )
else:
    df["env_sensitivity_temperate_grassland_savanna_shrubland"] = False

# Optional combined environment if enough data.
df["env_combined_primary_or_temperate_grassland"] = (
    df["env_primary_north_midlatitude_30N_45N"]
    | df["env_sensitivity_temperate_grassland_savanna_shrubland"]
)

TRAITS = [c for c in ["rooting_depth", "p50", "psi50", "isohydricity"] if c in df.columns]
CLIMATE_CONTROLS = [c for c in ["aridity", "aridity_index", "mean_annual_precipitation", "mean_annual_temperature", "mean_lai", "growing_season_mean_lai"] if c in df.columns]
SOIL_CONTROLS = [c for c in ["soil_sand", "soil_silt", "soil_clay"] if c in df.columns]
OUTCOMES = [c for c in ["latent_post_slope", "latent_slope_change", "latent_satbreak_probability", "p_satbreak", "p_threshold_like"] if c in df.columns]

ENVIRONMENTS = [
    ("primary_north_midlatitude_30N_45N", "env_primary_north_midlatitude_30N_45N"),
    ("sensitivity_temperate_grassland_savanna_shrubland", "env_sensitivity_temperate_grassland_savanna_shrubland"),
    ("combined_primary_or_temperate_grassland", "env_combined_primary_or_temperate_grassland"),
]

coverage_rows = []
for env_name, env_col in ENVIRONMENTS:
    sub = df[df[env_col]].copy()
    for c in OUTCOMES + TRAITS + CLIMATE_CONTROLS + SOIL_CONTROLS:
        if c in sub.columns:
            coverage_rows.append({
                "environment": env_name,
                "column": c,
                "n_nonmissing": int(sub[c].notna().sum()),
                "n_unique": int(sub[c].nunique(dropna=True)),
                "min": float(sub[c].min()) if sub[c].notna().any() else np.nan,
                "max": float(sub[c].max()) if sub[c].notna().any() else np.nan,
            })
coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(TAB / "Table_PRODUCT02fc_environment_locked_coverage.csv", index=False)

model_rows = []
residual_rows = []
perm_rows = []

for env_name, env_col in ENVIRONMENTS:
    sub = df[df[env_col]].copy()
    sub.to_csv(DATA / f"{env_name}_input_points.csv", index=False)

    for outcome in OUTCOMES:
        if outcome not in sub.columns or sub[outcome].notna().sum() < 8:
            continue

        # Model families.
        model_specs = {
            "climate_only": CLIMATE_CONTROLS,
            "soil_only": SOIL_CONTROLS,
            "climate_plus_soil": CLIMATE_CONTROLS + SOIL_CONTROLS,
            "traits_only": TRAITS,
            "rooting_only": ["rooting_depth"] if "rooting_depth" in TRAITS else [],
            "traits_plus_climate": TRAITS + CLIMATE_CONTROLS,
            "traits_plus_climate_soil": TRAITS + CLIMATE_CONTROLS + SOIL_CONTROLS,
        }

        base_fit = fit_ols(sub, outcome, CLIMATE_CONTROLS + SOIL_CONTROLS)
        full_fit = fit_ols(sub, outcome, TRAITS + CLIMATE_CONTROLS + SOIL_CONTROLS)

        for model_name, predictors in model_specs.items():
            predictors = [p for p in predictors if p in sub.columns]
            if not predictors:
                continue

            fit = fit_ols(sub, outcome, predictors)
            if fit is None:
                continue

            cv_rmse, root_sign_stability, preds = loocv_rmse(sub, outcome, predictors)

            row = {
                "environment": env_name,
                "outcome": outcome,
                "model": model_name,
                "n": fit["n"],
                "predictors_used": fit["predictors_used"],
                "r2": fit["r2"],
                "adj_r2": fit["adj_r2"],
                "rmse": fit["rmse"],
                "loocv_rmse": cv_rmse,
                "rooting_depth_loo_sign_stability": root_sign_stability,
                "coef_rooting_depth": fit["coefs"].get("rooting_depth", np.nan),
                "coef_p50": fit["coefs"].get("p50", np.nan),
                "coef_psi50": fit["coefs"].get("psi50", np.nan),
                "coef_isohydricity": fit["coefs"].get("isohydricity", np.nan),
            }

            if base_fit is not None and model_name == "traits_plus_climate_soil":
                row["delta_r2_vs_climate_soil"] = fit["r2"] - base_fit["r2"]
                row["delta_adj_r2_vs_climate_soil"] = fit["adj_r2"] - base_fit["adj_r2"]
            elif model_name == "traits_only":
                climate_fit = fit_ols(sub, outcome, CLIMATE_CONTROLS)
                row["delta_r2_vs_climate_only"] = fit["r2"] - climate_fit["r2"] if climate_fit else np.nan
            model_rows.append(row)

        # Climate/soil residual trait analysis.
        controls = CLIMATE_CONTROLS + SOIL_CONTROLS
        resid_pack = residualize(sub, outcome, controls)
        if resid_pack is not None:
            resid_df, control_fit = resid_pack

            for trait in TRAITS:
                if trait not in resid_df.columns:
                    continue
                rho, p = spearman_perm(resid_df[trait], resid_df["climate_control_residual"])

                simple = fit_ols(resid_df, "climate_control_residual", [trait])
                if simple is None:
                    continue

                cv_rmse, sign_stability, preds = loocv_rmse(resid_df, "climate_control_residual", [trait])

                residual_rows.append({
                    "environment": env_name,
                    "outcome": outcome,
                    "control_set": "climate_plus_soil",
                    "trait": trait,
                    "n": simple["n"],
                    "control_r2_on_original_outcome": control_fit["r2"],
                    "trait_r2_on_control_residual": simple["r2"],
                    "trait_adj_r2_on_control_residual": simple["adj_r2"],
                    "trait_coef_on_control_residual": simple["coefs"].get(trait, np.nan),
                    "spearman_trait_vs_residual": rho,
                    "perm_p_spearman": p,
                    "loocv_rmse": cv_rmse,
                    "loo_sign_stability": sign_stability,
                })

models = pd.DataFrame(model_rows)
residuals = pd.DataFrame(residual_rows)

if len(models):
    models = models.sort_values(
        ["environment", "outcome", "model"],
        ascending=True
    )
if len(residuals):
    residuals = residuals.sort_values(
        ["trait_r2_on_control_residual", "spearman_trait_vs_residual", "n"],
        ascending=[False, False, False],
    )

models.to_csv(TAB / "Table_PRODUCT02fd_environment_locked_models.csv", index=False)
residuals.to_csv(TAB / "Table_PRODUCT02fe_climate_soil_residual_trait_tests.csv", index=False)

# Decision logic: mentor-style success = trait explains >=20% climate/soil residual variance
# and survives LOOCV direction stability.
candidate = pd.DataFrame()
if len(residuals):
    candidate = residuals[
        (residuals["trait"].eq("rooting_depth"))
        & (residuals["trait_r2_on_control_residual"] >= 0.20)
        & (residuals["loo_sign_stability"].fillna(0) >= 0.80)
        & (residuals["perm_p_spearman"].fillna(1) <= 0.15)
    ].head(1)

if len(candidate):
    b = candidate.iloc[0]
    verdict = "ENVIRONMENT_LOCKED_TRAIT_ANALYSIS_PASSES_MENTOR_CRITERION"
    safe_claim = (
        f"Within the pre-locked environment {b['environment']}, effective rooting depth explains "
        f"{b['trait_r2_on_control_residual']:.3f} of climate/soil-residual variation in {b['outcome']} "
        f"(n={int(b['n'])}, Spearman residual association={b['spearman_trait_vs_residual']:.3f}, "
        f"permutation p={b['perm_p_spearman']:.4f}, LOO sign stability={b['loo_sign_stability']:.3f}). "
        "This matches the mentor-specified trait pathway more closely than the earlier discovery scan."
    )
else:
    verdict = "ENVIRONMENT_LOCKED_TRAIT_ANALYSIS_DOES_NOT_PASS_FULL_MENTOR_CRITERION"
    safe_claim = (
        "The prior scan found a strong rooting-depth association, but after locking the environment and controlling "
        "for climate and soil texture, the trait effect did not meet the >20% climate/soil-residual variance plus "
        "LOOCV criterion. This would mean the honest claim is trait-consistent but not a full mentor-style trait proof."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "primary_environment": "north_midlatitude_30N_45N",
    "primary_trait": "rooting_depth",
    "primary_outcome": "latent_post_slope",
    "n_environments_tested": len(ENVIRONMENTS),
    "n_model_rows": int(len(models)),
    "n_residual_trait_tests": int(len(residuals)),
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_MENTOR_ALIGNED_TRAIT_SECTION" if "PASSES" in verdict else "REPORT_TRAIT_SIGNAL_AS_ASSOCIATIONAL_AND_CONSIDER_BAYESIAN_MODEL",
}])
decision.to_csv(TAB / "Table_PRODUCT02ff_environment_locked_trait_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    primary = df[df["env_primary_north_midlatitude_30N_45N"]].copy()
    primary = primary.dropna(subset=["rooting_depth", "latent_post_slope"])

    if len(primary) >= 6:
        plt.figure(figsize=(6.5, 4.8))
        plt.scatter(primary["rooting_depth"], primary["latent_post_slope"], alpha=0.8)
        zfit = np.polyfit(primary["rooting_depth"], primary["latent_post_slope"], 1)
        xs = np.linspace(primary["rooting_depth"].min(), primary["rooting_depth"].max(), 100)
        plt.plot(xs, zfit[0] * xs + zfit[1], linestyle="--")
        plt.xlabel("Effective rooting depth")
        plt.ylabel("Latent post-stress response slope")
        plt.title("Environment-locked rooting-depth trait effect")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02m_environment_locked_rooting_depth_effect.png", dpi=220)
        plt.close()

    if len(residuals):
        plot = residuals.head(12).copy()
        labels = plot["environment"].astype(str) + " | " + plot["outcome"].astype(str) + " | " + plot["trait"].astype(str)
        vals = plot["trait_r2_on_control_residual"]
        plt.figure(figsize=(10, 6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Trait R² on climate/soil residual")
        plt.ylabel("Environment | outcome | trait")
        plt.title("Trait contribution after climate/soil controls")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02n_trait_residual_variance_explained.png", dpi=220)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6AC environment-locked full trait analysis")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Safe claim")
report.append("")
report.append(safe_claim)
report.append("")
report.append("## Why this stage exists")
report.append("")
report.append("Stage 1B.6Z discovered that rooting depth was strongest in north-midlatitude 30N-45N systems. This stage locks that environment first, then tests whether rooting depth still explains the response after climate and soil controls.")
report.append("")
report.append("## Coverage")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage rows.")
report.append("```")
report.append("")
report.append("## Model comparison")
report.append("")
report.append("```text")
report.append(models.head(80).to_string(index=False) if len(models) else "No model rows.")
report.append("```")
report.append("")
report.append("## Climate/soil residual trait tests")
report.append("")
report.append("```text")
report.append(residuals.head(80).to_string(index=False) if len(residuals) else "No residual trait rows.")
report.append("```")
report.append("")
report.append("## Mentor criterion")
report.append("")
report.append("The mentor-style criterion is: inside the relevant environment, the core plant trait should explain at least 20% of climate/soil-residual response variance and survive leave-one-out direction stability.")
report.append("")
report.append(f"Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6AC_ENVIRONMENT_LOCKED_TRAIT_MODEL_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6AC_environment_locked_trait_model",
    "status": verdict,
    "safe_claim": safe_claim,
    "outputs": {
        "coverage": str(TAB / "Table_PRODUCT02fc_environment_locked_coverage.csv"),
        "models": str(TAB / "Table_PRODUCT02fd_environment_locked_models.csv"),
        "residual_trait_tests": str(TAB / "Table_PRODUCT02fe_climate_soil_residual_trait_tests.csv"),
        "decision": str(TAB / "Table_PRODUCT02ff_environment_locked_trait_decision.csv"),
        "report": str(TXT / "STAGE1B6AC_ENVIRONMENT_LOCKED_TRAIT_MODEL_REPORT.md"),
    }
}
(TAB / "STAGE1B6AC_ENVIRONMENT_LOCKED_TRAIT_MODEL_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02fc_environment_locked_coverage.csv")
print("WROTE", TAB / "Table_PRODUCT02fd_environment_locked_models.csv")
print("WROTE", TAB / "Table_PRODUCT02fe_climate_soil_residual_trait_tests.csv")
print("WROTE", TAB / "Table_PRODUCT02ff_environment_locked_trait_decision.csv")
print("WROTE", TXT / "STAGE1B6AC_ENVIRONMENT_LOCKED_TRAIT_MODEL_REPORT.md")
print("WROTE figures to", FIG)
