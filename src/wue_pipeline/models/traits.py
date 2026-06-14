"""Trait models for conditional Phase 4."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.inspection import permutation_importance


def standardize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        sd = out[c].std()
        out[c] = (out[c] - out[c].mean()) / (sd if sd else 1)
    return out


def prepare_trait_frame(response: pd.DataFrame, traits_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["lat", "lon"]
    # Avoid duplicate covariates from earlier gate tables. Climate/trait sources are authoritative here.
    drop_cols = [c for c in ["aridity_index", "lai", "map", "mat", "psi50", "isohydricity", "rooting_depth"] if c in response.columns]
    response_clean = response.drop(columns=drop_cols)
    df = response_clean.merge(traits_df, on=keys, how="inner").merge(climate_df, on=keys, how="inner")
    if "psi50" in df and "psi50_abs" not in df:
        df["psi50_abs"] = df["psi50"].abs()
    if "slope_change" not in df and {"pre_slope", "post_slope"}.issubset(df.columns):
        df["slope_change"] = df["post_slope"] - df["pre_slope"]
    return df.replace([np.inf, -np.inf], np.nan)


def random_forest_trait_analysis(df: pd.DataFrame, y_col: str, trait_cols: list[str], climate_cols: list[str], n_estimators: int = 500, seed: int = 42) -> dict:
    cols = trait_cols + climate_cols
    d = df[[y_col] + cols].dropna().copy()
    d = standardize_columns(d, cols)
    Xc = d[climate_cols]
    Xf = d[cols]
    y = d[y_col]
    rf_c = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=3)
    rf_f = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1, min_samples_leaf=3)
    rf_c.fit(Xc, y)
    rf_f.fit(Xf, y)
    r2_c = rf_c.score(Xc, y)
    r2_f = rf_f.score(Xf, y)
    perm = permutation_importance(rf_f, Xf, y, n_repeats=5, random_state=seed, n_jobs=1)
    importances = pd.DataFrame({"variable": cols, "permutation_importance_mean": perm.importances_mean, "permutation_importance_sd": perm.importances_std}).sort_values("permutation_importance_mean", ascending=False)
    if len(d) < 20:
        # Tiny demo runs skip SHAP to keep the smoke test fast. Production runs use SHAP.
        shap_imp = pd.DataFrame({"variable": cols, "mean_abs_shap": np.nan, "reason": "skipped_for_small_demo_n"})
    else:
        try:
            import shap
            explainer = shap.TreeExplainer(rf_f)
            sv = explainer.shap_values(Xf)
            shap_imp = pd.DataFrame({"variable": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        except Exception as exc:
            shap_imp = pd.DataFrame({"variable": cols, "mean_abs_shap": np.nan, "reason": str(exc)})
    cv = KFold(n_splits=min(5, len(d)), shuffle=True, random_state=seed)
    cv_scores = cross_val_score(rf_f, Xf, y, cv=cv, scoring="r2") if len(d) >= 10 else np.array([np.nan])
    return {
        "r2_climate": float(r2_c),
        "r2_full": float(r2_f),
        "trait_unique_variance": float(r2_f - r2_c),
        "cv_r2_mean": float(np.nanmean(cv_scores)),
        "cv_r2_sd": float(np.nanstd(cv_scores)),
        "permutation_importance": importances,
        "shap_importance": shap_imp,
        "model": rf_f,
    }
