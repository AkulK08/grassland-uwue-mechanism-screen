from pathlib import Path
from datetime import datetime
import json
import math
import itertools
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6w2_true_trait_environment_effect")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6w2")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

BASE = Path("data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv")
FITS = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")

COVARIATE_SOURCES = [
    Path("results/paper_point_geography_thesis_lock/tables/Table72_high_vpd_point_geography.csv"),
    Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
    Path("data/external/soilgrids_texture_by_point.csv"),
    Path("data/external/aridity_by_point.csv"),
]

SEED = 20260629
rng = np.random.default_rng(SEED)

OUTCOME = "satellite_limitation_mean_fraction"
MAX_NEAREST_KM = 75.0
MIN_ENV_N = 4
N_PERM = 10000

BIO_CLIMATE_SOIL_CANDIDATES = [
    "p50",
    "psi50",
    "isohydricity",
    "rooting_depth",
    "soil_sand",
    "soil_clay",
    "soil_silt",
    "soil_sand_mean",
    "soil_clay_mean",
    "soil_silt_mean",
    "soil_texture_coarse_index",
    "soil_texture_fine_index",
    "mean_vpd",
    "mean_soil_moisture",
    "mean_annual_temperature",
    "mean_temperature",
    "mean_annual_precipitation",
    "mean_precipitation",
    "mean_lai",
    "growing_season_mean_lai",
    "aridity_index",
    "aridity",
    "lat_aridity",
    "lon_aridity",
]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def normalize_source(df):
    cols = {c.lower(): c for c in df.columns}
    id_col = None
    for k in ["point_id", "site", "site_id", "tower_id", "id"]:
        if k in cols:
            id_col = cols[k]
            break

    lat_col = None
    for k in ["lat", "latitude"]:
        if k in cols:
            lat_col = cols[k]
            break

    lon_col = None
    for k in ["lon", "longitude"]:
        if k in cols:
            lon_col = cols[k]
            break

    out = df.copy()
    if id_col:
        out["__point_id_norm"] = out[id_col].astype(str)
    else:
        out["__point_id_norm"] = ""

    if lat_col:
        out["__lat_norm"] = num(out[lat_col])
    else:
        out["__lat_norm"] = np.nan

    if lon_col:
        out["__lon_norm"] = num(out[lon_col])
    else:
        out["__lon_norm"] = np.nan

    return out

def useful_cols(df):
    cols = []
    for c in df.columns:
        cl = c.lower()
        if c.startswith("__"):
            continue
        if c in ["point_id"]:
            continue
        if c in BIO_CLIMATE_SOIL_CANDIDATES:
            cols.append(c)
        elif any(k in cl for k in [
            "p50", "psi50", "iso", "rooting",
            "soil", "sand", "clay", "silt",
            "aridity", "vpd", "moisture", "precip",
            "temperature", "lai"
        ]):
            cols.append(c)
    return sorted(set(cols))

def slope_r2(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return {
        "n": int(len(x)),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "r2": float(r2),
        "spearman_r": rho,
        "abs_spearman_r": abs(rho),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "effect_range": float(beta[1] * (np.max(x) - np.min(x))),
        "abs_effect_range": abs(float(beta[1] * (np.max(x) - np.min(x)))),
    }

def perm_p(x, y, n_perm=N_PERM):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    obs = pd.Series(x).corr(pd.Series(y), method="spearman")
    vals = []
    for _ in range(n_perm):
        vals.append(pd.Series(x).corr(pd.Series(rng.permutation(y)), method="spearman"))
    vals = np.asarray(vals, dtype=float)
    return float((np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1))

def loo_stability(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4:
        return np.nan, np.nan, ""
    full = slope_r2(x, y)
    if not full:
        return np.nan, np.nan, ""
    full_sign = np.sign(full["slope"])
    slopes = []
    signs = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        out = slope_r2(x[keep], y[keep])
        if out:
            slopes.append(out["slope"])
            signs.append(np.sign(out["slope"]))
    if not slopes:
        return np.nan, np.nan, ""
    return (
        float(np.mean(np.asarray(signs) == full_sign)),
        float(np.median(slopes)),
        ";".join(str(round(v, 5)) for v in slopes),
    )

if not BASE.exists():
    raise FileNotFoundError(f"Missing base table: {BASE}")

base = pd.read_csv(BASE)
base["point_id"] = base["point_id"].astype(str)
base = base.drop_duplicates("point_id").copy()

# Attach fit-derived outcomes if needed.
if FITS.exists():
    fits = pd.read_csv(FITS)
    fits["point_id"] = fits["point_id"].astype(str)
    fits["is_limitation_like"] = fits["response_class"].isin(["breakdown", "saturation", "weakening"])
    wide = (
        fits.groupby(["point_id", "metric"])
        .agg(limitation_fraction=("is_limitation_like", "mean"))
        .reset_index()
        .pivot(index="point_id", columns="metric", values="limitation_fraction")
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.rename(columns={
        "log_wue": "limitation_fraction_log_wue_from_fits",
        "log_uwue": "limitation_fraction_log_uwue_from_fits",
    })
    for c in ["limitation_fraction_log_wue_from_fits", "limitation_fraction_log_uwue_from_fits"]:
        if c in base.columns:
            base = base.drop(columns=[c])
    base = base.merge(wide, on="point_id", how="left")

# Ensure base lat/lon.
if "lat" not in base.columns and "latitude" in base.columns:
    base["lat"] = num(base["latitude"])
if "lon" not in base.columns and "longitude" in base.columns:
    base["lon"] = num(base["longitude"])

base["lat"] = num(base["lat"])
base["lon"] = num(base["lon"])

audit_rows = []
merged = base.copy()

for source in COVARIATE_SOURCES:
    if not source.exists():
        audit_rows.append({
            "source": str(source),
            "exists": False,
            "used": False,
            "reason": "missing",
        })
        continue

    raw = pd.read_csv(source)
    src = normalize_source(raw)
    cols = useful_cols(src)

    if not cols:
        audit_rows.append({
            "source": str(source),
            "exists": True,
            "used": False,
            "reason": "no_useful_covariate_columns",
            "source_rows": len(src),
        })
        continue

    # Exact point_id merge candidate.
    exact = src[["__point_id_norm"] + cols].drop_duplicates("__point_id_norm").copy()
    exact = exact.rename(columns={"__point_id_norm": "point_id"})
    exact["point_id"] = exact["point_id"].astype(str)

    exact_match_ids = set(merged["point_id"]).intersection(set(exact["point_id"]))
    exact_nonmissing_total = 0
    if exact_match_ids:
        temp = merged[["point_id"]].merge(exact, on="point_id", how="left")
        exact_nonmissing_total = int(temp[cols].notna().sum().sum())

    # Nearest-coordinate merge candidate.
    nearest_rows = []
    src_coord = src[src["__lat_norm"].notna() & src["__lon_norm"].notna()].copy()
    if len(src_coord):
        for _, br in merged[["point_id", "lat", "lon"]].drop_duplicates("point_id").iterrows():
            best_i = None
            best_d = np.inf
            for i, sr in src_coord.iterrows():
                d = haversine_km(br["lat"], br["lon"], sr["__lat_norm"], sr["__lon_norm"])
                if np.isfinite(d) and d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None and best_d <= MAX_NEAREST_KM:
                row = {"point_id": br["point_id"], f"__nearest_km__{source.name}": best_d}
                for c in cols:
                    row[c] = src_coord.loc[best_i, c]
                nearest_rows.append(row)

    nearest = pd.DataFrame(nearest_rows)
    nearest_nonmissing_total = int(nearest[cols].notna().sum().sum()) if len(nearest) else 0

    if exact_nonmissing_total >= nearest_nonmissing_total and exact_nonmissing_total > 0:
        chosen = exact
        method = "point_id_exact"
        chosen_nonmissing = exact_nonmissing_total
    elif nearest_nonmissing_total > 0:
        chosen = nearest
        method = f"nearest_latlon_within_{MAX_NEAREST_KM}km"
        chosen_nonmissing = nearest_nonmissing_total
    else:
        audit_rows.append({
            "source": str(source),
            "exists": True,
            "used": False,
            "reason": "no_matching_nonmissing_covariates",
            "source_rows": len(src),
            "candidate_cols": ";".join(cols),
            "exact_match_ids": len(exact_match_ids),
            "exact_nonmissing_total": exact_nonmissing_total,
            "nearest_nonmissing_total": nearest_nonmissing_total,
        })
        continue

    # Add with source suffix if necessary, then coalesce into canonical names.
    add = chosen.copy()
    new_cols = []
    for c in cols:
        if c in merged.columns:
            new_name = f"{c}__rescued_{source.stem}"
            add = add.rename(columns={c: new_name})
            merged[new_name] = merged["point_id"].map(add.set_index("point_id")[new_name])
            # coalesce into original if original all missing
            if merged[c].notna().sum() == 0:
                merged[c] = merged[new_name]
            new_cols.append(new_name)
        else:
            merged[c] = merged["point_id"].map(add.set_index("point_id")[c])
            new_cols.append(c)

    # Add distance cols if present.
    for c in add.columns:
        if c.startswith("__nearest_km__"):
            merged[c] = merged["point_id"].map(add.set_index("point_id")[c])

    audit_rows.append({
        "source": str(source),
        "exists": True,
        "used": True,
        "method": method,
        "source_rows": len(src),
        "candidate_cols": ";".join(cols),
        "exact_match_ids": len(exact_match_ids),
        "exact_nonmissing_total": exact_nonmissing_total,
        "nearest_nonmissing_total": nearest_nonmissing_total,
        "chosen_nonmissing_total": chosen_nonmissing,
        "added_cols": ";".join(new_cols),
    })

audit = pd.DataFrame(audit_rows)
audit.to_csv(TAB / "Table_PRODUCT02du_covariate_rescue_source_audit.csv", index=False)

# Coverage after rescue.
coverage_rows = []
for c in BIO_CLIMATE_SOIL_CANDIDATES:
    if c in merged.columns:
        s = num(merged[c])
        coverage_rows.append({
            "covariate": c,
            "n_nonmissing": int(s.notna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            "usable": bool(s.notna().sum() >= 4 and s.nunique(dropna=True) >= 2),
        })

coverage = pd.DataFrame(coverage_rows).sort_values(["usable", "n_nonmissing"], ascending=[False, False])
coverage.to_csv(TAB / "Table_PRODUCT02dv_true_trait_covariate_coverage_after_rescue.csv", index=False)

merged.to_csv(DATA / "true_trait_environment_input_covariate_rescued.csv", index=False)

usable_covariates = coverage.loc[coverage["usable"], "covariate"].tolist() if len(coverage) else []

# Outcomes.
outcomes = [
    "satellite_limitation_mean_fraction",
    "limitation_fraction_log_wue_from_fits",
    "limitation_fraction_log_uwue_from_fits",
]
outcomes = [c for c in outcomes if c in merged.columns and num(merged[c]).notna().sum() >= 4]

# Environments.
envs = []
merged["env_Great_Plains_core"] = merged["point_id"].isin(["US-Ne1", "US-Ne2", "US-Ne3"])
merged["env_US_sites"] = merged["point_id"].astype(str).str.startswith("US-")
merged["env_nonUS_sites"] = ~merged["env_US_sites"]
if "latitude_band_handbuilt" in merged.columns:
    merged["env_midlat_35_50"] = merged["latitude_band_handbuilt"].astype(str).eq("mid_lat_35_50")
    merged["env_high_lat_ge50"] = merged["latitude_band_handbuilt"].astype(str).eq("high_lat_ge50")
if "longitude_sector_handbuilt" in merged.columns:
    merged["env_eastern_NA"] = merged["longitude_sector_handbuilt"].astype(str).eq("eastern_NA")
    merged["env_western_or_central_NA"] = merged["longitude_sector_handbuilt"].astype(str).eq("western_or_central_NA")
if "broad_region_handbuilt" in merged.columns:
    for level in merged["broad_region_handbuilt"].dropna().astype(str).unique():
        safe = level.replace(" ", "_").replace("/", "_")
        merged[f"env_region_{safe}"] = merged["broad_region_handbuilt"].astype(str).eq(level)

for c in merged.columns:
    if c.startswith("env_"):
        mask = merged[c].astype(bool)
        if mask.sum() >= 2 and (~mask).sum() >= 2:
            envs.append((c, mask))

scan_rows = []
interaction_rows = []

for outcome in outcomes:
    y_all = num(merged[outcome])

    for cov in usable_covariates:
        x_all = zscore(merged[cov])

        # Global covariate effect.
        fit = slope_r2(x_all, y_all)
        if fit:
            loo, loo_slope, loo_slopes = loo_stability(x_all, y_all)
            scan_rows.append({
                "scope": "GLOBAL_ALL_13",
                "environment": "ALL",
                "environment_n": int(len(merged)),
                "outcome": outcome,
                "covariate": cov,
                "covariate_type": "true_trait_climate_soil",
                **fit,
                "perm_p_spearman": perm_p(x_all, y_all),
                "loo_sign_stability": loo,
                "loo_median_slope": loo_slope,
                "loo_slopes": loo_slopes,
                "claim_strength": "global_screen",
            })

        # Within-environment covariate effect.
        for env_name, mask in envs:
            env_n = int(mask.sum())
            xx = x_all[mask]
            yy = y_all[mask]
            fit = slope_r2(xx, yy)
            if not fit:
                continue

            loo, loo_slope, loo_slopes = loo_stability(xx, yy)
            p = perm_p(xx, yy) if env_n >= 4 else np.nan

            if (
                env_n >= MIN_ENV_N
                and fit["abs_effect_range"] >= 0.20
                and fit["abs_spearman_r"] >= 0.70
                and (pd.isna(p) or p <= 0.15)
                and (pd.isna(loo) or loo >= 0.80)
            ):
                strength = "candidate_big_true_trait_environment_effect"
            elif env_n < MIN_ENV_N and fit["abs_effect_range"] >= 0.20:
                strength = "hypothesis_only_small_n"
            else:
                strength = "weak_or_exploratory"

            scan_rows.append({
                "scope": "WITHIN_ENVIRONMENT",
                "environment": env_name,
                "environment_n": env_n,
                "outcome": outcome,
                "covariate": cov,
                "covariate_type": "true_trait_climate_soil",
                **fit,
                "perm_p_spearman": p,
                "loo_sign_stability": loo,
                "loo_median_slope": loo_slope,
                "loo_slopes": loo_slopes,
                "claim_strength": strength,
            })

            # Interaction using all sites: y ~ cov + env + cov*env.
            tmp = pd.DataFrame({
                "y": y_all,
                "x": x_all,
                "env": mask.astype(float),
            }).dropna()
            if len(tmp) >= 8 and tmp["x"].nunique() >= 2 and tmp["env"].nunique() == 2:
                X0 = np.column_stack([np.ones(len(tmp)), tmp["x"], tmp["env"]])
                X1 = np.column_stack([np.ones(len(tmp)), tmp["x"], tmp["env"], tmp["x"] * tmp["env"]])
                y = tmp["y"].to_numpy(dtype=float)

                b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
                b1, *_ = np.linalg.lstsq(X1, y, rcond=None)

                rss0 = float(np.sum((y - X0 @ b0) ** 2))
                rss1 = float(np.sum((y - X1 @ b1) ** 2))
                tss = float(np.sum((y - y.mean()) ** 2))
                r20 = 1 - rss0 / tss if tss > 0 else np.nan
                r21 = 1 - rss1 / tss if tss > 0 else np.nan
                delta = r21 - r20 if np.isfinite(r20) and np.isfinite(r21) else np.nan

                if env_n >= MIN_ENV_N and np.isfinite(delta) and delta >= 0.10 and abs(float(b1[3])) >= 0.10:
                    strength_i = "candidate_true_trait_environment_interaction"
                else:
                    strength_i = "exploratory_or_small_n"

                interaction_rows.append({
                    "environment": env_name,
                    "environment_n": env_n,
                    "outcome": outcome,
                    "covariate": cov,
                    "n_total": int(len(tmp)),
                    "coef_covariate_main": float(b1[1]),
                    "coef_env_main": float(b1[2]),
                    "coef_covariate_x_env": float(b1[3]),
                    "r2_without_interaction": float(r20),
                    "r2_with_interaction": float(r21),
                    "delta_r2_interaction": float(delta),
                    "interaction_abs_effect": abs(float(b1[3])),
                    "claim_strength": strength_i,
                })

scan = pd.DataFrame(scan_rows)
interactions = pd.DataFrame(interaction_rows)

if len(scan):
    priority = {
        "candidate_big_true_trait_environment_effect": 0,
        "global_screen": 1,
        "hypothesis_only_small_n": 2,
        "weak_or_exploratory": 3,
    }
    scan["_priority"] = scan["claim_strength"].map(priority).fillna(9)
    scan = scan.sort_values(
        ["_priority", "abs_effect_range", "abs_spearman_r", "environment_n"],
        ascending=[True, False, False, False]
    ).drop(columns=["_priority"])

if len(interactions):
    priority_i = {
        "candidate_true_trait_environment_interaction": 0,
        "exploratory_or_small_n": 1,
    }
    interactions["_priority"] = interactions["claim_strength"].map(priority_i).fillna(9)
    interactions = interactions.sort_values(
        ["_priority", "delta_r2_interaction", "interaction_abs_effect", "environment_n"],
        ascending=[True, False, False, False]
    ).drop(columns=["_priority"])

scan.to_csv(TAB / "Table_PRODUCT02dw_true_trait_covariate_environment_scan.csv", index=False)
interactions.to_csv(TAB / "Table_PRODUCT02dx_true_trait_covariate_environment_interactions.csv", index=False)

# Great Plains diagnostic.
gp = scan[scan["environment"].eq("env_Great_Plains_core")].copy() if len(scan) else pd.DataFrame()
gp.to_csv(TAB / "Table_PRODUCT02dy_true_trait_great_plains_diagnostic.csv", index=False)

candidate = scan[scan["claim_strength"].eq("candidate_big_true_trait_environment_effect")].head(1) if len(scan) else pd.DataFrame()
candidate_i = interactions[interactions["claim_strength"].eq("candidate_true_trait_environment_interaction")].head(1) if len(interactions) else pd.DataFrame()

if len(candidate):
    b = candidate.iloc[0]
    verdict = "TRUE_TRAIT_CLIMATE_SOIL_EFFECT_IN_SPECIFIC_ENVIRONMENT_FOUND"
    safe_claim = (
        f"After covariate rescue, the strongest defensible environment-conditioned covariate effect is {b['covariate']} "
        f"within {b['environment']} (n={int(b['environment_n'])}), with effect_range={b['effect_range']:.3f}, "
        f"Spearman r={b['spearman_r']:.3f}, permutation p={b['perm_p_spearman']:.3f}, "
        f"and leave-one-out sign stability={b['loo_sign_stability']:.3f}. This supports a trait/climate/soil-conditioned "
        "environment hypothesis, but still should be framed as exploratory because the site count is small."
    )
elif len(candidate_i):
    b = candidate_i.iloc[0]
    verdict = "TRUE_TRAIT_CLIMATE_SOIL_BY_ENVIRONMENT_INTERACTION_FOUND"
    safe_claim = (
        f"After covariate rescue, the strongest interaction is {b['covariate']} × {b['environment']} "
        f"for {b['outcome']} (environment n={int(b['environment_n'])}), with ΔR²={b['delta_r2_interaction']:.3f} "
        f"and interaction coefficient={b['coef_covariate_x_env']:.3f}. This supports an environment-dependent "
        "trait/climate/soil mechanism hypothesis, not causal proof."
    )
else:
    verdict = "NO_TRUE_TRAIT_CLIMATE_SOIL_EFFECT_FOUND_AFTER_RESCUE"
    safe_claim = (
        "After covariate rescue, no defensible trait/climate/soil effect within a specific environment passed the screening thresholds. "
        "Do not present the latitude/longitude result as trait proof. The strongest defensible result remains spatial/biome heterogeneity."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_sites": int(merged["point_id"].nunique()),
    "n_covariates_usable_after_rescue": int(len(usable_covariates)),
    "usable_covariates": ";".join(usable_covariates),
    "n_environment_masks": int(len(envs)),
    "n_trait_environment_tests": int(len(scan)),
    "n_interaction_tests": int(len(interactions)),
    "n_candidate_big_true_trait_environment_effects": int((scan["claim_strength"].eq("candidate_big_true_trait_environment_effect")).sum()) if len(scan) else 0,
    "n_candidate_true_trait_environment_interactions": int((interactions["claim_strength"].eq("candidate_true_trait_environment_interaction")).sum()) if len(interactions) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_MECHANISM_RESULTS_SECTION_OR_EXPAND_SITE_SET",
}])
decision.to_csv(TAB / "Table_PRODUCT02dz_true_trait_environment_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(scan):
        figdf = scan.head(15).copy()
        labels = figdf["covariate"].astype(str) + " | " + figdf["environment"].astype(str)
        vals = figdf["effect_range"]
        plt.figure(figsize=(11, 6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range on limitation fraction")
        plt.ylabel("Covariate | environment")
        plt.title("Top true trait/climate/soil effects by environment")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02e_true_trait_environment_effects.png", dpi=200)
        plt.close()

    if len(coverage):
        cfig = coverage.sort_values("n_nonmissing", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(cfig["covariate"], cfig["n_nonmissing"])
        plt.xlabel("Nonmissing final-13 sites")
        plt.ylabel("Covariate")
        plt.title("Covariate rescue coverage")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02f_covariate_rescue_coverage.png", dpi=200)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6W.2 true trait/climate/soil × environment effect scan")
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
report.append("## Covariate rescue source audit")
report.append("")
report.append("```text")
report.append(audit.to_string(index=False) if len(audit) else "No source audit rows.")
report.append("```")
report.append("")
report.append("## Coverage after rescue")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage rows.")
report.append("```")
report.append("")
report.append("## Top true trait/climate/soil environment effects")
report.append("")
report.append("```text")
report.append(scan.head(40).to_string(index=False) if len(scan) else "No scan rows.")
report.append("```")
report.append("")
report.append("## Top true trait/climate/soil × environment interactions")
report.append("")
report.append("```text")
report.append(interactions.head(40).to_string(index=False) if len(interactions) else "No interaction rows.")
report.append("```")
report.append("")
report.append("## Great Plains diagnostic")
report.append("")
report.append("```text")
report.append(gp.head(40).to_string(index=False) if len(gp) else "No Great Plains diagnostic rows.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("- This scan excludes lat/lon as trait variables.")
report.append("- If the decision finds no true trait/climate/soil effect, do not rebrand spatial latitude/longitude as trait proof.")
report.append("- Great Plains-only n=3 remains hypothesis-generating unless additional nearby sites are added.")
report.append(f"- Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6W2_TRUE_TRAIT_ENVIRONMENT_EFFECT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6W.2_true_trait_environment_effect",
    "status": str(decision["verdict"].iloc[0]),
    "safe_claim": str(decision["safe_claim"].iloc[0]),
    "outputs": {
        "rescued_input": str(DATA / "true_trait_environment_input_covariate_rescued.csv"),
        "source_audit": str(TAB / "Table_PRODUCT02du_covariate_rescue_source_audit.csv"),
        "coverage": str(TAB / "Table_PRODUCT02dv_true_trait_covariate_coverage_after_rescue.csv"),
        "scan": str(TAB / "Table_PRODUCT02dw_true_trait_covariate_environment_scan.csv"),
        "interactions": str(TAB / "Table_PRODUCT02dx_true_trait_covariate_environment_interactions.csv"),
        "great_plains": str(TAB / "Table_PRODUCT02dy_true_trait_great_plains_diagnostic.csv"),
        "decision": str(TAB / "Table_PRODUCT02dz_true_trait_environment_decision.csv"),
        "report": str(TXT / "STAGE1B6W2_TRUE_TRAIT_ENVIRONMENT_EFFECT_REPORT.md"),
    },
}
(TAB / "STAGE1B6W2_TRUE_TRAIT_ENVIRONMENT_EFFECT_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "true_trait_environment_input_covariate_rescued.csv")
print("WROTE", TAB / "Table_PRODUCT02du_covariate_rescue_source_audit.csv")
print("WROTE", TAB / "Table_PRODUCT02dv_true_trait_covariate_coverage_after_rescue.csv")
print("WROTE", TAB / "Table_PRODUCT02dw_true_trait_covariate_environment_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02dx_true_trait_covariate_environment_interactions.csv")
print("WROTE", TAB / "Table_PRODUCT02dy_true_trait_great_plains_diagnostic.csv")
print("WROTE", TAB / "Table_PRODUCT02dz_true_trait_environment_decision.csv")
print("WROTE", TXT / "STAGE1B6W2_TRUE_TRAIT_ENVIRONMENT_EFFECT_REPORT.md")
print("WROTE figures to", FIG)
