from pathlib import Path
from datetime import datetime
import json
import math
import itertools
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6x_expanded_trait_mechanism_screen")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6x")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 20260629
rng = np.random.default_rng(SEED)

MIN_ENV_N = 8
N_PERM = 5000

SEARCH_ROOTS = [
    Path("results"),
    Path("data/processed"),
    Path("data/external"),
    Path("data/raw"),
    Path("data/raw_local"),
]

PRIORITY_FILES = [
    Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
    Path("results/paper_point_geography_thesis_lock/tables/Table72_high_vpd_point_geography.csv"),
    Path("results/final_nonwriting_lock/files/phase19_tower_satellite_comparison.csv"),
    Path("results/final_nonwriting_lock 2/files/phase19_tower_satellite_comparison.csv"),
]

OUTCOME_CANDIDATES = [
    "latent_satbreak_probability",
    "latent_satbreak_probability_direct",
    "satellite_limitation_mean_fraction",
    "limitation_fraction",
    "satbreak_probability",
    "satbreak_fraction",
    "breakdown_fraction",
    "saturation_fraction",
    "limitation_like_fraction",
    "latent_slope_change",
    "slope_change",
    "median_slope_change",
    "post_slope",
    "latent_post_slope",
]

COVARIATE_CANDIDATES = [
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
    "aridity",
    "aridity_index",
]

ENV_CANDIDATES = [
    "geo_continent",
    "geo_subregion",
    "geo_country",
    "eco_biome",
    "eco_realm",
    "eco_ecoregion",
    "hydroclimatic_regime",
    "aridity_quartile",
    "mean_vpd_quartile",
    "latitude_band",
    "longitude_sector",
    "named_geographic_regions",
    "region_North_American_Great_Plains",
    "region_Pampas",
    "region_Cerrado",
    "region_Australian_grasslands",
    "region_Deccan_semiarid_grasslands",
    "high_vpd_gt_2p26",
    "low_latitude_high_vpd",
    "sahel_and_high_vpd",
]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def safe_read(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows)
        except Exception:
            return None

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def usable_numeric_cols(df, candidates, min_n=8):
    out = []
    for c in candidates:
        if c in df.columns:
            s = num(df[c])
            if s.notna().sum() >= min_n and s.nunique(dropna=True) >= 2:
                out.append(c)
    return out

def slope_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None

    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    effect_range = float(beta[1] * (np.max(x) - np.min(x)))

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
        "effect_range": effect_range,
        "abs_effect_range": abs(effect_range),
    }

def perm_p(x, y, n_perm=N_PERM):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 8 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
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
    if len(x) < 8:
        return np.nan, np.nan, ""

    full = slope_fit(x, y)
    if full is None:
        return np.nan, np.nan, ""

    full_sign = np.sign(full["slope"])
    slopes = []
    signs = []

    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        fit = slope_fit(x[keep], y[keep])
        if fit:
            slopes.append(fit["slope"])
            signs.append(np.sign(fit["slope"]))

    if not slopes:
        return np.nan, np.nan, ""

    return (
        float(np.mean(np.asarray(signs) == full_sign)),
        float(np.median(slopes)),
        ";".join(str(round(v, 5)) for v in slopes[:50]),
    )

def classify_big_effect(row):
    n = row.get("n", 0)
    abs_effect = row.get("abs_effect_range", np.nan)
    abs_rho = row.get("abs_spearman_r", np.nan)
    p = row.get("perm_p_spearman", np.nan)
    loo = row.get("loo_sign_stability", np.nan)

    if n >= 20 and abs_effect >= 0.20 and abs_rho >= 0.45 and (pd.isna(p) or p <= 0.10) and (pd.isna(loo) or loo >= 0.80):
        return "strong_expanded_trait_environment_effect"
    if n >= 12 and abs_effect >= 0.25 and abs_rho >= 0.55 and (pd.isna(p) or p <= 0.15) and (pd.isna(loo) or loo >= 0.80):
        return "candidate_expanded_trait_environment_effect"
    if n >= 8 and abs_effect >= 0.30 and abs_rho >= 0.65:
        return "small_n_candidate_trait_environment_effect"
    return "weak_or_exploratory"

# ---------------------------------------------------------------------
# 1. Inventory candidate CSVs.
# ---------------------------------------------------------------------
csv_paths = []
for root in SEARCH_ROOTS:
    if root.exists():
        csv_paths.extend(root.rglob("*.csv"))

csv_paths = sorted(set(csv_paths))
priority_set = {str(p) for p in PRIORITY_FILES if p.exists()}

inventory_rows = []

for p in csv_paths:
    sample = safe_read(p, nrows=500)
    if sample is None or sample.empty:
        continue

    cols = list(sample.columns)
    lower = [c.lower() for c in cols]

    outcome_hits = [c for c in OUTCOME_CANDIDATES if c.lower() in lower]
    cov_hits = [c for c in COVARIATE_CANDIDATES if c.lower() in lower]
    env_hits = [c for c in ENV_CANDIDATES if c.lower() in lower]

    if not outcome_hits and not cov_hits:
        continue

    score = 0
    score += 10 * len(outcome_hits)
    score += 3 * len(cov_hits)
    score += 1 * len(env_hits)
    if str(p) in priority_set:
        score += 20

    inventory_rows.append({
        "path": str(p),
        "priority_file": str(p) in priority_set,
        "n_sample_rows": len(sample),
        "n_columns": len(cols),
        "outcome_hits": ";".join(outcome_hits),
        "covariate_hits": ";".join(cov_hits),
        "env_hits": ";".join(env_hits),
        "score": score,
    })

inventory = pd.DataFrame(inventory_rows).sort_values("score", ascending=False)
inventory.to_csv(TAB / "Table_PRODUCT02eh_expanded_mechanism_candidate_inventory.csv", index=False)

if inventory.empty:
    raise RuntimeError("No candidate mechanism files found.")

# ---------------------------------------------------------------------
# 2. Pick best file with outcome + covariates.
# ---------------------------------------------------------------------
chosen_path = None
chosen_df = None
chosen_info = None

for _, row in inventory.iterrows():
    p = Path(row["path"])
    df = safe_read(p)
    if df is None or df.empty:
        continue

    outcomes = usable_numeric_cols(df, OUTCOME_CANDIDATES, min_n=8)
    covariates = usable_numeric_cols(df, COVARIATE_CANDIDATES, min_n=8)

    if outcomes and covariates:
        chosen_path = p
        chosen_df = df.copy()
        chosen_info = {
            "path": str(p),
            "outcomes": outcomes,
            "covariates": covariates,
            "n_rows": len(df),
        }
        break

if chosen_df is None:
    # fallback: best candidate, but mark failure.
    chosen_path = Path(inventory.iloc[0]["path"])
    chosen_df = safe_read(chosen_path)
    chosen_info = {
        "path": str(chosen_path),
        "outcomes": usable_numeric_cols(chosen_df, OUTCOME_CANDIDATES, min_n=8) if chosen_df is not None else [],
        "covariates": usable_numeric_cols(chosen_df, COVARIATE_CANDIDATES, min_n=8) if chosen_df is not None else [],
        "n_rows": len(chosen_df) if chosen_df is not None else 0,
    }

chosen_outcomes = chosen_info["outcomes"]
chosen_covariates = chosen_info["covariates"]

# Normalize ID/lat/lon.
id_col = find_col(chosen_df, ["point_id", "site", "site_id", "tower_id", "id"])
lat_col = find_col(chosen_df, ["lat", "latitude", "lat_aridity"])
lon_col = find_col(chosen_df, ["lon", "longitude", "lon_aridity"])

if id_col:
    chosen_df["point_id_norm"] = chosen_df[id_col].astype(str)
else:
    chosen_df["point_id_norm"] = np.arange(len(chosen_df)).astype(str)

if lat_col:
    chosen_df["lat_norm"] = num(chosen_df[lat_col])
else:
    chosen_df["lat_norm"] = np.nan

if lon_col:
    chosen_df["lon_norm"] = num(chosen_df[lon_col])
else:
    chosen_df["lon_norm"] = np.nan

# Add basic environment columns if absent.
if "latitude_band_expanded" not in chosen_df.columns and chosen_df["lat_norm"].notna().any():
    chosen_df["latitude_band_expanded"] = pd.cut(
        chosen_df["lat_norm"],
        bins=[-90, 30, 45, 60, 90],
        labels=["lt30", "30_45", "45_60", "ge60"],
        include_lowest=True,
    ).astype(str)

if "longitude_sector_expanded" not in chosen_df.columns and chosen_df["lon_norm"].notna().any():
    def lonsector(x):
        if pd.isna(x): return "unknown"
        if x < -100: return "western_NA_or_Pacific"
        if x < -70: return "eastern_NA"
        if x < 40: return "Europe_Africa"
        return "Asia_Oceania"
    chosen_df["longitude_sector_expanded"] = chosen_df["lon_norm"].apply(lonsector)

# Add region flags if coordinate-based.
if "env_North_American_Great_Plains_proxy" not in chosen_df.columns:
    chosen_df["env_North_American_Great_Plains_proxy"] = (
        chosen_df["lat_norm"].between(30, 55)
        & chosen_df["lon_norm"].between(-110, -90)
    )

if "env_low_mid_latitude_proxy" not in chosen_df.columns:
    chosen_df["env_low_mid_latitude_proxy"] = chosen_df["lat_norm"].abs() < 45

if "env_high_vpd_proxy" not in chosen_df.columns and "mean_vpd" in chosen_df.columns:
    vpd = num(chosen_df["mean_vpd"])
    chosen_df["env_high_vpd_proxy"] = vpd >= vpd.quantile(0.75)

if "env_dry_aridity_proxy" not in chosen_df.columns:
    arid_col = "aridity_index" if "aridity_index" in chosen_df.columns else ("aridity" if "aridity" in chosen_df.columns else None)
    if arid_col:
        ai = num(chosen_df[arid_col])
        chosen_df["env_dry_aridity_proxy"] = ai <= ai.quantile(0.25)

chosen_df.to_csv(DATA / "expanded_trait_mechanism_input.csv", index=False)

# ---------------------------------------------------------------------
# 3. Build environment masks.
# ---------------------------------------------------------------------
env_masks = []

# Categorical columns.
for col in list(chosen_df.columns):
    cl = col.lower()
    if col in ENV_CANDIDATES or col.startswith("env_") or any(k in cl for k in ["biome", "realm", "region", "latitude_band", "longitude_sector", "hydroclimatic"]):
        vals = chosen_df[col]
        if vals.dtype == bool:
            m = vals.fillna(False).astype(bool)
            if m.sum() >= MIN_ENV_N and (~m).sum() >= MIN_ENV_N:
                env_masks.append((f"{col}=True", m))
        else:
            vc = vals.dropna().astype(str).value_counts()
            for level, n in vc.items():
                if n >= MIN_ENV_N and (len(chosen_df) - n) >= MIN_ENV_N and level.lower() not in ["nan", "none", "unknown"]:
                    m = vals.astype(str).eq(level)
                    env_masks.append((f"{col}=={level}", m))

# Deduplicate masks by name.
dedup = {}
for name, mask in env_masks:
    dedup[name] = mask
env_masks = list(dedup.items())

# ---------------------------------------------------------------------
# 4. Scan covariate effects globally, within environments, and interactions.
# ---------------------------------------------------------------------
scan_rows = []
interaction_rows = []

for outcome in chosen_outcomes:
    y_all = num(chosen_df[outcome])

    for cov in chosen_covariates:
        x_all = zscore(chosen_df[cov])

        # global
        f = slope_fit(x_all, y_all)
        if f:
            loo, loo_slope, loo_slopes = loo_stability(x_all, y_all)
            p = perm_p(x_all, y_all)
            row = {
                "scope": "GLOBAL_EXPANDED",
                "environment": "ALL",
                "environment_n": int(y_all.notna().sum()),
                "outcome": outcome,
                "covariate": cov,
                **f,
                "perm_p_spearman": p,
                "loo_sign_stability": loo,
                "loo_median_slope": loo_slope,
                "loo_slopes": loo_slopes,
            }
            row["claim_strength"] = classify_big_effect(row)
            scan_rows.append(row)

        # within env
        for env_name, mask in env_masks:
            env_n = int(mask.sum())
            xx = x_all[mask]
            yy = y_all[mask]
            f = slope_fit(xx, yy)
            if not f:
                continue

            loo, loo_slope, loo_slopes = loo_stability(xx, yy)
            p = perm_p(xx, yy)
            row = {
                "scope": "WITHIN_ENVIRONMENT_EXPANDED",
                "environment": env_name,
                "environment_n": env_n,
                "outcome": outcome,
                "covariate": cov,
                **f,
                "perm_p_spearman": p,
                "loo_sign_stability": loo,
                "loo_median_slope": loo_slope,
                "loo_slopes": loo_slopes,
            }
            row["claim_strength"] = classify_big_effect(row)
            scan_rows.append(row)

            # interaction
            tmp = pd.DataFrame({"y": y_all, "x": x_all, "env": mask.astype(float)}).dropna()
            if len(tmp) >= max(20, MIN_ENV_N * 2) and tmp["x"].nunique() >= 2 and tmp["env"].nunique() == 2:
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

                strength = "weak_or_exploratory_interaction"
                if env_n >= 12 and np.isfinite(delta) and delta >= 0.05 and abs(float(b1[3])) >= 0.05:
                    strength = "candidate_expanded_trait_environment_interaction"
                if env_n >= 20 and np.isfinite(delta) and delta >= 0.08 and abs(float(b1[3])) >= 0.08:
                    strength = "strong_expanded_trait_environment_interaction"

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
                    "claim_strength": strength,
                })

scan = pd.DataFrame(scan_rows)
interactions = pd.DataFrame(interaction_rows)

if len(scan):
    priority = {
        "strong_expanded_trait_environment_effect": 0,
        "candidate_expanded_trait_environment_effect": 1,
        "small_n_candidate_trait_environment_effect": 2,
        "weak_or_exploratory": 3,
    }
    scan["_priority"] = scan["claim_strength"].map(priority).fillna(9)
    scan = scan.sort_values(
        ["_priority", "abs_effect_range", "abs_spearman_r", "environment_n"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority"])

if len(interactions):
    priority_i = {
        "strong_expanded_trait_environment_interaction": 0,
        "candidate_expanded_trait_environment_interaction": 1,
        "weak_or_exploratory_interaction": 2,
    }
    interactions["_priority"] = interactions["claim_strength"].map(priority_i).fillna(9)
    interactions = interactions.sort_values(
        ["_priority", "delta_r2_interaction", "interaction_abs_effect", "environment_n"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority"])

scan.to_csv(TAB / "Table_PRODUCT02ei_expanded_trait_environment_effect_scan.csv", index=False)
interactions.to_csv(TAB / "Table_PRODUCT02ej_expanded_trait_environment_interaction_scan.csv", index=False)

# Great Plains proxy diagnostic.
gp = scan[scan["environment"].astype(str).str.contains("Great_Plains|Great Plains", case=False, regex=True)].copy() if len(scan) else pd.DataFrame()
gp.to_csv(TAB / "Table_PRODUCT02ek_expanded_great_plains_trait_diagnostic.csv", index=False)

# Coverage table for chosen file.
coverage_rows = []
for c in chosen_covariates:
    s = num(chosen_df[c])
    coverage_rows.append({
        "covariate": c,
        "n_nonmissing": int(s.notna().sum()),
        "n_unique": int(s.nunique(dropna=True)),
        "min": float(s.min(skipna=True)),
        "max": float(s.max(skipna=True)),
    })
coverage = pd.DataFrame(coverage_rows).sort_values("n_nonmissing", ascending=False)
coverage.to_csv(TAB / "Table_PRODUCT02el_expanded_covariate_coverage.csv", index=False)

selected_info = pd.DataFrame([{
    "selected_file": str(chosen_path),
    "n_rows": int(len(chosen_df)),
    "id_col": id_col,
    "lat_col": lat_col,
    "lon_col": lon_col,
    "outcomes": ";".join(chosen_outcomes),
    "covariates": ";".join(chosen_covariates),
    "n_environment_masks": int(len(env_masks)),
}])
selected_info.to_csv(TAB / "Table_PRODUCT02em_expanded_mechanism_selected_dataset.csv", index=False)

# Decision.
strong = scan[scan["claim_strength"].eq("strong_expanded_trait_environment_effect")].head(1) if len(scan) else pd.DataFrame()
candidate = scan[scan["claim_strength"].isin(["strong_expanded_trait_environment_effect", "candidate_expanded_trait_environment_effect"])].head(1) if len(scan) else pd.DataFrame()
interaction_candidate = interactions[interactions["claim_strength"].isin(["strong_expanded_trait_environment_interaction", "candidate_expanded_trait_environment_interaction"])].head(1) if len(interactions) else pd.DataFrame()

if len(strong):
    b = strong.iloc[0]
    verdict = "STRONG_EXPANDED_TRAIT_ENVIRONMENT_EFFECT_FOUND"
    safe_claim = (
        f"In the expanded covariate-covered mechanism screen, {b['covariate']} shows a strong environment-specific association "
        f"with {b['outcome']} within {b['environment']} (n={int(b['environment_n'])}), with effect_range={b['effect_range']:.3f}, "
        f"Spearman r={b['spearman_r']:.3f}, permutation p={b['perm_p_spearman']:.3f}, and LOO sign stability={b['loo_sign_stability']:.3f}. "
        "This supports a large trait/climate/soil-conditioned effect in that environment, while the strict final-13 analysis remains the primary response-inference set."
    )
elif len(candidate):
    b = candidate.iloc[0]
    verdict = "CANDIDATE_EXPANDED_TRAIT_ENVIRONMENT_EFFECT_FOUND"
    safe_claim = (
        f"In the expanded covariate-covered mechanism screen, {b['covariate']} shows a candidate environment-specific association "
        f"with {b['outcome']} within {b['environment']} (n={int(b['environment_n'])}), with effect_range={b['effect_range']:.3f}, "
        f"Spearman r={b['spearman_r']:.3f}, permutation p={b['perm_p_spearman']:.3f}, and LOO sign stability={b['loo_sign_stability']:.3f}. "
        "Frame this as exploratory mechanism evidence, not causal proof."
    )
elif len(interaction_candidate):
    b = interaction_candidate.iloc[0]
    verdict = "CANDIDATE_EXPANDED_TRAIT_ENVIRONMENT_INTERACTION_FOUND"
    safe_claim = (
        f"In the expanded covariate-covered mechanism screen, {b['covariate']} × {b['environment']} is the strongest interaction "
        f"for {b['outcome']} (environment n={int(b['environment_n'])}), with ΔR²={b['delta_r2_interaction']:.3f}. "
        "Frame this as exploratory environment-dependent mechanism evidence."
    )
elif not chosen_outcomes or not chosen_covariates:
    verdict = "NO_EXPANDED_MECHANISM_DATASET_WITH_BOTH_OUTCOME_AND_COVARIATES"
    safe_claim = (
        "No expanded local dataset with both response outcomes and true trait/climate/soil covariates was found. "
        "More data assembly is required before claiming trait mechanism."
    )
else:
    verdict = "NO_BIG_EXPANDED_TRAIT_ENVIRONMENT_EFFECT_FOUND"
    safe_claim = (
        "Expanded covariate-covered screening did not find a large trait/climate/soil effect that passed the screening thresholds. "
        "Keep spatial/biome heterogeneity as the mechanism framing."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "selected_file": str(chosen_path),
    "n_rows_selected": int(len(chosen_df)),
    "n_outcomes": int(len(chosen_outcomes)),
    "n_covariates": int(len(chosen_covariates)),
    "n_environment_masks": int(len(env_masks)),
    "n_scan_rows": int(len(scan)),
    "n_interaction_rows": int(len(interactions)),
    "n_strong_effects": int((scan["claim_strength"].eq("strong_expanded_trait_environment_effect")).sum()) if len(scan) else 0,
    "n_candidate_effects": int((scan["claim_strength"].eq("candidate_expanded_trait_environment_effect")).sum()) if len(scan) else 0,
    "n_candidate_interactions": int((interactions["claim_strength"].isin(["strong_expanded_trait_environment_interaction", "candidate_expanded_trait_environment_interaction"])).sum()) if len(interactions) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_MECHANISM_RESULTS_SECTION" if "FOUND" in verdict else "ASSEMBLE_ADDITIONAL_MECHANISM_COVARIATES",
}])
decision.to_csv(TAB / "Table_PRODUCT02en_expanded_trait_mechanism_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(scan):
        figdf = scan.head(15).copy()
        labels = figdf["covariate"].astype(str) + " | " + figdf["environment"].astype(str)
        vals = figdf["effect_range"]
        plt.figure(figsize=(12, 6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range on response outcome")
        plt.ylabel("Covariate | environment")
        plt.title("Top expanded trait/climate/soil environment effects")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02i_expanded_trait_environment_effects.png", dpi=200)
        plt.close()

    if len(gp):
        gpfig = gp.head(12).copy()
        labels = gpfig["covariate"].astype(str) + " | " + gpfig["outcome"].astype(str)
        vals = gpfig["effect_range"]
        plt.figure(figsize=(10, 5))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range")
        plt.ylabel("Covariate | outcome")
        plt.title("Expanded Great Plains proxy trait diagnostic")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02j_expanded_great_plains_trait_diagnostic.png", dpi=200)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6X expanded trait/environment mechanism screen")
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
report.append("## Selected expanded mechanism dataset")
report.append("")
report.append("```text")
report.append(selected_info.to_string(index=False))
report.append("```")
report.append("")
report.append("## Expanded covariate coverage")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage rows.")
report.append("```")
report.append("")
report.append("## Top expanded trait/environment effects")
report.append("")
report.append("```text")
report.append(scan.head(40).to_string(index=False) if len(scan) else "No scan rows.")
report.append("```")
report.append("")
report.append("## Top expanded trait/environment interactions")
report.append("")
report.append("```text")
report.append(interactions.head(40).to_string(index=False) if len(interactions) else "No interaction rows.")
report.append("```")
report.append("")
report.append("## Expanded Great Plains proxy diagnostic")
report.append("")
report.append("```text")
report.append(gp.head(40).to_string(index=False) if len(gp) else "No Great Plains proxy rows.")
report.append("```")
report.append("")
report.append("## Candidate file inventory")
report.append("")
report.append("```text")
report.append(inventory.head(40).to_string(index=False) if len(inventory) else "No inventory rows.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("- Strict final-13 remains the primary product/tower-centered response inference.")
report.append("- This expanded screen is for mechanism discovery only.")
report.append("- A strong/candidate expanded trait effect can support a mechanism hypothesis in a specific environment.")
report.append("- Do not merge this into a strict 3x3 PML claim.")
report.append(f"- Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6X_EXPANDED_TRAIT_MECHANISM_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6X_expanded_trait_mechanism_screen",
    "status": str(decision["verdict"].iloc[0]),
    "safe_claim": str(decision["safe_claim"].iloc[0]),
    "outputs": {
        "inventory": str(TAB / "Table_PRODUCT02eh_expanded_mechanism_candidate_inventory.csv"),
        "selected_dataset": str(TAB / "Table_PRODUCT02em_expanded_mechanism_selected_dataset.csv"),
        "coverage": str(TAB / "Table_PRODUCT02el_expanded_covariate_coverage.csv"),
        "scan": str(TAB / "Table_PRODUCT02ei_expanded_trait_environment_effect_scan.csv"),
        "interactions": str(TAB / "Table_PRODUCT02ej_expanded_trait_environment_interaction_scan.csv"),
        "great_plains": str(TAB / "Table_PRODUCT02ek_expanded_great_plains_trait_diagnostic.csv"),
        "decision": str(TAB / "Table_PRODUCT02en_expanded_trait_mechanism_decision.csv"),
        "report": str(TXT / "STAGE1B6X_EXPANDED_TRAIT_MECHANISM_REPORT.md"),
    },
}
(TAB / "STAGE1B6X_EXPANDED_TRAIT_MECHANISM_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "expanded_trait_mechanism_input.csv")
print("WROTE", TAB / "Table_PRODUCT02eh_expanded_mechanism_candidate_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT02em_expanded_mechanism_selected_dataset.csv")
print("WROTE", TAB / "Table_PRODUCT02el_expanded_covariate_coverage.csv")
print("WROTE", TAB / "Table_PRODUCT02ei_expanded_trait_environment_effect_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02ej_expanded_trait_environment_interaction_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02ek_expanded_great_plains_trait_diagnostic.csv")
print("WROTE", TAB / "Table_PRODUCT02en_expanded_trait_mechanism_decision.csv")
print("WROTE", TXT / "STAGE1B6X_EXPANDED_TRAIT_MECHANISM_REPORT.md")
