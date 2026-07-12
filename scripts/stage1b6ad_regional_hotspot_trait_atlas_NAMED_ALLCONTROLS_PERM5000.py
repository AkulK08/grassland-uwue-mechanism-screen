from pathlib import Path
from datetime import datetime
import json
import math
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6ad_regional_hotspot_trait_atlas_NAMED_ALLCONTROLS_PERM5000")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6ad")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SRC = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")
if not SRC.exists():
    raise FileNotFoundError(f"Missing required table: {SRC}")

SEED = 20260630
rng = np.random.default_rng(SEED)
N_PERM = 5000

MIN_N_EXPLORATORY = 6
MIN_N_CASE = 8
MIN_N_MAIN = 20

df = pd.read_csv(SRC)

def num(s):
    return pd.to_numeric(s, errors="coerce")

def z(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def safe_col(c):
    return str(c).replace(" ", "_").replace("/", "_").replace("&", "and").replace(",", "").replace("(", "").replace(")", "")

for c in df.columns:
    if c in [
        "lat", "lon", "latitude", "longitude",
        "latent_post_slope", "latent_slope_change",
        "latent_satbreak_probability", "p_satbreak", "p_threshold_like",
        "rooting_depth", "p50", "psi50", "isohydricity",
        "aridity", "aridity_index",
        "mean_annual_precipitation", "mean_precipitation",
        "mean_annual_temperature", "mean_temperature",
        "mean_lai", "growing_season_mean_lai",
        "soil_sand", "soil_silt", "soil_clay",
        "mean_vpd", "mean_soil_moisture"
    ]:
        df[c] = num(df[c])

if "lat" not in df.columns and "latitude" in df.columns:
    df["lat"] = df["latitude"]
if "lon" not in df.columns and "longitude" in df.columns:
    df["lon"] = df["longitude"]

required = ["lat", "lon"]
for r in required:
    if r not in df.columns:
        raise ValueError(f"Missing required coordinate column: {r}")

OUTCOMES = [c for c in ["latent_post_slope", "latent_slope_change", "p_threshold_like", "p_satbreak", "latent_satbreak_probability"] if c in df.columns]
TRAITS = [c for c in ["rooting_depth", "p50", "psi50", "isohydricity"] if c in df.columns]

CONTROL_SETS = {
    "none": [],
    "soil_texture_only": ["soil_sand", "soil_silt", "soil_clay"],
    "climate_lai_only": ["aridity", "mean_annual_precipitation", "mean_annual_temperature", "mean_lai"],
    "parsimonious_aridity_temp_lai_soil": ["aridity", "mean_annual_temperature", "mean_lai", "soil_sand"],
    "full_climate_soil_lai": [
        "aridity",
        "mean_annual_precipitation",
        "mean_annual_temperature",
        "mean_lai",
        "growing_season_mean_lai",
        "soil_sand",
        "soil_silt",
        "soil_clay",
    ],
}

def fit_ols(data, y_col, x_cols):
    x_cols = [c for c in x_cols if c in data.columns]
    cols = [y_col] + x_cols
    d = data[cols].copy()
    for c in cols:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < max(6, len(x_cols) + 3):
        return None

    y = d[y_col].to_numpy(float)
    parts = [np.ones(len(d))]
    kept = []
    for c in x_cols:
        xc = z(d[c]).to_numpy(float)
        if np.isfinite(xc).all() and np.nanstd(xc) > 0:
            parts.append(xc)
            kept.append(c)

    X = np.column_stack(parts)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean())**2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1-r2) * (len(y)-1) / max(1, len(y)-X.shape[1]) if np.isfinite(r2) else np.nan

    coefs = {"intercept": float(beta[0])}
    for c, b in zip(kept, beta[1:]):
        coefs[c] = float(b)

    return {
        "n": int(len(d)),
        "index": d.index,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "resid": resid,
        "pred": pred,
        "predictors_used": ";".join(kept),
        "coefs": coefs,
        "fit_data": d,
    }

def spearman_perm(x, y):
    d = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(d) < MIN_N_CASE or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan, np.nan

    obs = float(d["x"].corr(d["y"], method="spearman"))
    xx = d["x"].to_numpy()
    yy = d["y"].to_numpy()
    vals = []
    for _ in range(N_PERM):
        vals.append(pd.Series(xx).corr(pd.Series(rng.permutation(yy)), method="spearman"))
    vals = np.asarray(vals)
    p = float((np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1))
    return obs, p

def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order]
    m = len(ranked)
    qvals = ranked * m / np.arange(1, m+1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    q[order] = np.minimum(qvals, 1.0)
    return q

def loo_sign_stability_for_trait(data, y_col, controls, trait):
    needed = [y_col, trait] + [c for c in controls if c in data.columns]
    d = data[needed].copy()
    for c in needed:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < MIN_N_CASE:
        return np.nan, np.nan

    signs = []
    slopes = []

    full_control = fit_ols(d, y_col, controls)
    if full_control is None:
        return np.nan, np.nan

    resid_df = d.loc[full_control["index"]].copy()
    resid_df["control_residual"] = full_control["resid"]

    full_trait = fit_ols(resid_df, "control_residual", [trait])
    if full_trait is None:
        return np.nan, np.nan

    full_sign = np.sign(full_trait["coefs"].get(trait, np.nan))
    if not np.isfinite(full_sign):
        return np.nan, np.nan

    for i in range(len(resid_df)):
        train = resid_df.drop(resid_df.index[i])
        f = fit_ols(train, "control_residual", [trait])
        if f is None:
            continue
        slope = f["coefs"].get(trait, np.nan)
        if np.isfinite(slope):
            slopes.append(slope)
            signs.append(np.sign(slope))

    if len(signs) == 0:
        return np.nan, np.nan

    return float(np.mean(np.asarray(signs) == full_sign)), float(np.median(slopes))

def add_candidate(label, family, mask, meta, candidates):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    n = int(mask.sum())
    if n >= MIN_N_EXPLORATORY and n < len(df):
        candidates.append({
            "region_id": safe_col(label),
            "region_label": label,
            "region_family": family,
            "n_points": n,
            **meta,
            "mask": mask,
        })

candidates = []

# Handbuilt scientific regions.
handbuilt = {
    "US_West_Coast_Mediterranean_grassland_proxy": (df["lat"].between(32, 43) & df["lon"].between(-125, -114)),
    "California_Sierra_foothill_grassland_proxy": (df["lat"].between(36, 40) & df["lon"].between(-123, -119)),
    "US_Great_Plains_proxy": (df["lat"].between(35, 50) & df["lon"].between(-106, -90)),
    "US_Southwest_dryland_proxy": (df["lat"].between(28, 40) & df["lon"].between(-120, -100)),
    "North_midlatitude_30N_45N": (df["lat"].between(30, 45)),
    "Temperate_midlatitude_35N_50N": (df["lat"].between(35, 50)),
    "Sahel_proxy": (df["lat"].between(10, 17) & df["lon"].between(-18, 15)),
    "West_African_Sahel_core_proxy": (df["lat"].between(10.5, 16.5) & df["lon"].between(-3, 3)),
    "East_Central_Asia_steppe_proxy": (df["lat"].between(35, 55) & df["lon"].between(80, 125)),
    "Australia_grassland_proxy": (df["lat"].between(-40, -20) & df["lon"].between(110, 155)),
}
for label, mask in handbuilt.items():
    add_candidate(label, "handbuilt_region", mask, {}, candidates)

# Categorical columns if present.
for col in ["eco_biome", "eco_realm", "aridity_quartile", "mean_vpd_quartile", "hydroclimatic_regime", "longitude_sector", "latitude_band"]:
    if col in df.columns:
        for val, sub in df.groupby(col, dropna=True):
            add_candidate(f"{col}={val}", f"categorical_{col}", df.index.isin(sub.index), {col: str(val)}, candidates)

# Lat/lon grid windows.
for lat_step, lon_step in [(10, 10), (15, 15), (20, 20)]:
    lat_min = math.floor(df["lat"].min() / lat_step) * lat_step
    lat_max = math.ceil(df["lat"].max() / lat_step) * lat_step
    lon_min = math.floor(df["lon"].min() / lon_step) * lon_step
    lon_max = math.ceil(df["lon"].max() / lon_step) * lon_step
    for la in np.arange(lat_min, lat_max, lat_step):
        for lo in np.arange(lon_min, lon_max, lon_step):
            mask = df["lat"].between(la, la + lat_step) & df["lon"].between(lo, lo + lon_step)
            add_candidate(
                f"grid_{lat_step}x{lon_step}_lat{la:.0f}_{la+lat_step:.0f}_lon{lo:.0f}_{lo+lon_step:.0f}",
                f"grid_{lat_step}x{lon_step}",
                mask,
                {"lat_min": la, "lat_max": la + lat_step, "lon_min": lo, "lon_max": lo + lon_step},
                candidates
            )

# KNN windows in geographic space.
coords = df[["lat", "lon"]].to_numpy(float)
valid_coords = np.isfinite(coords).all(axis=1)
for k in [6, 8, 10, 12, 15, 20, 25, 30]:
    if valid_coords.sum() < k:
        continue
    valid_idx = np.where(valid_coords)[0]
    vc = coords[valid_idx]
    for center_pos, orig_i in enumerate(valid_idx):
        dlat = vc[:,0] - coords[orig_i,0]
        dlon = (vc[:,1] - coords[orig_i,1]) * np.cos(np.deg2rad(coords[orig_i,0]))
        dist = np.sqrt(dlat**2 + dlon**2)
        nearest_valid = valid_idx[np.argsort(dist)[:k]]
        mask = df.index.isin(df.index[nearest_valid])
        add_candidate(
            f"KNN_k{k}_center_{coords[orig_i,0]:.3f}_{coords[orig_i,1]:.3f}",
            f"knn_k{k}",
            mask,
            {"center_lat": coords[orig_i,0], "center_lon": coords[orig_i,1], "k": k},
            candidates
        )

# De-duplicate exact masks.
seen = set()
unique_candidates = []
for c in candidates:
    key = tuple(np.where(c["mask"].to_numpy())[0])
    if key in seen:
        continue
    seen.add(key)
    unique_candidates.append(c)
candidates = unique_candidates
candidates = [c for c in candidates if not (str(c['region_family']).startswith('knn') or str(c['region_family']).startswith('grid'))]

region_rows = []
trait_rows = []

for c in candidates:
    mask = c["mask"]
    inside = df[mask].copy()
    outside = df[~mask].copy()

    row = {
        "region_id": c["region_id"],
        "region_label": c["region_label"],
        "region_family": c["region_family"],
        "n_points": int(len(inside)),
        "fraction_points": float(len(inside) / len(df)),
        "lat_min": float(inside["lat"].min()),
        "lat_max": float(inside["lat"].max()),
        "lon_min": float(inside["lon"].min()),
        "lon_max": float(inside["lon"].max()),
        "lat_center": float(inside["lat"].median()),
        "lon_center": float(inside["lon"].median()),
    }

    if "p_threshold_like" in df.columns:
        row["median_p_threshold_inside"] = float(inside["p_threshold_like"].median())
        row["median_p_threshold_outside"] = float(outside["p_threshold_like"].median())
        row["delta_median_p_threshold"] = row["median_p_threshold_inside"] - row["median_p_threshold_outside"]
        high_cut = df["p_threshold_like"].quantile(0.75)
        row["threshold_hotspot_fraction_inside"] = float((inside["p_threshold_like"] >= high_cut).mean())
        row["threshold_hotspot_fraction_outside"] = float((outside["p_threshold_like"] >= high_cut).mean())
        row["threshold_hotspot_risk_ratio"] = (
            row["threshold_hotspot_fraction_inside"] / row["threshold_hotspot_fraction_outside"]
            if row["threshold_hotspot_fraction_outside"] > 0 else np.nan
        )

    if "latent_post_slope" in df.columns:
        row["median_latent_post_slope_inside"] = float(inside["latent_post_slope"].median())
        row["median_latent_post_slope_outside"] = float(outside["latent_post_slope"].median())
        row["delta_median_latent_post_slope"] = row["median_latent_post_slope_inside"] - row["median_latent_post_slope_outside"]

    if "latent_slope_change" in df.columns:
        row["median_latent_slope_change_inside"] = float(inside["latent_slope_change"].median())
        row["median_latent_slope_change_outside"] = float(outside["latent_slope_change"].median())
        row["delta_median_latent_slope_change"] = row["median_latent_slope_change_inside"] - row["median_latent_slope_change_outside"]

    # Phenotype score emphasizes threshold enrichment and negative/high-stress weakening.
    row["phenotype_score"] = 0.0
    if np.isfinite(row.get("delta_median_p_threshold", np.nan)):
        row["phenotype_score"] += abs(row["delta_median_p_threshold"]) * 10
    if np.isfinite(row.get("threshold_hotspot_risk_ratio", np.nan)):
        row["phenotype_score"] += min(row["threshold_hotspot_risk_ratio"], 5)
    if np.isfinite(row.get("delta_median_latent_post_slope", np.nan)):
        row["phenotype_score"] += min(abs(row["delta_median_latent_post_slope"]) / 5, 5)

    region_rows.append(row)

    for outcome in OUTCOMES:
        for control_name, controls_raw in CONTROL_SETS.items():
            controls = [x for x in controls_raw if x in inside.columns]
            # Full control requires larger n to avoid nonsense.
            if control_name == "full_climate_soil_lai" and len(inside) < 30:
                continue
            if control_name == "parsimonious_aridity_temp_lai_soil" and len(inside) < 14:
                continue

            control_fit = fit_ols(inside, outcome, controls) if controls else None

            if controls and control_fit is None:
                continue

            if controls:
                resid_df = inside.loc[control_fit["index"]].copy()
                resid_df["control_residual"] = control_fit["resid"]
                y_for_trait = "control_residual"
                control_r2 = control_fit["r2"]
                control_adj_r2 = control_fit["adj_r2"]
                predictors_used = control_fit["predictors_used"]
            else:
                resid_df = inside.copy()
                resid_df["control_residual"] = resid_df[outcome]
                y_for_trait = "control_residual"
                control_r2 = np.nan
                control_adj_r2 = np.nan
                predictors_used = ""

            for trait in TRAITS:
                if trait not in resid_df.columns:
                    continue
                if resid_df[trait].notna().sum() < MIN_N_CASE or resid_df[trait].nunique(dropna=True) < 2:
                    continue

                trait_fit = fit_ols(resid_df, y_for_trait, [trait])
                if trait_fit is None:
                    continue

                rho, p = spearman_perm(resid_df[trait], resid_df[y_for_trait])
                loo_stab, loo_med = loo_sign_stability_for_trait(inside, outcome, controls, trait)

                trait_rows.append({
                    "region_id": c["region_id"],
                    "region_label": c["region_label"],
                    "region_family": c["region_family"],
                    "n_region_points": int(len(inside)),
                    "outcome": outcome,
                    "control_set": control_name,
                    "controls_used": predictors_used,
                    "trait": trait,
                    "n_trait_test": trait_fit["n"],
                    "control_r2": control_r2,
                    "control_adj_r2": control_adj_r2,
                    "trait_r2_on_residual": trait_fit["r2"],
                    "trait_adj_r2_on_residual": trait_fit["adj_r2"],
                    "trait_coef_on_residual": trait_fit["coefs"].get(trait, np.nan),
                    "spearman_trait_vs_residual": rho,
                    "perm_p_spearman": p,
                    "loo_sign_stability": loo_stab,
                    "loo_median_slope": loo_med,
                })

regions = pd.DataFrame(region_rows)
traits = pd.DataFrame(trait_rows)

if len(regions):
    regions = regions.sort_values(["phenotype_score", "n_points"], ascending=[False, False])

if len(traits):
    traits["q_spearman_bh_all_tests"] = bh_qvalues(traits["perm_p_spearman"].to_numpy())
    traits["passes_case_threshold"] = (
        (traits["n_trait_test"] >= MIN_N_CASE)
        & (traits["trait_r2_on_residual"] >= 0.20)
        & (traits["perm_p_spearman"] <= 0.10)
        & (traits["loo_sign_stability"].fillna(0) >= 0.80)
    )
    traits["passes_main_threshold"] = (
        (traits["n_trait_test"] >= MIN_N_MAIN)
        & (traits["trait_r2_on_residual"] >= 0.20)
        & (traits["q_spearman_bh_all_tests"] <= 0.10)
        & (traits["loo_sign_stability"].fillna(0) >= 0.80)
        & (~traits["region_family"].str.startswith("knn"))
    )
    traits["passes_full_control"] = traits["control_set"].eq("full_climate_soil_lai") & traits["passes_case_threshold"]
    traits["publication_rank_score"] = (
        traits["trait_r2_on_residual"].fillna(0)
        + traits["loo_sign_stability"].fillna(0)
        + (1 - traits["q_spearman_bh_all_tests"].fillna(1))
        + np.log1p(traits["n_trait_test"]) / 5
    )
    traits = traits.sort_values(
        ["passes_main_threshold", "passes_case_threshold", "publication_rank_score", "trait_r2_on_residual"],
        ascending=[False, False, False, False]
    )

regions.to_csv(TAB / "Table_PRODUCT02fk_regional_phenotype_hotspot_atlas.csv", index=False)
traits.to_csv(TAB / "Table_PRODUCT02fl_regional_trait_residual_atlas.csv", index=False)

# Merge top phenotype + trait summaries.
top_regions = regions.head(50).copy() if len(regions) else pd.DataFrame()
top_traits = traits.head(80).copy() if len(traits) else pd.DataFrame()

if len(top_traits):
    candidate_final = top_traits[
        (top_traits["passes_main_threshold"]) 
        | (top_traits["passes_case_threshold"])
    ].copy()
else:
    candidate_final = pd.DataFrame()

candidate_final.to_csv(TAB / "Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv", index=False)

if len(candidate_final):
    b = candidate_final.iloc[0]
    verdict = "SMALL_REGION_TRAIT_MECHANISM_CANDIDATES_FOUND"
    safe_claim = (
        f"Regional hotspot scanning found candidate small-region trait mechanisms. Top candidate: {b['region_label']} "
        f"({b['region_family']}), outcome={b['outcome']}, trait={b['trait']}, controls={b['control_set']}, "
        f"n={int(b['n_trait_test'])}, residual trait R2={b['trait_r2_on_residual']:.3f}, "
        f"Spearman r={b['spearman_trait_vs_residual']:.3f}, p={b['perm_p_spearman']:.4f}, "
        f"BH q={b['q_spearman_bh_all_tests']:.4f}, LOO sign stability={b['loo_sign_stability']:.3f}. "
        "Treat KNN/grid results as discovery unless replicated in named ecological regions."
    )
else:
    verdict = "NO_SMALL_REGION_TRAIT_MECHANISM_SURVIVES_SCREEN"
    safe_claim = (
        "Small-region scanning did not find a robust trait mechanism under the specified residual R2, permutation, "
        "LOOCV, and multiple-testing criteria. The paper should keep the broader environment-locked rooting-depth result."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_candidate_regions": int(len(regions)),
    "n_trait_tests": int(len(traits)),
    "n_case_passes": int(traits["passes_case_threshold"].sum()) if len(traits) else 0,
    "n_main_passes": int(traits["passes_main_threshold"].sum()) if len(traits) else 0,
    "n_full_control_case_passes": int(traits["passes_full_control"].sum()) if len(traits) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "CONFIRM_TOP_REGIONS_WITH_HELD_OUT_OR_PREDEFINED_REGION_TESTS",
}])
decision.to_csv(TAB / "Table_PRODUCT02fn_regional_hotspot_trait_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(regions):
        plot = regions.head(20).copy()
        plt.figure(figsize=(11, 6))
        plt.barh(plot["region_label"][::-1], plot["phenotype_score"][::-1])
        plt.xlabel("Phenotype hotspot score")
        plt.ylabel("Candidate region")
        plt.title("Top regional threshold/response phenotype hotspots")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02q_top_regional_phenotype_hotspots.png", dpi=220)
        plt.close()

    if len(traits):
        plot = traits.head(20).copy()
        labels = plot["region_label"].astype(str) + " | " + plot["outcome"].astype(str) + " | " + plot["control_set"].astype(str) + " | " + plot["trait"].astype(str)
        plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], plot["trait_r2_on_residual"][::-1])
        plt.xlabel("Trait R2 on residual")
        plt.ylabel("Region | outcome | controls | trait")
        plt.title("Top small-region trait-residual effects")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02r_top_regional_trait_residual_effects.png", dpi=220)
        plt.close()

    # Map-like scatter.
    if "p_threshold_like" in df.columns:
        plt.figure(figsize=(9, 5))
        sc = plt.scatter(df["lon"], df["lat"], c=df["p_threshold_like"], s=30)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Point-level threshold-like response probability")
        plt.colorbar(sc, label="p_threshold_like")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02s_threshold_probability_map.png", dpi=220)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6AD regional hotspot + trait mechanism atlas")
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
report.append("This stage searches for small regions where the response phenotype is strongest, then tests whether rooting depth/P50/isohydricity explain residual response variation within those regions after controls. It is a discovery atlas; final claims require confirmation in named ecological regions or held-out data.")
report.append("")
report.append("## Top phenotype hotspot regions")
report.append("")
report.append("```text")
report.append(top_regions.head(40).to_string(index=False) if len(top_regions) else "No region rows.")
report.append("```")
report.append("")
report.append("## Top regional trait-residual tests")
report.append("")
report.append("```text")
report.append(top_traits.head(60).to_string(index=False) if len(top_traits) else "No trait rows.")
report.append("```")
report.append("")
report.append("## Candidate mechanisms")
report.append("")
report.append("```text")
report.append(candidate_final.head(60).to_string(index=False) if len(candidate_final) else "No candidate mechanisms passed thresholds.")
report.append("```")
report.append("")
report.append("## Interpretation rule")
report.append("")
report.append("- Named ecological regions with n >= 20 and BH q <= 0.10 can be manuscript candidates.")
report.append("- KNN/grid windows are discovery only unless replicated in named ecological regions.")
report.append("- Full climate+soil+LAI controls are only considered meaningful for n >= 30.")
report.append("- Very small regions can be case studies, not the main proof.")
report.append("")
report.append(f"Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6AD_REGIONAL_HOTSPOT_TRAIT_ATLAS_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6AD_regional_hotspot_trait_atlas",
    "status": verdict,
    "safe_claim": safe_claim,
    "outputs": {
        "regions": str(TAB / "Table_PRODUCT02fk_regional_phenotype_hotspot_atlas.csv"),
        "traits": str(TAB / "Table_PRODUCT02fl_regional_trait_residual_atlas.csv"),
        "candidates": str(TAB / "Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv"),
        "decision": str(TAB / "Table_PRODUCT02fn_regional_hotspot_trait_decision.csv"),
        "report": str(TXT / "STAGE1B6AD_REGIONAL_HOTSPOT_TRAIT_ATLAS_REPORT.md"),
    }
}
(TAB / "STAGE1B6AD_REGIONAL_HOTSPOT_TRAIT_ATLAS_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02fk_regional_phenotype_hotspot_atlas.csv")
print("WROTE", TAB / "Table_PRODUCT02fl_regional_trait_residual_atlas.csv")
print("WROTE", TAB / "Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv")
print("WROTE", TAB / "Table_PRODUCT02fn_regional_hotspot_trait_decision.csv")
print("WROTE", TXT / "STAGE1B6AD_REGIONAL_HOTSPOT_TRAIT_ATLAS_REPORT.md")
print("WROTE figures to", FIG)
