from pathlib import Path
from datetime import datetime
import json
import itertools
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6w_trait_environment_effect")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6w")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

INPUT = Path("data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv")
FITS = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")

SEED = 20260629
rng = np.random.default_rng(SEED)

OUTCOME = "satellite_limitation_mean_fraction"
MIN_ENV_N_FOR_LOCK = 4
N_PERM = 10000

def num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

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
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    corr = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return {
        "n": int(len(x)),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "r2": float(r2),
        "spearman_r": corr,
        "abs_spearman_r": abs(corr),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "effect_range": float(beta[1] * (np.max(x) - np.min(x))),
    }

def perm_p_spearman(x, y, n_perm=N_PERM):
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

def leave_one_out_sign_stability(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    signs = []
    slopes = []
    if len(x) < 4:
        return np.nan, np.nan, ""
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        out = slope_r2(x[keep], y[keep])
        if out:
            slopes.append(out["slope"])
            signs.append(np.sign(out["slope"]))
    if not slopes:
        return np.nan, np.nan, ""
    full = slope_r2(x, y)
    full_sign = np.sign(full["slope"]) if full else np.nan
    stable = np.mean(np.asarray(signs) == full_sign)
    return float(stable), float(np.nanmedian(slopes)), ";".join([str(round(v, 4)) for v in slopes])

def binary_env_from_column(df, col, level):
    return df[col].astype(str).eq(str(level))

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT)
df["point_id"] = df["point_id"].astype(str)
df = df.drop_duplicates("point_id").copy()
df[OUTCOME] = num(df[OUTCOME])

# Build extra outcome summaries from raw fits if present.
if FITS.exists():
    fits = pd.read_csv(FITS)
    fits["point_id"] = fits["point_id"].astype(str)
    fits["is_limitation_like"] = fits["response_class"].isin(["breakdown", "saturation", "weakening"])
    metric_summary = fits.groupby(["point_id", "metric"]).agg(
        n_fits=("response_class", "size"),
        n_ok=("fit_status", lambda s: int((s == "OK").sum())),
        limitation_fraction=("is_limitation_like", "mean"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
    ).reset_index()
    metric_wide = metric_summary.pivot(index="point_id", columns="metric", values="limitation_fraction").reset_index()
    metric_wide.columns = [str(c) for c in metric_wide.columns]
    metric_wide = metric_wide.rename(columns={
        "log_wue": "limitation_fraction_log_wue_from_fits",
        "log_uwue": "limitation_fraction_log_uwue_from_fits",
    })
    df = df.merge(metric_wide, on="point_id", how="left")

# Candidate traits and covariates.
trait_candidates = [
    "p50", "isohydricity", "rooting_depth",
    "soil_sand", "soil_clay", "soil_silt",
    "soil_sand_mean", "soil_clay_mean", "soil_silt_mean",
    "soil_texture_coarse_index", "soil_texture_fine_index",
    "mean_vpd", "mean_soil_moisture",
    "mean_annual_temperature", "mean_annual_precipitation",
    "mean_lai", "growing_season_mean_lai",
    "aridity_index", "aridity",
    "lat", "lon", "abs_lat",
]
trait_candidates = [c for c in trait_candidates if c in df.columns]

# Keep a trait coverage audit.
trait_audit_rows = []
for c in trait_candidates:
    s = num(df[c])
    trait_audit_rows.append({
        "candidate": c,
        "n_nonmissing": int(s.notna().sum()),
        "n_unique": int(s.nunique(dropna=True)),
        "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
        "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
        "usable_for_trait_scan": bool(s.notna().sum() >= 4 and s.nunique(dropna=True) >= 2),
    })
trait_audit = pd.DataFrame(trait_audit_rows)
trait_audit.to_csv(TAB / "Table_PRODUCT02dp_trait_covariate_coverage_audit.csv", index=False)

trait_candidates = [
    r["candidate"] for _, r in trait_audit.iterrows()
    if bool(r["usable_for_trait_scan"])
]

# Candidate environments.
envs = []

# Hand-built environments.
df["env_Great_Plains_core"] = df["point_id"].isin(["US-Ne1", "US-Ne2", "US-Ne3"])
df["env_US_sites"] = df["point_id"].astype(str).str.startswith("US-")
df["env_nonUS_sites"] = ~df["env_US_sites"]
df["env_US_midlat_or_lowlat"] = df["env_US_sites"] & df.get("latitude_band_handbuilt", "").astype(str).isin(["low_mid_lat_lt35", "mid_lat_35_50"])
df["env_NA_moderate_high_signal_region"] = df["point_id"].isin(["US-CMW", "US-Ne1", "US-Ne2", "US-Ne3", "US-Ton", "US-Dk1", "US-Var", "US-Cop", "US-SP1"])
df["env_high_lat_or_temperate_nonUS"] = df["point_id"].isin(["CA-SF3", "NL-Hrw", "RU-NeC", "CN-HaM"])

for c in [
    "env_Great_Plains_core",
    "env_US_sites",
    "env_nonUS_sites",
    "env_US_midlat_or_lowlat",
    "env_NA_moderate_high_signal_region",
    "env_high_lat_or_temperate_nonUS",
]:
    envs.append((c, df[c].astype(bool), "handbuilt"))

# All categorical levels with at least 2 in-level and 2 out-level.
cat_cols = [
    "broad_region_handbuilt",
    "us_vs_nonus",
    "latitude_band_handbuilt",
    "longitude_sector_handbuilt",
    "eco_biome",
    "eco_realm",
    "hydroclimatic_regime",
    "aridity_quartile",
    "mean_vpd_quartile",
    "geo_continent",
    "geo_subregion",
]
for col in cat_cols:
    if col not in df.columns:
        continue
    vals = df[col].dropna().astype(str)
    for level, n in vals.value_counts().items():
        mask = binary_env_from_column(df, col, level)
        if mask.sum() >= 2 and (~mask).sum() >= 2:
            safe_name = f"env_{col}__{str(level).replace(' ', '_').replace('/', '_')}"
            df[safe_name] = mask
            envs.append((safe_name, df[safe_name].astype(bool), f"{col}=={level}"))

# Continuous outcomes to scan.
outcomes = [OUTCOME]
for c in ["limitation_fraction_log_wue_from_fits", "limitation_fraction_log_uwue_from_fits"]:
    if c in df.columns:
        outcomes.append(c)

scan_rows = []
interaction_rows = []

for outcome in outcomes:
    y_all = num(df[outcome])

    for trait in trait_candidates:
        x_all = zscore(df[trait])

        # Global trait effect.
        global_fit = slope_r2(x_all, y_all)
        if global_fit:
            loo_stable, loo_median_slope, loo_slopes = leave_one_out_sign_stability(x_all, y_all)
            scan_rows.append({
                "scope": "GLOBAL_ALL_13",
                "environment": "ALL",
                "environment_source": "global",
                "environment_n": int(df.shape[0]),
                "outcome": outcome,
                "trait": trait,
                "trait_z": True,
                **global_fit,
                "perm_p_spearman": perm_p_spearman(x_all, y_all),
                "loo_sign_stability": loo_stable,
                "loo_median_slope": loo_median_slope,
                "loo_slopes": loo_slopes,
                "claim_strength": "global_trait_screen_only",
            })

        # Within-environment trait effects.
        for env_name, mask, env_src in envs:
            env_n = int(mask.sum())
            if env_n < 2:
                continue

            xx = x_all[mask]
            yy = y_all[mask]
            fit = slope_r2(xx, yy)
            if fit:
                loo_stable, loo_median_slope, loo_slopes = leave_one_out_sign_stability(xx, yy)
                if env_n >= MIN_ENV_N_FOR_LOCK and abs(fit["effect_range"]) >= 0.20 and abs(fit["spearman_r"]) >= 0.70:
                    strength = "candidate_big_environment_trait_effect"
                elif env_n < MIN_ENV_N_FOR_LOCK and abs(fit["effect_range"]) >= 0.20:
                    strength = "hypothesis_only_small_n_environment"
                else:
                    strength = "weak_or_exploratory"

                scan_rows.append({
                    "scope": "WITHIN_ENVIRONMENT",
                    "environment": env_name,
                    "environment_source": env_src,
                    "environment_n": env_n,
                    "outcome": outcome,
                    "trait": trait,
                    "trait_z": True,
                    **fit,
                    "perm_p_spearman": perm_p_spearman(xx, yy) if env_n >= 4 else np.nan,
                    "loo_sign_stability": loo_stable,
                    "loo_median_slope": loo_median_slope,
                    "loo_slopes": loo_slopes,
                    "claim_strength": strength,
                })

            # Trait x environment interaction using all sites:
            # y ~ trait + env + trait*env, tiny-n screen.
            if env_n >= 2 and (~mask).sum() >= 2:
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
                    pred0 = X0 @ b0
                    rss0 = float(np.sum((y - pred0) ** 2))

                    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
                    pred1 = X1 @ b1
                    rss1 = float(np.sum((y - pred1) ** 2))

                    tss = float(np.sum((y - y.mean()) ** 2))
                    r2_0 = 1 - rss0 / tss if tss > 0 else np.nan
                    r2_1 = 1 - rss1 / tss if tss > 0 else np.nan
                    delta_r2 = r2_1 - r2_0 if np.isfinite(r2_0) and np.isfinite(r2_1) else np.nan

                    interaction_rows.append({
                        "environment": env_name,
                        "environment_source": env_src,
                        "environment_n": env_n,
                        "outcome": outcome,
                        "trait": trait,
                        "n_total": int(len(tmp)),
                        "coef_trait_main": float(b1[1]),
                        "coef_env_main": float(b1[2]),
                        "coef_trait_x_env": float(b1[3]),
                        "r2_without_interaction": float(r2_0),
                        "r2_with_interaction": float(r2_1),
                        "delta_r2_interaction": float(delta_r2),
                        "interaction_abs_effect": abs(float(b1[3])),
                        "claim_strength": (
                            "candidate_trait_environment_interaction"
                            if env_n >= MIN_ENV_N_FOR_LOCK and np.isfinite(delta_r2) and delta_r2 >= 0.10 and abs(float(b1[3])) >= 0.10
                            else "exploratory_or_small_n"
                        ),
                    })

scan = pd.DataFrame(scan_rows)
interactions = pd.DataFrame(interaction_rows)

if len(scan):
    scan["abs_effect_range"] = scan["effect_range"].abs()
    scan["abs_slope"] = scan["slope"].abs()
    scan = scan.sort_values(
        ["claim_strength", "environment_n", "abs_effect_range", "abs_spearman_r"],
        ascending=[True, False, False, False]
    )
    # Custom priority sort: candidate rows first.
    priority = {
        "candidate_big_environment_trait_effect": 0,
        "hypothesis_only_small_n_environment": 1,
        "global_trait_screen_only": 2,
        "weak_or_exploratory": 3,
    }
    scan["_priority"] = scan["claim_strength"].map(priority).fillna(9)
    scan = scan.sort_values(["_priority", "abs_effect_range", "abs_spearman_r", "environment_n"], ascending=[True, False, False, False])
    scan = scan.drop(columns=["_priority"])

if len(interactions):
    priority2 = {
        "candidate_trait_environment_interaction": 0,
        "exploratory_or_small_n": 1,
    }
    interactions["_priority"] = interactions["claim_strength"].map(priority2).fillna(9)
    interactions = interactions.sort_values(["_priority", "delta_r2_interaction", "interaction_abs_effect", "environment_n"], ascending=[True, False, False, False])
    interactions = interactions.drop(columns=["_priority"])

scan.to_csv(TAB / "Table_PRODUCT02dq_trait_effects_within_environment_scan.csv", index=False)
interactions.to_csv(TAB / "Table_PRODUCT02dr_trait_by_environment_interaction_scan.csv", index=False)

# Great Plains special diagnostic.
gp = scan[(scan["environment"].eq("env_Great_Plains_core"))].copy() if len(scan) else pd.DataFrame()
gp.to_csv(TAB / "Table_PRODUCT02ds_great_plains_trait_effect_diagnostic.csv", index=False)

# Select locked result.
candidate = scan[scan["claim_strength"].eq("candidate_big_environment_trait_effect")].head(1) if len(scan) else pd.DataFrame()
candidate_interaction = interactions[interactions["claim_strength"].eq("candidate_trait_environment_interaction")].head(1) if len(interactions) else pd.DataFrame()

if len(candidate):
    best = candidate.iloc[0].to_dict()
    verdict = "BIG_TRAIT_EFFECT_IN_SPECIFIC_ENVIRONMENT_FOUND"
    safe_claim = (
        f"Exploratory conditional analysis identifies a large {best['trait']} association within "
        f"{best['environment']} ({best['environment_source']}), with n={int(best['environment_n'])}, "
        f"effect_range={best['effect_range']:.3f}, Spearman r={best['spearman_r']:.3f}, "
        f"and leave-one-out sign stability={best['loo_sign_stability']:.3f}. "
        "This supports a specific trait-conditioned environment hypothesis, but remains small-n and should be framed as strong exploratory evidence rather than causal proof."
    )
elif len(candidate_interaction):
    best = candidate_interaction.iloc[0].to_dict()
    verdict = "TRAIT_BY_ENVIRONMENT_INTERACTION_FOUND"
    safe_claim = (
        f"Exploratory interaction screening identifies {best['trait']} × {best['environment']} as a large interaction "
        f"for {best['outcome']}, with delta_R2={best['delta_r2_interaction']:.3f} and interaction coefficient "
        f"{best['coef_trait_x_env']:.3f}. This supports targeted trait-by-environment follow-up, not causal proof."
    )
elif len(gp):
    topgp = gp.sort_values("abs_effect_range", ascending=False).iloc[0].to_dict()
    verdict = "GREAT_PLAINS_TRAIT_EFFECT_ONLY_HYPOTHESIS_SMALL_N"
    safe_claim = (
        f"Great Plains-only trait screening is underpowered because it has only n={int(topgp['environment_n'])} sites. "
        f"The largest Great Plains candidate is {topgp['trait']} with effect_range={topgp['effect_range']:.3f} "
        f"and Spearman r={topgp['spearman_r']:.3f}, but this is hypothesis-generating only."
    )
else:
    verdict = "NO_DEFENSIBLE_TRAIT_ENVIRONMENT_EFFECT_FOUND"
    safe_claim = (
        "No defensible large trait effect within a specific environment was found under the current n=13 screen. "
        "Use spatial/biome heterogeneity as the main result and treat traits as future/hypothesis-generating."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_sites": int(df["point_id"].nunique()),
    "n_trait_covariates_usable": int(len(trait_candidates)),
    "n_environment_masks_scanned": int(len(envs)),
    "n_within_environment_trait_tests": int(len(scan)),
    "n_trait_environment_interaction_tests": int(len(interactions)),
    "n_candidate_big_environment_trait_effects": int((scan["claim_strength"].eq("candidate_big_environment_trait_effect")).sum()) if len(scan) else 0,
    "n_candidate_trait_environment_interactions": int((interactions["claim_strength"].eq("candidate_trait_environment_interaction")).sum()) if len(interactions) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_MECHANISM_RESULTS_SECTION_OR_EXPAND_SITE_SET",
}])
decision.to_csv(TAB / "Table_PRODUCT02dt_trait_environment_effect_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt
    if len(scan):
        figdf = scan.head(15).copy()
        labels = figdf["trait"].astype(str) + " | " + figdf["environment"].astype(str)
        vals = figdf["effect_range"]
        plt.figure(figsize=(10, 6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Trait effect range on limitation fraction")
        plt.ylabel("Trait | environment")
        plt.title("Top conditional trait effects by environment")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02c_top_trait_environment_effects.png", dpi=200)
        plt.close()

    if len(gp):
        gpfig = gp.sort_values("abs_effect_range", ascending=False).head(10)
        labels = gpfig["trait"].astype(str)
        vals = gpfig["effect_range"]
        plt.figure(figsize=(8, 4))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range on limitation fraction")
        plt.ylabel("Trait/covariate")
        plt.title("Great Plains trait-effect diagnostic")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02d_great_plains_trait_diagnostic.png", dpi=200)
        plt.close()
    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6W trait × environment effect scan")
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
report.append("## Trait/covariate coverage audit")
report.append("")
report.append("```text")
report.append(trait_audit.to_string(index=False) if len(trait_audit) else "No trait/covariate audit.")
report.append("```")
report.append("")
report.append("## Top within-environment trait effects")
report.append("")
report.append("```text")
report.append(scan.head(40).to_string(index=False) if len(scan) else "No within-environment scan rows.")
report.append("```")
report.append("")
report.append("## Top trait × environment interactions")
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
report.append("## Interpretation rules")
report.append("")
report.append("- A Great Plains-only result with n=3 is hypothesis-generating, not proof.")
report.append("- A stronger claim requires an environment with at least 4 independent sites plus large effect range and leave-one-out sign stability.")
report.append("- If no trait effect passes, keep spatial/biome heterogeneity as the main result and frame trait analysis as exploratory context.")
report.append("")
report.append("## Outputs")
report.append("")
report.append("- Trait coverage audit: `results/stage1b6w_trait_environment_effect/tables/Table_PRODUCT02dp_trait_covariate_coverage_audit.csv`")
report.append("- Within-environment trait scan: `results/stage1b6w_trait_environment_effect/tables/Table_PRODUCT02dq_trait_effects_within_environment_scan.csv`")
report.append("- Trait × environment interaction scan: `results/stage1b6w_trait_environment_effect/tables/Table_PRODUCT02dr_trait_by_environment_interaction_scan.csv`")
report.append("- Great Plains diagnostic: `results/stage1b6w_trait_environment_effect/tables/Table_PRODUCT02ds_great_plains_trait_effect_diagnostic.csv`")
report.append("- Decision: `results/stage1b6w_trait_environment_effect/tables/Table_PRODUCT02dt_trait_environment_effect_decision.csv`")
report.append(f"- Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6W_TRAIT_ENVIRONMENT_EFFECT_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6W_trait_environment_effect",
    "status": str(decision["verdict"].iloc[0]),
    "safe_claim": str(decision["safe_claim"].iloc[0]),
    "outputs": {
        "trait_audit": str(TAB / "Table_PRODUCT02dp_trait_covariate_coverage_audit.csv"),
        "within_environment_scan": str(TAB / "Table_PRODUCT02dq_trait_effects_within_environment_scan.csv"),
        "interaction_scan": str(TAB / "Table_PRODUCT02dr_trait_by_environment_interaction_scan.csv"),
        "great_plains_diagnostic": str(TAB / "Table_PRODUCT02ds_great_plains_trait_effect_diagnostic.csv"),
        "decision": str(TAB / "Table_PRODUCT02dt_trait_environment_effect_decision.csv"),
        "report": str(TXT / "STAGE1B6W_TRAIT_ENVIRONMENT_EFFECT_REPORT.md"),
    },
}
(TAB / "STAGE1B6W_TRAIT_ENVIRONMENT_EFFECT_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02dp_trait_covariate_coverage_audit.csv")
print("WROTE", TAB / "Table_PRODUCT02dq_trait_effects_within_environment_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02dr_trait_by_environment_interaction_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02ds_great_plains_trait_effect_diagnostic.csv")
print("WROTE", TAB / "Table_PRODUCT02dt_trait_environment_effect_decision.csv")
print("WROTE", TXT / "STAGE1B6W_TRAIT_ENVIRONMENT_EFFECT_REPORT.md")
print("WROTE figures to", FIG)
