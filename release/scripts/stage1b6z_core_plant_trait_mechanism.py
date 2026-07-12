from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6z_core_plant_trait_mechanism")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6z")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

SEED = 20260629
rng = np.random.default_rng(SEED)
N_PERM = 10000
MIN_N_GLOBAL = 10
MIN_N_ENV = 6

DATASETS = [
    {
        "name": "point_geography_high_vpd_23",
        "path": Path("results/paper_point_geography_thesis_lock/tables/Table72_high_vpd_point_geography.csv"),
        "id_col": "point_id",
        "outcomes": [
            "latent_satbreak_probability",
            "latent_satbreak_probability_direct",
            "latent_post_slope",
            "latent_slope_change",
            "p_satbreak",
            "p_threshold_like",
        ],
        "traits": ["p50", "rooting_depth", "isohydricity"],
    },
    {
        "name": "trait_model_ready_allpoints",
        "path": Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
        "id_col": "point_id",
        "outcomes": [
            "sat_or_breakdown_rate",
            "breakdown_rate",
            "saturation_rate",
            "median_slope_change",
            "median_post_slope",
            "median_pre_slope",
        ],
        "traits": ["psi50", "rooting_depth", "isohydricity"],
    },
    {
        "name": "point_geography_all199_if_present",
        "path": Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv"),
        "id_col": "point_id",
        "outcomes": [
            "latent_satbreak_probability",
            "latent_satbreak_probability_direct",
            "latent_post_slope",
            "latent_slope_change",
            "p_satbreak",
            "p_threshold_like",
        ],
        "traits": ["p50", "rooting_depth", "isohydricity"],
    },
]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def fit_line(x, y):
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

    full = fit_line(x, y)
    if full is None:
        return np.nan, np.nan, ""

    full_sign = np.sign(full["slope"])
    slopes = []
    signs = []

    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        f = fit_line(x[keep], y[keep])
        if f:
            slopes.append(f["slope"])
            signs.append(np.sign(f["slope"]))

    if not slopes:
        return np.nan, np.nan, ""

    return (
        float(np.mean(np.asarray(signs) == full_sign)),
        float(np.median(slopes)),
        ";".join(str(round(v, 5)) for v in slopes[:80]),
    )

def classify_effect(row):
    n = row["n"]
    abs_effect = row["abs_effect_range"]
    abs_r = row["abs_spearman_r"]
    p = row["perm_p_spearman"]
    loo = row["loo_sign_stability"]

    if n >= 20 and abs_effect >= 0.15 and abs_r >= 0.35 and (pd.isna(p) or p <= 0.10) and (pd.isna(loo) or loo >= 0.80):
        return "strong_core_trait_effect"

    if n >= 12 and abs_effect >= 0.20 and abs_r >= 0.45 and (pd.isna(p) or p <= 0.15) and (pd.isna(loo) or loo >= 0.75):
        return "candidate_core_trait_effect"

    if n >= 6 and abs_effect >= 0.25 and abs_r >= 0.60:
        return "small_n_trait_signal"

    return "weak_or_exploratory"

def add_environment_columns(df):
    out = df.copy()

    if "lat" in out.columns:
        lat = num(out["lat"])
        out["env_low_mid_latitude"] = lat.abs() < 45
        out["env_high_latitude"] = lat.abs() >= 45

    if "lon" in out.columns:
        lon = num(out["lon"])
        out["env_great_plains_proxy"] = lat_between(out, 30, 55) & lon.between(-110, -90)
        out["env_western_na_proxy"] = lon.between(-125, -100)
        out["env_eastern_na_proxy"] = lon.between(-100, -70)

    if "aridity_quartile" in out.columns:
        for level in out["aridity_quartile"].dropna().astype(str).unique():
            out[f"env_aridity_quartile_{level}"] = out["aridity_quartile"].astype(str).eq(level)

    if "aridity_index" in out.columns:
        ai = num(out["aridity_index"])
        if ai.notna().sum() >= 8:
            out["env_driest_tertile"] = ai <= ai.quantile(1/3)
            out["env_wettest_tertile"] = ai >= ai.quantile(2/3)

    if "mean_vpd_quartile" in out.columns:
        for level in out["mean_vpd_quartile"].dropna().astype(str).unique():
            out[f"env_mean_vpd_quartile_{level}"] = out["mean_vpd_quartile"].astype(str).eq(level)

    for col in ["longitude_sector", "latitude_band", "hydroclimatic_regime", "eco_biome", "eco_realm"]:
        if col in out.columns:
            vals = out[col].dropna().astype(str).value_counts()
            for level, n in vals.items():
                if n >= MIN_N_ENV:
                    safe = str(level).replace(" ", "_").replace("/", "_").replace("=", "_")
                    out[f"env_{col}_{safe}"] = out[col].astype(str).eq(level)

    return out

def lat_between(df, lo, hi):
    if "lat" not in df.columns:
        return pd.Series(False, index=df.index)
    return num(df["lat"]).between(lo, hi)

def aggregate_dataset(raw, id_col, outcomes, traits):
    df = raw.copy()

    available_outcomes = [c for c in outcomes if c in df.columns]
    available_traits = [c for c in traits if c in df.columns]

    keep_cols = [id_col]
    for c in ["lat", "lon", "aridity_index", "aridity_quartile", "mean_vpd_quartile", "longitude_sector", "latitude_band", "hydroclimatic_regime", "eco_biome", "eco_realm"]:
        if c in df.columns:
            keep_cols.append(c)
    keep_cols += available_outcomes + available_traits
    keep_cols = list(dict.fromkeys(keep_cols))

    df = df[keep_cols].copy()
    df[id_col] = df[id_col].astype(str)

    agg = {}
    for c in available_outcomes + available_traits:
        agg[c] = lambda s: pd.to_numeric(s, errors="coerce").median()
    for c in keep_cols:
        if c not in agg and c != id_col:
            agg[c] = "first"

    out = df.groupby(id_col, dropna=False).agg(agg).reset_index()
    out = out.rename(columns={id_col: "point_id"})
    return out, available_outcomes, available_traits

inventory_rows = []
all_scan_rows = []
all_interaction_rows = []
dataset_objects = {}

for ds in DATASETS:
    path = ds["path"]
    name = ds["name"]

    if not path.exists():
        inventory_rows.append({
            "dataset": name,
            "path": str(path),
            "exists": False,
            "usable": False,
            "reason": "missing",
        })
        continue

    raw = pd.read_csv(path)
    id_col = ds["id_col"]
    if id_col not in raw.columns:
        inventory_rows.append({
            "dataset": name,
            "path": str(path),
            "exists": True,
            "usable": False,
            "reason": f"missing_id_col_{id_col}",
            "n_raw_rows": len(raw),
        })
        continue

    df, outcomes, traits = aggregate_dataset(raw, id_col, ds["outcomes"], ds["traits"])
    df = add_environment_columns(df)
    dataset_objects[name] = df

    n_trait_usable = 0
    trait_coverage = []
    for t in traits:
        s = num(df[t])
        usable = s.notna().sum() >= 6 and s.nunique(dropna=True) >= 2
        n_trait_usable += int(usable)
        trait_coverage.append(f"{t}:{int(s.notna().sum())}")

    n_outcome_usable = 0
    outcome_coverage = []
    for o in outcomes:
        s = num(df[o])
        usable = s.notna().sum() >= 6 and s.nunique(dropna=True) >= 2
        n_outcome_usable += int(usable)
        outcome_coverage.append(f"{o}:{int(s.notna().sum())}")

    usable_dataset = n_trait_usable > 0 and n_outcome_usable > 0

    inventory_rows.append({
        "dataset": name,
        "path": str(path),
        "exists": True,
        "usable": usable_dataset,
        "n_raw_rows": len(raw),
        "n_points_after_aggregation": len(df),
        "available_outcomes": ";".join(outcomes),
        "available_traits": ";".join(traits),
        "trait_coverage": ";".join(trait_coverage),
        "outcome_coverage": ";".join(outcome_coverage),
    })

    if not usable_dataset:
        continue

    df.to_csv(DATA / f"{name}_point_level_trait_input.csv", index=False)

    env_cols = []
    for c in df.columns:
        if c.startswith("env_"):
            m = df[c].fillna(False).astype(bool)
            if m.sum() >= MIN_N_ENV and (~m).sum() >= MIN_N_ENV:
                env_cols.append(c)

    for outcome in outcomes:
        y_all = num(df[outcome])
        if y_all.notna().sum() < 6 or y_all.nunique(dropna=True) < 2:
            continue

        for trait in traits:
            x_all = zscore(df[trait])
            if x_all.notna().sum() < 6 or x_all.nunique(dropna=True) < 2:
                continue

            # Global trait effect.
            f = fit_line(x_all, y_all)
            if f:
                loo, loo_slope, loo_slopes = loo_stability(x_all, y_all)
                row = {
                    "dataset": name,
                    "scope": "GLOBAL_POINT_LEVEL",
                    "environment": "ALL",
                    "environment_n": int(y_all.notna().sum()),
                    "outcome": outcome,
                    "trait": trait,
                    **f,
                    "perm_p_spearman": perm_p(x_all, y_all),
                    "loo_sign_stability": loo,
                    "loo_median_slope": loo_slope,
                    "loo_slopes": loo_slopes,
                }
                row["claim_strength"] = classify_effect(row)
                all_scan_rows.append(row)

            # Within-environment trait effect.
            for env in env_cols:
                mask = df[env].fillna(False).astype(bool)
                env_n = int(mask.sum())

                xx = x_all[mask]
                yy = y_all[mask]
                f = fit_line(xx, yy)
                if f:
                    loo, loo_slope, loo_slopes = loo_stability(xx, yy)
                    row = {
                        "dataset": name,
                        "scope": "WITHIN_ENVIRONMENT_POINT_LEVEL",
                        "environment": env,
                        "environment_n": env_n,
                        "outcome": outcome,
                        "trait": trait,
                        **f,
                        "perm_p_spearman": perm_p(xx, yy),
                        "loo_sign_stability": loo,
                        "loo_median_slope": loo_slope,
                        "loo_slopes": loo_slopes,
                    }
                    row["claim_strength"] = classify_effect(row)
                    all_scan_rows.append(row)

                # Trait x environment interaction.
                tmp = pd.DataFrame({
                    "y": y_all,
                    "x": x_all,
                    "env": mask.astype(float),
                }).dropna()

                if len(tmp) >= 12 and tmp["x"].nunique() >= 2 and tmp["env"].nunique() == 2:
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
                    if env_n >= 10 and np.isfinite(delta) and delta >= 0.05 and abs(float(b1[3])) >= 0.05:
                        strength = "candidate_core_trait_environment_interaction"
                    if env_n >= 20 and np.isfinite(delta) and delta >= 0.08 and abs(float(b1[3])) >= 0.08:
                        strength = "strong_core_trait_environment_interaction"

                    all_interaction_rows.append({
                        "dataset": name,
                        "environment": env,
                        "environment_n": env_n,
                        "outcome": outcome,
                        "trait": trait,
                        "n_total": int(len(tmp)),
                        "coef_trait_main": float(b1[1]),
                        "coef_env_main": float(b1[2]),
                        "coef_trait_x_env": float(b1[3]),
                        "r2_without_interaction": float(r20),
                        "r2_with_interaction": float(r21),
                        "delta_r2_interaction": float(delta),
                        "interaction_abs_effect": abs(float(b1[3])),
                        "claim_strength": strength,
                    })

inventory = pd.DataFrame(inventory_rows)
scan = pd.DataFrame(all_scan_rows)
interactions = pd.DataFrame(all_interaction_rows)

inventory.to_csv(TAB / "Table_PRODUCT02es_core_trait_dataset_inventory.csv", index=False)

if len(scan):
    priority = {
        "strong_core_trait_effect": 0,
        "candidate_core_trait_effect": 1,
        "small_n_trait_signal": 2,
        "weak_or_exploratory": 3,
    }
    scan["_priority"] = scan["claim_strength"].map(priority).fillna(9)
    scan = scan.sort_values(
        ["_priority", "abs_effect_range", "abs_spearman_r", "environment_n"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority"])

if len(interactions):
    priority_i = {
        "strong_core_trait_environment_interaction": 0,
        "candidate_core_trait_environment_interaction": 1,
        "weak_or_exploratory_interaction": 2,
    }
    interactions["_priority"] = interactions["claim_strength"].map(priority_i).fillna(9)
    interactions = interactions.sort_values(
        ["_priority", "delta_r2_interaction", "interaction_abs_effect", "environment_n"],
        ascending=[True, False, False, False],
    ).drop(columns=["_priority"])

scan.to_csv(TAB / "Table_PRODUCT02et_core_trait_effect_scan.csv", index=False)
interactions.to_csv(TAB / "Table_PRODUCT02eu_core_trait_environment_interaction_scan.csv", index=False)

# Great Plains-specific diagnostic.
gp = pd.DataFrame()
if len(scan):
    gp = scan[scan["environment"].astype(str).str.contains("great_plains", case=False, regex=False)].copy()
gp.to_csv(TAB / "Table_PRODUCT02ev_core_trait_great_plains_diagnostic.csv", index=False)

# Select best result.
strong = scan[scan["claim_strength"].eq("strong_core_trait_effect")].head(1) if len(scan) else pd.DataFrame()
candidate = scan[scan["claim_strength"].isin(["strong_core_trait_effect", "candidate_core_trait_effect"])].head(1) if len(scan) else pd.DataFrame()
interaction_candidate = interactions[interactions["claim_strength"].isin(["strong_core_trait_environment_interaction", "candidate_core_trait_environment_interaction"])].head(1) if len(interactions) else pd.DataFrame()

if len(strong):
    b = strong.iloc[0]
    verdict = "STRONG_CORE_PLANT_TRAIT_EFFECT_FOUND"
    safe_claim = (
        f"Core plant-trait screening identifies {b['trait']} as a strong predictor of {b['outcome']} "
        f"within {b['environment']} in dataset {b['dataset']} (n={int(b['n'])}; effect_range={b['effect_range']:.3f}; "
        f"Spearman r={b['spearman_r']:.3f}; permutation p={b['perm_p_spearman']:.3f}; "
        f"LOO sign stability={b['loo_sign_stability']:.3f}). This is trait-based mechanism evidence, still observational."
    )
elif len(candidate):
    b = candidate.iloc[0]
    verdict = "CANDIDATE_CORE_PLANT_TRAIT_EFFECT_FOUND"
    safe_claim = (
        f"Core plant-trait screening identifies {b['trait']} as a candidate predictor of {b['outcome']} "
        f"within {b['environment']} in dataset {b['dataset']} (n={int(b['n'])}; effect_range={b['effect_range']:.3f}; "
        f"Spearman r={b['spearman_r']:.3f}; permutation p={b['perm_p_spearman']:.3f}; "
        f"LOO sign stability={b['loo_sign_stability']:.3f}). This supports a trait-based mechanism hypothesis, not causal proof."
    )
elif len(interaction_candidate):
    b = interaction_candidate.iloc[0]
    verdict = "CANDIDATE_CORE_PLANT_TRAIT_BY_ENVIRONMENT_INTERACTION_FOUND"
    safe_claim = (
        f"Core plant-trait interaction screening identifies {b['trait']} × {b['environment']} as the strongest interaction "
        f"for {b['outcome']} in dataset {b['dataset']} (environment n={int(b['environment_n'])}; "
        f"ΔR²={b['delta_r2_interaction']:.3f}; interaction coefficient={b['coef_trait_x_env']:.3f}). "
        "This supports a trait-by-environment mechanism hypothesis, not causal proof."
    )
else:
    verdict = "NO_DEFENSIBLE_CORE_PLANT_TRAIT_EFFECT_FOUND"
    safe_claim = (
        "No core plant-trait effect using only p50/psi50, rooting depth, or isohydricity passed the screening thresholds. "
        "Do not use LAI, climate, soil, latitude, or longitude as a substitute for plant-trait proof."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_datasets_checked": int(len(DATASETS)),
    "n_usable_datasets": int(inventory["usable"].sum()) if "usable" in inventory.columns else 0,
    "n_trait_effect_rows": int(len(scan)),
    "n_trait_interaction_rows": int(len(interactions)),
    "n_strong_core_trait_effects": int((scan["claim_strength"].eq("strong_core_trait_effect")).sum()) if len(scan) else 0,
    "n_candidate_core_trait_effects": int((scan["claim_strength"].eq("candidate_core_trait_effect")).sum()) if len(scan) else 0,
    "n_candidate_core_trait_interactions": int((interactions["claim_strength"].isin(["strong_core_trait_environment_interaction", "candidate_core_trait_environment_interaction"])).sum()) if len(interactions) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_TRAIT_MECHANISM_SECTION_IF_FOUND_OR_REPORT_NO_CORE_TRAIT_PROOF",
}])
decision.to_csv(TAB / "Table_PRODUCT02ew_core_trait_mechanism_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(scan):
        figdf = scan.head(15).copy()
        labels = figdf["trait"].astype(str) + " | " + figdf["environment"].astype(str) + " | " + figdf["outcome"].astype(str)
        vals = figdf["effect_range"]
        plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range")
        plt.ylabel("Trait | environment | outcome")
        plt.title("Top core plant-trait mechanism effects")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02k_core_trait_effects.png", dpi=200)
        plt.close()

    if len(interactions):
        ifig = interactions.head(15).copy()
        labels = ifig["trait"].astype(str) + " × " + ifig["environment"].astype(str)
        vals = ifig["delta_r2_interaction"]
        plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Interaction ΔR²")
        plt.ylabel("Trait × environment")
        plt.title("Top core trait × environment interactions")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02l_core_trait_interactions.png", dpi=200)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6Z core plant-trait mechanism screen")
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
report.append("## Dataset inventory")
report.append("")
report.append("```text")
report.append(inventory.to_string(index=False))
report.append("```")
report.append("")
report.append("## Top core plant-trait effects")
report.append("")
report.append("```text")
report.append(scan.head(50).to_string(index=False) if len(scan) else "No core trait effect rows.")
report.append("```")
report.append("")
report.append("## Top core plant-trait × environment interactions")
report.append("")
report.append("```text")
report.append(interactions.head(50).to_string(index=False) if len(interactions) else "No core trait interaction rows.")
report.append("```")
report.append("")
report.append("## Great Plains diagnostic")
report.append("")
report.append("```text")
report.append(gp.head(50).to_string(index=False) if len(gp) else "No Great Plains core trait rows.")
report.append("```")
report.append("")
report.append("## Strict interpretation")
report.append("")
report.append("- This stage allows only p50/psi50, rooting depth, and isohydricity as predictors.")
report.append("- LAI, aridity, VPD, soil, latitude, longitude, biome, and region are not treated as plant traits here.")
report.append("- Environment terms are allowed only as strata or interactions.")
report.append("- If this stage finds no effect, the paper cannot honestly claim core plant-trait proof.")
report.append(f"- Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6Z_CORE_PLANT_TRAIT_MECHANISM_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6Z_core_plant_trait_mechanism",
    "status": str(decision["verdict"].iloc[0]),
    "safe_claim": str(decision["safe_claim"].iloc[0]),
    "outputs": {
        "inventory": str(TAB / "Table_PRODUCT02es_core_trait_dataset_inventory.csv"),
        "trait_scan": str(TAB / "Table_PRODUCT02et_core_trait_effect_scan.csv"),
        "interactions": str(TAB / "Table_PRODUCT02eu_core_trait_environment_interaction_scan.csv"),
        "great_plains": str(TAB / "Table_PRODUCT02ev_core_trait_great_plains_diagnostic.csv"),
        "decision": str(TAB / "Table_PRODUCT02ew_core_trait_mechanism_decision.csv"),
        "report": str(TXT / "STAGE1B6Z_CORE_PLANT_TRAIT_MECHANISM_REPORT.md"),
    },
}
(TAB / "STAGE1B6Z_CORE_PLANT_TRAIT_MECHANISM_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02es_core_trait_dataset_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT02et_core_trait_effect_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02eu_core_trait_environment_interaction_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02ev_core_trait_great_plains_diagnostic.csv")
print("WROTE", TAB / "Table_PRODUCT02ew_core_trait_mechanism_decision.csv")
print("WROTE", TXT / "STAGE1B6Z_CORE_PLANT_TRAIT_MECHANISM_REPORT.md")
