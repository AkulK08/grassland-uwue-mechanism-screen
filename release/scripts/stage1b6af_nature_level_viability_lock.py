from pathlib import Path
from datetime import datetime
import json
import math
import time
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

OUT = Path("results/stage1b6af_nature_level_viability_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6af")
for p in [TAB, TXT, FIG, DATA]:
    p.mkdir(parents=True, exist_ok=True)

SRC = Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv")
if not SRC.exists():
    raise FileNotFoundError(f"Missing required point table: {SRC}")

SEED = 20260630
rng = np.random.default_rng(SEED)

# Keep this moderate so the stage should finish quickly.
N_PERM = 999

MIN_N_CASE = 8
MIN_N_MAIN = 20
MIN_N_FULL_CONTROL = 30

def num(s):
    return pd.to_numeric(s, errors="coerce")

def z(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def rank_arr(x):
    return pd.Series(x).rank(method="average").to_numpy(float)

def fast_spearman_perm(x, y, n_perm=N_PERM):
    d = pd.DataFrame({"x": num(x), "y": num(y)}).dropna()
    if len(d) < MIN_N_CASE or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan, np.nan

    rx = rank_arr(d["x"].to_numpy())
    ry = rank_arr(d["y"].to_numpy())
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    obs = float(np.mean(rx * ry))

    hits = 0
    for _ in range(n_perm):
        rp = rng.permutation(ry)
        val = float(np.mean(rx * rp))
        if abs(val) >= abs(obs):
            hits += 1

    p = float((hits + 1) / (n_perm + 1))
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
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[order] = np.minimum(qv, 1.0)
    return q

def fit_ols(data, y_col, x_cols):
    x_cols = [c for c in x_cols if c in data.columns]
    cols = [y_col] + x_cols
    d = data[cols].copy()
    for c in cols:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < max(MIN_N_CASE, len(x_cols) + 4):
        return None

    y = d[y_col].to_numpy(float)
    parts = [np.ones(len(d))]
    kept = []

    for c in x_cols:
        xc = z(d[c]).to_numpy(float)
        if np.isfinite(xc).all() and np.nanstd(xc) > 0:
            parts.append(xc)
            kept.append(c)

    if len(parts) == 0:
        return None

    X = np.column_stack(parts)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1 - r2) * (len(y) - 1) / max(1, len(y) - X.shape[1]) if np.isfinite(r2) else np.nan

    coefs = {"intercept": float(beta[0])}
    for c, b in zip(kept, beta[1:]):
        coefs[c] = float(b)

    return {
        "n": int(len(d)),
        "index": d.index,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "resid": resid,
        "pred": pred,
        "coefs": coefs,
        "predictors_used": ";".join(kept),
        "fit_data": d,
    }

def loo_trait_sign_stability(data, outcome, controls, trait):
    controls = [c for c in controls if c in data.columns]
    needed = [outcome, trait] + controls
    d = data[needed].copy()
    for c in needed:
        d[c] = num(d[c])
    d = d.dropna()

    if len(d) < MIN_N_CASE:
        return np.nan, np.nan

    if controls:
        cf = fit_ols(d, outcome, controls)
        if cf is None:
            return np.nan, np.nan
        rd = d.loc[cf["index"]].copy()
        rd["residual"] = cf["resid"]
    else:
        rd = d.copy()
        rd["residual"] = rd[outcome]

    full = fit_ols(rd, "residual", [trait])
    if full is None:
        return np.nan, np.nan

    full_slope = full["coefs"].get(trait, np.nan)
    if not np.isfinite(full_slope):
        return np.nan, np.nan

    full_sign = np.sign(full_slope)
    slopes = []
    signs = []

    for i in range(len(rd)):
        train = rd.drop(rd.index[i])
        f = fit_ols(train, "residual", [trait])
        if f is None:
            continue
        slope = f["coefs"].get(trait, np.nan)
        if np.isfinite(slope):
            slopes.append(slope)
            signs.append(np.sign(slope))

    if not signs:
        return np.nan, np.nan

    return float(np.mean(np.asarray(signs) == full_sign)), float(np.median(slopes))

def infer_ok_count(d):
    for c in ["status", "fit_status", "model_status", "ok"]:
        if c in d.columns:
            s = d[c].astype(str).str.lower()
            return int((s.str.contains("ok") | s.str.contains("true") | s.eq("1")).sum())
    return int(len(d))

def unique_count_any(d, names):
    for c in names:
        if c in d.columns:
            return int(d[c].nunique(dropna=True))
    return np.nan

def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# Load main point-level response + trait table.
df = pd.read_csv(SRC)
for c in df.columns:
    if c in [
        "lat", "lon", "latitude", "longitude",
        "latent_post_slope", "latent_slope_change",
        "p_threshold_like", "p_satbreak", "latent_satbreak_probability",
        "rooting_depth", "p50", "psi50", "isohydricity",
        "aridity", "aridity_index",
        "mean_annual_precipitation", "mean_precipitation",
        "mean_annual_temperature", "mean_temperature",
        "mean_lai", "growing_season_mean_lai",
        "soil_sand", "soil_silt", "soil_clay",
        "mean_vpd", "mean_soil_moisture",
    ]:
        df[c] = num(df[c])

if "lat" not in df.columns and "latitude" in df.columns:
    df["lat"] = df["latitude"]
if "lon" not in df.columns and "longitude" in df.columns:
    df["lon"] = df["longitude"]

# Named regimes only. No random KNN/grid windows.
regions = {
    "Temperate_Grasslands_Savannas_Shrublands": (
        df["eco_biome"].astype(str).str.contains("Temperate Grasslands", case=False, na=False)
        if "eco_biome" in df.columns else pd.Series(False, index=df.index)
    ),
    "Temperate_midlatitude_35N_50N": df["lat"].between(35, 50),
    "North_midlatitude_30N_45N": df["lat"].between(30, 45),
    "High_VPD_quartile_Q4": (
        df["mean_vpd_quartile"].astype(str).eq("vpd_Q4_high")
        if "mean_vpd_quartile" in df.columns else pd.Series(False, index=df.index)
    ),
    "Low_VPD_quartile_Q1": (
        df["mean_vpd_quartile"].astype(str).eq("vpd_Q1_low")
        if "mean_vpd_quartile" in df.columns else pd.Series(False, index=df.index)
    ),
    "Sahel_proxy": df["lat"].between(10, 17) & df["lon"].between(-18, 15),
    "Sahel_high_vpd": (
        df["hydroclimatic_regime"].astype(str).eq("sahel_high_vpd")
        if "hydroclimatic_regime" in df.columns else pd.Series(False, index=df.index)
    ),
    "Americas_east": (
        df["longitude_sector"].astype(str).eq("americas_east")
        if "longitude_sector" in df.columns else pd.Series(False, index=df.index)
    ),
    "East_central_Asia_steppe": (
        df["longitude_sector"].astype(str).eq("east_central_asia")
        if "longitude_sector" in df.columns else pd.Series(False, index=df.index)
    ),
    "US_West_Coast_proxy": df["lat"].between(32, 43) & df["lon"].between(-125, -114),
}

outcomes = [c for c in ["latent_post_slope", "p_threshold_like", "latent_slope_change"] if c in df.columns]
traits = [c for c in ["rooting_depth", "p50", "psi50", "isohydricity"] if c in df.columns]

control_sets = {
    "none": [],
    "soil_texture_only": ["soil_sand", "soil_silt", "soil_clay"],
    "climate_lai_only": ["aridity", "mean_annual_precipitation", "mean_annual_temperature", "mean_lai"],
    "parsimonious_climate_soil_lai": ["aridity", "mean_annual_temperature", "mean_lai", "soil_sand"],
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

# Create task list.
tasks = []
coverage_rows = []

for region_name, mask in regions.items():
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    sub = df[mask].copy()

    coverage_rows.append({
        "region": region_name,
        "n_points": int(len(sub)),
        "lat_min": float(sub["lat"].min()) if len(sub) else np.nan,
        "lat_max": float(sub["lat"].max()) if len(sub) else np.nan,
        "lon_min": float(sub["lon"].min()) if len(sub) else np.nan,
        "lon_max": float(sub["lon"].max()) if len(sub) else np.nan,
        "rooting_depth_n": int(sub["rooting_depth"].notna().sum()) if "rooting_depth" in sub.columns else 0,
        "p50_n": int(sub["p50"].notna().sum()) if "p50" in sub.columns else 0,
        "isohydricity_n": int(sub["isohydricity"].notna().sum()) if "isohydricity" in sub.columns else 0,
        "median_p_threshold_like": float(sub["p_threshold_like"].median()) if "p_threshold_like" in sub.columns and len(sub) else np.nan,
        "median_latent_post_slope": float(sub["latent_post_slope"].median()) if "latent_post_slope" in sub.columns and len(sub) else np.nan,
        "median_latent_slope_change": float(sub["latent_slope_change"].median()) if "latent_slope_change" in sub.columns and len(sub) else np.nan,
    })

    if len(sub) < MIN_N_CASE:
        continue

    for outcome in outcomes:
        for trait in traits:
            if trait not in sub.columns or outcome not in sub.columns:
                continue
            for control_name, controls in control_sets.items():
                if control_name == "full_climate_soil_lai" and len(sub) < MIN_N_FULL_CONTROL:
                    continue
                if control_name == "parsimonious_climate_soil_lai" and len(sub) < 14:
                    continue
                tasks.append((region_name, mask, outcome, trait, control_name, controls))

rows = []
start = time.time()
iterator = tqdm(tasks, desc="Named-region trait tests", unit="test") if tqdm else tasks

for idx, task in enumerate(iterator, start=1):
    region_name, mask, outcome, trait, control_name, controls = task
    sub = df[mask].copy()

    controls = [c for c in controls if c in sub.columns and sub[c].notna().sum() >= MIN_N_CASE and sub[c].nunique(dropna=True) >= 2]

    if control_name != "none" and not controls:
        continue

    if controls:
        cf = fit_ols(sub, outcome, controls)
        if cf is None:
            continue
        rd = sub.loc[cf["index"]].copy()
        rd["residual"] = cf["resid"]
        control_r2 = cf["r2"]
        control_adj_r2 = cf["adj_r2"]
        controls_used = cf["predictors_used"]
    else:
        rd = sub.copy()
        rd["residual"] = rd[outcome]
        control_r2 = np.nan
        control_adj_r2 = np.nan
        controls_used = ""

    if trait not in rd.columns or rd[trait].notna().sum() < MIN_N_CASE or rd[trait].nunique(dropna=True) < 2:
        continue

    tf = fit_ols(rd, "residual", [trait])
    if tf is None:
        continue

    rho, p = fast_spearman_perm(rd[trait], rd["residual"], N_PERM)
    loo, loo_med = loo_trait_sign_stability(sub, outcome, controls, trait)

    rows.append({
        "region": region_name,
        "outcome": outcome,
        "trait": trait,
        "control_set": control_name,
        "controls_used": controls_used,
        "n_region_points": int(mask.sum()),
        "n_test": tf["n"],
        "control_r2": control_r2,
        "control_adj_r2": control_adj_r2,
        "trait_r2_on_residual": tf["r2"],
        "trait_adj_r2_on_residual": tf["adj_r2"],
        "trait_coef_on_residual": tf["coefs"].get(trait, np.nan),
        "spearman_r": rho,
        "perm_p": p,
        "loo_sign_stability": loo,
        "loo_median_slope": loo_med,
    })

    if not tqdm and idx % 25 == 0:
        print(f"[{idx}/{len(tasks)}] tests complete...")

elapsed = time.time() - start

coverage = pd.DataFrame(coverage_rows)
res = pd.DataFrame(rows)

if len(res):
    res["bh_q_all_named_tests"] = bh_qvalues(res["perm_p"].to_numpy())

    res["passes_discovery_named"] = (
        (res["n_test"] >= MIN_N_CASE)
        & (res["trait_r2_on_residual"] >= 0.20)
        & (res["perm_p"] <= 0.10)
        & (res["loo_sign_stability"].fillna(0) >= 0.80)
    )

    res["passes_case_named"] = (
        (res["n_test"] >= MIN_N_CASE)
        & (res["trait_r2_on_residual"] >= 0.20)
        & (res["perm_p"] <= 0.05)
        & (res["loo_sign_stability"].fillna(0) >= 0.80)
        & (~res["control_set"].eq("none"))
    )

    res["passes_main_named"] = (
        (res["n_test"] >= MIN_N_MAIN)
        & (res["trait_r2_on_residual"] >= 0.20)
        & (res["bh_q_all_named_tests"] <= 0.10)
        & (res["loo_sign_stability"].fillna(0) >= 0.80)
        & (~res["control_set"].eq("none"))
    )

    res["passes_full_control"] = (
        res["control_set"].eq("full_climate_soil_lai")
        & (res["n_test"] >= MIN_N_FULL_CONTROL)
        & (res["trait_r2_on_residual"] >= 0.15)
        & (res["perm_p"] <= 0.10)
        & (res["loo_sign_stability"].fillna(0) >= 0.80)
    )

    res["publication_rank_score"] = (
        res["trait_r2_on_residual"].fillna(0)
        + res["loo_sign_stability"].fillna(0)
        + (1 - res["perm_p"].fillna(1))
        + np.log1p(res["n_test"]) / 5
        + np.where(res["control_set"].eq("full_climate_soil_lai"), 0.30, 0)
        + np.where(res["control_set"].eq("parsimonious_climate_soil_lai"), 0.20, 0)
        + np.where(res["control_set"].eq("soil_texture_only"), 0.15, 0)
        - np.where(res["control_set"].eq("none"), 0.25, 0)
    )

    res = res.sort_values(
        ["passes_main_named", "passes_case_named", "passes_full_control", "publication_rank_score"],
        ascending=[False, False, False, False],
    )

coverage.to_csv(TAB / "Table_PRODUCT02fr_named_regime_coverage.csv", index=False)
res.to_csv(TAB / "Table_PRODUCT02fs_named_regime_trait_tests.csv", index=False)

top = res.head(40).copy() if len(res) else pd.DataFrame()
top.to_csv(TAB / "Table_PRODUCT02ft_top_named_regime_trait_results.csv", index=False)

# Summarize strict matrix / tower evidence from prior outputs if available.
evidence = []

strict_path = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")
if strict_path.exists():
    d = try_read_csv(strict_path)
    if d is not None:
        evidence.append({
            "pillar": "strict_2x2_response_models",
            "available": True,
            "primary_number": len(d),
            "secondary_number": infer_ok_count(d),
            "description": "Strict 2x2 threshold-response fits; primary=all rows, secondary=OK-like rows.",
        })
else:
    evidence.append({
        "pillar": "strict_2x2_response_models",
        "available": False,
        "primary_number": np.nan,
        "secondary_number": np.nan,
        "description": "Strict 2x2 fit table not found at expected path.",
    })

# Prior AC.2 environment-locked result if available.
ac2_path = Path("results/stage1b6ac2_environment_locked_trait_residual_fix/tables/Table_PRODUCT02fi_ac2_residual_trait_tests_FIXED.csv")
if ac2_path.exists():
    d = try_read_csv(ac2_path)
    if d is not None and len(d):
        passes = int(d.get("passes_reviewer_20pct_residual_variance", pd.Series(False, index=d.index)).astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        best = d.sort_values("trait_r2_on_control_residual", ascending=False).iloc[0]
        evidence.append({
            "pillar": "environment_locked_controlled_trait_AC2",
            "available": True,
            "primary_number": int(len(d)),
            "secondary_number": passes,
            "description": f"AC.2 residual trait tests; best={best.get('environment','?')} {best.get('trait','?')}->{best.get('outcome','?')} R2={best.get('trait_r2_on_control_residual',np.nan):.3f}.",
        })
else:
    evidence.append({
        "pillar": "environment_locked_controlled_trait_AC2",
        "available": False,
        "primary_number": np.nan,
        "secondary_number": np.nan,
        "description": "AC.2 controlled residual trait table not found.",
    })

# Prior 1B.6AD atlas if available.
ad_path = Path("results/stage1b6ad_regional_hotspot_trait_atlas/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv")
if ad_path.exists():
    d = try_read_csv(ad_path)
    if d is not None and len(d):
        r = d.iloc[0]
        evidence.append({
            "pillar": "regional_hotspot_atlas_AD",
            "available": True,
            "primary_number": int(r.get("n_trait_tests", np.nan)),
            "secondary_number": int(r.get("n_case_passes", np.nan)),
            "description": f"Regional discovery atlas; verdict={r.get('verdict','?')}; full-control passes={r.get('n_full_control_case_passes','?')}.",
        })
else:
    evidence.append({
        "pillar": "regional_hotspot_atlas_AD",
        "available": False,
        "primary_number": np.nan,
        "secondary_number": np.nan,
        "description": "AD regional hotspot decision table not found.",
    })

evidence_df = pd.DataFrame(evidence)
evidence_df.to_csv(TAB / "Table_PRODUCT02fu_prior_evidence_pillars.csv", index=False)

# Nature-level viability decision.
n_tests = int(len(res))
n_discovery = int(res["passes_discovery_named"].sum()) if len(res) else 0
n_case = int(res["passes_case_named"].sum()) if len(res) else 0
n_main = int(res["passes_main_named"].sum()) if len(res) else 0
n_full = int(res["passes_full_control"].sum()) if len(res) else 0

rooting_named = res[(res["trait"] == "rooting_depth")] if len(res) else pd.DataFrame()
rooting_case = rooting_named[rooting_named["passes_case_named"]] if len(rooting_named) else pd.DataFrame()
rooting_main = rooting_named[rooting_named["passes_main_named"]] if len(rooting_named) else pd.DataFrame()

if n_main > 0 and len(rooting_main) > 0:
    verdict = "NATURE_STYLE_NAMED_REGIME_TRAIT_SIGNAL_STRONG"
    b = rooting_main.iloc[0]
    claim_strength = "main_text_candidate"
elif n_case > 0 and len(rooting_case) > 0:
    verdict = "NATURE_STYLE_NAMED_REGIME_TRAIT_SIGNAL_MODERATE"
    b = rooting_case.iloc[0]
    claim_strength = "controlled_case_candidate"
elif n_discovery > 0:
    verdict = "NATURE_STYLE_SIGNAL_DISCOVERY_ONLY"
    b = res[res["passes_discovery_named"]].iloc[0]
    claim_strength = "discovery_only"
elif len(res):
    verdict = "NATURE_STYLE_SIGNAL_NOT_READY"
    b = res.iloc[0]
    claim_strength = "not_ready"
else:
    verdict = "NO_TESTS_RAN"
    b = None
    claim_strength = "not_ready"

if b is not None:
    safe_claim = (
        f"Top named-regime signal: {b['region']}; trait={b['trait']}; outcome={b['outcome']}; "
        f"controls={b['control_set']}; n={int(b['n_test'])}; residual trait R2={b['trait_r2_on_residual']:.3f}; "
        f"Spearman r={b['spearman_r']:.3f}; permutation p={b['perm_p']:.4f}; "
        f"BH q={b['bh_q_all_named_tests']:.4f}; LOO sign stability={b['loo_sign_stability']:.3f}. "
        f"Claim strength={claim_strength}."
    )
else:
    safe_claim = "No named-regime trait tests ran."

if n_full > 0:
    paper_frame = "Full climate+soil+LAI controlled trait mechanism is available in at least one named regime."
elif n_case > 0:
    paper_frame = "Use controlled named-regime trait mechanism, but state that strongest controlled passes are not full climate+soil+LAI universal proof."
elif n_discovery > 0:
    paper_frame = "Use as discovery atlas only; more confirmation needed before writing a Nature-style mechanism claim."
else:
    paper_frame = "Do not write Nature-style mechanism claim yet; results are not strong enough."

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_perm": N_PERM,
    "runtime_seconds": round(elapsed, 2),
    "n_named_regions": int(len(coverage)),
    "n_named_trait_tests": n_tests,
    "n_discovery_named_passes": n_discovery,
    "n_controlled_case_passes": n_case,
    "n_main_named_passes": n_main,
    "n_full_climate_soil_lai_passes": n_full,
    "verdict": verdict,
    "claim_strength": claim_strength,
    "safe_claim": safe_claim,
    "paper_frame": paper_frame,
    "blocking_next_stage": False,
    "next_stage": "WRITE_NATURE_STYLE_RESULTS_SKELETON" if claim_strength in ["main_text_candidate", "controlled_case_candidate"] else "CONFIRM_OR_DOWNGRADE_BEFORE_WRITING",
}])
decision.to_csv(TAB / "Table_PRODUCT02fv_nature_level_viability_decision.csv", index=False)

# Compact claim-number table for writing.
claim_numbers = []
if b is not None:
    claim_numbers.append({
        "use": "top_named_regime_trait_signal",
        "number": f"{b['region']} | {b['trait']}->{b['outcome']} | controls={b['control_set']} | n={int(b['n_test'])} | R2={b['trait_r2_on_residual']:.3f} | rho={b['spearman_r']:.3f} | p={b['perm_p']:.4f} | q={b['bh_q_all_named_tests']:.4f} | LOO={b['loo_sign_stability']:.3f}",
    })
claim_numbers.append({"use": "named_regime_tests", "number": f"{n_tests} tests across {len(coverage)} named regimes with {N_PERM} permutations"})
claim_numbers.append({"use": "controlled_case_passes", "number": str(n_case)})
claim_numbers.append({"use": "main_named_passes", "number": str(n_main)})
claim_numbers.append({"use": "full_control_passes", "number": str(n_full)})
pd.DataFrame(claim_numbers).to_csv(TAB / "Table_PRODUCT02fw_writing_claim_numbers.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(top):
        plot = top.head(20).copy()
        labels = plot["region"].astype(str) + " | " + plot["trait"].astype(str) + "→" + plot["outcome"].astype(str) + " | " + plot["control_set"].astype(str)
        plt.figure(figsize=(12, 7))
        plt.barh(labels[::-1], plot["trait_r2_on_residual"][::-1])
        plt.xlabel("Trait R2 on residual response")
        plt.ylabel("Named regime | trait → outcome | controls")
        plt.title("Top named-regime trait mechanisms")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02t_top_named_regime_trait_mechanisms.png", dpi=220)
        plt.close()

    if "p_threshold_like" in df.columns:
        plt.figure(figsize=(9, 5))
        sc = plt.scatter(df["lon"], df["lat"], c=df["p_threshold_like"], s=30)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Threshold-like response probability across point dataset")
        plt.colorbar(sc, label="p_threshold_like")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02u_threshold_like_response_map.png", dpi=220)
        plt.close()

    if b is not None:
        mask = pd.Series(regions[b["region"]], index=df.index).fillna(False).astype(bool)
        sub = df[mask].copy()
        controls = control_sets.get(b["control_set"], [])
        controls = [c for c in controls if c in sub.columns]
        if controls:
            cf = fit_ols(sub, b["outcome"], controls)
            if cf is not None:
                rd = sub.loc[cf["index"]].copy()
                rd["residual"] = cf["resid"]
            else:
                rd = pd.DataFrame()
        else:
            rd = sub.copy()
            rd["residual"] = rd[b["outcome"]]

        if len(rd) and b["trait"] in rd.columns:
            pdat = rd[[b["trait"], "residual"]].dropna()
            if len(pdat) >= 8:
                plt.figure(figsize=(6.5, 4.8))
                plt.scatter(pdat[b["trait"]], pdat["residual"], alpha=0.85)
                coef = np.polyfit(pdat[b["trait"]], pdat["residual"], 1)
                xs = np.linspace(pdat[b["trait"]].min(), pdat[b["trait"]].max(), 100)
                plt.plot(xs, coef[0] * xs + coef[1], linestyle="--")
                plt.xlabel(b["trait"])
                plt.ylabel(f"{b['outcome']} residual")
                plt.title(f"Top named-regime signal: {b['region']}")
                plt.tight_layout()
                plt.savefig(FIG / "Figure_PRODUCT02v_top_signal_scatter.png", dpi=220)
                plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6AF Nature-level viability lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append(f"Runtime seconds: {elapsed:.2f}")
report.append(f"Permutations per test: {N_PERM}")
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
report.append("## Paper frame")
report.append("")
report.append(paper_frame)
report.append("")
report.append("## Prior evidence pillars")
report.append("")
report.append("```text")
report.append(evidence_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Named-regime coverage")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False))
report.append("```")
report.append("")
report.append("## Top named-regime trait results")
report.append("")
report.append("```text")
report.append(top.head(40).to_string(index=False) if len(top) else "No named-regime tests passed or no tests ran.")
report.append("```")
report.append("")
report.append("## Writing claim numbers")
report.append("")
report.append("```text")
report.append(pd.DataFrame(claim_numbers).to_string(index=False))
report.append("```")
report.append("")
report.append("## Interpretation rule")
report.append("")
report.append("- Main Nature-style result requires named regime, n >= 20, controlled model, R2 >= 0.20, BH q <= 0.10, and LOO sign stability >= 0.80.")
report.append("- Controlled case result requires named regime, n >= 8, non-null controls, R2 >= 0.20, permutation p <= 0.05, and LOO sign stability >= 0.80.")
report.append("- No-control results are discovery only.")
report.append("- Full climate+soil+LAI controls are only meaningful when n >= 30.")
report.append("- The paper should not claim universal grassland WUE breakdown.")
report.append("- The paper should not claim full causal proof; it should frame results as product-screened, tower-informed, trait-conditioned mechanism evidence.")
report.append("")
report.append(f"Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6AF_NATURE_LEVEL_VIABILITY_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6AF_nature_level_viability_lock",
    "status": verdict,
    "claim_strength": claim_strength,
    "safe_claim": safe_claim,
    "paper_frame": paper_frame,
    "runtime_seconds": round(elapsed, 2),
    "outputs": {
        "coverage": str(TAB / "Table_PRODUCT02fr_named_regime_coverage.csv"),
        "tests": str(TAB / "Table_PRODUCT02fs_named_regime_trait_tests.csv"),
        "top_results": str(TAB / "Table_PRODUCT02ft_top_named_regime_trait_results.csv"),
        "evidence_pillars": str(TAB / "Table_PRODUCT02fu_prior_evidence_pillars.csv"),
        "decision": str(TAB / "Table_PRODUCT02fv_nature_level_viability_decision.csv"),
        "claim_numbers": str(TAB / "Table_PRODUCT02fw_writing_claim_numbers.csv"),
        "report": str(TXT / "STAGE1B6AF_NATURE_LEVEL_VIABILITY_LOCK_REPORT.md"),
    },
}
(TAB / "STAGE1B6AF_NATURE_LEVEL_VIABILITY_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02fr_named_regime_coverage.csv")
print("WROTE", TAB / "Table_PRODUCT02fs_named_regime_trait_tests.csv")
print("WROTE", TAB / "Table_PRODUCT02ft_top_named_regime_trait_results.csv")
print("WROTE", TAB / "Table_PRODUCT02fu_prior_evidence_pillars.csv")
print("WROTE", TAB / "Table_PRODUCT02fv_nature_level_viability_decision.csv")
print("WROTE", TAB / "Table_PRODUCT02fw_writing_claim_numbers.csv")
print("WROTE", TXT / "STAGE1B6AF_NATURE_LEVEL_VIABILITY_LOCK_REPORT.md")
print("WROTE figures to", FIG)
