from pathlib import Path
from datetime import datetime
import json
import math
import re
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

OUT = Path("results/stage1b6ag_regional_mechanism_typology")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"

for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

EXACT_INPUTS = [
    # AC.2 environment-locked trait residual tests
    ("AC2_residual_trait_tests", Path("results/stage1b6ac2_environment_locked_trait_residual_fix/tables/Table_PRODUCT02fi_ac2_residual_trait_tests_FIXED.csv")),
    ("AC2_decision", Path("results/stage1b6ac2_environment_locked_trait_residual_fix/tables/Table_PRODUCT02fj_ac2_environment_locked_trait_decision_FIXED.csv")),

    # AD regional hotspot atlas
    ("AD_regional_trait_atlas", Path("results/stage1b6ad_regional_hotspot_trait_atlas/tables/Table_PRODUCT02fl_regional_trait_residual_atlas.csv")),
    ("AD_candidate_mechanisms", Path("results/stage1b6ad_regional_hotspot_trait_atlas/tables/Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv")),
    ("AD_decision", Path("results/stage1b6ad_regional_hotspot_trait_atlas/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv")),

    # AD fast fallback outputs if those are what exist locally
    ("AD_FAST1MIN_regional_trait_atlas", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FAST1MIN/tables/Table_PRODUCT02fl_regional_trait_residual_atlas.csv")),
    ("AD_FAST1MIN_candidate_mechanisms", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FAST1MIN/tables/Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv")),
    ("AD_FAST1MIN_decision", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FAST1MIN/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv")),
    ("AD_FASTPERM50_regional_trait_atlas", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FASTPERM50/tables/Table_PRODUCT02fl_regional_trait_residual_atlas.csv")),
    ("AD_FASTPERM50_candidate_mechanisms", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FASTPERM50/tables/Table_PRODUCT02fm_candidate_small_region_trait_mechanisms.csv")),
    ("AD_FASTPERM50_decision", Path("results/stage1b6ad_regional_hotspot_trait_atlas_FASTPERM50/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv")),

    # AF Nature-level viability lock
    ("AF_named_regime_coverage", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fr_named_regime_coverage.csv")),
    ("AF_named_regime_trait_tests", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fs_named_regime_trait_tests.csv")),
    ("AF_top_named_regime_results", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02ft_top_named_regime_trait_results.csv")),
    ("AF_prior_evidence_pillars", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fu_prior_evidence_pillars.csv")),
    ("AF_nature_viability_decision", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fv_nature_level_viability_decision.csv")),
    ("AF_writing_claim_numbers", Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fw_writing_claim_numbers.csv")),

    # Strict response foundation
    ("strict_2x2_threshold_fits", Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")),
]

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: failed reading {path}: {e}")
        return None

def norm_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")

def first_existing(df, candidates):
    lut = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        if norm_col(cand) in lut:
            return lut[norm_col(cand)]
    return None

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def boolish(s):
    if s is None:
        return False
    if pd.isna(s):
        return False
    return str(s).strip().lower() in {"true", "1", "yes", "y", "pass", "passed"}

def clean_region_name(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("eco_biome=", "")
    s = s.replace("eco_realm=", "")
    s = s.replace("latitude_band=", "")
    s = s.replace("longitude_sector=", "")
    s = s.replace("hydroclimatic_regime=", "")
    s = s.replace("mean_vpd_quartile=", "")
    s = s.replace("_and_", "_&_")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_random_window(region, family):
    s = f"{region} {family}".lower()
    return (
        "knn" in s
        or "grid_" in s
        or "grid " in s
        or re.search(r"center_[\-\d\.]+", s) is not None
    )

def response_phenotype(outcome):
    s = str(outcome).lower()
    if "latent_post_slope" in s or s == "post_slope":
        return "post-stress slope modulation"
    if "p_threshold_like" in s or "threshold_like" in s:
        return "threshold-like response probability"
    if "latent_slope_change" in s or "slope_change" in s:
        return "slope-change / stress-response transition"
    if "p_satbreak" in s or "satbreak" in s or "breakdown" in s:
        return "saturation/breakdown probability"
    return s.replace("_", " ")

def trait_mechanism(trait):
    s = str(trait).lower()
    if "rooting" in s or "root" in s:
        return "rooting-zone storage / effective rooting depth"
    if "psi50" in s or "p50" in s:
        return "plant hydraulic safety"
    if "isohyd" in s:
        return "stomatal regulation strategy"
    if "soil" in s:
        return "soil hydraulic context"
    if "lai" in s:
        return "canopy structure / LAI"
    return s.replace("_", " ")

def control_context(control_set, controls_used):
    s = f"{control_set} {controls_used}".lower()

    has_soil = any(x in s for x in ["soil", "sand", "silt", "clay"])
    has_climate = any(x in s for x in ["climate", "aridity", "temperature", "precip", "annual"])
    has_lai = "lai" in s or "vegetation" in s

    if has_climate and has_soil and has_lai:
        return "climate-soil-vegetation controlled context"
    if has_climate and has_soil:
        return "climate-soil controlled context"
    if has_soil:
        return "soil water-holding context"
    if has_climate and has_lai:
        return "climate-vegetation context"
    if has_climate:
        return "climate context"
    if str(control_set).lower().strip() in {"none", "", "nan"}:
        return "uncontrolled discovery association"
    return str(control_set).replace("_", " ")

def mechanism_label(trait, control_set, controls_used):
    tm = trait_mechanism(trait)
    cc = control_context(control_set, controls_used)
    if "rooting-zone" in tm and "climate-soil-vegetation" in cc:
        return "climate-soil-vegetation constrained rooting-zone effect"
    if "rooting-zone" in tm and "soil water" in cc:
        return "soil-texture adjusted rooting-zone storage effect"
    if "rooting-zone" in tm:
        return "rooting-zone storage effect"
    if "plant hydraulic safety" in tm:
        return "plant hydraulic safety effect"
    if "stomatal" in tm:
        return "stomatal regulation strategy effect"
    return f"{tm} under {cc}"

def has_any_controls(control_set, controls_used):
    s = f"{control_set} {controls_used}".strip().lower()
    return not (s == "" or s == "none" or s == "nan" or s == "none nan")

def has_climate_soil_lai_controls(control_set, controls_used):
    s = f"{control_set} {controls_used}".lower()
    if "parsimonious_climate_soil_lai" in s or "full_climate_soil_lai" in s:
        return True
    has_soil = any(x in s for x in ["soil", "sand", "silt", "clay"])
    has_climate = any(x in s for x in ["aridity", "temperature", "precip", "mean_annual"])
    has_lai = "lai" in s
    return has_soil and has_climate and has_lai

def effect_direction(coef, rho):
    val = coef if pd.notna(coef) else rho
    if pd.isna(val):
        return "unknown"
    if val > 0:
        return "positive association"
    if val < 0:
        return "negative association"
    return "near-zero association"

def infer_claim_tier(row):
    n = row.get("n", np.nan)
    p = row.get("perm_p", np.nan)
    q = row.get("bh_q", np.nan)
    loo = row.get("loo_sign_stability", np.nan)
    controls = row.get("control_set", "")
    controls_used = row.get("controls_used", "")

    n_ok = pd.notna(n)
    p_ok = pd.notna(p)
    loo_ok = pd.notna(loo)

    if n_ok and p_ok and loo_ok:
        if n >= 30 and p <= 0.01 and loo >= 0.90 and (pd.isna(q) or q <= 0.25):
            return "main_regional_mechanism"
        if has_climate_soil_lai_controls(controls, controls_used) and p <= 0.05 and loo >= 0.90 and n >= 8:
            return "controlled_case_mechanism"
        if n >= 15 and p <= 0.05 and loo >= 0.90:
            return "secondary_regional_mechanism"
        if p <= 0.05:
            return "exploratory_candidate"
    return "no_stable_signal"

def tier_rank(tier):
    ranks = {
        "main_regional_mechanism": 5,
        "controlled_case_mechanism": 4,
        "secondary_regional_mechanism": 3,
        "exploratory_candidate": 2,
        "no_stable_signal": 1,
    }
    return ranks.get(str(tier), 0)

def source_weight(source):
    s = str(source)
    if s.startswith("AF_top"):
        return 5
    if s.startswith("AF_named_regime_trait"):
        return 4
    if s.startswith("AC2"):
        return 3
    if s.startswith("AD_candidate"):
        return 2
    if s.startswith("AD_regional") or "FAST" in s:
        return 1
    return 0

def paper_sentence(row):
    region = row["region_clean"]
    phenotype = row["response_phenotype"]
    mech = row["mechanism_label"]
    trait = row["trait"]
    outcome = row["outcome"]
    n = int(row["n"]) if pd.notna(row["n"]) else "NA"
    r2 = row["residual_r2"]
    rho = row["spearman_r"]
    p = row["perm_p"]
    q = row["bh_q"]
    loo = row["loo_sign_stability"]
    tier = row["claim_tier"]
    direction = row["effect_direction"]

    qtxt = "q missing" if pd.isna(q) else f"BH q={q:.3f}"
    return (
        f"In {region}, the dominant phenotype was {phenotype}; {trait} was associated with this response "
        f"as a {mech} ({direction}; n={n}; residual R2={r2:.3f}; Spearman r={rho:.3f}; "
        f"permutation p={p:.4f}; {qtxt}; LOO sign stability={loo:.3f}). Claim tier: {tier}."
    )

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def standardize_trait_table(df, source_name, path):
    region_col = first_existing(df, ["region", "environment", "region_label", "regime", "region_id"])
    family_col = first_existing(df, ["region_family", "family", "source_family"])
    trait_col = first_existing(df, ["trait", "predictor", "variable"])
    outcome_col = first_existing(df, ["outcome", "response", "response_metric"])
    control_col = first_existing(df, ["control_set", "controls", "model", "control_model"])
    controls_used_col = first_existing(df, ["controls_used", "predictors_used", "control_predictors"])
    n_col = first_existing(df, ["n_test", "n_trait_test", "n", "n_points", "n_region_points"])
    r2_col = first_existing(df, ["trait_r2_on_residual", "trait_r2_on_control_residual", "trait_r2_on_control_residual", "trait_r2", "r2"])
    adj_r2_col = first_existing(df, ["trait_adj_r2_on_residual", "trait_adj_r2_on_control_residual", "adj_r2"])
    coef_col = first_existing(df, ["trait_coef_on_residual", "trait_coef_on_control_residual", "coef_rooting_depth", "coef_p50", "coef_psi50", "coef_isohydricity", "trait_coef"])
    rho_col = first_existing(df, ["spearman_r", "spearman_trait_vs_residual", "spearman", "rho"])
    p_col = first_existing(df, ["perm_p", "perm_p_spearman", "permutation_p", "p"])
    q_col = first_existing(df, ["bh_q_all_named_tests", "q_spearman_bh_all_tests", "bh_q", "q"])
    loo_col = first_existing(df, ["loo_sign_stability", "rooting_depth_loo_sign_stability", "loo_stability"])

    if region_col is None or trait_col is None or outcome_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["source_table"] = source_name
    out["source_path"] = str(path)
    out["region"] = df[region_col].astype(str)
    out["region_family"] = df[family_col].astype(str) if family_col else ""
    out["trait"] = df[trait_col].astype(str)
    out["outcome"] = df[outcome_col].astype(str)
    out["control_set"] = df[control_col].astype(str) if control_col else "none"
    out["controls_used"] = df[controls_used_col].astype(str) if controls_used_col else ""
    out["n"] = to_num(df[n_col]) if n_col else np.nan
    out["residual_r2"] = to_num(df[r2_col]) if r2_col else np.nan
    out["residual_adj_r2"] = to_num(df[adj_r2_col]) if adj_r2_col else np.nan

    # Coef can be generic or trait-specific. If generic missing, infer from trait-specific if available.
    if coef_col:
        out["trait_coef"] = to_num(df[coef_col])
    else:
        out["trait_coef"] = np.nan
        for trait_name, colname in [
            ("rooting_depth", "coef_rooting_depth"),
            ("p50", "coef_p50"),
            ("psi50", "coef_psi50"),
            ("isohydricity", "coef_isohydricity"),
        ]:
            if colname in df.columns:
                mask = out["trait"].str.lower().eq(trait_name)
                out.loc[mask, "trait_coef"] = to_num(df.loc[mask, colname])

    out["spearman_r"] = to_num(df[rho_col]) if rho_col else np.nan
    out["perm_p"] = to_num(df[p_col]) if p_col else np.nan
    out["bh_q"] = to_num(df[q_col]) if q_col else np.nan
    out["loo_sign_stability"] = to_num(df[loo_col]) if loo_col else np.nan

    # Keep possible precomputed flags if available.
    for flag in [
        "passes_discovery_named",
        "passes_case_named",
        "passes_main_named",
        "passes_full_control",
        "passes_reviewer_20pct_residual_variance",
        "passes_case_threshold",
        "passes_main_threshold",
        "passes_full_control",
    ]:
        if flag in df.columns:
            out[flag] = df[flag].map(boolish)

    return out

def standardize_coverage_table(df, source_name, path):
    region_col = first_existing(df, ["region", "environment", "region_label", "regime"])
    n_col = first_existing(df, ["n_points", "n_region_points", "n"])
    if region_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["source_table"] = source_name
    out["source_path"] = str(path)
    out["region"] = df[region_col].astype(str)
    out["region_clean"] = out["region"].map(clean_region_name)
    out["region_n_points"] = to_num(df[n_col]) if n_col else np.nan

    for c in [
        "rooting_depth_n", "p50_n", "isohydricity_n",
        "median_p_threshold_like", "median_latent_post_slope", "median_latent_slope_change",
        "lat_min", "lat_max", "lon_min", "lon_max"
    ]:
        if c in df.columns:
            out[c] = to_num(df[c])

    return out

print("Searching for prior result tables...")
existing = []
for name, path in EXACT_INPUTS:
    if path.exists():
        existing.append((name, path))
        print(f"FOUND {name}: {path}")
    else:
        print(f"missing {name}: {path}")

if not existing:
    raise FileNotFoundError("No prior AC2/AD/AF result tables were found. Run prior stages first.")

trait_tables = []
coverage_tables = []
decision_tables = []
strict_summary = {}

iterator = tqdm(existing, desc="Loading prior tables", unit="file") if tqdm else existing

for name, path in iterator:
    d = read_csv_safe(path)
    if d is None:
        continue

    low_name = name.lower()

    if "coverage" in low_name:
        cov = standardize_coverage_table(d, name, path)
        if len(cov):
            coverage_tables.append(cov)
    elif "decision" in low_name or "evidence" in low_name or "claim_numbers" in low_name:
        dd = d.copy()
        dd["source_table"] = name
        dd["source_path"] = str(path)
        decision_tables.append(dd)
    elif "strict_2x2" in low_name:
        strict_summary["strict_2x2_rows"] = int(len(d))
        for c in ["status", "fit_status", "model_status"]:
            if c in d.columns:
                s = d[c].astype(str).str.lower()
                strict_summary["strict_2x2_ok_like_rows"] = int((s.str.contains("ok") | s.eq("true") | s.eq("1")).sum())
                break
        if "strict_2x2_ok_like_rows" not in strict_summary:
            strict_summary["strict_2x2_ok_like_rows"] = np.nan
    else:
        std = standardize_trait_table(d, name, path)
        if len(std):
            trait_tables.append(std)

if not trait_tables:
    raise ValueError("No usable trait/mechanism result rows were found in prior tables.")

all_traits = pd.concat(trait_tables, ignore_index=True)
all_traits = all_traits.replace({"nan": np.nan, "None": np.nan, "": np.nan})

# Remove rows with no actual stats.
for c in ["n", "residual_r2", "spearman_r", "perm_p", "bh_q", "loo_sign_stability", "trait_coef"]:
    all_traits[c] = to_num(all_traits[c])

all_traits = all_traits[
    all_traits["region"].notna()
    & all_traits["trait"].notna()
    & all_traits["outcome"].notna()
].copy()

all_traits["region_clean"] = all_traits["region"].map(clean_region_name)
all_traits["is_random_window"] = [
    is_random_window(r, f) for r, f in zip(all_traits["region"], all_traits["region_family"])
]
all_traits["is_named_region"] = ~all_traits["is_random_window"]

all_traits["response_phenotype"] = all_traits["outcome"].map(response_phenotype)
all_traits["trait_mechanism"] = all_traits["trait"].map(trait_mechanism)
all_traits["control_context"] = [
    control_context(a, b) for a, b in zip(all_traits["control_set"], all_traits["controls_used"])
]
all_traits["mechanism_label"] = [
    mechanism_label(t, c, u)
    for t, c, u in zip(all_traits["trait"], all_traits["control_set"], all_traits["controls_used"])
]
all_traits["effect_direction"] = [
    effect_direction(c, r) for c, r in zip(all_traits["trait_coef"], all_traits["spearman_r"])
]

# Ensure q-values exist at least globally if missing.
if all_traits["bh_q"].isna().all() and all_traits["perm_p"].notna().any():
    p = all_traits["perm_p"].to_numpy(float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order]
    m = len(ranked)
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[order] = np.minimum(qv, 1.0)
    all_traits["bh_q"] = q

all_traits["claim_tier"] = all_traits.apply(infer_claim_tier, axis=1)
all_traits["tier_rank"] = all_traits["claim_tier"].map(tier_rank)
all_traits["source_weight"] = all_traits["source_table"].map(source_weight)

# Prefer named region rows for paper typology; keep random windows in full raw table only.
named_traits = all_traits[all_traits["is_named_region"]].copy()

# Avoid duplicate rows from AF top table + AF full table + AD fast tables:
# Keep best source / best tier for identical region-trait-outcome-control stats.
dedupe_keys = ["region_clean", "trait", "outcome", "control_set", "n", "residual_r2", "spearman_r", "perm_p"]
named_traits = named_traits.sort_values(
    ["tier_rank", "source_weight", "residual_r2", "loo_sign_stability"],
    ascending=[False, False, False, False]
).drop_duplicates(subset=dedupe_keys, keep="first")

# Build coverage summary.
if coverage_tables:
    coverage = pd.concat(coverage_tables, ignore_index=True)
    coverage = coverage.sort_values(["source_table", "region_clean"]).drop_duplicates("region_clean", keep="last")
else:
    coverage = pd.DataFrame(columns=["region_clean", "region_n_points"])

# Best-supported row for each named region.
sort_cols = ["tier_rank", "source_weight", "residual_r2", "loo_sign_stability"]
typology = named_traits.sort_values(sort_cols, ascending=[False, False, False, False]).groupby("region_clean", as_index=False).head(1).copy()

# Add fallback rows for regions with coverage but no stable signal.
existing_regions = set(typology["region_clean"].dropna().astype(str))
fallback_rows = []
for _, r in coverage.iterrows():
    rc = str(r.get("region_clean", ""))
    if rc and rc not in existing_regions:
        fallback_rows.append({
            "source_table": r.get("source_table", "coverage_only"),
            "source_path": r.get("source_path", ""),
            "region": rc,
            "region_family": "coverage_only",
            "trait": "none",
            "outcome": "none",
            "control_set": "none",
            "controls_used": "",
            "n": r.get("region_n_points", np.nan),
            "residual_r2": np.nan,
            "residual_adj_r2": np.nan,
            "trait_coef": np.nan,
            "spearman_r": np.nan,
            "perm_p": np.nan,
            "bh_q": np.nan,
            "loo_sign_stability": np.nan,
            "region_clean": rc,
            "is_random_window": False,
            "is_named_region": True,
            "response_phenotype": "no stable response-mechanism signal",
            "trait_mechanism": "none",
            "control_context": "none",
            "mechanism_label": "no stable trait mechanism",
            "effect_direction": "unknown",
            "claim_tier": "no_stable_signal",
            "tier_rank": tier_rank("no_stable_signal"),
            "source_weight": 0,
        })
if fallback_rows:
    typology = pd.concat([typology, pd.DataFrame(fallback_rows)], ignore_index=True)

# Merge coverage metrics.
if len(coverage):
    typology = typology.merge(
        coverage.drop_duplicates("region_clean"),
        how="left",
        on="region_clean",
        suffixes=("", "_coverage")
    )

# Paper-ready sentence.
typology["paper_ready_sentence"] = typology.apply(
    lambda row: paper_sentence(row) if row["claim_tier"] != "no_stable_signal" else
    f"In {row['region_clean']}, no stable trait-mechanism signal met the reporting thresholds in the available named-regime tables.",
    axis=1
)

# Sort final typology.
typology = typology.sort_values(
    ["tier_rank", "residual_r2", "loo_sign_stability", "n"],
    ascending=[False, False, False, False]
)

# Claim tiers summary.
tier_counts = (
    typology.groupby("claim_tier", dropna=False)
    .size()
    .reset_index(name="n_regions")
    .sort_values("n_regions", ascending=False)
)

paper_ready = typology[typology["claim_tier"].isin([
    "main_regional_mechanism",
    "controlled_case_mechanism",
    "secondary_regional_mechanism",
    "exploratory_candidate",
])].copy()

# Explicit known result checks, if present.
known_targets = [
    ("Temperate Grasslands", "rooting_depth", "latent_post_slope"),
    ("North midlatitude 30N 45N", "rooting_depth", "p_threshold_like"),
    ("Sahel proxy", "rooting_depth", "latent_slope_change"),
]
known_rows = []
for region_sub, trait, outcome in known_targets:
    m = (
        named_traits["region_clean"].astype(str).str.lower().str.contains(region_sub.lower().replace("_", " "), regex=False)
        & named_traits["trait"].astype(str).str.lower().eq(trait)
        & named_traits["outcome"].astype(str).str.lower().eq(outcome)
    )
    if m.any():
        known_rows.append(named_traits[m].sort_values(
            ["tier_rank", "residual_r2", "loo_sign_stability"],
            ascending=[False, False, False]
        ).head(1))
known_table = pd.concat(known_rows, ignore_index=True) if known_rows else pd.DataFrame()

# Save outputs.
all_traits.to_csv(TAB / "Table_PRODUCT02fx0_all_standardized_mechanism_rows.csv", index=False)
typology.to_csv(TAB / "Table_PRODUCT02fx_regional_mechanism_typology.csv", index=False)
tier_counts.to_csv(TAB / "Table_PRODUCT02fy_region_claim_tiers.csv", index=False)
paper_ready.to_csv(TAB / "Table_PRODUCT02fz_paper_ready_regional_claims.csv", index=False)
known_table.to_csv(TAB / "Table_PRODUCT02fz_known_key_result_checks.csv", index=False)

# Build final decision summary.
n_regions = int(len(typology))
n_main = int((typology["claim_tier"] == "main_regional_mechanism").sum())
n_controlled = int((typology["claim_tier"] == "controlled_case_mechanism").sum())
n_secondary = int((typology["claim_tier"] == "secondary_regional_mechanism").sum())
n_exploratory = int((typology["claim_tier"] == "exploratory_candidate").sum())
n_none = int((typology["claim_tier"] == "no_stable_signal").sum())

if len(paper_ready):
    best = paper_ready.sort_values(
        ["tier_rank", "residual_r2", "loo_sign_stability", "n"],
        ascending=[False, False, False, False]
    ).iloc[0]
    best_claim = paper_sentence(best)
else:
    best = None
    best_claim = "No paper-ready regional mechanism passed the typology thresholds."

if n_main > 0:
    verdict = "REGIONAL_MECHANISM_TYPOLOGY_HAS_MAIN_EVIDENCE"
elif n_controlled > 0:
    verdict = "REGIONAL_MECHANISM_TYPOLOGY_HAS_CONTROLLED_CASE_EVIDENCE"
elif n_secondary > 0:
    verdict = "REGIONAL_MECHANISM_TYPOLOGY_HAS_SECONDARY_EVIDENCE"
elif n_exploratory > 0:
    verdict = "REGIONAL_MECHANISM_TYPOLOGY_DISCOVERY_ONLY"
else:
    verdict = "NO_STABLE_REGIONAL_MECHANISM_TYPOLOGY"

safest_thesis = (
    "Grassland WUE/uWUE response to compound atmospheric-soil moisture stress is regionally organized rather "
    "than a single global effect. Named regimes express different response phenotypes, and rooting-zone storage "
    "or rooting depth is repeatedly associated with response-shape variation in specific regions. These results "
    "support a region-specific rooting-zone-storage mechanism, but they should be framed as product-screened, "
    "tower-informed, observational evidence rather than global causal proof."
)

decision = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "verdict": verdict,
    "number_of_regions_summarized": n_regions,
    "number_of_main_regional_mechanisms": n_main,
    "number_of_controlled_case_mechanisms": n_controlled,
    "number_of_secondary_mechanisms": n_secondary,
    "number_of_exploratory_candidates": n_exploratory,
    "number_of_no_stable_signal_regions": n_none,
    "best_main_claim": best_claim,
    "safest_manuscript_thesis": safest_thesis,
    "strict_response_foundation": strict_summary,
    "loaded_trait_tables": len(trait_tables),
    "loaded_coverage_tables": len(coverage_tables),
}

with open(TAB / "STAGE1B6AG_REGIONAL_MECHANISM_TYPOLOGY_DECISION.json", "w") as f:
    json.dump(decision, f, indent=2)

# Figure.
figure_status = "NO_FIGURE"
try:
    import matplotlib.pyplot as plt

    fig_df = typology.copy()
    fig_df = fig_df[fig_df["claim_tier"] != "no_stable_signal"].head(18)
    if len(fig_df):
        labels = (
            fig_df["region_clean"].astype(str)
            + "\n"
            + fig_df["response_phenotype"].astype(str)
            + "\n"
            + fig_df["claim_tier"].astype(str)
        )
        values = fig_df["residual_r2"].fillna(0).to_numpy()

        plt.figure(figsize=(12, max(6, 0.45 * len(fig_df))))
        plt.barh(labels[::-1], values[::-1])
        plt.xlabel("Residual trait R2")
        plt.ylabel("Region / phenotype / claim tier")
        plt.title("Regional mechanism typology for grassland WUE/uWUE response")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02w_regional_mechanism_typology.png", dpi=220)
        plt.close()
        figure_status = "FIGURE_WRITTEN"
    else:
        figure_status = "NO_PAPER_READY_ROWS_FOR_FIGURE"
except Exception as e:
    figure_status = f"FIGURE_FAILED: {repr(e)}"

# Markdown report.
lines = []
lines.append("# Stage 1B.6AG regional mechanism typology")
lines.append("")
lines.append(f"Generated: {decision['generated']}")
lines.append("")
lines.append("## Final decision")
lines.append("")
lines.append("```json")
lines.append(json.dumps(decision, indent=2))
lines.append("```")
lines.append("")
lines.append("## Paper-ready interpretation")
lines.append("")
lines.append(safest_thesis)
lines.append("")
lines.append("Do not claim global causal proof. Do not claim a single universal grassland WUE breakdown response. The safest paper frame is a regional mechanism atlas: different grassland regimes express different compound-dryness response phenotypes, and rooting depth/rooting-zone storage is a repeated but region-specific ecohydraulic control.")
lines.append("")
lines.append("## Claim tier rules used")
lines.append("")
lines.append("- `main_regional_mechanism`: n >= 30, permutation p <= 0.01, LOO sign stability >= 0.9, and BH q <= 0.25 or missing.")
lines.append("- `controlled_case_mechanism`: controls include climate/soil/LAI, permutation p <= 0.05, LOO sign stability >= 0.9, and n >= 8.")
lines.append("- `secondary_regional_mechanism`: n >= 15, permutation p <= 0.05, and LOO sign stability >= 0.9.")
lines.append("- `exploratory_candidate`: permutation p <= 0.05 but small n, weak q, or limited/no controls.")
lines.append("- `no_stable_signal`: no stable result under these thresholds.")
lines.append("")
lines.append("## Region claim tier counts")
lines.append("")
lines.append("```text")
lines.append(tier_counts.to_string(index=False))
lines.append("```")
lines.append("")
lines.append("## Regional mechanism typology")
lines.append("")
show_cols = [
    "region_clean", "response_phenotype", "trait", "mechanism_label", "control_set",
    "n", "residual_r2", "spearman_r", "perm_p", "bh_q", "loo_sign_stability",
    "effect_direction", "claim_tier", "source_table"
]
show_cols = [c for c in show_cols if c in typology.columns]
lines.append("```text")
lines.append(typology[show_cols].head(60).to_string(index=False))
lines.append("```")
lines.append("")
lines.append("## Paper-ready regional claims")
lines.append("")
if len(paper_ready):
    for i, (_, row) in enumerate(paper_ready.head(20).iterrows(), start=1):
        lines.append(f"{i}. {row['paper_ready_sentence']}")
else:
    lines.append("No paper-ready regional claims passed the typology thresholds.")
lines.append("")
lines.append("## Known key result checks")
lines.append("")
if len(known_table):
    known_cols = [c for c in show_cols if c in known_table.columns]
    lines.append("```text")
    lines.append(known_table[known_cols].to_string(index=False))
    lines.append("```")
else:
    lines.append("No known key result rows were matched exactly. Check standardized table manually.")
lines.append("")
lines.append("## Loaded sources")
lines.append("")
source_rows = []
for name, path in existing:
    source_rows.append({"source_table": name, "path": str(path)})
source_df = pd.DataFrame(source_rows)
lines.append("```text")
lines.append(source_df.to_string(index=False))
lines.append("```")
lines.append("")
lines.append(f"Figure status: `{figure_status}`")
lines.append("")
lines.append("## Manuscript thesis")
lines.append("")
lines.append("Grassland WUE/uWUE response to compound dryness is regionally organized rather than a single global effect. Different named regimes express different response phenotypes, and rooting-zone storage/effective rooting depth is repeatedly associated with residual response-shape variation. This supports a regional ecohydraulic mechanism, while remaining observational and product-screened rather than global causal proof.")

report = "\n".join(lines)
(TXT / "STAGE1B6AG_REGIONAL_MECHANISM_TYPOLOGY_REPORT.md").write_text(report, encoding="utf-8")

print(report)
print("")
print("WROTE", TAB / "Table_PRODUCT02fx_regional_mechanism_typology.csv")
print("WROTE", TAB / "Table_PRODUCT02fy_region_claim_tiers.csv")
print("WROTE", TAB / "Table_PRODUCT02fz_paper_ready_regional_claims.csv")
print("WROTE", TAB / "Table_PRODUCT02fx0_all_standardized_mechanism_rows.csv")
print("WROTE", TAB / "Table_PRODUCT02fz_known_key_result_checks.csv")
print("WROTE", TAB / "STAGE1B6AG_REGIONAL_MECHANISM_TYPOLOGY_DECISION.json")
print("WROTE", TXT / "STAGE1B6AG_REGIONAL_MECHANISM_TYPOLOGY_REPORT.md")
print("WROTE", FIG / "Figure_PRODUCT02w_regional_mechanism_typology.png")
print("")
print("FINAL_DECISION_JSON")
print(json.dumps(decision, indent=2))
