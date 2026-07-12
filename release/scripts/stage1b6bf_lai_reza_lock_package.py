from pathlib import Path
import re
import json
import hashlib
import subprocess
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bf_lai_reza_lock_package"
TAB = OUT / "tables"
FIG = OUT / "figures"
TXT = OUT / "text"
for p in [TAB, FIG, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT_CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]
POINT_INPUT = next((p for p in POINT_CANDIDATES if p.exists()), None)
if POINT_INPUT is None:
    raise SystemExit("No point-level input table found.")

OBS_INPUT = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"
LC_FLAGS_INPUT = ROOT / "results/stage1b6be_full_reza_lai_artifact_screen/tables/POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv"

# ======================================================================================
# Utility functions
# ======================================================================================

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def safe_name(s):
    s = norm(s)
    if not s:
        s = "x"
    if re.match(r"^[0-9]", s):
        s = "x_" + s
    return s[:80]

def sha256_file(path):
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def first_existing(cols, *names):
    cols = list(cols)
    for n in names:
        if n in cols:
            return n
    low = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in low:
            return low[norm(n)]
    return None

def find_cols(cols, patterns):
    out = []
    for c in cols:
        lc = norm(c)
        if any(re.search(p, lc) for p in patterns):
            out.append(c)
    return out

def zscore(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def formula_vars(formula):
    toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula)
    return sorted(set(t for t in toks if t not in {"C", "I", "Q"}))

def fit_ols(data, formula, cov_type="HC3"):
    vars_needed = formula_vars(formula)
    missing = [v for v in vars_needed if v not in data.columns]
    if missing:
        return None, pd.DataFrame(), f"MISSING_VARS: {missing}"
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(30, len(vars_needed) + 8):
        return None, use, "N_TOO_SMALL"
    try:
        fit = smf.ols(formula, data=use).fit(cov_type=cov_type)
        return fit, use, "FIT_OK"
    except Exception as e:
        return None, use, f"FIT_FAIL: {e}"

def compare_model(data, label, full_formula, reduced_formula, focal_terms, family="", note=""):
    full, use_full, status = fit_ols(data, full_formula)
    red, use_red, red_status = fit_ols(data, reduced_formula)
    rows = []

    if full is None or red is None:
        for term in focal_terms:
            rows.append({
                "test_label": label,
                "family": family,
                "status": status,
                "reduced_status": red_status,
                "focal_term": term,
                "n": len(use_full),
                "note": note,
                "full_formula": full_formula,
                "reduced_formula": reduced_formula,
            })
        return rows

    try:
        full_nr = smf.ols(full_formula, data=use_full).fit()
        red_same = smf.ols(reduced_formula, data=use_full).fit()
        nested_p = float(full_nr.compare_f_test(red_same)[1])
    except Exception:
        nested_p = np.nan

    ci = full.conf_int()

    for term in focal_terms:
        rows.append({
            "test_label": label,
            "family": family,
            "status": "FIT_OK",
            "focal_term": term,
            "n": int(full.nobs),
            "coef": full.params.get(term, np.nan),
            "se_hc3": full.bse.get(term, np.nan),
            "p": full.pvalues.get(term, np.nan),
            "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
            "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
            "ci_excludes_zero": bool(ci.loc[term, 0] * ci.loc[term, 1] > 0) if term in ci.index else False,
            "full_r2": full.rsquared,
            "reduced_r2": red.rsquared,
            "delta_r2": full.rsquared - red.rsquared,
            "full_aic": full.aic,
            "reduced_aic": red.aic,
            "delta_aic_full_minus_reduced": full.aic - red.aic,
            "nested_f_p": nested_p,
            "note": note,
            "full_formula": full_formula,
            "reduced_formula": reduced_formula,
        })
    return rows

def p_adjust_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = ranked * m / np.arange(1, m + 1)
    vals = np.minimum.accumulate(vals[::-1])[::-1]
    q[order] = np.minimum(vals, 1.0)
    return q

def p_adjust_by(pvals):
    p = np.asarray(pvals, dtype=float)
    q_bh = p_adjust_bh(p)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q_bh
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    return np.minimum(q_bh * harmonic, 1.0)

def p_adjust_holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return adj
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    vals = (m - np.arange(0, m)) * ranked
    vals = np.maximum.accumulate(vals)
    vals = np.minimum(vals, 1.0)
    adj[order] = vals
    return adj

def add_adjustments(df, p_col="p", group_col="family"):
    if len(df) == 0 or p_col not in df.columns:
        return df
    df = df.copy()
    df["bh_q"] = np.nan
    df["by_q"] = np.nan
    df["holm_p"] = np.nan
    for fam, idx in df.groupby(group_col).groups.items():
        p = df.loc[idx, p_col].values
        df.loc[idx, "bh_q"] = p_adjust_bh(p)
        df.loc[idx, "by_q"] = p_adjust_by(p)
        df.loc[idx, "holm_p"] = p_adjust_holm(p)
    return df

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return None

def permutation_spearman(x, y, n_perm=10000, seed=123):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4:
        return np.nan, np.nan, len(x)
    obs = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        vals.append(stats.spearmanr(x, yp).correlation)
    vals = np.asarray(vals)
    p = (np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1)
    return float(obs), float(p), len(x)

# ======================================================================================
# Read point table and canonical variables
# ======================================================================================

raw = pd.read_csv(POINT_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()
cols = list(raw.columns)

sources = {
    "point_id": first_existing(cols, "point_id"),
    "y": first_existing(cols, "latent_slope_change"),
    "post": first_existing(cols, "latent_post_slope"),
    "sat": first_existing(cols, "latent_satbreak_probability", "p_satbreak", "p_threshold_like"),
    "vpd": first_existing(cols, "mean_vpd", "mean_obs_vpd"),
    "lai": first_existing(cols, "growing_season_mean_lai", "mean_lai"),
    "mat": first_existing(cols, "mean_annual_temperature", "mean_temperature"),
    "map": first_existing(cols, "mean_annual_precipitation", "mean_precipitation"),
    "arid": first_existing(cols, "aridity"),
    "sm": first_existing(cols, "mean_soil_moisture", "mean_obs_soil_moisture"),
    "lat": first_existing(cols, "lat", "latitude"),
    "lon": first_existing(cols, "lon", "longitude"),
    "c4": first_existing(cols, "c4_fraction", "c4_fraction_raw"),
    "rooting_depth": first_existing(cols, "rooting_depth"),
    "p50": first_existing(cols, "p50", "psi50"),
    "sand": first_existing(cols, "soil_sand", "sand", "sand_fraction"),
    "clay": first_existing(cols, "soil_clay", "clay", "clay_fraction"),
    "silt": first_existing(cols, "soil_silt", "silt", "silt_fraction"),
    "soil_texture_pc1_existing": first_existing(cols, "soil_texture_pc1", "soil_texture"),
}

gs_temp_candidates = find_cols(cols, [
    r"growing.*season.*temp",
    r"season.*mean.*temp",
    r"gs.*temp",
    r"gseason.*temp",
])
sources["gs_temp"] = gs_temp_candidates[0] if gs_temp_candidates else None

required = ["point_id", "y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"]
missing = [k for k in required if sources.get(k) is None]
if missing:
    raise SystemExit(f"Missing required canonical variables: {missing}")

d = pd.DataFrame(index=raw.index)
for canon, src in sources.items():
    if src is None:
        continue
    if canon == "point_id":
        d[canon] = raw[src].astype(str)
    else:
        d[canon] = pd.to_numeric(raw[src], errors="coerce")

d["abs_lat"] = d["lat"].abs()
d["sahel_broad"] = d["lat"].between(10, 20) & d["lon"].between(-20, 40)
d["sahel_core"] = d["lat"].between(12, 18) & d["lon"].between(-17, 35)
d["has_c4"] = d["c4"].notna() if "c4" in d.columns else False
d["region_block"] = (
    pd.cut(d["lat"], [-90, -30, 0, 30, 60, 90], labels=False).astype(str)
    + "_"
    + pd.cut(d["lon"], [-180, -90, 0, 90, 180], labels=False).astype(str)
)

# Soil texture PC1.
soil_texture_note = ""
if "soil_texture_pc1_existing" in d.columns and d["soil_texture_pc1_existing"].notna().sum() >= 40:
    d["soil_texture_pc1"] = d["soil_texture_pc1_existing"]
    soil_texture_note = "Used existing soil_texture_pc1 column."
elif all(c in d.columns for c in ["sand", "clay", "silt"]) and d[["sand", "clay", "silt"]].notna().sum().min() >= 40:
    soil = d[["sand", "clay", "silt"]].copy()
    soil_std = soil.apply(zscore)
    use = soil_std.dropna()
    pc1 = pd.Series(np.nan, index=d.index)
    X = use.values
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    scores = X @ vt[0, :]
    pc1.loc[use.index] = scores
    if pc1.corr(d["clay"]) < 0:
        pc1 = -pc1
    d["soil_texture_pc1"] = pc1
    soil_texture_note = "Computed soil_texture_pc1 from sand/clay/silt using SVD; sign oriented positive with clay."
else:
    d["soil_texture_pc1"] = np.nan
    soil_texture_note = "No usable soil texture PC1 or sand/clay/silt combination found."

# Merge previous land-cover flags if available.
landcover_note = ""
if LC_FLAGS_INPUT.exists() and "point_id" in d.columns:
    lc = pd.read_csv(LC_FLAGS_INPUT, low_memory=False)
    lc = lc.loc[:, ~lc.columns.duplicated()].copy()
    if "point_id" in lc.columns:
        lc["point_id"] = lc["point_id"].astype(str)
        keep = ["point_id"]
        for c in [
            "any_cropland_managed_irrigation_flag",
            "n_cropland_managed_irrigation_flags",
            "any_natural_grassland_indicator",
            "n_natural_grassland_indicators",
        ]:
            if c in lc.columns:
                keep.append(c)
        lc = lc[keep].groupby("point_id", dropna=False).first().reset_index()
        d = d.merge(lc, on="point_id", how="left")
        landcover_note = f"Merged point-level land-cover flags from {LC_FLAGS_INPUT}."
    else:
        landcover_note = "Land-cover flags file existed but had no point_id."
else:
    landcover_note = "No previous point-level land-cover flags file found; defaulted to no flags."

if "any_cropland_managed_irrigation_flag" not in d.columns:
    d["any_cropland_managed_irrigation_flag"] = False
if "any_natural_grassland_indicator" not in d.columns:
    d["any_natural_grassland_indicator"] = True
if "n_cropland_managed_irrigation_flags" not in d.columns:
    d["n_cropland_managed_irrigation_flags"] = 0
if "n_natural_grassland_indicators" not in d.columns:
    d["n_natural_grassland_indicators"] = np.nan

d["any_cropland_managed_irrigation_flag"] = d["any_cropland_managed_irrigation_flag"].fillna(False).astype(bool)
d["any_natural_grassland_indicator"] = d["any_natural_grassland_indicator"].fillna(False).astype(bool)
d["n_cropland_managed_irrigation_flags"] = pd.to_numeric(d["n_cropland_managed_irrigation_flags"], errors="coerce").fillna(0)

# Standardized columns.
for c in list(d.columns):
    if c in ["point_id", "region_block", "sahel_broad", "sahel_core", "has_c4", "any_cropland_managed_irrigation_flag", "any_natural_grassland_indicator"]:
        continue
    if pd.api.types.is_numeric_dtype(d[c]):
        d[c + "_z"] = zscore(d[c])

# Full Reza control set.
BASE_CONTROLS = ["vpd_z", "arid_z", "mat_z", "map_z", "sm_z", "soil_texture_pc1_z", "lat_z", "lon_z"]
BASE_CONTROLS = [c for c in BASE_CONTROLS if c in d.columns and d[c].notna().sum() >= 40]

BASE_CONTROLS_NO_MAT = [c for c in BASE_CONTROLS if c != "mat_z"]

MAIN_FULL = "y_z ~ lai_z + " + " + ".join(BASE_CONTROLS)
MAIN_REDUCED = "y_z ~ " + " + ".join(BASE_CONTROLS)

INT_FULL = "y_z ~ lai_z * mat_z + " + " + ".join(BASE_CONTROLS_NO_MAT)
INT_REDUCED = "y_z ~ lai_z + mat_z + " + " + ".join(BASE_CONTROLS_NO_MAT)

# ======================================================================================
# 1. Algorithm dependency table
# ======================================================================================

def classify_gpp_product(name):
    s = str(name).lower()
    if "pml" in s:
        return {
            "component": "GPP",
            "direct_optical_lai_input": True,
            "direct_fpar_input": False,
            "optical_vegetation_proxy": "MODIS_LAI",
            "microwave_vegetation_proxy": "none",
            "internal_gpp_et_coupling": True,
            "dependency_class": "very_high_direct_LAI_and_coupled",
            "dependency_rank": 4,
            "algorithm_note": "PML-V2 GPP is estimated inside a coupled water-carbon framework using MODIS LAI.",
            "citation_key": "PML_V2_algorithm_paper_or_docs",
        }
    if "modis" in s or "mod17" in s:
        return {
            "component": "GPP",
            "direct_optical_lai_input": False,
            "direct_fpar_input": True,
            "optical_vegetation_proxy": "MODIS_FPAR",
            "microwave_vegetation_proxy": "none",
            "internal_gpp_et_coupling": False,
            "dependency_class": "high_direct_FPAR",
            "dependency_rank": 3,
            "algorithm_note": "MOD17 GPP uses MODIS FPAR with light-use-efficiency constraints.",
            "citation_key": "MOD17_ATBD",
        }
    if "gosif" in s or "sif" in s:
        return {
            "component": "GPP",
            "direct_optical_lai_input": False,
            "direct_fpar_input": False,
            "optical_vegetation_proxy": "MODIS_EVI_used_in_gapfilled_SIF_training",
            "microwave_vegetation_proxy": "none",
            "internal_gpp_et_coupling": False,
            "dependency_class": "intermediate_indirect_EVI_proxy_not_direct_LAI",
            "dependency_rank": 1,
            "algorithm_note": "GOSIF is SIF-based but the gap-filled gridded product uses MODIS EVI, an optical canopy proxy, in the reconstruction model.",
            "citation_key": "GOSIF_GOSIF_GPP_paper",
        }
    return {
        "component": "GPP",
        "direct_optical_lai_input": None,
        "direct_fpar_input": None,
        "optical_vegetation_proxy": "unknown",
        "microwave_vegetation_proxy": "unknown",
        "internal_gpp_et_coupling": None,
        "dependency_class": "unknown",
        "dependency_rank": np.nan,
        "algorithm_note": "Unknown GPP product; manually verify.",
        "citation_key": "manual_check_needed",
    }

def classify_et_product(name):
    s = str(name).lower()
    if "pml" in s:
        return {
            "component": "ET",
            "direct_optical_lai_input": True,
            "direct_fpar_input": False,
            "optical_vegetation_proxy": "MODIS_LAI",
            "microwave_vegetation_proxy": "none",
            "internal_gpp_et_coupling": True,
            "dependency_class": "very_high_direct_LAI_and_coupled",
            "dependency_rank": 4,
            "algorithm_note": "PML-V2 ET directly uses MODIS LAI and couples canopy conductance/photosynthesis with transpiration.",
            "citation_key": "PML_V2_algorithm_paper_or_docs",
        }
    if "modis" in s or "mod16" in s:
        return {
            "component": "ET",
            "direct_optical_lai_input": True,
            "direct_fpar_input": True,
            "optical_vegetation_proxy": "MODIS_LAI_FPAR",
            "microwave_vegetation_proxy": "none",
            "internal_gpp_et_coupling": False,
            "dependency_class": "high_direct_LAI_FPAR",
            "dependency_rank": 3,
            "algorithm_note": "MOD16 ET uses MODIS LAI/FPAR vegetation dynamics in the Penman-Monteith ET framework.",
            "citation_key": "MOD16_ATBD",
        }
    if "gleam" in s:
        return {
            "component": "ET",
            "direct_optical_lai_input": False,
            "direct_fpar_input": False,
            "optical_vegetation_proxy": "none",
            "microwave_vegetation_proxy": "VOD",
            "internal_gpp_et_coupling": False,
            "dependency_class": "lowest_direct_LAI_entanglement_nonoptical_VOD",
            "dependency_rank": 0,
            "algorithm_note": "GLEAM uses Priestley-Taylor potential evaporation and an evaporative stress factor informed by microwave VOD and root-zone soil moisture, not optical LAI.",
            "citation_key": "GLEAM_algorithm_papers",
        }
    return {
        "component": "ET",
        "direct_optical_lai_input": None,
        "direct_fpar_input": None,
        "optical_vegetation_proxy": "unknown",
        "microwave_vegetation_proxy": "unknown",
        "internal_gpp_et_coupling": None,
        "dependency_class": "unknown",
        "dependency_rank": np.nan,
        "algorithm_note": "Unknown ET product; manually verify.",
        "citation_key": "manual_check_needed",
    }

product_dependency_rows = []

if OBS_INPUT.exists():
    obs_head = pd.read_csv(OBS_INPUT, nrows=5, low_memory=False)
    obs_cols = list(obs_head.columns)
    gpp_col = first_existing(obs_cols, "gpp_product", "gpp")
    et_col = first_existing(obs_cols, "et_product", "et")
    obs_all_for_products = pd.read_csv(OBS_INPUT, usecols=lambda c: c in [gpp_col, et_col, "product_combo"], low_memory=False)
    if gpp_col and gpp_col in obs_all_for_products.columns:
        for gp in sorted(obs_all_for_products[gpp_col].dropna().astype(str).unique()):
            row = {"product": gp, **classify_gpp_product(gp)}
            product_dependency_rows.append(row)
    if et_col and et_col in obs_all_for_products.columns:
        for et in sorted(obs_all_for_products[et_col].dropna().astype(str).unique()):
            row = {"product": et, **classify_et_product(et)}
            product_dependency_rows.append(row)

# Add expected products if absent.
expected_gpp = ["GOSIF_GPP", "MODIS_GPP_MOD17", "PML_GPP"]
expected_et = ["GLEAM_ET", "MODIS_ET_MOD16", "PML_ET"]
existing_products = set([r["product"] for r in product_dependency_rows])
for gp in expected_gpp:
    if gp not in existing_products:
        product_dependency_rows.append({"product": gp, **classify_gpp_product(gp)})
for et in expected_et:
    if et not in existing_products:
        product_dependency_rows.append({"product": et, **classify_et_product(et)})

product_dependency = pd.DataFrame(product_dependency_rows).drop_duplicates(subset=["product", "component"])
product_dependency.to_csv(TAB / "ALGORITHM_DEPENDENCY_TABLE.csv", index=False)

# ======================================================================================
# 2. Exact 3x3 product matrix
# ======================================================================================

product_matrix_rows = []
combo_dependency_rows = []

def combo_dependency(gpp_name, et_name):
    g = classify_gpp_product(gpp_name)
    e = classify_et_product(et_name)
    ranks = [g.get("dependency_rank", np.nan), e.get("dependency_rank", np.nan)]
    ranks_ok = [r for r in ranks if pd.notna(r)]
    max_rank = max(ranks_ok) if ranks_ok else np.nan
    sum_rank = sum(ranks_ok) if ranks_ok else np.nan

    if max_rank == 0:
        combo_class = "lowest_direct_LAI_entanglement"
    elif max_rank <= 1:
        combo_class = "low_to_intermediate_indirect_optical_proxy"
    elif max_rank == 3:
        combo_class = "one_or_more_direct_FPAR_LAI_components"
    elif max_rank >= 4:
        combo_class = "high_direct_LAI_or_coupled_product"
    else:
        combo_class = "unknown"

    return {
        "gpp_product": gpp_name,
        "et_product": et_name,
        "gpp_dependency_class": g["dependency_class"],
        "et_dependency_class": e["dependency_class"],
        "gpp_dependency_rank": g["dependency_rank"],
        "et_dependency_rank": e["dependency_rank"],
        "combo_dependency_rank_max": max_rank,
        "combo_dependency_rank_sum": sum_rank,
        "combo_dependency_class": combo_class,
        "least_directly_LAI_dependent_overall_pair": bool(("gosif" in str(gpp_name).lower() or "sif" in str(gpp_name).lower()) and "gleam" in str(et_name).lower()),
    }

if OBS_INPUT.exists():
    obs = pd.read_csv(OBS_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
    obs = obs.loc[:, ~obs.columns.duplicated()].copy()
    obs_cols = list(obs.columns)

    point_col = first_existing(obs_cols, "point_id")
    gpp_col = first_existing(obs_cols, "gpp_product", "gpp")
    et_col = first_existing(obs_cols, "et_product", "et")
    slope_col = first_existing(obs_cols, "slope_change", "latent_slope_change")

    if point_col and gpp_col and et_col and slope_col:
        obs[point_col] = obs[point_col].astype(str)
        obs[gpp_col] = obs[gpp_col].astype(str)
        obs[et_col] = obs[et_col].astype(str)
        obs[slope_col] = pd.to_numeric(obs[slope_col], errors="coerce")

        combo_point = (
            obs.dropna(subset=[point_col, gpp_col, et_col, slope_col])
               .groupby([point_col, gpp_col, et_col], dropna=False)[slope_col]
               .mean()
               .reset_index()
               .rename(columns={point_col: "point_id", gpp_col: "gpp_product", et_col: "et_product", slope_col: "y_product"})
        )

        combos = combo_point[["gpp_product", "et_product"]].drop_duplicates().sort_values(["gpp_product", "et_product"])
        combo_keys = []
        for _, r in combos.iterrows():
            key = safe_name(f"{r['gpp_product']}__X__{r['et_product']}")
            combo_keys.append((r["gpp_product"], r["et_product"], key))

        # Build wide matrix for common complete-case sample.
        wide_parts = []
        for gp, et, key in combo_keys:
            sub = combo_point[(combo_point["gpp_product"] == gp) & (combo_point["et_product"] == et)][["point_id", "y_product"]].copy()
            sub = sub.rename(columns={"y_product": key})
            wide_parts.append(sub)

        wide = d[["point_id"] + [c for c in d.columns if c != "point_id"]].copy()
        for sub in wide_parts:
            wide = wide.merge(sub, on="point_id", how="left")

        base_needed = ["lai_z"] + BASE_CONTROLS
        combo_outcome_cols = [key for _, _, key in combo_keys]
        common_mask = wide[base_needed + combo_outcome_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)

        scenario_masks = {
            "all_available_complete_case": pd.Series(True, index=wide.index),
            "cropland_clean_available_complete_case": ~wide["any_cropland_managed_irrigation_flag"],
            "natural_grass_indicator_available_complete_case": wide["any_natural_grassland_indicator"],
            "all_common_complete_case_across_product_combos": common_mask,
            "cropland_clean_common_complete_case_across_product_combos": common_mask & (~wide["any_cropland_managed_irrigation_flag"]),
        }

        for gp, et, key in combo_keys:
            dep = combo_dependency(gp, et)
            combo_dependency_rows.append(dep)

            for scenario, mask in scenario_masks.items():
                tmp = wide.loc[mask].copy()
                tmp["y_alt_z"] = zscore(tmp[key])

                full = "y_alt_z ~ lai_z + " + " + ".join(BASE_CONTROLS)
                reduced = "y_alt_z ~ " + " + ".join(BASE_CONTROLS)

                rows = compare_model(
                    tmp,
                    f"{scenario}__{gp}__X__{et}",
                    full,
                    reduced,
                    ["lai_z"],
                    family="exact_3x3_product_matrix_LAI_main_effect",
                    note="Outcome is point-level mean slope_change for this exact GPP×ET product combination. Predictor is growing-season LAI. Controls are full Reza controls."
                )

                for row in rows:
                    row.update(dep)
                    row["sample_mode"] = scenario
                    product_matrix_rows.append(row)

        product_matrix = pd.DataFrame(product_matrix_rows)
        if len(product_matrix):
            product_matrix = add_adjustments(product_matrix, p_col="p", group_col="sample_mode")
            product_matrix.to_csv(TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv", index=False)

        pd.DataFrame(combo_dependency_rows).drop_duplicates(
            subset=["gpp_product", "et_product"]
        ).to_csv(TAB / "PRODUCT_COMBINATION_DEPENDENCY_TABLE.csv", index=False)
    else:
        pd.DataFrame([{
            "status": "OBS_TABLE_MISSING_REQUIRED_COLUMNS",
            "point_col": point_col,
            "gpp_col": gpp_col,
            "et_col": et_col,
            "slope_col": slope_col,
        }]).to_csv(TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv", index=False)
else:
    pd.DataFrame([{"status": "OBS_INPUT_NOT_FOUND", "path": str(OBS_INPUT)}]).to_csv(TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv", index=False)

# Product forest plot.
try:
    pm = pd.read_csv(TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv")
    plot_df = pm[
        (pm["status"] == "FIT_OK")
        & (pm["sample_mode"] == "all_common_complete_case_across_product_combos")
    ].copy()
    if len(plot_df) == 0:
        plot_df = pm[(pm["status"] == "FIT_OK") & (pm["sample_mode"] == "all_available_complete_case")].copy()
    if len(plot_df):
        plot_df["label"] = plot_df["gpp_product"].astype(str) + " × " + plot_df["et_product"].astype(str)
        plot_df = plot_df.sort_values(["combo_dependency_rank_max", "label"])
        y_pos = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(plot_df))))
        ax.errorbar(
            plot_df["coef"],
            y_pos,
            xerr=[
                plot_df["coef"] - plot_df["ci_low"],
                plot_df["ci_high"] - plot_df["coef"],
            ],
            fmt="o",
            capsize=3,
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["label"])
        ax.set_xlabel("LAI coefficient in product-specific slope-change model")
        ax.set_title("LAI effect by exact GPP × ET product combination")
        fig.tight_layout()
        fig.savefig(FIG / "FIG_product_dependency_forest_plot_LAI.png", dpi=220)
        plt.close(fig)
except Exception as e:
    (TXT / "PRODUCT_FOREST_PLOT_ERROR.txt").write_text(str(e))

# ======================================================================================
# 3. Full discovery-family FDR reconstruction
# ======================================================================================

# Discovery family is intentionally locked to trait / canopy / soil-structure predictors,
# not pure climate controls like VPD, MAT, aridity, precipitation, soil moisture, lat, lon.
candidate_predictors = []

manual_candidates = {
    "growing_season_mean_lai": sources.get("lai"),
    "rooting_depth": sources.get("rooting_depth"),
    "p50_or_psi50": sources.get("p50"),
    "c4_fraction": sources.get("c4"),
    "soil_texture_pc1": "soil_texture_pc1",
    "soil_sand": sources.get("sand"),
    "soil_clay": sources.get("clay"),
    "soil_silt": sources.get("silt"),
}

for canonical, col in manual_candidates.items():
    if col is not None and col in d.columns and d[col].notna().sum() >= 40:
        candidate_predictors.append((canonical, col, "manual_trait_structure_family"))

# Add any extra trait-like columns present in raw but avoid climate/geography/outcome/product names.
trait_patterns = [
    r"\blai\b", r"leaf", r"canopy", r"root", r"p50", r"psi50", r"hydraulic",
    r"isohyd", r"trait", r"sand", r"clay", r"silt", r"texture", r"\bc4\b"
]
exclude_patterns = [
    r"vpd", r"temp", r"precip", r"arid", r"moisture", r"\blat\b", r"\blon\b",
    r"response", r"slope", r"satbreak", r"threshold", r"uncertainty",
    r"product", r"metric", r"stress", r"year", r"date"
]

for col in cols:
    lc = norm(col)
    if any(re.search(p, lc) for p in trait_patterns) and not any(re.search(p, lc) for p in exclude_patterns):
        if col in raw.columns:
            vals = pd.to_numeric(raw[col], errors="coerce")
            if vals.notna().sum() >= 40:
                canonical = lc
                if all(col != existing_col for _, existing_col, _ in candidate_predictors):
                    d_col = f"raw_{safe_name(col)}"
                    d[d_col] = vals
                    d[d_col + "_z"] = zscore(vals)
                    candidate_predictors.append((canonical, d_col, "regex_trait_structure_family"))

# Remove duplicate columns by canonical name.
seen = set()
unique_candidates = []
for canonical, col, source_type in candidate_predictors:
    if (canonical, col) not in seen:
        seen.add((canonical, col))
        unique_candidates.append((canonical, col, source_type))
candidate_predictors = unique_candidates

fdr_rows = []

for canonical, col, source_type in candidate_predictors:
    zcol = col + "_z" if col + "_z" in d.columns else None
    if zcol is None:
        continue

    controls = [c for c in BASE_CONTROLS if c != zcol]
    if zcol == "soil_texture_pc1_z":
        controls = [c for c in controls if c != "soil_texture_pc1_z"]

    full = "y_z ~ " + zcol + " + " + " + ".join(controls)
    reduced = "y_z ~ " + " + ".join(controls)

    rows = compare_model(
        d,
        f"discovery_family_main__{canonical}",
        full,
        reduced,
        [zcol],
        family="A_trait_structure_main_effects_on_latent_slope_change",
        note="Discovery-family FDR reconstruction: unique trait/canopy/soil-structure main effects on latent_slope_change under full Reza controls."
    )

    for row in rows:
        row["canonical_predictor"] = canonical
        row["original_column"] = col
        row["candidate_source_type"] = source_type
        row["included_in_primary_lai_family"] = True
        fdr_rows.append(row)

# Interaction family: trait / structure × MAT.
for canonical, col, source_type in candidate_predictors:
    zcol = col + "_z" if col + "_z" in d.columns else None
    if zcol is None or zcol == "mat_z":
        continue

    controls = [c for c in BASE_CONTROLS if c not in [zcol, "mat_z"]]
    if zcol == "soil_texture_pc1_z":
        controls = [c for c in controls if c != "soil_texture_pc1_z"]

    full = "y_z ~ " + zcol + " * mat_z + " + " + ".join(controls)
    reduced = "y_z ~ " + zcol + " + mat_z + " + " + ".join(controls)
    focal = f"{zcol}:mat_z"

    rows = compare_model(
        d,
        f"discovery_family_interaction_with_MAT__{canonical}",
        full,
        reduced,
        [focal],
        family="B_trait_structure_x_MAT_interactions_on_latent_slope_change",
        note="Discovery-family FDR reconstruction: trait/canopy/soil-structure × MAT interactions under full Reza controls."
    )

    for row in rows:
        row["canonical_predictor"] = canonical
        row["original_column"] = col
        row["candidate_source_type"] = source_type
        row["included_in_primary_lai_family"] = True
        fdr_rows.append(row)

fdr = pd.DataFrame(fdr_rows)
if len(fdr):
    fdr = add_adjustments(fdr, p_col="p", group_col="family")
    fdr = fdr.sort_values(["family", "bh_q", "p"])
fdr.to_csv(TAB / "FULL_DISCOVERY_FDR_TABLE.csv", index=False)

# Compact LAI q-value extract.
lai_fdr = fdr[
    fdr["canonical_predictor"].astype(str).str.contains("lai", case=False, na=False)
].copy() if len(fdr) else pd.DataFrame()
lai_fdr.to_csv(TAB / "LAI_DISCOVERY_Q_VALUE_EXTRACT.csv", index=False)

# ======================================================================================
# 4. Domain sensitivity table
# ======================================================================================

domain_masks = {
    "all_points": pd.Series(True, index=d.index),
    "cropland_clean": ~d["any_cropland_managed_irrigation_flag"],
    "natural_grassland_indicator_only": d["any_natural_grassland_indicator"],
    "cropland_clean_and_no_sahel_broad": (~d["any_cropland_managed_irrigation_flag"]) & (~d["sahel_broad"]),
    "exclude_sahel_broad": ~d["sahel_broad"],
    "exclude_sahel_core": ~d["sahel_core"],
    "warm_only_MAT_gt_0C": d["mat"] > 0,
    "warm_only_MAT_gt_2C": d["mat"] > 2.08,
    "abs_lat_le_48": d["abs_lat"] <= 48,
    "c4_covered_domain_only": d["has_c4"],
}

domain_rows = []
range_rows = []

for name, mask in domain_masks.items():
    sub = d.loc[mask].copy()

    domain_rows += compare_model(
        sub,
        f"{name}__LAI_main",
        MAIN_FULL,
        MAIN_REDUCED,
        ["lai_z"],
        family="domain_sensitivity_LAI_main",
        note="Locked domain sensitivity. No new cutoffs searched."
    )

    domain_rows += compare_model(
        sub,
        f"{name}__LAI_x_MAT",
        INT_FULL,
        INT_REDUCED,
        ["lai_z:mat_z"],
        family="domain_sensitivity_LAI_x_MAT",
        note="Locked domain sensitivity. No new cutoffs searched."
    )

    # Range restriction diagnostics.
    tmp = sub[["y", "lai", "mat", "vpd", "arid", "sm", "lat", "lon"]].replace([np.inf, -np.inf], np.nan)
    rr = {
        "domain": name,
        "n_raw": len(sub),
        "n_complete_main_model": int(sub[formula_vars(MAIN_FULL)].dropna().shape[0]) if all(v in sub.columns for v in formula_vars(MAIN_FULL)) else np.nan,
    }
    for c in ["y", "lai", "mat", "vpd"]:
        x = pd.to_numeric(tmp[c], errors="coerce")
        rr[f"{c}_n"] = int(x.notna().sum())
        rr[f"{c}_mean"] = float(x.mean()) if x.notna().any() else np.nan
        rr[f"{c}_sd"] = float(x.std()) if x.notna().any() else np.nan
        rr[f"{c}_min"] = float(x.min()) if x.notna().any() else np.nan
        rr[f"{c}_max"] = float(x.max()) if x.notna().any() else np.nan
    rr["lai_mat_spearman"] = stats.spearmanr(tmp["lai"], tmp["mat"], nan_policy="omit").correlation if tmp[["lai", "mat"]].dropna().shape[0] >= 4 else np.nan
    rr["lai_vpd_spearman"] = stats.spearmanr(tmp["lai"], tmp["vpd"], nan_policy="omit").correlation if tmp[["lai", "vpd"]].dropna().shape[0] >= 4 else np.nan
    range_rows.append(rr)

domain = pd.DataFrame(domain_rows)
if len(domain):
    domain = add_adjustments(domain, p_col="p", group_col="family")
domain.to_csv(TAB / "DOMAIN_SENSITIVITY_LAI.csv", index=False)
pd.DataFrame(range_rows).to_csv(TAB / "DOMAIN_RANGE_RESTRICTION_DIAGNOSTICS.csv", index=False)

# Domain plot.
try:
    dp = domain[(domain["status"] == "FIT_OK") & (domain["family"] == "domain_sensitivity_LAI_main")].copy()
    if len(dp):
        dp["domain"] = dp["test_label"].str.replace("__LAI_main", "", regex=False)
        y_pos = np.arange(len(dp))
        fig, ax = plt.subplots(figsize=(8, max(5, 0.42 * len(dp))))
        ax.errorbar(
            dp["coef"],
            y_pos,
            xerr=[dp["coef"] - dp["ci_low"], dp["ci_high"] - dp["coef"]],
            fmt="o",
            capsize=3,
        )
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(dp["domain"])
        ax.set_xlabel("LAI coefficient")
        ax.set_title("Domain sensitivity: LAI main effect")
        fig.tight_layout()
        fig.savefig(FIG / "FIG_domain_sensitivity_LAI_main.png", dpi=220)
        plt.close(fig)
except Exception as e:
    (TXT / "DOMAIN_PLOT_ERROR.txt").write_text(str(e))

# ======================================================================================
# 5. Strict tower inclusion table and provisional tower directional test
# ======================================================================================

tower_candidate_paths = [
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table101_tower_landcover_spatial_trait_annotation.csv",
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table110_high_priority_grassland_open_tower_sites_for_satellite_extraction.csv",
    ROOT / "results/tower_validation_broad_inventory/tables/Table89_tower_response_phenotypes_primary_by_site.csv",
    ROOT / "results/stage1b6aw_clean_crop_tower_controls/tables/Table_PRODUCT03ej_clean_tower_landcover_rows_Table101_tower_landcover_spatial_trait_annotation.csv",
    ROOT / "results/stage1b6ay_final_reza_audit/tables/Table_PRODUCT03fe_target_tower_landcover_details.csv",
    ROOT / "data/raw/towers/fluxnet_v13_harmonization_summary.csv",
    ROOT / "data/raw/towers/fluxnet_v13_harmonization_summary_FIXED.csv",
]

tower_tables = []
for path in tower_candidate_paths:
    if not path.exists():
        continue
    try:
        t = pd.read_csv(path, low_memory=False).replace([np.inf, -np.inf], np.nan)
        t = t.loc[:, ~t.columns.duplicated()].copy()
        t["_source_path"] = str(path)
        tower_tables.append(t)
    except Exception:
        pass

tower_inclusion_rows = []
tower_test_source = None

def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).lower().strip()
    return s in ["true", "1", "yes", "y"]

# Pick richest tower table for provisional directional test.
for t in tower_tables:
    tcols = list(t.columns)
    site_col = first_existing(tcols, "site", "site_id", "target_id", "tower_id", "_site_clean")
    slope_col = first_existing(tcols, "slope_change", "tower_slope_change")
    near_col = first_existing(tcols, "nearest_sat_point_id", "nearest_point_id")
    lat_col = first_existing(tcols, "tower_lat", "lat", "latitude")
    lon_col = first_existing(tcols, "tower_lon", "lon", "longitude")
    if site_col and slope_col and (near_col or (lat_col and lon_col)):
        tower_test_source = t.copy()
        break

# Build inclusion table from all tower tables, one row per site per source.
for t in tower_tables:
    tcols = list(t.columns)
    site_col = first_existing(tcols, "site", "site_id", "target_id", "tower_id", "_site_clean")
    if site_col is None:
        continue

    igbp_col = first_existing(tcols, "igbp_final", "igbp_class", "igbp")
    strict_col = first_existing(tcols, "is_strict_grassland_tower", "is_strict_grassland_gra")
    crop_col = first_existing(tcols, "crop_or_cro_flag")
    pass_lc_col = first_existing(tcols, "passes_landcover_screen_lenient")
    source_path = str(t["_source_path"].iloc[0]) if "_source_path" in t.columns and len(t) else ""

    rows = t.copy()
    rows[site_col] = rows[site_col].astype(str)

    for site, g in rows.groupby(site_col, dropna=False):
        igbp_vals = []
        if igbp_col:
            igbp_vals = [str(v) for v in g[igbp_col].dropna().unique()]
        igbp_join = ";".join(igbp_vals)
        igbp_upper = igbp_join.upper()

        is_us_ne = site.upper() in ["US-NE1", "US-NE2", "US-NE3"]

        crop_flag = False
        if crop_col:
            crop_flag = any(parse_bool(v) for v in g[crop_col].dropna().tolist())
        if "CRO" in igbp_upper:
            crop_flag = True
        if is_us_ne:
            crop_flag = True

        strict_flag = False
        if strict_col:
            strict_flag = any(parse_bool(v) for v in g[strict_col].dropna().tolist())
        if "GRA" in igbp_upper:
            strict_flag = True

        passes_lc = None
        if pass_lc_col:
            vals = [parse_bool(v) for v in g[pass_lc_col].dropna().tolist()]
            passes_lc = any(vals) if vals else None

        reasons = []
        if crop_flag:
            reasons.append("exclude_crop_or_CRO_or_US_Ne_managed_ag")
        if not strict_flag:
            reasons.append("exclude_not_strict_GRA")
        if passes_lc is False:
            reasons.append("fails_lenient_landcover_screen")

        include_strict = (not crop_flag) and strict_flag

        tower_inclusion_rows.append({
            "site": site,
            "source_path": source_path,
            "igbp_values": igbp_join,
            "is_US_Ne1_2_3": is_us_ne,
            "crop_or_cro_or_managed_ag_flag": crop_flag,
            "strict_grassland_flag": strict_flag,
            "passes_landcover_screen_lenient": passes_lc,
            "include_in_primary_strict_tower_test": include_strict,
            "exclusion_reason": "; ".join(reasons) if reasons else "",
        })

tower_inclusion = pd.DataFrame(tower_inclusion_rows)
if len(tower_inclusion):
    tower_inclusion = tower_inclusion.drop_duplicates()
    tower_inclusion.to_csv(TAB / "STRICT_TOWER_SITE_INCLUSION.csv", index=False)
else:
    pd.DataFrame([{"status": "NO_TOWER_TABLES_FOUND"}]).to_csv(TAB / "STRICT_TOWER_SITE_INCLUSION.csv", index=False)

tower_test_rows = []
tower_detail_rows = []

if tower_test_source is not None and len(tower_inclusion):
    t = tower_test_source.copy()
    tcols = list(t.columns)
    site_col = first_existing(tcols, "site", "site_id", "target_id", "tower_id", "_site_clean")
    slope_col = first_existing(tcols, "slope_change", "tower_slope_change")
    near_col = first_existing(tcols, "nearest_sat_point_id", "nearest_point_id")
    lat_col = first_existing(tcols, "tower_lat", "lat", "latitude")
    lon_col = first_existing(tcols, "tower_lon", "lon", "longitude")

    t[site_col] = t[site_col].astype(str)
    include_sites = set(
        tower_inclusion[tower_inclusion["include_in_primary_strict_tower_test"] == True]["site"].astype(str)
    )

    primary = t[t[site_col].astype(str).isin(include_sites)].copy()

    # One row per site: prefer primary fit if available.
    if "is_primary_tower_fit" in primary.columns:
        primary["_primary_rank_tmp"] = primary["is_primary_tower_fit"].apply(lambda x: 0 if parse_bool(x) else 1)
        primary = primary.sort_values(["_primary_rank_tmp"])
    primary = primary.groupby(site_col, dropna=False).first().reset_index()

    primary["tower_response"] = pd.to_numeric(primary[slope_col], errors="coerce")

    if near_col and near_col in primary.columns:
        primary["point_id_for_lai"] = primary[near_col].astype(str)
        lai_lookup = d[["point_id", "lai"]].dropna().drop_duplicates("point_id")
        primary = primary.merge(lai_lookup, left_on="point_id_for_lai", right_on="point_id", how="left")
        lai_source_note = "Used satellite LAI at nearest_sat_point_id. This is a PROVISIONAL tower-side predictor, not true independent tower-coordinate LAI extraction."
    elif lat_col and lon_col:
        # nearest point fallback
        pts = d[["point_id", "lat", "lon", "lai"]].dropna().copy()
        out_lai = []
        out_pid = []
        for _, r in primary.iterrows():
            lat = pd.to_numeric(r[lat_col], errors="coerce")
            lon = pd.to_numeric(r[lon_col], errors="coerce")
            if pd.isna(lat) or pd.isna(lon) or len(pts) == 0:
                out_lai.append(np.nan)
                out_pid.append(None)
                continue
            dist2 = (pts["lat"] - lat) ** 2 + (pts["lon"] - lon) ** 2
            j = dist2.idxmin()
            out_lai.append(pts.loc[j, "lai"])
            out_pid.append(pts.loc[j, "point_id"])
        primary["lai"] = out_lai
        primary["point_id_for_lai"] = out_pid
        lai_source_note = "Used nearest satellite point by lat/lon. This is a PROVISIONAL tower-side predictor, not true independent tower-coordinate LAI extraction."
    else:
        lai_source_note = "No tower LAI or nearest satellite point available."

    primary["lai_z_tower_sample"] = zscore(primary["lai"])
    primary["tower_response_z"] = zscore(primary["tower_response"])
    primary.to_csv(TAB / "STRICT_TOWER_DIRECTIONAL_TEST_SITE_VALUES.csv", index=False)

    rho, perm_p, n_tower = permutation_spearman(primary["lai"], primary["tower_response"], n_perm=10000)

    loo_signs = []
    ok_primary = primary[["lai", "tower_response"]].dropna()
    if len(ok_primary) >= 4:
        for i in range(len(ok_primary)):
            sub = ok_primary.drop(ok_primary.index[i])
            if len(sub) >= 3:
                rr = stats.spearmanr(sub["lai"], sub["tower_response"]).correlation
                loo_signs.append(np.sign(rr))
    loo_sign_consistency = np.nan
    if loo_signs and np.isfinite(rho) and rho != 0:
        loo_sign_consistency = float(np.mean(np.asarray(loo_signs) == np.sign(rho)))

    tower_test_rows.append({
        "tower_test_status": "FIT_OK" if n_tower >= 4 else "UNDERPOWERED_OR_INSUFFICIENT_STRICT_TOWERS",
        "n_strict_towers_with_lai_and_response": n_tower,
        "spearman_rho_lai_vs_tower_response": rho,
        "permutation_p_two_sided": perm_p,
        "leave_one_out_sign_consistency": loo_sign_consistency,
        "lai_source_note": lai_source_note,
        "outcome_column": slope_col,
        "source_table": str(tower_test_source["_source_path"].iloc[0]) if "_source_path" in tower_test_source.columns else "",
        "interpretation_rule": "Same negative direction as satellite supports LAI; null/underpowered is inconclusive; opposite stable direction challenges satellite interpretation.",
    })
else:
    tower_test_rows.append({
        "tower_test_status": "NO_USABLE_TOWER_TEST_SOURCE",
        "n_strict_towers_with_lai_and_response": 0,
        "spearman_rho_lai_vs_tower_response": np.nan,
        "permutation_p_two_sided": np.nan,
        "leave_one_out_sign_consistency": np.nan,
        "lai_source_note": "No usable strict tower source table found. Need true tower-coordinate LAI extraction.",
    })

pd.DataFrame(tower_test_rows).to_csv(TAB / "TOWER_LAI_DIRECTIONAL_TEST.csv", index=False)

(TXT / "TOWER_TRUE_LAI_EXTRACTION_TODO.txt").write_text(
"""True independent tower LAI extraction still needed if strict tower test is underpowered or uses nearest satellite proxy.

Required implementation:
1. Take strict tower coordinates only: IGBP GRA, non-CRO, exclude US-Ne1/US-Ne2/US-Ne3, pass quality/record-length screens.
2. Extract the same growing-season LAI product used in the satellite predictor directly at tower coordinates.
3. Use the same years / climatology window as the satellite predictor.
4. Recompute tower uWUE response from tower-derived GPP, ET from latent heat, and tower VPD.
5. Run site-level Spearman/permutation and leave-one-site-out checks.
6. Report as directional independent support only unless n and uncertainty justify stronger language.
"""
)

# ======================================================================================
# 6. Conditional effect diagnostic for LAI × MAT
# ======================================================================================

try:
    fit, use, status = fit_ols(d, INT_FULL)
    if fit is not None and "lai_z:mat_z" in fit.params.index:
        mat_grid = np.linspace(use["mat_z"].quantile(0.02), use["mat_z"].quantile(0.98), 150)
        b_lai = fit.params["lai_z"]
        b_int = fit.params["lai_z:mat_z"]
        cov = fit.cov_params()
        slopes = []
        los = []
        his = []
        for m in mat_grid:
            slope = b_lai + b_int * m
            var = cov.loc["lai_z", "lai_z"] + (m ** 2) * cov.loc["lai_z:mat_z", "lai_z:mat_z"] + 2 * m * cov.loc["lai_z", "lai_z:mat_z"]
            se = np.sqrt(max(var, 0))
            slopes.append(slope)
            los.append(slope - 1.96 * se)
            his.append(slope + 1.96 * se)

        diag = pd.DataFrame({
            "mat_z": mat_grid,
            "conditional_LAI_slope": slopes,
            "ci_low": los,
            "ci_high": his,
        })
        diag.to_csv(TAB / "CONDITIONAL_LAI_SLOPE_OVER_MAT.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(diag["mat_z"], diag["conditional_LAI_slope"])
        ax.fill_between(diag["mat_z"], diag["ci_low"], diag["ci_high"], alpha=0.25)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Mean annual temperature, standardized")
        ax.set_ylabel("Conditional LAI slope")
        ax.set_title("Locked LAI × MAT diagnostic")
        fig.tight_layout()
        fig.savefig(FIG / "FIG_conditional_LAI_slope_over_MAT.png", dpi=220)
        plt.close(fig)
except Exception as e:
    (TXT / "CONDITIONAL_SLOPE_ERROR.txt").write_text(str(e))

# ======================================================================================
# 7. Analysis lock JSON
# ======================================================================================

input_files = {
    "point_input": str(POINT_INPUT),
    "point_input_sha256": sha256_file(POINT_INPUT),
    "phase8_observations": str(OBS_INPUT) if OBS_INPUT.exists() else None,
    "phase8_observations_sha256": sha256_file(OBS_INPUT) if OBS_INPUT.exists() else None,
    "landcover_flags_input": str(LC_FLAGS_INPUT) if LC_FLAGS_INPUT.exists() else None,
    "landcover_flags_input_sha256": sha256_file(LC_FLAGS_INPUT) if LC_FLAGS_INPUT.exists() else None,
}

analysis_lock = {
    "stage": "stage1b6bf_lai_reza_lock_package",
    "date_note": "Generated by local script at runtime; no new discovery variables allowed after this lock.",
    "git_commit": get_git_commit(),
    "input_files": input_files,
    "primary_predictor": {
        "name": "growing_season_mean_lai",
        "canonical_source_column": sources.get("lai"),
        "standardized_column": "lai_z",
    },
    "primary_outcome": {
        "name": "latent_slope_change",
        "canonical_source_column": sources.get("y"),
        "standardized_column": "y_z",
    },
    "primary_claim": "Growing-season canopy structure / LAI is associated with latent grassland uWUE slope-change beyond VPD and full hydroclimate/geography/soil-texture controls.",
    "secondary_claim": "LAI × mean annual temperature is a cross-regime diagnostic interaction, not a universal within-warm-grassland mechanism.",
    "full_control_set": BASE_CONTROLS,
    "soil_texture_definition": soil_texture_note,
    "growing_season_temperature_column_found": sources.get("gs_temp"),
    "landcover_mask_primary": "Exclude points where any_cropland_managed_irrigation_flag is True; report all-points and clean-subset results.",
    "fixed_domain_sensitivity_set": list(domain_masks.keys()),
    "fdr_families": {
        "A_trait_structure_main_effects_on_latent_slope_change": "All locked trait/canopy/soil-structure main effects tested against latent_slope_change with full controls.",
        "B_trait_structure_x_MAT_interactions_on_latent_slope_change": "All locked trait/canopy/soil-structure × MAT interactions tested with full controls.",
    },
    "product_matrix_definition": "Point-level mean slope_change by exact GPP × ET product combination from phase8 observation table; fit LAI with full controls for each combination.",
    "least_directly_lai_dependent_pair_definition": "GOSIF or SIF-based GPP combined with GLEAM ET; this is least directly dependent on the MODIS LAI predictor but not fully vegetation-independent because GOSIF uses EVI in reconstruction.",
    "tower_primary_mask": "Strict GRA only, non-CRO, no US-Ne1/US-Ne2/US-Ne3, pass quality/record-length screens when available.",
    "tower_language_rule": "Use 'limited independent anchor' or 'directional independent support'; do not write 'tower validation' unless strict tower test is significant and stable.",
    "pass_fail_criteria": {
        "primary_lai_main": "BH q < 0.05 in family A, CI excludes zero in full-control model, survives cropland-clean sample, and direction consistent in GOSIF×GLEAM.",
        "secondary_lai_x_mat": "BH q < 0.05 in family B and disclosed as cross-regime if warm-only fails.",
        "tower": "Same direction with exact/permutation evidence is support; underpowered same-direction is limited support; unstable/opposite direction is a challenge.",
    },
    "no_more_discovery_scans": True,
}

(TAB / "ANALYSIS_LOCK.json").write_text(json.dumps(analysis_lock, indent=2))

# ======================================================================================
# 8. Feature implementation notes
# ======================================================================================

feature_notes = f"""
Stage1B6BF feature implementation notes
================================================================================

1. Algorithm-dependency table
Implemented in: ALGORITHM_DEPENDENCY_TABLE.csv
How:
- Product names were read from table_latent_model_observations.csv when present.
- Each GPP/ET product was classified by hard-coded algorithm lineage rules:
  * GLEAM ET = lowest direct LAI entanglement because it is classified as Priestley-Taylor/VOD/root-zone-soil-moisture based rather than optical LAI based.
  * MOD16 ET = direct MODIS LAI/FPAR dependency.
  * PML ET/GPP = direct MODIS LAI dependency and internally coupled water-carbon model.
  * MOD17 GPP = direct MODIS FPAR dependency.
  * GOSIF/GOSIF-GPP = intermediate; not direct LAI, but gap-filled SIF reconstruction uses MODIS EVI, an optical vegetation proxy.
- Output includes dependency_class, dependency_rank, and algorithm_note for every product.

2. Exact 3×3 product matrix
Implemented in: EXACT_3x3_PRODUCT_MATRIX_LAI.csv and PRODUCT_COMBINATION_DEPENDENCY_TABLE.csv
How:
- The script reads table_latent_model_observations.csv.
- It groups observations by point_id × gpp_product × et_product and averages slope_change.
- For every exact GPP×ET pair, it fits:
    product_specific_slope_change_z ~ lai_z + {BASE_CONTROLS}
  versus:
    product_specific_slope_change_z ~ {BASE_CONTROLS}
- It runs both available-complete-case and common-complete-case-across-product-combos versions.
- It also runs cropland-clean and natural-grassland-indicator versions.
- BH q-values are computed within each sample_mode across the product combinations.

3. Full discovery-family FDR reconstruction
Implemented in: FULL_DISCOVERY_FDR_TABLE.csv and LAI_DISCOVERY_Q_VALUE_EXTRACT.csv
How:
- The primary family is locked to trait/canopy/soil-structure predictors, not climate/geography controls.
- Manual candidates include LAI, rooting depth, p50/psi50, C4 fraction, soil texture PC1, sand, clay, and silt when available.
- Additional trait-like columns are included only if their names match trait/canopy/soil regexes and do not match climate/geography/outcome/product regexes.
- Every main-effect model uses full Reza controls.
- Separate FDR families are used for:
  A. trait/structure main effects on latent_slope_change
  B. trait/structure × MAT interactions on latent_slope_change
- BH q, Benjamini-Yekutieli q, and Holm-adjusted p-values are reported.

4. Domain sensitivity
Implemented in: DOMAIN_SENSITIVITY_LAI.csv and DOMAIN_RANGE_RESTRICTION_DIAGNOSTICS.csv
How:
- The fixed domains are all-points, cropland-clean, natural-grassland-indicator-only, no-Sahel, MAT > 0°C, MAT > 2.08°C, |lat| ≤ 48°, and C4-covered domain.
- No new thresholds are searched.
- Each domain reports the LAI main effect and LAI×MAT interaction under the same full Reza control set.
- Range diagnostics report LAI/MAT/VPD ranges and LAI-MAT correlations by domain.

5. Cropland / managed-system screen
Implemented via merged flags from:
{LC_FLAGS_INPUT if LC_FLAGS_INPUT.exists() else "No prior flag table found"}
How:
- If the stage1b6be point-level flag table exists, it is merged by point_id.
- Primary clean mask excludes any_cropland_managed_irrigation_flag == True.
- Natural-grassland sensitivity uses any_natural_grassland_indicator == True.
- The script does not silently discard flagged points; it reports both all-points and clean-subset models.

6. Tower inclusion/exclusion table
Implemented in: STRICT_TOWER_SITE_INCLUSION.csv
How:
- The script scans known tower metadata / response tables already in the repo.
- A tower is included in the strict primary tower set only if it is strict GRA and not crop/CRO/US-Ne1/US-Ne2/US-Ne3.
- US-Ne1/US-Ne2/US-Ne3 are automatically excluded as managed cropland/agriculture.
- Every exclusion reason is printed.

7. Provisional tower directional test
Implemented in: TOWER_LAI_DIRECTIONAL_TEST.csv and STRICT_TOWER_DIRECTIONAL_TEST_SITE_VALUES.csv
How:
- If a usable tower response table exists, the script takes strict included towers only.
- It uses slope_change / tower_slope_change as the tower response.
- If no true tower-coordinate LAI exists, it uses nearest_sat_point_id or nearest satellite point as a PROVISIONAL predictor and labels it as such.
- It runs Spearman correlation plus exact-style permutation p-value and leave-one-site-out sign consistency.
- It also writes TOWER_TRUE_LAI_EXTRACTION_TODO.txt describing the proper independent extraction still needed.

8. Conditional LAI×MAT diagnostic
Implemented in: CONDITIONAL_LAI_SLOPE_OVER_MAT.csv and FIG_conditional_LAI_slope_over_MAT.png
How:
- Uses the locked interaction model:
    y_z ~ lai_z * mat_z + controls_without_mat
- Computes the conditional LAI slope over the observed MAT range:
    slope_LAI(MAT) = beta_LAI + beta_LAIxMAT * MAT
- Reports 95% confidence bands from the model covariance matrix.
- This is a diagnostic of the locked interaction, not a new threshold search.

9. Analysis lock
Implemented in: ANALYSIS_LOCK.json
How:
- Saves primary predictor, primary outcome, controls, soil texture definition, FDR families, product-matrix definition, tower-test rules, pass/fail criteria, file hashes, and git commit.
- Explicitly sets no_more_discovery_scans = true.
"""

(TXT / "FEATURE_IMPLEMENTATION_NOTES.txt").write_text(feature_notes)

# ======================================================================================
# Final memo
# ======================================================================================

def show_csv(path, n=25):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    try:
        x = pd.read_csv(p)
        if len(x) == 0:
            return "EMPTY"
        return x.head(n).to_string(index=False)
    except Exception as e:
        return f"READ_ERROR: {e}"

def show_text(path, n_chars=12000):
    p = TXT / path
    if not p.exists():
        return "MISSING"
    s = p.read_text()
    return s[:n_chars]

memo = []
memo.append("Stage1B6BF LAI Reza lock package")
memo.append("=" * 100)
memo.append("")
memo.append(f"Point input: {POINT_INPUT}")
memo.append(f"Observation input: {OBS_INPUT if OBS_INPUT.exists() else 'MISSING'}")
memo.append(f"Rows in point table: {len(d)}")
memo.append("")
memo.append("Canonical source columns:")
for k, v in sources.items():
    memo.append(f"- {k}: {v}")
memo.append("")
memo.append(f"Soil texture: {soil_texture_note}")
memo.append(f"Land-cover flags: {landcover_note}")
memo.append(f"Full controls: {BASE_CONTROLS}")
memo.append("")
memo.append("LAI discovery q-value extract:")
memo.append(show_csv("LAI_DISCOVERY_Q_VALUE_EXTRACT.csv", 40))
memo.append("")
memo.append("Algorithm-dependency table:")
memo.append(show_csv("ALGORITHM_DEPENDENCY_TABLE.csv", 40))
memo.append("")
memo.append("Exact 3×3 product matrix, key rows:")
try:
    pm = pd.read_csv(TAB / "EXACT_3x3_PRODUCT_MATRIX_LAI.csv")
    if "status" in pm.columns:
        key_pm = pm[
            (pm["status"] == "FIT_OK")
            & (
                pm["sample_mode"].astype(str).str.contains("common_complete_case", case=False, na=False)
                | pm["sample_mode"].astype(str).str.contains("available_complete_case", case=False, na=False)
            )
        ].copy()
        cols_keep = [c for c in [
            "sample_mode", "gpp_product", "et_product", "combo_dependency_class",
            "least_directly_LAI_dependent_overall_pair", "n", "coef", "p", "bh_q",
            "ci_low", "ci_high", "delta_r2", "delta_aic_full_minus_reduced"
        ] if c in key_pm.columns]
        memo.append(key_pm[cols_keep].head(80).to_string(index=False))
    else:
        memo.append(pm.head(20).to_string(index=False))
except Exception as e:
    memo.append(f"READ_ERROR: {e}")
memo.append("")
memo.append("Domain sensitivity:")
memo.append(show_csv("DOMAIN_SENSITIVITY_LAI.csv", 80))
memo.append("")
memo.append("Tower directional test:")
memo.append(show_csv("TOWER_LAI_DIRECTIONAL_TEST.csv", 20))
memo.append("")
memo.append("Strict tower inclusion table:")
memo.append(show_csv("STRICT_TOWER_SITE_INCLUSION.csv", 80))
memo.append("")
memo.append("Feature implementation notes:")
memo.append(show_text("FEATURE_IMPLEMENTATION_NOTES.txt"))
memo.append("")
memo.append("Important files:")
for f in [
    "ALGORITHM_DEPENDENCY_TABLE.csv",
    "PRODUCT_COMBINATION_DEPENDENCY_TABLE.csv",
    "EXACT_3x3_PRODUCT_MATRIX_LAI.csv",
    "FULL_DISCOVERY_FDR_TABLE.csv",
    "LAI_DISCOVERY_Q_VALUE_EXTRACT.csv",
    "DOMAIN_SENSITIVITY_LAI.csv",
    "DOMAIN_RANGE_RESTRICTION_DIAGNOSTICS.csv",
    "STRICT_TOWER_SITE_INCLUSION.csv",
    "TOWER_LAI_DIRECTIONAL_TEST.csv",
    "STRICT_TOWER_DIRECTIONAL_TEST_SITE_VALUES.csv",
    "CONDITIONAL_LAI_SLOPE_OVER_MAT.csv",
    "ANALYSIS_LOCK.json",
]:
    memo.append(f"- {TAB / f}")
for f in [
    "FEATURE_IMPLEMENTATION_NOTES.txt",
    "TOWER_TRUE_LAI_EXTRACTION_TODO.txt",
]:
    memo.append(f"- {TXT / f}")
for f in [
    "FIG_product_dependency_forest_plot_LAI.png",
    "FIG_domain_sensitivity_LAI_main.png",
    "FIG_conditional_LAI_slope_over_MAT.png",
]:
    memo.append(f"- {FIG / f}")

(TXT / "READ_ME_lai_reza_lock_package.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_lai_reza_lock_package.txt'}")
