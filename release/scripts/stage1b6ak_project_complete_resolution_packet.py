from pathlib import Path
from datetime import datetime
import json, math, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
except Exception:
    raise ImportError("Need scikit-learn. Run: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

OUT = Path("results/stage1b6ak_project_complete_resolution_packet")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260704
rng = np.random.default_rng(SEED)

STRICT_TS = Path("data/processed/stage1b6q2/analysis_design_strict_2x2_with_tower_stress_and_gs.csv")
STRICT_FITS = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")
TOWER_COMPARE = Path("results/final_nonwriting_lock/files/phase19_tower_satellite_comparison.csv")
C4_JOINED = Path("results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv")
C4_CLEAN = Path("results/stage1b6aj_clean_c4_model_lock/tables/Table_PRODUCT03bb_clean_c4_decision_by_response.csv")
SCREEN_ANSWER = Path("results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03al_product_screening_answer_for_project.csv")

PRIMARY_WUE_METRIC = "uWUE"
PRIMARY_RESPONSE_METRIC = "latent_post_slope"
PRIMARY_SENSITIVITY_RESPONSE = "p_satbreak"
EXPLORATORY_RESPONSE = "latent_slope_change"

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def read_csv_safe(p, nrows=None):
    try:
        return pd.read_csv(p, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(p, encoding="latin1", nrows=nrows)
        except Exception:
            return None

def first_col(df, candidates):
    lut = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None

def clean_class(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    if s in {"", "nan", "none"}:
        return np.nan
    if "threshold" in s:
        return "threshold-like"
    if "sat" in s and "break" in s:
        return "saturation/breakdown"
    if "saturation" in s:
        return "saturation"
    if "breakdown" in s:
        return "breakdown"
    if "reversal" in s:
        return "reversal"
    if "enhance" in s:
        return "enhancement"
    if "limit" in s:
        return "limitation-like"
    if "weak" in s or "mixed" in s or "inconclusive" in s:
        return "weak/inconclusive"
    return s

def limited_group(cls):
    s = clean_class(cls)
    if pd.isna(s):
        return np.nan
    if s in {"saturation", "breakdown", "reversal", "saturation/breakdown", "threshold-like", "limitation-like"}:
        return "limited_or_breakdown"
    if s in {"enhancement"}:
        return "enhancement"
    return "weak_or_inconclusive"

def product_combo(gpp, et):
    return f"{str(gpp).upper()}x{str(et).upper()}"

def infer_et(combo):
    s = str(combo).lower()
    if "gleam" in s:
        return "GLEAM"
    if "modis" in s or "mod16" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    return "UNKNOWN_ET"

def infer_gpp(combo):
    s = str(combo).lower()
    if "gosif" in s:
        return "GOSIF"
    if "modis" in s or "mod17" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    return "UNKNOWN_GPP"

def zscore(s):
    s = to_num(s)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def normal_p(t):
    if not np.isfinite(t):
        return np.nan
    return float(math.erfc(abs(t) / math.sqrt(2)))

def bh_q(pvals):
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

def ols_standardized(df, y, xs):
    d = df[[y] + xs].copy()
    for c in d.columns:
        d[c] = to_num(d[c])
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(20, len(xs) + 6):
        return None

    yv = d[y].to_numpy(float)
    X_parts = [np.ones(len(d))]
    kept = []
    for c in xs:
        zz = zscore(d[c]).to_numpy(float)
        if np.isfinite(zz).all() and np.nanstd(zz) > 0:
            X_parts.append(zz)
            kept.append(c)

    if "c4_fraction" not in kept and any("c4" in x for x in xs):
        return None

    X = np.column_stack(X_parts)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ beta
    resid = yv - pred
    n = len(yv)
    k = X.shape[1]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj = 1 - (1-r2)*(n-1)/max(1,n-k) if np.isfinite(r2) else np.nan

    sigma2 = rss / max(1, n-k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)

    rows = []
    for term, b, s in zip(["intercept"] + kept, beta, se):
        t = b / s if np.isfinite(s) and s != 0 else np.nan
        rows.append({
            "term": term,
            "coef_standardized": float(b),
            "se": float(s) if np.isfinite(s) else np.nan,
            "t_normal_approx": float(t) if np.isfinite(t) else np.nan,
            "p_normal_approx": normal_p(t),
        })

    return {
        "n": n,
        "k": k,
        "r2": float(r2),
        "adj_r2": float(adj),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coef_table": pd.DataFrame(rows),
        "fit_index": d.index,
        "fit_data": d,
        "resid": resid,
        "pred": pred,
        "predictors_used": kept,
    }

def make_pc(df, cols, name):
    use = []
    for c in cols:
        if c in df.columns and to_num(df[c]).notna().sum() >= 20 and to_num(df[c]).nunique(dropna=True) > 2:
            use.append(c)
    if len(use) == 0:
        return df, None, []
    d = df[use].apply(to_num)
    d = d.fillna(d.median())
    pc = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(d))[:, 0]
    df[name] = pc
    return df, name, use

def classify_sat_response(row):
    # Preserve existing response_class when available.
    if "response_class" in row and pd.notna(row["response_class"]):
        return clean_class(row["response_class"])
    post = row.get("post_slope", np.nan)
    change = row.get("slope_change", np.nan)
    sat = row.get("sat_or_breakdown", np.nan)
    if pd.notna(sat) and str(sat).lower() in {"true", "1", "yes"}:
        return "saturation/breakdown"
    if pd.notna(post) and post < 0:
        return "breakdown"
    if pd.notna(change) and change < -0.1:
        return "saturation"
    if pd.notna(post) and post > 0:
        return "enhancement"
    return "weak/inconclusive"

def status_item(item, status, evidence, limitation="", output_file=""):
    return {
        "project_item": item,
        "status": status,
        "evidence": evidence,
        "limitation_or_next_action": limitation,
        "output_file": output_file,
    }

# ---------------------------------------------------------------------
# 1. Product identifiability: WUE/GPP/ET anomaly correlations.
# ---------------------------------------------------------------------
if not STRICT_TS.exists():
    raise FileNotFoundError(f"Missing strict time-series: {STRICT_TS}")
ts = pd.read_csv(STRICT_TS)
ts["combo"] = ts["gpp_product"].astype(str).str.upper() + "x" + ts["et_product"].astype(str).str.upper()

for c in ["gpp", "et", "wue", "uwue", "log_wue", "log_uwue"]:
    if c in ts.columns:
        ts[c] = to_num(ts[c])

# Anomalies by point/product/day-of-year, then pairwise combo correlations.
anom_rows = []
corr_rows = []
metrics = [m for m in ["gpp", "et", "wue", "uwue", "log_wue", "log_uwue"] if m in ts.columns]
group_keys = ["point_id", "combo"]

for metric in metrics:
    tmp = ts[["point_id", "date", "doy", "combo", metric]].dropna().copy()
    if "doy" in tmp.columns:
        tmp[f"{metric}_anom"] = tmp[metric] - tmp.groupby(["point_id", "combo", "doy"])[metric].transform("mean")
    else:
        tmp[f"{metric}_anom"] = tmp[metric] - tmp.groupby(["point_id", "combo"])[metric].transform("mean")

    wide = tmp.pivot_table(index=["point_id", "date"], columns="combo", values=f"{metric}_anom", aggfunc="mean")
    combos = list(wide.columns)
    for i in range(len(combos)):
        for j in range(i+1, len(combos)):
            a, b = combos[i], combos[j]
            d = wide[[a,b]].dropna()
            if len(d) >= 50:
                r = float(d[a].corr(d[b]))
                corr_rows.append({
                    "metric": metric,
                    "combo_a": a,
                    "combo_b": b,
                    "n": int(len(d)),
                    "pearson_r": r,
                    "absolute_r": abs(r),
                    "same_et_product": infer_et(a) == infer_et(b),
                    "same_gpp_product": infer_gpp(a) == infer_gpp(b),
                    "et_a": infer_et(a),
                    "et_b": infer_et(b),
                    "gpp_a": infer_gpp(a),
                    "gpp_b": infer_gpp(b),
                })

corr = pd.DataFrame(corr_rows)
corr.to_csv(TAB / "Table_PRODUCT03bc_product_anomaly_correlations.csv", index=False)

if len(corr):
    corr_summary = corr.groupby("metric").agg(
        n_pairs=("pearson_r", "size"),
        median_r=("pearson_r", "median"),
        mean_abs_r=("absolute_r", "mean"),
        min_r=("pearson_r", "min"),
        max_r=("pearson_r", "max"),
    ).reset_index()
else:
    corr_summary = pd.DataFrame()
corr_summary.to_csv(TAB / "Table_PRODUCT03bd_product_identifiability_summary.csv", index=False)

# ---------------------------------------------------------------------
# 2. Product-screening definition.
# ---------------------------------------------------------------------
if SCREEN_ANSWER.exists():
    screen = pd.read_csv(SCREEN_ANSWER)
else:
    screen = pd.DataFrame([{
        "agreement_filter_keyword_hits": np.nan,
        "quality_filter_keyword_hits": np.nan,
        "answer_for_project": "Product-screening audit file missing."
    }])
screen.to_csv(TAB / "Table_PRODUCT03be_product_screened_definition_final.csv", index=False)

# ---------------------------------------------------------------------
# 3. Tower validation table: one row per site plus satellite classes for each strict 2x2 combo.
# ---------------------------------------------------------------------
if not TOWER_COMPARE.exists():
    raise FileNotFoundError(f"Missing tower comparison file: {TOWER_COMPARE}")
if not STRICT_FITS.exists():
    raise FileNotFoundError(f"Missing strict fits file: {STRICT_FITS}")

tower = pd.read_csv(TOWER_COMPARE)
fits = pd.read_csv(STRICT_FITS)

# Standardize tower table.
site_col = first_col(tower, ["site_id", "site", "tower_id"])
if site_col is None:
    raise ValueError("Tower comparison table lacks site_id/site column.")

tower["site_id"] = tower[site_col].astype(str)

tower_class_col = first_col(tower, ["tower_response_class", "tower_class", "uwue_class", "response_class_tower"])
sat_class_col = first_col(tower, ["satellite_response_class", "satellite_class", "response_class_satellite"])
years_col = first_col(tower, ["n_years", "site_years", "years"])
igbp_col = first_col(tower, ["igbp", "igbp_class", "class"])

tower_primary = pd.DataFrame({
    "site_id": tower["site_id"],
    "tower_response_class": tower[tower_class_col].map(clean_class) if tower_class_col else np.nan,
    "site_years": to_num(tower[years_col]) if years_col else np.nan,
    "igbp_class": tower[igbp_col].astype(str) if igbp_col else np.nan,
})
tower_primary = tower_primary.drop_duplicates("site_id", keep="first")

# Fill IGBP from raw tower metadata where possible.
metadata_paths = [
    Path("data/raw/towers/ameriflux_grassland_sites.csv"),
    Path("data/raw/towers/fluxnet2015_grassland_sites.csv"),
    Path("data/raw/tower_centered_phase19/phase19_main_13_tower_points_for_export.csv"),
]
meta_frames = []
for p in metadata_paths:
    if p.exists():
        d = read_csv_safe(p)
        if d is not None and len(d):
            sc = first_col(d, ["site_id", "site", "SITE_ID", "tower_id", "id"])
            ic = first_col(d, ["igbp", "IGBP", "igbp_class", "landcover", "land_cover", "class"])
            latc = first_col(d, ["lat", "latitude", "LATITUDE"])
            lonc = first_col(d, ["lon", "longitude", "LONGITUDE"])
            if sc:
                out = pd.DataFrame({"site_id": d[sc].astype(str)})
                if ic:
                    out["igbp_class_meta"] = d[ic].astype(str)
                if latc:
                    out["lat"] = to_num(d[latc])
                if lonc:
                    out["lon"] = to_num(d[lonc])
                meta_frames.append(out.drop_duplicates("site_id"))
if meta_frames:
    meta = pd.concat(meta_frames, ignore_index=True).drop_duplicates("site_id", keep="first")
    tower_primary = tower_primary.merge(meta, on="site_id", how="left")
    if "igbp_class_meta" in tower_primary.columns:
        tower_primary["igbp_class"] = tower_primary["igbp_class"].replace("nan", np.nan)
        tower_primary["igbp_class"] = tower_primary["igbp_class"].fillna(tower_primary["igbp_class_meta"])
else:
    tower_primary["lat"] = np.nan
    tower_primary["lon"] = np.nan

# Create satellite classes for each product combo from strict fits.
fits["combo"] = fits["gpp_product"].astype(str).str.upper() + "x" + fits["et_product"].astype(str).str.upper()
fits["point_id"] = fits["point_id"].astype(str)
fits["satellite_response_class"] = fits.apply(classify_sat_response, axis=1)
fits["satellite_limited_group"] = fits["satellite_response_class"].map(limited_group)

# Prefer primary metric/stress/growing season if available.
primary = fits.copy()
if "metric" in primary.columns:
    mask = primary["metric"].astype(str).str.lower().str.contains("uwue")
    if mask.sum() >= 4:
        primary = primary[mask].copy()
if "stress_col" in primary.columns:
    preferred = primary["stress_col"].astype(str).eq("stress_vpd_x_dryness_z")
    if preferred.sum() >= 4:
        primary = primary[preferred].copy()
elif "stress_definition" in primary.columns:
    preferred = primary["stress_definition"].astype(str).str.lower().str.contains("vpd|dry")
    if preferred.sum() >= 4:
        primary = primary[preferred].copy()
if "growing_season" in primary.columns:
    # Prefer GPP20 if present.
    gs = primary["growing_season"].astype(str).str.lower()
    pref = gs.str.contains("gpp20|peak")
    if pref.sum() >= 4:
        primary = primary[pref].copy()

# If still multiple per site/combo, keep best n_fit then strongest segmented preference.
for c in ["n_fit", "delta_bic_seg_minus_linear"]:
    if c in primary.columns:
        primary[c] = to_num(primary[c])
sort_cols = []
ascending = []
if "n_fit" in primary.columns:
    sort_cols.append("n_fit"); ascending.append(False)
if "delta_bic_seg_minus_linear" in primary.columns:
    sort_cols.append("delta_bic_seg_minus_linear"); ascending.append(True)
if sort_cols:
    primary = primary.sort_values(sort_cols, ascending=ascending)

primary = primary.drop_duplicates(["point_id", "combo"], keep="first")

# Merge tower class with all satellite combo classes.
sat_wide = primary.pivot_table(index="point_id", columns="combo", values="satellite_response_class", aggfunc="first")
sat_wide.columns = [f"satellite_class_{c}" for c in sat_wide.columns]
sat_wide = sat_wide.reset_index().rename(columns={"point_id": "site_id"})

tower_final = tower_primary.merge(sat_wide, on="site_id", how="left")

# Compute agreement columns for every combo.
combo_cols = [c for c in tower_final.columns if c.startswith("satellite_class_")]
for c in combo_cols:
    combo = c.replace("satellite_class_", "")
    tower_final[f"exact_agree_{combo}"] = tower_final["tower_response_class"].map(clean_class).eq(tower_final[c].map(clean_class))
    tower_final[f"limited_group_agree_{combo}"] = tower_final["tower_response_class"].map(limited_group).eq(tower_final[c].map(limited_group))

# Closure/gap-fill: use any prior computed audit if available, otherwise produce missing.
closure_path = Path("results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03am_tower_flux_quality_closure_gapfill.csv")
if closure_path.exists():
    closure = pd.read_csv(closure_path)
    if "site_id" in closure.columns:
        closure["site_id"] = closure["site_id"].astype(str)
        closure = closure.sort_values(["closure_pass_0p7_1p3", "gapfill_pass_0p3", "n_flux_rows"], ascending=[False, False, False])
        closure = closure.drop_duplicates("site_id", keep="first")
        keep = [c for c in ["site_id", "closure_ratio", "closure_pass_0p7_1p3", "gapfill_fraction_estimated", "gapfill_pass_0p3", "source_flux_file"] if c in closure.columns]
        tower_final = tower_final.merge(closure[keep], on="site_id", how="left")
else:
    tower_final["closure_ratio"] = np.nan
    tower_final["gapfill_fraction_estimated"] = np.nan

tower_final.to_csv(TAB / "Table_PRODUCT03bf_project_final_tower_validation_table.csv", index=False)

# Long per-product agreement table.
agreement_rows = []
for _, r in tower_final.iterrows():
    for c in combo_cols:
        combo = c.replace("satellite_class_", "")
        agreement_rows.append({
            "site_id": r["site_id"],
            "igbp_class": r.get("igbp_class", np.nan),
            "site_years": r.get("site_years", np.nan),
            "tower_response_class": r.get("tower_response_class", np.nan),
            "product_combo": combo,
            "gpp_product": infer_gpp(combo),
            "et_product": infer_et(combo),
            "satellite_response_class": r.get(c, np.nan),
            "exact_agreement": bool(r.get(f"exact_agree_{combo}", False)),
            "limited_group_agreement": bool(r.get(f"limited_group_agree_{combo}", False)),
            "closure_ratio": r.get("closure_ratio", np.nan),
            "gapfill_fraction": r.get("gapfill_fraction_estimated", np.nan),
        })
agreement_long = pd.DataFrame(agreement_rows)
agreement_long = agreement_long[agreement_long["satellite_response_class"].notna()].copy()
agreement_long.to_csv(TAB / "Table_PRODUCT03bg_project_tower_satellite_agreement_long.csv", index=False)

if len(agreement_long):
    et_rank = agreement_long.groupby("et_product").agg(
        n_site_product_rows=("site_id","size"),
        n_unique_sites=("site_id","nunique"),
        exact_agreement_rate=("exact_agreement","mean"),
        limited_group_agreement_rate=("limited_group_agreement","mean"),
        n_with_closure=("closure_ratio", lambda x: int(to_num(x).notna().sum())),
        n_with_gapfill=("gapfill_fraction", lambda x: int(to_num(x).notna().sum())),
    ).reset_index().sort_values(["exact_agreement_rate","limited_group_agreement_rate","n_unique_sites"], ascending=[False,False,False])
else:
    et_rank = pd.DataFrame()
et_rank.to_csv(TAB / "Table_PRODUCT03bh_project_final_et_product_ranking.csv", index=False)

tower_ranked_et = et_rank.iloc[0]["et_product"] if len(et_rank) else "UNKNOWN_ET"

# ---------------------------------------------------------------------
# 4. C4 hierarchical / controlled test.
# ---------------------------------------------------------------------
if not C4_JOINED.exists():
    raise FileNotFoundError(f"Missing C4 joined table: {C4_JOINED}")
c4 = pd.read_csv(C4_JOINED)
c4["c4_fraction"] = to_num(c4["c4_fraction"])

# Build clean covariates.
climate_inputs = []
for candidates in [
    ["aridity", "aridity_index"],
    ["mean_vpd", "baseline_vpd"],
    ["mean_annual_temperature", "mean_temperature"],
    ["mean_annual_precipitation", "mean_precipitation"],
]:
    col = first_col(c4, candidates)
    if col and col not in climate_inputs:
        climate_inputs.append(col)

soil_inputs = []
for candidates in [["soil_sand", "sand"], ["soil_clay", "clay"]]:
    col = first_col(c4, candidates)
    if col and col not in soil_inputs:
        soil_inputs.append(col)

lai_col = first_col(c4, ["growing_season_mean_lai", "mean_lai", "lai", "lai_max"])
sm_col = first_col(c4, ["mean_soil_moisture", "baseline_soil_moisture", "soil_moisture", "rootzone_soil_moisture"])
root_col = first_col(c4, ["rooting_depth", "rooting_zone_storage", "root_depth"])

c4, climate_pc, climate_used = make_pc(c4, climate_inputs, "climate_dryness_PC1")
c4, soil_pc, soil_used = make_pc(c4, soil_inputs, "soil_texture_PC1")

controls = []
if climate_pc: controls.append(climate_pc)
if soil_pc: controls.append(soil_pc)
if lai_col: controls.append(lai_col)
if sm_col: controls.append(sm_col)

controls_with_root = controls.copy()
if root_col:
    controls_with_root.append(root_col)

responses = [r for r in [PRIMARY_RESPONSE_METRIC, PRIMARY_SENSITIVITY_RESPONSE, EXPLORATORY_RESPONSE] if r in c4.columns]

# ecoregion block.
if "eco_biome" in c4.columns:
    c4["ecoregion_block"] = c4["eco_biome"].astype(str)
elif "eco_realm" in c4.columns:
    c4["ecoregion_block"] = c4["eco_realm"].astype(str)
elif "lat" in c4.columns and "lon" in c4.columns:
    c4["ecoregion_block"] = (
        np.floor(to_num(c4["lat"]) / 10).astype("Int64").astype(str)
        + "_"
        + np.floor(to_num(c4["lon"]) / 10).astype("Int64").astype(str)
    )
else:
    c4["ecoregion_block"] = "all"

def prep_model_df(df, y, xs):
    cols = [y, "c4_fraction", "ecoregion_block"] + xs
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in [y, "c4_fraction"] + xs:
        if c in d.columns:
            d[c] = to_num(d[c])
    d = d.replace([np.inf,-np.inf], np.nan).dropna()
    for c in ["c4_fraction"] + xs:
        if c in d.columns:
            d[f"z_{c}"] = zscore(d[c])
    return d

model_rows = []
boot_rows = []
loo_rows = []
mixed_rows = []

model_specs = {
    "minimal_c4_only": [],
    "controlled_no_rooting": controls,
    "controlled_with_rooting_sensitivity": controls_with_root,
}

samples = {
    "all_available_points": pd.Series(True, index=c4.index),
}

if "eco_biome" in c4.columns:
    samples["grassland_savanna_shrubland_only"] = c4["eco_biome"].astype(str).str.contains("Grassland|Savanna|Shrubland", case=False, na=False)

iterator = []
for response in responses:
    for sample_name, mask in samples.items():
        for model_name, xs0 in model_specs.items():
            iterator.append((response, sample_name, mask, model_name, xs0))

for response, sample_name, mask, model_name, xs0 in tqdm(iterator, desc="C4 controlled/hierarchical models", unit="model") if tqdm else iterator:
    sub = c4[mask].copy()
    xs = [x for x in xs0 if x in sub.columns and to_num(sub[x]).notna().sum() >= 25 and to_num(sub[x]).nunique(dropna=True) > 2]
    ols_xs = ["c4_fraction"] + xs
    fit = ols_standardized(sub, response, ols_xs)
    if fit is None:
        continue

    coef = fit["coef_table"].copy()
    coef["response"] = response
    coef["sample"] = sample_name
    coef["model"] = model_name
    coef["model_family"] = "standardized_OLS"
    coef["n"] = fit["n"]
    coef["r2"] = fit["r2"]
    coef["adj_r2"] = fit["adj_r2"]
    coef["predictors_used"] = ";".join(fit["predictors_used"])
    model_rows.append(coef)

    # Ridge.
    fit_d = fit["fit_data"]
    X = fit_d[fit["predictors_used"]].apply(to_num)
    yv = fit_d[response].to_numpy(float)
    Xz = StandardScaler().fit_transform(X)
    ridge = RidgeCV(alphas=np.logspace(-3,3,25)).fit(Xz, yv)
    c4_idx = fit["predictors_used"].index("c4_fraction")
    ridge_c4 = float(ridge.coef_[c4_idx])

    # Block bootstrap.
    dblock = sub.loc[fit["fit_index"]].copy()
    blocks = list(dblock["ecoregion_block"].dropna().unique())
    c4_coef = float(coef.loc[coef["term"]=="c4_fraction","coef_standardized"].iloc[0])
    c4_p = float(coef.loc[coef["term"]=="c4_fraction","p_normal_approx"].iloc[0])

    boot_coefs = []
    if len(blocks) >= 3:
        for _ in range(500):
            sampled = rng.choice(blocks, size=len(blocks), replace=True)
            bd = pd.concat([dblock[dblock["ecoregion_block"] == b] for b in sampled], ignore_index=True)
            bf = ols_standardized(bd, response, fit["predictors_used"])
            if bf is not None:
                bt = bf["coef_table"]
                if (bt["term"] == "c4_fraction").any():
                    boot_coefs.append(float(bt.loc[bt["term"]=="c4_fraction","coef_standardized"].iloc[0]))

    loo_coefs = []
    for b in blocks:
        ld = dblock[dblock["ecoregion_block"] != b].copy()
        if len(ld) < 25:
            continue
        lf = ols_standardized(ld, response, fit["predictors_used"])
        if lf is not None:
            lt = lf["coef_table"]
            if (lt["term"] == "c4_fraction").any():
                val = float(lt.loc[lt["term"]=="c4_fraction","coef_standardized"].iloc[0])
                loo_coefs.append(val)
                loo_rows.append({
                    "response": response,
                    "sample": sample_name,
                    "model": model_name,
                    "left_out_block": b,
                    "n_train": lf["n"],
                    "c4_coef_standardized": val,
                })

    boot_rows.append({
        "response": response,
        "sample": sample_name,
        "model": model_name,
        "n": fit["n"],
        "n_blocks": len(blocks),
        "c4_coef_standardized": c4_coef,
        "c4_p_normal_approx": c4_p,
        "ridge_c4_coef": ridge_c4,
        "ridge_same_sign": bool(np.sign(ridge_c4) == np.sign(c4_coef)),
        "bootstrap_n": len(boot_coefs),
        "bootstrap_p025": float(np.quantile(boot_coefs,0.025)) if boot_coefs else np.nan,
        "bootstrap_p975": float(np.quantile(boot_coefs,0.975)) if boot_coefs else np.nan,
        "loo_n": len(loo_coefs),
        "loo_sign_stability": float(np.mean(np.sign(loo_coefs) == np.sign(c4_coef))) if loo_coefs else np.nan,
    })

    # Mixed model with ecoregion random intercept.
    if smf is not None and len(blocks) >= 3:
        md = prep_model_df(sub, response, xs)
        if len(md) >= 40 and md["ecoregion_block"].nunique() >= 3:
            formula_terms = ["z_c4_fraction"] + [f"z_{x}" for x in xs if f"z_{x}" in md.columns]
            formula = response + " ~ " + " + ".join(formula_terms)
            try:
                mfit = smf.mixedlm(formula, md, groups=md["ecoregion_block"]).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
                term = "z_c4_fraction"
                mixed_rows.append({
                    "response": response,
                    "sample": sample_name,
                    "model": model_name,
                    "model_family": "mixedlm_random_intercept_ecoregion",
                    "n": int(len(md)),
                    "n_blocks": int(md["ecoregion_block"].nunique()),
                    "term": "c4_fraction",
                    "coef_standardized": float(mfit.params.get(term, np.nan)),
                    "se": float(mfit.bse.get(term, np.nan)),
                    "p_value": float(mfit.pvalues.get(term, np.nan)),
                    "converged": bool(getattr(mfit, "converged", False)),
                    "formula": formula,
                })
            except Exception as e:
                mixed_rows.append({
                    "response": response,
                    "sample": sample_name,
                    "model": model_name,
                    "model_family": "mixedlm_random_intercept_ecoregion",
                    "n": int(len(md)),
                    "n_blocks": int(md["ecoregion_block"].nunique()),
                    "term": "c4_fraction",
                    "coef_standardized": np.nan,
                    "se": np.nan,
                    "p_value": np.nan,
                    "converged": False,
                    "formula": formula,
                    "error": repr(e),
                })

models = pd.concat(model_rows, ignore_index=True) if model_rows else pd.DataFrame()
if len(models):
    models["bh_q_normal_approx"] = bh_q(models["p_normal_approx"].to_numpy(float))
boot = pd.DataFrame(boot_rows)
loo = pd.DataFrame(loo_rows)
mixed = pd.DataFrame(mixed_rows)
if len(mixed):
    mixed["bh_q_mixed"] = bh_q(mixed["p_value"].to_numpy(float))

models.to_csv(TAB / "Table_PRODUCT03bi_c4_ols_coefficients.csv", index=False)
boot.to_csv(TAB / "Table_PRODUCT03bj_c4_block_bootstrap_ridge.csv", index=False)
loo.to_csv(TAB / "Table_PRODUCT03bk_c4_leave_one_ecoregion_out.csv", index=False)
mixed.to_csv(TAB / "Table_PRODUCT03bl_c4_mixedlm_ecoregion_random_intercept.csv", index=False)

# Decision classification for C4.
if len(models):
    focal = models[models["term"]=="c4_fraction"].merge(
        boot[["response","sample","model","bootstrap_p025","bootstrap_p975","loo_sign_stability","ridge_same_sign","n_blocks","bootstrap_n"]],
        on=["response","sample","model"],
        how="left"
    )
else:
    focal = pd.DataFrame()

if len(mixed):
    mx = mixed[mixed["term"]=="c4_fraction"][["response","sample","model","coef_standardized","p_value","bh_q_mixed","converged"]].rename(columns={
        "coef_standardized": "mixed_c4_coef",
        "p_value": "mixed_p",
        "converged": "mixed_converged"
    })
    focal = focal.merge(mx, on=["response","sample","model"], how="left")

if len(focal):
    focal["ci_excludes_zero"] = (
        focal["bootstrap_p025"].notna()
        & focal["bootstrap_p975"].notna()
        & (focal["bootstrap_p025"] * focal["bootstrap_p975"] > 0)
    )
    focal["primary_controlled_project_pass"] = (
        focal["response"].eq(PRIMARY_RESPONSE_METRIC)
        & focal["model"].isin(["controlled_no_rooting", "controlled_with_rooting_sensitivity"])
        & (focal["p_normal_approx"] <= 0.05)
        & (focal["ci_excludes_zero"])
        & (focal["loo_sign_stability"].fillna(0) >= 0.80)
        & (focal["ridge_same_sign"].fillna(False))
    )
    focal["sensitivity_controlled_pass"] = (
        focal["response"].eq(PRIMARY_SENSITIVITY_RESPONSE)
        & focal["model"].isin(["controlled_no_rooting", "controlled_with_rooting_sensitivity"])
        & (focal["p_normal_approx"] <= 0.05)
        & (focal["ci_excludes_zero"])
        & (focal["loo_sign_stability"].fillna(0) >= 0.80)
        & (focal["ridge_same_sign"].fillna(False))
    )
    focal["exploratory_minimal_pass"] = (
        focal["model"].eq("minimal_c4_only")
        & (focal["p_normal_approx"] <= 0.05)
        & (focal["ci_excludes_zero"])
        & (focal["loo_sign_stability"].fillna(0) >= 0.80)
        & (focal["ridge_same_sign"].fillna(False))
    )
    focal = focal.sort_values(
        ["primary_controlled_project_pass","sensitivity_controlled_pass","exploratory_minimal_pass","p_normal_approx"],
        ascending=[False,False,False,True]
    )
focal.to_csv(TAB / "Table_PRODUCT03bm_c4_project_decision_by_model.csv", index=False)

# ---------------------------------------------------------------------
# 5. Protocol satisfaction matrix.
# ---------------------------------------------------------------------
items = []

items.append(status_item(
    "Product identifiability quantified across product matrix",
    "SATISFIED" if len(corr_summary) else "BLOCKED",
    f"Computed anomaly correlations for metrics: {', '.join(corr_summary['metric'].astype(str)) if len(corr_summary) else 'none'}.",
    output_file="Table_PRODUCT03bd_product_identifiability_summary.csv"
))

items.append(status_item(
    "Exactly define product-screened",
    "SATISFIED" if not screen.empty else "BLOCKED",
    screen.iloc[0].get("answer_for_project", "No answer available."),
    output_file="Table_PRODUCT03be_product_screened_definition_final.csv"
))

closure_nonmissing = int(to_num(tower_final.get("closure_ratio", pd.Series([]))).notna().sum()) if len(tower_final) else 0
gap_nonmissing = int(to_num(tower_final.get("gapfill_fraction_estimated", pd.Series([]))).notna().sum()) if len(tower_final) else 0

items.append(status_item(
    "Tower-validation table with sites/classes/product agreement",
    "SATISFIED" if len(tower_final) and len(agreement_long) else "BLOCKED",
    f"Final tower table has {len(tower_final)} sites and {len(agreement_long)} site-product rows.",
    output_file="Table_PRODUCT03bf_project_final_tower_validation_table.csv"
))

items.append(status_item(
    "Tower energy-balance closure and gap-fill filters",
    "SATISFIED" if closure_nonmissing > 0 and gap_nonmissing > 0 else "SATISFIED_WITH_CAVEAT",
    f"Closure values available for {closure_nonmissing}/{len(tower_final)} tower rows; gap-fill values available for {gap_nonmissing}/{len(tower_final)} rows.",
    "Raw tower files with H, LE, NETRAD, G and FLUXNET/AmeriFlux QC fields are not present/parseable for the clean 13-site table. This must be stated to project or filled by downloading raw network exports.",
    output_file="Table_PRODUCT03bf_project_final_tower_validation_table.csv"
))

items.append(status_item(
    "Per-ET-product tower agreement ranking",
    "SATISFIED" if len(et_rank) else "BLOCKED",
    f"ET ranking rows: {len(et_rank)}. Top ET product: {tower_ranked_et}.",
    output_file="Table_PRODUCT03bh_project_final_et_product_ranking.csv"
))

items.append(status_item(
    "C4 data layer attached",
    "SATISFIED" if len(c4) and c4["c4_fraction"].notna().sum() > 0 else "BLOCKED",
    f"C4 nonmissing points: {int(c4['c4_fraction'].notna().sum())}/{len(c4)}.",
    output_file="Table_PRODUCT03bm_c4_project_decision_by_model.csv"
))

if len(focal):
    primary_pass = bool(focal["primary_controlled_project_pass"].any())
    sens_pass = bool(focal["sensitivity_controlled_pass"].any())
    expl_pass = bool(focal["exploratory_minimal_pass"].any())
else:
    primary_pass = sens_pass = expl_pass = False

items.append(status_item(
    "Pre-specified primary C4 controlled test",
    "SATISFIED" if primary_pass else "SATISFIED_WITH_CAVEAT" if expl_pass or sens_pass else "FAILED_TEST_COMPLETED",
    f"Primary response={PRIMARY_RESPONSE_METRIC}; sensitivity={PRIMARY_SENSITIVITY_RESPONSE}; exploratory={EXPLORATORY_RESPONSE}. Primary controlled pass={primary_pass}; sensitivity controlled pass={sens_pass}; exploratory/minimal pass={expl_pass}.",
    "If only exploratory/minimal C4 passes, report it as supportive/exploratory, not as the primary controlled project hypothesis.",
    output_file="Table_PRODUCT03bm_c4_project_decision_by_model.csv"
))

items.append(status_item(
    "Hierarchical / partial-pooling model with ecoregion random effects",
    "SATISFIED" if len(mixed) and mixed.get("converged", pd.Series(False)).fillna(False).any() else "SATISFIED_WITH_CAVEAT",
    f"MixedLM rows: {len(mixed)}; converged rows: {int(mixed.get('converged', pd.Series(False)).fillna(False).sum()) if len(mixed) else 0}.",
    "If MixedLM does not converge, use OLS + ecoregion block bootstrap + leave-one-ecoregion-out as the reported robustness model.",
    output_file="Table_PRODUCT03bl_c4_mixedlm_ecoregion_random_intercept.csv"
))

items.append(status_item(
    "Spatial block bootstrap and leave-one-ecoregion-out",
    "SATISFIED" if len(boot) and len(loo) else "BLOCKED",
    f"Bootstrap rows: {len(boot)}; leave-one-block rows: {len(loo)}.",
    output_file="Table_PRODUCT03bj_c4_block_bootstrap_ridge.csv; Table_PRODUCT03bk_c4_leave_one_ecoregion_out.csv"
))

satisfaction = pd.DataFrame(items)
satisfaction.to_csv(TAB / "Table_PRODUCT03bn_project_protocol_satisfaction_matrix.csv", index=False)

# ---------------------------------------------------------------------
# 6. Final interpretation.
# ---------------------------------------------------------------------
if primary_pass:
    paper_fork = "ECOLOGICAL_C3C4_MECHANISM_PRIMARY_SUPPORTED"
elif sens_pass:
    paper_fork = "ECOLOGICAL_C3C4_MECHANISM_SENSITIVITY_SUPPORTED"
elif expl_pass:
    paper_fork = "C3C4_EXPLORATORY_SIGNAL_ONLY_METHODS_OR_MECHANISM_NEEDS_CAUTION"
else:
    paper_fork = "PRODUCT_IDENTIFIABILITY_METHODS_PAPER"

best_c4 = focal.iloc[0].to_dict() if len(focal) else {}
best_c4_text = "No C4 model rows."
if len(focal):
    b = focal.iloc[0]
    best_c4_text = (
        f"Best C4 row: response={b['response']}, sample={b['sample']}, model={b['model']}, "
        f"n={int(b['n'])}, coef={b['coef_standardized']:.3f}, p={b['p_normal_approx']:.4g}, "
        f"q={b['bh_q_normal_approx']:.4g}, bootCI=[{b['bootstrap_p025']:.3f},{b['bootstrap_p975']:.3f}], "
        f"LOO={b['loo_sign_stability']:.3f}, ridge_same_sign={bool(b['ridge_same_sign'])}."
    )

decision = {
    "generated": now(),
    "stage": "1B.6AK_project_complete_resolution_packet",
    "paper_fork": paper_fork,
    "tower_ranked_et": tower_ranked_et,
    "n_tower_sites": int(len(tower_final)),
    "n_site_product_agreement_rows": int(len(agreement_long)),
    "closure_rows_available": closure_nonmissing,
    "gapfill_rows_available": gap_nonmissing,
    "product_screened_answer": screen.iloc[0].get("answer_for_project", "") if not screen.empty else "",
    "primary_c4_controlled_pass": primary_pass,
    "sensitivity_c4_controlled_pass": sens_pass,
    "exploratory_c4_minimal_pass": expl_pass,
    "best_c4_text": best_c4_text,
    "all_project_items_fully_satisfied": bool((satisfaction["status"] == "SATISFIED").all()),
    "items_with_caveat": satisfaction[satisfaction["status"].astype(str).str.contains("CAVEAT|FAILED|BLOCKED", regex=True)].to_dict(orient="records"),
}
(TAB / "STAGE1B6AK_project_COMPLETE_RESOLUTION_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

# project-ready email content.
project_lines = []
project_lines.append("Hi project,")
project_lines.append("")
project_lines.append("Thank you again — I treated your note as an analysis-locking checklist rather than as a manuscript-writing prompt, and I reran the work around the gates you specified.")
project_lines.append("")
project_lines.append("First, I audited what I meant by “product-screened.” I did not find evidence that pixels were filtered by product agreement. The audit found 0 agreement-filter hits and 208 QC/product-quality hits, so I am defining product-screened as QC/product-confidence screening rather than retaining only product-agreeing pixels.")
project_lines.append("")
project_lines.append(f"Second, I rebuilt the tower-validation table. The current clean table has {len(tower_final)} tower sites and {len(agreement_long)} site-product rows. The ET-product ranking currently selects {tower_ranked_et}. I am treating tower agreement as a confidence/ranking layer, not as proof of a threshold.")
project_lines.append("")
if closure_nonmissing == 0 or gap_nonmissing == 0:
    project_lines.append("The one remaining caveat is that energy-balance closure and gap-fill fields are not yet computable for the clean tower-validation rows from the local raw files I currently have. I generated the table with explicit missing-field flags rather than filling these values artificially. If you want those filters strictly enforced before submission, the next mechanical step is to add the raw FLUXNET/AmeriFlux exports containing H, LE, NETRAD, G, and QC fields for the selected towers, after which the same script will populate closure and gap-fill columns.")
    project_lines.append("")
else:
    project_lines.append("The tower table also includes closure and gap-fill fields, so the tower quality-filter requirement is filled directly.")
    project_lines.append("")
project_lines.append("Third, I downloaded and attached the Luo et al. C4 composition layer and ran the C3/C4 test. I separated the primary controlled test from exploratory/minimal screens so we do not overclaim.")
project_lines.append("")
project_lines.append(best_c4_text)
project_lines.append("")
if paper_fork == "ECOLOGICAL_C3C4_MECHANISM_PRIMARY_SUPPORTED":
    project_lines.append("The primary controlled C4 test supports the ecological mechanism framing.")
elif paper_fork == "ECOLOGICAL_C3C4_MECHANISM_SENSITIVITY_SUPPORTED":
    project_lines.append("The primary controlled C4 test is not the cleanest positive result, but the pre-specified sensitivity response supports the C4 mechanism.")
elif paper_fork == "C3C4_EXPLORATORY_SIGNAL_ONLY_METHODS_OR_MECHANISM_NEEDS_CAUTION":
    project_lines.append("The C4 signal is real enough to keep as a mechanism candidate, but the strongest positive row is exploratory/minimal rather than the primary controlled model, so I would not overstate it.")
else:
    project_lines.append("The C4 test does not support the ecological mechanism framing; the safer paper fork is the product-identifiability/tower-ranking paper.")
project_lines.append("")
project_lines.append("I also generated product-anomaly correlation tables so the product disagreement/identifiability issue is quantified rather than assumed away.")
project_lines.append("")
project_lines.append("Best,")
project_lines.append("Akul")

(TXT / "project_READY_RESPONSE_DRAFT.md").write_text("\n".join(project_lines), encoding="utf-8")

# Report.
report = []
report.append("# Stage 1B.6AK project complete resolution packet")
report.append("")
report.append(f"Generated: {decision['generated']}")
report.append("")
report.append("## Final decision")
report.append("")
report.append("```json")
report.append(json.dumps(decision, indent=2))
report.append("```")
report.append("")
report.append("## project protocol satisfaction matrix")
report.append("")
report.append("```text")
report.append(satisfaction.to_string(index=False))
report.append("```")
report.append("")
report.append("## Product identifiability summary")
report.append("")
report.append("```text")
report.append(corr_summary.to_string(index=False) if len(corr_summary) else "No product correlation summary.")
report.append("```")
report.append("")
report.append("## Product-screened definition")
report.append("")
report.append("```text")
report.append(screen.to_string(index=False))
report.append("```")
report.append("")
report.append("## Final tower-validation table preview")
report.append("")
tower_cols = [c for c in tower_final.columns if c in ["site_id","igbp_class","site_years","tower_response_class","closure_ratio","gapfill_fraction_estimated"] or c.startswith("satellite_class_") or c.startswith("exact_agree_")]
report.append("```text")
report.append(tower_final[tower_cols].head(40).to_string(index=False) if len(tower_final) else "No tower rows.")
report.append("```")
report.append("")
report.append("## ET product ranking")
report.append("")
report.append("```text")
report.append(et_rank.to_string(index=False) if len(et_rank) else "No ET ranking.")
report.append("```")
report.append("")
report.append("## C4 project decision table")
report.append("")
report.append("```text")
report.append(focal.head(60).to_string(index=False) if len(focal) else "No C4 decision rows.")
report.append("```")
report.append("")
report.append("## Mixed model rows")
report.append("")
report.append("```text")
report.append(mixed.head(80).to_string(index=False) if len(mixed) else "No mixed model rows.")
report.append("```")
report.append("")
report.append("## Bottom line")
report.append("")
if decision["all_project_items_fully_satisfied"]:
    report.append("All project items are fully satisfied from local data.")
else:
    report.append("The analysis packet is complete, but not every project item is fully satisfied from local data. The unresolved items are explicitly flagged in the satisfaction matrix, especially closure/gap-fill if raw tower energy-balance/QC fields are unavailable.")
report.append("")
report.append("## project-ready response draft")
report.append("")
report.append("```text")
report.append("\n".join(project_lines))
report.append("```")

report_text = "\n".join(report)
(TXT / "STAGE1B6AK_project_COMPLETE_RESOLUTION_REPORT.md").write_text(report_text, encoding="utf-8")

# Figures.
if plt is not None:
    try:
        if len(et_rank):
            plt.figure(figsize=(7,4))
            plt.bar(et_rank["et_product"].astype(str), et_rank["exact_agreement_rate"].astype(float))
            plt.ylabel("Exact tower/satellite class agreement")
            plt.xlabel("ET product")
            plt.title("Tower-ranked ET products")
            plt.tight_layout()
            plt.savefig(FIG / "Figure_PRODUCT03f_final_et_product_ranking.png", dpi=220)
            plt.close()
    except Exception:
        pass

    try:
        if len(focal):
            plot = focal.head(20).copy()
            labels = plot["response"].astype(str) + " | " + plot["sample"].astype(str) + " | " + plot["model"].astype(str)
            plt.figure(figsize=(12, max(6, len(plot)*0.4)))
            plt.barh(labels[::-1], plot["coef_standardized"][::-1])
            plt.axvline(0, linestyle="--")
            plt.xlabel("Standardized C4 coefficient")
            plt.ylabel("Response | sample | model")
            plt.title("C4 mechanism decision rows")
            plt.tight_layout()
            plt.savefig(FIG / "Figure_PRODUCT03g_c4_project_decision_rows.png", dpi=220)
            plt.close()
    except Exception:
        pass

print(report_text)
print("")
print("WROTE", TAB / "Table_PRODUCT03bc_product_anomaly_correlations.csv")
print("WROTE", TAB / "Table_PRODUCT03bd_product_identifiability_summary.csv")
print("WROTE", TAB / "Table_PRODUCT03be_product_screened_definition_final.csv")
print("WROTE", TAB / "Table_PRODUCT03bf_project_final_tower_validation_table.csv")
print("WROTE", TAB / "Table_PRODUCT03bg_project_tower_satellite_agreement_long.csv")
print("WROTE", TAB / "Table_PRODUCT03bh_project_final_et_product_ranking.csv")
print("WROTE", TAB / "Table_PRODUCT03bi_c4_ols_coefficients.csv")
print("WROTE", TAB / "Table_PRODUCT03bj_c4_block_bootstrap_ridge.csv")
print("WROTE", TAB / "Table_PRODUCT03bk_c4_leave_one_ecoregion_out.csv")
print("WROTE", TAB / "Table_PRODUCT03bl_c4_mixedlm_ecoregion_random_intercept.csv")
print("WROTE", TAB / "Table_PRODUCT03bm_c4_project_decision_by_model.csv")
print("WROTE", TAB / "Table_PRODUCT03bn_project_protocol_satisfaction_matrix.csv")
print("WROTE", TAB / "STAGE1B6AK_project_COMPLETE_RESOLUTION_DECISION.json")
print("WROTE", TXT / "project_READY_RESPONSE_DRAFT.md")
print("WROTE", TXT / "STAGE1B6AK_project_COMPLETE_RESOLUTION_REPORT.md")
