from pathlib import Path
import re, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6be_FULL_STRICT_lai_artifact_screen"
TAB = OUT / "tables"
TXT = OUT / "text"
for p in [TAB, TXT]:
    p.mkdir(parents=True, exist_ok=True)

POINT_CANDIDATES = [
    ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv",
    ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv",
    ROOT / "results/trait_framework/phase8/table_latent_response_by_point.csv",
]
POINT_INPUT = next((p for p in POINT_CANDIDATES if p.exists()), None)
if POINT_INPUT is None:
    raise SystemExit("No point-level input table found.")

OBS_INPUT = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"

raw = pd.read_csv(POINT_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
raw = raw.loc[:, ~raw.columns.duplicated()].copy()

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")

def first_existing(cols, *names):
    for n in names:
        if n in cols:
            return n
    low = {norm(c): c for c in cols}
    for n in names:
        if norm(n) in low:
            return low[norm(n)]
    return None

def find_col_contains(cols, patterns):
    hits = []
    for c in cols:
        lc = norm(c)
        if any(re.search(p, lc) for p in patterns):
            hits.append(c)
    return hits

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
    if len(use) < max(40, len(vars_needed) + 8):
        return None, use, "N_TOO_SMALL"
    try:
        fit = smf.ols(formula, data=use).fit(cov_type=cov_type)
        return fit, use, "FIT_OK"
    except Exception as e:
        return None, use, f"FIT_FAIL: {e}"

def compare(data, label, full_formula, reduced_formula, focal_terms, note=""):
    full, use_full, status = fit_ols(data, full_formula)
    red, use_red, red_status = fit_ols(data, reduced_formula)

    rows = []
    if full is None or red is None:
        for term in focal_terms:
            rows.append({
                "test_label": label,
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

def bootstrap_term(data, formula, term, n_boot=1000, seed=123):
    fit, use, status = fit_ols(data, formula)
    if fit is None:
        return {"boot_n": 0, "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_median": np.nan, "boot_pass": False}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(use), len(use))
        bs = use.iloc[idx].copy()
        try:
            f = smf.ols(formula, data=bs).fit()
            vals.append(f.params.get(term, np.nan))
        except Exception:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 100:
        return {"boot_n": len(vals), "boot_ci_low": np.nan, "boot_ci_high": np.nan, "boot_median": np.nan, "boot_pass": False}
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return {
        "boot_n": len(vals),
        "boot_median": float(np.median(vals)),
        "boot_ci_low": lo,
        "boot_ci_high": hi,
        "boot_pass": bool(lo * hi > 0),
    }

def leave_region_out(data, formula, term):
    if "region_block" not in data.columns:
        return {"lro_n": 0, "lro_median": np.nan, "lro_sign_stability": np.nan}
    vars_needed = formula_vars(formula) + ["region_block"]
    use = data[vars_needed].replace([np.inf, -np.inf], np.nan).dropna()
    vals = []
    for r in sorted(use["region_block"].dropna().unique()):
        train = use[use["region_block"] != r].copy()
        if len(train) < max(40, len(vars_needed) + 8):
            continue
        try:
            f = smf.ols(formula, data=train).fit()
            vals.append(f.params.get(term, np.nan))
        except Exception:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) == 0:
        return {"lro_n": 0, "lro_median": np.nan, "lro_sign_stability": np.nan}
    med = float(np.median(vals))
    return {
        "lro_n": len(vals),
        "lro_median": med,
        "lro_sign_stability": float(np.mean(np.sign(vals) == np.sign(med))) if med != 0 else np.nan,
    }

# ------------------------------------------------------------
# Canonical point-level data
# ------------------------------------------------------------

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
    "lat": first_existing(cols, "lat"),
    "lon": first_existing(cols, "lon"),
    "sand": first_existing(cols, "soil_sand", "sand", "sand_fraction"),
    "clay": first_existing(cols, "soil_clay", "clay", "clay_fraction"),
    "silt": first_existing(cols, "soil_silt", "silt", "silt_fraction"),
    "soil_texture_pc1_existing": first_existing(cols, "soil_texture_pc1", "soil_texture"),
    "c4": first_existing(cols, "c4_fraction"),
    "root": first_existing(cols, "rooting_depth"),
}

# Search for growing-season temperature variants.
gs_temp_candidates = find_col_contains(cols, [
    r"growing.*season.*temp",
    r"season.*mean.*temp",
    r"gs.*temp",
    r"gseason.*temp",
])
sources["gs_temp"] = gs_temp_candidates[0] if gs_temp_candidates else None

required = ["y", "vpd", "lai", "mat", "map", "arid", "sm", "lat", "lon"]
missing = [k for k in required if sources[k] is None]
if missing:
    raise SystemExit(f"Missing required source columns: {missing}")

d = pd.DataFrame(index=raw.index)
for canon, src in sources.items():
    if src is None:
        continue
    if canon == "point_id":
        d[canon] = raw[src]
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
    soil_texture_note = "Using existing soil_texture_pc1 column."
elif all(c in d.columns for c in ["sand", "clay", "silt"]) and d[["sand", "clay", "silt"]].notna().sum().min() >= 40:
    soil = d[["sand", "clay", "silt"]].copy()
    soil_std = soil.apply(zscore)
    use = soil_std.dropna()
    pc1 = pd.Series(np.nan, index=d.index)
    # SVD PC1; sign oriented so higher clay tends positive if possible.
    X = use.values
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    scores = X @ vt[0, :]
    pc1.loc[use.index] = scores
    if "clay" in soil.columns and pc1.corr(d["clay"]) < 0:
        pc1 = -pc1
    d["soil_texture_pc1"] = pc1
    soil_texture_note = "Computed soil_texture_pc1 from sand/clay/silt using SVD; oriented positive with clay."
else:
    d["soil_texture_pc1"] = np.nan
    soil_texture_note = "No usable soil texture PC1 or sand/clay/silt combination found."

# Standardized columns.
for c in [x for x in d.columns if x not in ["point_id", "region_block", "sahel_broad", "sahel_core", "has_c4"]]:
    d[c + "_z"] = zscore(d[c])

# ------------------------------------------------------------
# Land-cover / cropland / managed-system leakage audit
# ------------------------------------------------------------

LC_PATTERNS = [
    r"crop", r"cropland", r"irrig", r"landcover", r"land_cover", r"\blc\b",
    r"igbp", r"esa", r"modis.*class", r"pasture", r"grass", r"managed",
    r"maize", r"corn", r"sorghum", r"millet", r"sugarcane", r"sugar_cane",
    r"cultiv", r"agric", r"hay", r"mow", r"fertil",
]
CROP_TERMS = [
    "crop", "cropland", "cultivated", "agriculture", "agricultural", "maize",
    "corn", "sorghum", "millet", "sugarcane", "sugar cane", "irrigated",
    "managed", "fertilized", "mowed", "hay",
]
NATURAL_GRASS_TERMS = ["grassland", "natural grass", "rangeland", "savanna", "steppe", "prairie"]

def safe_read_head(path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=5, low_memory=False)
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

def safe_read_cols(path, cols_to_read):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, usecols=lambda c: c in cols_to_read, low_memory=False)
        if path.suffix.lower() in [".parquet", ".pq"]:
            return pd.read_parquet(path, columns=[c for c in cols_to_read])
    except Exception:
        return None
    return None

scan_roots = [ROOT / "results", ROOT / "data" / "processed", ROOT / "data" / "raw"]
all_tabular_files = []
for sr in scan_roots:
    if sr.exists():
        all_tabular_files += list(sr.rglob("*.csv"))
        all_tabular_files += list(sr.rglob("*.parquet"))
        all_tabular_files += list(sr.rglob("*.pq"))

# Avoid extremely large raw timeseries, but keep inventories.
MAX_SIZE = 150 * 1024 * 1024

inventory = []
merge_candidates = []
tower_candidates = []

for path in sorted(set(all_tabular_files)):
    try:
        size = path.stat().st_size
    except Exception:
        continue

    if size > MAX_SIZE:
        continue

    head = safe_read_head(path)
    if head is None:
        continue

    fcols = list(head.columns)
    lc_cols = find_col_contains(fcols, LC_PATTERNS)
    point_col = first_existing(fcols, "point_id")
    lat_col = first_existing(fcols, "lat", "latitude")
    lon_col = first_existing(fcols, "lon", "longitude")
    path_l = str(path).lower()

    is_towerish = any(t in path_l for t in ["tower", "ameriflux", "fluxnet", "icos", "ozflux", "us-ne"])

    inventory.append({
        "path": str(path),
        "size_mb": size / 1024 / 1024,
        "n_cols": len(fcols),
        "has_point_id": point_col is not None,
        "has_lat_lon": lat_col is not None and lon_col is not None,
        "n_landcover_like_cols": len(lc_cols),
        "landcover_like_cols": "; ".join(map(str, lc_cols[:30])),
        "is_towerish": is_towerish,
        "columns": "; ".join(map(str, fcols[:100])),
    })

    if point_col is not None and lc_cols:
        merge_candidates.append((path, point_col, lc_cols))

    if is_towerish or lc_cols:
        tower_candidates.append(path)

pd.DataFrame(inventory).sort_values(
    ["n_landcover_like_cols", "is_towerish", "has_point_id"],
    ascending=[False, False, False],
).to_csv(TAB / "LANDCOVER_CROPLAND_FILE_INVENTORY.csv", index=False)

# Merge point-level land-cover-like variables where possible.
lc_merged = d[["point_id", "lat", "lon"]].copy() if "point_id" in d.columns else d[["lat", "lon"]].copy()
merged_lc_cols = []

for i, (path, point_col, lc_cols) in enumerate(merge_candidates[:40]):
    cols_to_read = [point_col] + lc_cols
    tmp = safe_read_cols(path, cols_to_read)
    if tmp is None or point_col not in tmp.columns:
        continue

    tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()
    tmp = tmp.dropna(subset=[point_col])
    if len(tmp) == 0:
        continue

    # Keep one row per point_id; mostly these are point tables.
    tmp = tmp.groupby(point_col, dropna=False).first().reset_index()
    prefix = f"lc{i:02d}_{path.stem[:24]}"

    rename = {point_col: "point_id"}
    for c in tmp.columns:
        if c != point_col:
            newc = f"{prefix}__{norm(c)[:60]}"
            rename[c] = newc
            merged_lc_cols.append(newc)

    tmp = tmp.rename(columns=rename)
    try:
        lc_merged = lc_merged.merge(tmp, on="point_id", how="left")
    except Exception:
        continue

lc_merged = lc_merged.loc[:, ~lc_merged.columns.duplicated()].copy()

def series_indicates_issue(colname, s):
    lc = norm(colname)

    # Positive clean/pass columns: False means possible issue.
    if any(k in lc for k in ["crop_clean", "no_crop", "non_crop", "grassland_clean", "natural_grassland_clean"]):
        vals = s
        if vals.dtype == bool:
            return vals.eq(False)
        nums = pd.to_numeric(vals, errors="coerce")
        if nums.notna().any():
            return nums.eq(0)
        txt = vals.astype(str).str.lower()
        return txt.isin(["false", "0", "no", "fail", "failed"])

    # Issue columns.
    if any(k in lc for k in ["crop", "cropland", "irrig", "maize", "corn", "sorghum", "millet", "sugarcane", "managed", "cultiv", "agric"]):
        nums = pd.to_numeric(s, errors="coerce")
        if nums.notna().any():
            return nums.fillna(0) > 0
        txt = s.astype(str).str.lower()
        return txt.apply(lambda v: any(t in v for t in CROP_TERMS))

    # Land-cover text columns.
    txt = s.astype(str).str.lower()
    return txt.apply(lambda v: any(t in v for t in CROP_TERMS))

def series_indicates_natural_grass(colname, s):
    lc = norm(colname)
    if "grassland_clean" in lc or "natural_grassland" in lc:
        vals = s
        if vals.dtype == bool:
            return vals.eq(True)
        nums = pd.to_numeric(vals, errors="coerce")
        if nums.notna().any():
            return nums.eq(1)
    txt = s.astype(str).str.lower()
    return txt.apply(lambda v: any(t in v for t in NATURAL_GRASS_TERMS))

issue_flags = []
natural_flags = []

for c in merged_lc_cols:
    try:
        issue_flags.append(series_indicates_issue(c, lc_merged[c]).rename(c))
        natural_flags.append(series_indicates_natural_grass(c, lc_merged[c]).rename(c))
    except Exception:
        pass

if issue_flags:
    issue_mat = pd.concat(issue_flags, axis=1)
    lc_merged["any_cropland_managed_irrigation_flag"] = issue_mat.any(axis=1)
    lc_merged["n_cropland_managed_irrigation_flags"] = issue_mat.sum(axis=1)
else:
    lc_merged["any_cropland_managed_irrigation_flag"] = False
    lc_merged["n_cropland_managed_irrigation_flags"] = 0

if natural_flags:
    nat_mat = pd.concat(natural_flags, axis=1)
    lc_merged["any_natural_grassland_indicator"] = nat_mat.any(axis=1)
    lc_merged["n_natural_grassland_indicators"] = nat_mat.sum(axis=1)
else:
    lc_merged["any_natural_grassland_indicator"] = False
    lc_merged["n_natural_grassland_indicators"] = 0

lc_merged.to_csv(TAB / "POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv", index=False)

flag_cols_to_merge = ["point_id", "any_cropland_managed_irrigation_flag", "n_cropland_managed_irrigation_flags", "any_natural_grassland_indicator", "n_natural_grassland_indicators"]
if "point_id" in d.columns and "point_id" in lc_merged.columns:
    d = d.merge(lc_merged[flag_cols_to_merge], on="point_id", how="left")
else:
    for c in flag_cols_to_merge:
        if c != "point_id":
            d[c] = np.nan

d["any_cropland_managed_irrigation_flag"] = d["any_cropland_managed_irrigation_flag"].fillna(False).astype(bool)
d["n_cropland_managed_irrigation_flags"] = pd.to_numeric(d["n_cropland_managed_irrigation_flags"], errors="coerce").fillna(0)
d["any_natural_grassland_indicator"] = d["any_natural_grassland_indicator"].fillna(False).astype(bool)
d["n_natural_grassland_indicators"] = pd.to_numeric(d["n_natural_grassland_indicators"], errors="coerce").fillna(0)

d[d["any_cropland_managed_irrigation_flag"]].to_csv(TAB / "EXACT_POINTS_FLAGGED_CROPLAND_MANAGED_IRRIGATION.csv", index=False)

# ------------------------------------------------------------
# Tower land-cover audit
# ------------------------------------------------------------

tower_rows = []
target_tower_terms = ["us-ne1", "us-ne2", "us-ne3", "usne1", "usne2", "usne3"]

for path in sorted(set(tower_candidates)):
    try:
        size = path.stat().st_size
    except Exception:
        continue
    if size > MAX_SIZE:
        continue

    head = safe_read_head(path)
    if head is None:
        continue

    fcols = list(head.columns)
    lc_cols = find_col_contains(fcols, LC_PATTERNS)
    site_cols = find_col_contains(fcols, [r"site", r"station", r"tower", r"flux", r"id", r"name"])
    read_cols = list(dict.fromkeys(site_cols + lc_cols))

    n_rows = np.nan
    target_present = False
    target_rows_text = ""
    landcover_values = ""

    if read_cols:
        tmp = safe_read_cols(path, read_cols)
        if tmp is not None:
            n_rows = len(tmp)
            row_hit = pd.Series(False, index=tmp.index)
            for c in tmp.columns:
                col_txt = tmp[c].fillna("").astype(str).str.lower()
                row_hit = row_hit | col_txt.apply(lambda v: any(t in str(v) for t in target_tower_terms))
            target_present = bool(row_hit.any())
            if target_present:
                target_rows_text = tmp[row_hit].head(20).to_string(index=False)
            if lc_cols:
                vals = []
                for c in lc_cols[:20]:
                    vals.append(f"{c}: {tmp[c].dropna().astype(str).value_counts().head(10).to_dict()}")
                landcover_values = " | ".join(vals)

    tower_rows.append({
        "path": str(path),
        "size_mb": size / 1024 / 1024,
        "n_rows_read": n_rows,
        "site_like_cols": "; ".join(site_cols),
        "landcover_like_cols": "; ".join(lc_cols),
        "target_US_Ne_present": target_present,
        "target_rows_text": target_rows_text,
        "landcover_value_summary": landcover_values,
        "columns": "; ".join(fcols[:100]),
    })

tower_audit = pd.DataFrame(tower_rows)
if len(tower_audit):
    tower_audit = tower_audit.sort_values(["target_US_Ne_present", "landcover_like_cols"], ascending=[False, False])
tower_audit.to_csv(TAB / "TOWER_LANDCOVER_AND_US_NE_AUDIT.csv", index=False)

# ------------------------------------------------------------
# Full-control LAI models with soil texture and land-cover scenarios
# ------------------------------------------------------------

# full strict control set:
# VPD, aridity, temperature, precipitation, soil texture, productivity/canopy structure as focal LAI,
# baseline soil moisture, spatial terms.
base_controls = ["vpd_z", "arid_z", "mat_z", "map_z", "sm_z", "soil_texture_pc1_z", "lat_z", "lon_z"]
base_controls_no_mat = ["vpd_z", "arid_z", "map_z", "sm_z", "soil_texture_pc1_z", "lat_z", "lon_z"]

main_full = "y_z ~ lai_z + " + " + ".join(base_controls)
main_reduced = "y_z ~ " + " + ".join(base_controls)

int_full = "y_z ~ lai_z * mat_z + " + " + ".join(base_controls_no_mat)
int_reduced = "y_z ~ lai_z + mat_z + " + " + ".join(base_controls_no_mat)

if "gs_temp_z" in d.columns and d["gs_temp_z"].notna().sum() >= 60:
    gs_int_full = "y_z ~ lai_z * gs_temp_z + vpd_z + arid_z + map_z + sm_z + soil_texture_pc1_z + lat_z + lon_z"
    gs_int_reduced = "y_z ~ lai_z + gs_temp_z + vpd_z + arid_z + map_z + sm_z + soil_texture_pc1_z + lat_z + lon_z"
else:
    gs_int_full = None
    gs_int_reduced = None

scenario_masks = {
    "all_points_FULL_STRICT_controls": pd.Series(True, index=d.index),
    "exclude_sahel_broad": ~d["sahel_broad"],
    "exclude_sahel_core": ~d["sahel_core"],
    "exclude_cropland_managed_irrigation_flags": ~d["any_cropland_managed_irrigation_flag"],
    "natural_grassland_indicator_only": d["any_natural_grassland_indicator"],
    "exclude_cropland_flags_and_sahel": (~d["any_cropland_managed_irrigation_flag"]) & (~d["sahel_broad"]),
    "warm_only_mat_gt_0": d["mat"] > 0,
    "warm_only_mat_gt_2C": d["mat"] > 2.08,
    "exclude_high_lat_abs_gt_48": d["abs_lat"] <= 48,
    "c4_covered_domain_only": d["has_c4"],
}

model_rows = []
for scenario, mask in scenario_masks.items():
    sub = d.loc[mask].copy()

    model_rows += compare(
        sub,
        f"{scenario}__LAI_main_FULL_STRICT",
        main_full,
        main_reduced,
        ["lai_z"],
        note="Full strict controls: VPD, aridity, MAT, precipitation, soil moisture, soil texture PC1, lat/lon. LAI is focal predictor."
    )

    model_rows += compare(
        sub,
        f"{scenario}__LAI_x_temperature_FULL_STRICT",
        int_full,
        int_reduced,
        ["lai_z:mat_z"],
        note="Full strict interaction model with soil texture and hydroclimate/geography controls."
    )

    if gs_int_full is not None:
        model_rows += compare(
            sub,
            f"{scenario}__LAI_x_growing_season_temperature_FULL_STRICT",
            gs_int_full,
            gs_int_reduced,
            ["lai_z:gs_temp_z"],
            note="Sensitivity using growing-season temperature column if found."
        )

full_models = pd.DataFrame(model_rows)

# Add bootstrap and leave-region-out to all-points full-control canonical models.
stability_rows = []
for label, formula, term in [
    ("all_points_FULL_STRICT_controls__LAI_main_FULL_STRICT", main_full, "lai_z"),
    ("all_points_FULL_STRICT_controls__LAI_x_temperature_FULL_STRICT", int_full, "lai_z:mat_z"),
]:
    boot = bootstrap_term(d, formula, term)
    lro = leave_region_out(d, formula, term)
    stability_rows.append({"test_label": label, "focal_term": term, **boot, **lro})

stability = pd.DataFrame(stability_rows)
full_models = full_models.merge(stability, on=["test_label", "focal_term"], how="left")
full_models.to_csv(TAB / "FULL_STRICT_CONTROL_LAI_MODELS.csv", index=False)

# ------------------------------------------------------------
# Product-specific and GLEAM-centered confirmation
# ------------------------------------------------------------

product_rows = []
if OBS_INPUT.exists() and "point_id" in d.columns:
    obs = pd.read_csv(OBS_INPUT, low_memory=False).replace([np.inf, -np.inf], np.nan)
    obs = obs.loc[:, ~obs.columns.duplicated()].copy()

    if "point_id" in obs.columns and "slope_change" in obs.columns:
        def run_alt(label, obs_sub):
            if len(obs_sub) < 100:
                return
            alt = obs_sub.groupby("point_id", dropna=False)["slope_change"].mean().reset_index()
            alt = alt.rename(columns={"slope_change": "y_alt"})
            merged = d.merge(alt, on="point_id", how="left")
            merged["y_alt_z"] = zscore(merged["y_alt"])

            alt_main_full = "y_alt_z ~ lai_z + " + " + ".join(base_controls)
            alt_main_reduced = "y_alt_z ~ " + " + ".join(base_controls)

            alt_int_full = "y_alt_z ~ lai_z * mat_z + " + " + ".join(base_controls_no_mat)
            alt_int_reduced = "y_alt_z ~ lai_z + mat_z + " + " + ".join(base_controls_no_mat)

            for scenario, mask in {
                "all": pd.Series(True, index=merged.index),
                "exclude_cropland_managed_irrigation_flags": ~merged["any_cropland_managed_irrigation_flag"],
                "exclude_sahel_broad": ~merged["sahel_broad"],
            }.items():
                sub = merged.loc[mask].copy()
                rows = compare(
                    sub,
                    f"{label}__{scenario}__ALT_PRODUCT_MEAN_LAI_main",
                    alt_main_full,
                    alt_main_reduced,
                    ["lai_z"],
                    note="Alt outcome: mean slope_change over selected phase8 product/model observations."
                )
                product_rows.extend(rows)

                rows = compare(
                    sub,
                    f"{label}__{scenario}__ALT_PRODUCT_MEAN_LAI_x_temperature",
                    alt_int_full,
                    alt_int_reduced,
                    ["lai_z:mat_z"],
                    note="Alt outcome: mean slope_change over selected phase8 product/model observations."
                )
                product_rows.extend(rows)

        run_alt("ALL_PRODUCT_COMBOS", obs)

        if "et_product" in obs.columns:
            et_values = sorted([str(v) for v in obs["et_product"].dropna().unique()])
            for et in et_values:
                safe = norm(et)
                run_alt(f"ET_PRODUCT_{safe}", obs[obs["et_product"].astype(str).eq(et)].copy())

            gleam_hits = obs[obs["et_product"].astype(str).str.contains("GLEAM", case=False, na=False)].copy()
            if len(gleam_hits):
                run_alt("ET_PRODUCT_GLEAM_CENTERED", gleam_hits)

            nongleam_hits = obs[~obs["et_product"].astype(str).str.contains("GLEAM", case=False, na=False)].copy()
            if len(nongleam_hits):
                run_alt("ET_PRODUCT_NON_GLEAM", nongleam_hits)

        if "gpp_product" in obs.columns:
            gpp_values = sorted([str(v) for v in obs["gpp_product"].dropna().unique()])
            for gp in gpp_values:
                safe = norm(gp)
                run_alt(f"GPP_PRODUCT_{safe}", obs[obs["gpp_product"].astype(str).eq(gp)].copy())

        for axis in ["metric", "stress_definition", "growing_season", "co2_version"]:
            if axis in obs.columns:
                vals = sorted([str(v) for v in obs[axis].dropna().unique()])
                for val in vals[:30]:
                    run_alt(f"{axis.upper()}_{norm(val)}", obs[obs[axis].astype(str).eq(val)].copy())

product_models = pd.DataFrame(product_rows)
if len(product_models):
    product_models["coef_sign"] = np.sign(product_models["coef"])
product_models.to_csv(TAB / "PRODUCT_SPECIFIC_AND_GLEAM_LAI_MODELS.csv", index=False)

product_summary_rows = []
if len(product_models):
    ok = product_models[product_models["status"].eq("FIT_OK")].copy()
    for term, g in ok.groupby("focal_term"):
        med = np.nanmedian(g["coef"])
        product_summary_rows.append({
            "focal_term": term,
            "n_tests": len(g),
            "median_coef": float(med),
            "min_coef": float(np.nanmin(g["coef"])),
            "max_coef": float(np.nanmax(g["coef"])),
            "sign_consistency": float(np.mean(np.sign(g["coef"]) == np.sign(med))) if med != 0 else np.nan,
            "n_p_lt_0p05": int((g["p"] < 0.05).sum()),
            "n_ci_excludes_zero": int(g["ci_excludes_zero"].fillna(False).sum()),
            "median_delta_r2": float(np.nanmedian(g["delta_r2"])),
            "median_delta_aic": float(np.nanmedian(g["delta_aic_full_minus_reduced"])),
        })

    # GLEAM-specific row.
    gleam = ok[ok["test_label"].str.contains("GLEAM", case=False, na=False)]
    for term, g in gleam.groupby("focal_term"):
        med = np.nanmedian(g["coef"])
        product_summary_rows.append({
            "focal_term": term,
            "n_tests": len(g),
            "median_coef": float(med),
            "min_coef": float(np.nanmin(g["coef"])),
            "max_coef": float(np.nanmax(g["coef"])),
            "sign_consistency": float(np.mean(np.sign(g["coef"]) == np.sign(med))) if med != 0 else np.nan,
            "n_p_lt_0p05": int((g["p"] < 0.05).sum()),
            "n_ci_excludes_zero": int(g["ci_excludes_zero"].fillna(False).sum()),
            "median_delta_r2": float(np.nanmedian(g["delta_r2"])),
            "median_delta_aic": float(np.nanmedian(g["delta_aic_full_minus_reduced"])),
            "subset_note": "GLEAM-specific tests only",
        })

pd.DataFrame(product_summary_rows).to_csv(TAB / "PRODUCT_SPECIFIC_AND_GLEAM_LAI_SUMMARY.csv", index=False)

# ------------------------------------------------------------
# VIF / collinearity diagnostics for full controls
# ------------------------------------------------------------

vif_rows = []
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_cols = ["lai_z"] + base_controls
    x = d[vif_cols].replace([np.inf, -np.inf], np.nan).dropna()
    x_const = sm.add_constant(x, has_constant="add")
    for i, col in enumerate(x_const.columns):
        if col == "const":
            continue
        vif_rows.append({
            "variable": col,
            "vif": float(variance_inflation_factor(x_const.values, i)),
            "n": len(x),
        })
except Exception as e:
    vif_rows.append({"variable": "ERROR", "vif": np.nan, "n": np.nan, "error": str(e)})

pd.DataFrame(vif_rows).to_csv(TAB / "FULL_STRICT_VIF_DIAGNOSTICS.csv", index=False)

# ------------------------------------------------------------
# Decision table
# ------------------------------------------------------------

def row_lookup(table, contains, term):
    if table is None or len(table) == 0:
        return None
    q = table[
        table["test_label"].astype(str).str.contains(contains, case=False, na=False)
        & table["focal_term"].astype(str).eq(term)
        & table["status"].astype(str).eq("FIT_OK")
    ]
    if len(q) == 0:
        return None
    return q.iloc[0].to_dict()

main_all = row_lookup(full_models, "all_points_FULL_STRICT_controls__LAI_main", "lai_z")
int_all = row_lookup(full_models, "all_points_FULL_STRICT_controls__LAI_x_temperature", "lai_z:mat_z")
main_clean = row_lookup(full_models, "exclude_cropland_managed_irrigation_flags__LAI_main", "lai_z")
int_clean = row_lookup(full_models, "exclude_cropland_managed_irrigation_flags__LAI_x_temperature", "lai_z:mat_z")
main_warm = row_lookup(full_models, "warm_only_mat_gt_2C__LAI_main", "lai_z")
int_warm = row_lookup(full_models, "warm_only_mat_gt_2C__LAI_x_temperature", "lai_z:mat_z")

landcover_inventory = pd.read_csv(TAB / "LANDCOVER_CROPLAND_FILE_INVENTORY.csv")
n_lc_files = int((landcover_inventory["n_landcover_like_cols"] > 0).sum()) if len(landcover_inventory) else 0
n_flagged = int(d["any_cropland_managed_irrigation_flag"].sum())
n_natural = int(d["any_natural_grassland_indicator"].sum())

tower_target_present = False
if (TAB / "TOWER_LANDCOVER_AND_US_NE_AUDIT.csv").exists():
    ta = pd.read_csv(TAB / "TOWER_LANDCOVER_AND_US_NE_AUDIT.csv")
    if len(ta):
        tower_target_present = bool(ta["target_US_Ne_present"].fillna(False).any())

def pass_p(row):
    return row is not None and pd.notna(row.get("p")) and row.get("p") < 0.05 and bool(row.get("ci_excludes_zero"))

decision = {
    "point_input": str(POINT_INPUT),
    "n_points": int(len(d)),
    "soil_texture_note": soil_texture_note,
    "growing_season_temperature_column_found": sources.get("gs_temp"),
    "full_control_set_used": base_controls,
    "landcover_files_with_landcover_columns_found": n_lc_files,
    "point_level_cropland_managed_irrigation_flagged_n": n_flagged,
    "point_level_natural_grassland_indicator_n": n_natural,
    "tower_US_Ne1_2_3_present_in_scanned_tower_files": tower_target_present,
    "all_points_LAI_full_control_pass": pass_p(main_all),
    "all_points_LAIxTemp_full_control_pass": pass_p(int_all),
    "clean_landcover_LAI_full_control_pass": pass_p(main_clean),
    "clean_landcover_LAIxTemp_full_control_pass": pass_p(int_clean),
    "warm_only_LAI_full_control_pass": pass_p(main_warm),
    "warm_only_LAIxTemp_full_control_pass": pass_p(int_warm),
    "project_style_verdict": None,
    "recommended_claim": None,
    "required_remaining_caveats": [],
}

if n_lc_files == 0:
    decision["required_remaining_caveats"].append("No usable land-cover/cropland audit columns were found; cropland leakage cannot be ruled out.")
if n_flagged > 0:
    decision["required_remaining_caveats"].append("Some points are flagged as possible cropland/irrigated/managed; clean-subset results must be used.")
if sources.get("gs_temp") is None:
    decision["required_remaining_caveats"].append("No explicit growing-season temperature column found; mean annual temperature used.")
if not decision["warm_only_LAI_full_control_pass"]:
    decision["required_remaining_caveats"].append("LAI main effect weakens or fails in warm-only domain; do not frame as universal warm-grassland physiology.")
if not decision["warm_only_LAIxTemp_full_control_pass"]:
    decision["required_remaining_caveats"].append("LAI×temperature is primarily a cross-regime/global result; do not claim it within warm-only grasslands.")
if tower_target_present:
    decision["required_remaining_caveats"].append("US-Ne1/2/3 appear in scanned tower files; tower land-cover filtering needs manual confirmation/exclusion.")

if decision["all_points_LAI_full_control_pass"] and decision["all_points_LAIxTemp_full_control_pass"]:
    decision["project_style_verdict"] = "CONDITIONAL_PASS_AS_REGIME_DEPENDENCE_RESULT_NOT_UNIVERSAL_TRAIT_CAUSALITY"
    decision["recommended_claim"] = (
        "Canopy structure, measured by growing-season LAI, is associated with latent grassland uWUE slope-change "
        "beyond baseline VPD and full hydroclimate/geography/soil-texture controls, but the result should be framed "
        "as a climate-regime-dependent association rather than a universal causal LAI physiology effect."
    )
else:
    decision["project_style_verdict"] = "FAIL_OR_INCOMPLETE_FOR_MAIN_CLAIM"
    decision["recommended_claim"] = (
        "Do not make LAI the central claim unless the full-control and artifact-screen models pass."
    )

(TAB / "project_STYLE_PASS_FAIL_DECISION.json").write_text(json.dumps(decision, indent=2))

# ------------------------------------------------------------
# Memo
# ------------------------------------------------------------

def show(path, n=60):
    p = TAB / path
    if not p.exists():
        return "MISSING"
    x = pd.read_csv(p)
    if len(x) == 0:
        return "EMPTY"
    return x.head(n).to_string(index=False)

memo = []
memo.append("Stage1B6BE full strict LAI artifact screen")
memo.append("=" * 90)
memo.append("")
memo.append(f"Point input: {POINT_INPUT}")
memo.append(f"Rows: {len(d)}")
memo.append("")
memo.append("Core question:")
memo.append("- Not whether VPD matters. VPD is treated as a known baseline covariate.")
memo.append("- This asks whether LAI/canopy structure or LAI×temperature survives a full artifact screen.")
memo.append("")
memo.append("Canonical source columns:")
for k, v in sources.items():
    memo.append(f"- {k}: {v}")
memo.append("")
memo.append(f"Soil texture handling: {soil_texture_note}")
memo.append(f"Growing-season temperature column found: {sources.get('gs_temp')}")
memo.append("")
memo.append("strict decision:")
memo.append(json.dumps(decision, indent=2))
memo.append("")
memo.append("Full-control LAI models:")
memo.append(show("FULL_STRICT_CONTROL_LAI_MODELS.csv", 80))
memo.append("")
memo.append("Product-specific and GLEAM summary:")
memo.append(show("PRODUCT_SPECIFIC_AND_GLEAM_LAI_SUMMARY.csv", 40))
memo.append("")
memo.append("Land-cover/cropland file inventory:")
memo.append(show("LANDCOVER_CROPLAND_FILE_INVENTORY.csv", 40))
memo.append("")
memo.append("Tower land-cover and US-Ne audit:")
memo.append(show("TOWER_LANDCOVER_AND_US_NE_AUDIT.csv", 40))
memo.append("")
memo.append("VIF diagnostics:")
memo.append(show("FULL_STRICT_VIF_DIAGNOSTICS.csv", 30))
memo.append("")
memo.append("Important files:")
for f in [
    "project_STYLE_PASS_FAIL_DECISION.json",
    "FULL_STRICT_CONTROL_LAI_MODELS.csv",
    "PRODUCT_SPECIFIC_AND_GLEAM_LAI_MODELS.csv",
    "PRODUCT_SPECIFIC_AND_GLEAM_LAI_SUMMARY.csv",
    "LANDCOVER_CROPLAND_FILE_INVENTORY.csv",
    "POINT_LEVEL_LANDCOVER_CROPLAND_FLAGS.csv",
    "EXACT_POINTS_FLAGGED_CROPLAND_MANAGED_IRRIGATION.csv",
    "TOWER_LANDCOVER_AND_US_NE_AUDIT.csv",
    "FULL_STRICT_VIF_DIAGNOSTICS.csv",
]:
    memo.append(f"- {TAB / f}")

(TXT / "READ_ME_FULL_STRICT_lai_artifact_screen.txt").write_text("\n".join(memo))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_FULL_STRICT_lai_artifact_screen.txt'}")
