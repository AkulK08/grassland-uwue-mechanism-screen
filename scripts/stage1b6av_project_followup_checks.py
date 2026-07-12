from pathlib import Path
from datetime import datetime
import json
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6av_project_followup_checks"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

BASE_TABLE = ROOT / "results/stage1b6ai_project_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

project_DECISION = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/STAGE1B6AS_FINAL_FULL_STRICT_RIGOR_DECISION.json"
SITE_STATUS = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dh_final_FULL_STRICT_site_status.csv"
STRICT_RANK = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03di_final_FULL_STRICT_strict_et_ranking.csv"
SENS_RANK = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dj_final_FULL_STRICT_sensitivity_et_ranking.csv"

TOWER_LC_CANDIDATES = [
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table101_tower_landcover_spatial_trait_annotation.csv",
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table108_satellite_extraction_targets_all_49_tower_response_sites.csv",
]

CROP_WORDS = [
    "crop", "cropland", "cultivated", "agriculture", "agricultural",
    "maize", "corn", "sorghum", "millet", "sugarcane", "sugar cane",
    "irrigated", "irrigation", "fertilized", "fertilizer"
]

GRASS_WORDS = [
    "grassland", "grasslands", "savanna", "savannas", "shrubland", "shrublands", "pasture", "rangeland"
]

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def safe_read(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return None

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def zscore(s):
    s = to_num(s)
    if s.notna().sum() < 10:
        return s * np.nan
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd

def contains_any(text, words):
    t = str(text).lower()
    return any(w in t for w in words)

def all_csvs():
    files = []
    for root in [ROOT / "results", ROOT / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            sp = str(p)
            if "_project_raw_exports" in sp:
                continue
            if "stage1b6av_project_followup_checks" in sp:
                continue
            if ".ipynb_checkpoints" in sp:
                continue
            if p.stat().st_size < 50:
                continue
            files.append(p)
    return sorted(set(files))

CONTROL_SPECS = {
    "aridity": [
        r"^aridity$", r"aridity_index", r"aridity_quantile", r"^ai$", r"^ai_", r"dryness"
    ],
    "growing_season_temperature": [
        r"growing.*season.*temp", r"gs.*temp", r"season.*temp", r"tair", r"tmean",
        r"temperature", r"temp_mean", r"mat"
    ],
    "precipitation": [
        r"growing.*season.*precip", r"gs.*precip", r"season.*precip", r"precip",
        r"ppt", r"rain", r"map"
    ],
    "soil_texture_clay": [
        r"clay", r"soil.*clay"
    ],
    "soil_texture_sand": [
        r"sand", r"soil.*sand"
    ],
    "soil_texture_silt": [
        r"silt", r"soil.*silt"
    ],
    "lai_or_productivity": [
        r"^lai$", r"lai_mean", r"lai_peak", r"leaf_area",
        r"productivity", r"gpp_mean", r"gpp_peak", r"mean_gpp", r"peak_gpp",
        r"ndvi", r"evi", r"gosif", r"sif"
    ],
    "baseline_vpd": [
        r"baseline.*vpd", r"vpd.*baseline", r"mean.*vpd", r"vpd_mean",
        r"vpd_clim", r"vpd0", r"prestress.*vpd"
    ],
    "baseline_soil_moisture": [
        r"baseline.*soil.*moisture", r"soil.*moisture.*baseline",
        r"baseline.*sm", r"sm.*baseline", r"soil_moisture_mean",
        r"sm_mean", r"rootzone.*sm", r"swc", r"smap", r"prestress.*soil"
    ],
    "rooting_depth": [
        r"rooting_depth", r"root.*depth"
    ],
}

TRAIT_SPECS = {
    "c4_fraction": [
        r"^c4_fraction_raw$", r"^c4_fraction$", r"c4.*fraction", r"^c4_"
    ]
}

RESPONSE_SPECS = {
    "uwue_latent_slope_change": [
        r"^latent_slope_change$", r"uwue.*latent.*slope.*change",
        r"uwue.*slope.*change", r"wue.*slope.*change"
    ],
}

BAD_RESPONSE_WORDS = [
    "uncertainty", "range", "posterior_sd", "_sd", "stderr", "std", "se_",
    "ci_", "q_", "p_", "agreement", "disagreement", "correlation", "rmse",
    "metric_uncertainty", "product_uncertainty"
]

def pick_col(cols, patterns, exclude_words=None):
    exclude_words = exclude_words or []
    scored = []
    for c in cols:
        lc = norm(c)
        if any(w in lc for w in exclude_words):
            continue
        score = 0
        for i, pat in enumerate(patterns):
            if re.search(pat, lc):
                score += 100 - i
        if score > 0:
            scored.append((score, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]

def pick_response_col(cols):
    for response, pats in RESPONSE_SPECS.items():
        c = pick_col(cols, pats, BAD_RESPONSE_WORDS)
        if c:
            return c
    return None

def common_merge_keys(left, right):
    key_sets = [
        ["point_id"],
        ["site_id"],
        ["site"],
        ["tower_site"],
        ["pixel_id"],
        ["lat", "lon"],
        ["latitude", "longitude"],
    ]
    left_norm = {norm(c): c for c in left.columns}
    right_norm = {norm(c): c for c in right.columns}
    for ks in key_sets:
        if all(k in left_norm and k in right_norm for k in ks):
            return [left_norm[k] for k in ks], [right_norm[k] for k in ks]
    return None, None

def find_control_in_repo(base_df, role):
    # First try base table.
    c = pick_col(base_df.columns, CONTROL_SPECS[role])
    if c is not None:
        return base_df, c, "base_table"

    # Then try mergeable CSVs.
    for p in all_csvs():
        d0 = safe_read(p, nrows=5)
        if d0 is None or len(d0.columns) < 2:
            continue
        c2 = pick_col(d0.columns, CONTROL_SPECS[role])
        if c2 is None:
            continue

        d = safe_read(p)
        if d is None or c2 not in d.columns:
            continue

        left_keys, right_keys = common_merge_keys(base_df, d)
        if left_keys is None:
            continue

        keep = right_keys + [c2]
        tmp = d[keep].copy()
        tmp = tmp.drop_duplicates(right_keys)
        new_col = f"{role}__merged"
        tmp = tmp.rename(columns={c2: new_col})
        merged = base_df.merge(tmp, left_on=left_keys, right_on=right_keys, how="left")
        if new_col in merged.columns and pd.to_numeric(merged[new_col], errors="coerce").notna().sum() >= 30:
            return merged, new_col, str(p)

    return base_df, None, "missing_not_found_or_not_mergeable"

def fit_model(df, response, c4_col, controls, label):
    cols = [response, c4_col] + controls
    use = df[cols].copy()
    use = use.replace([np.inf, -np.inf], np.nan)

    missing = []
    for c in cols:
        missing.append({
            "model_label": label,
            "column": c,
            "nonmissing_n": int(pd.to_numeric(use[c], errors="coerce").notna().sum()),
            "missing_n": int(pd.to_numeric(use[c], errors="coerce").isna().sum()),
        })

    use = use.dropna()
    if len(use) < max(40, len(cols) + 10):
        return None, pd.DataFrame(missing), None

    zuse = pd.DataFrame(index=use.index)
    for c in cols:
        zuse[c] = zscore(use[c])
    zuse = zuse.dropna()

    X = sm.add_constant(zuse[[c4_col] + controls])
    y = zuse[response]
    m = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for term in [c4_col] + controls:
        rows.append({
            "model_label": label,
            "response": response,
            "term": term,
            "n": int(m.nobs),
            "coef_standardized": float(m.params.get(term, np.nan)),
            "se_hc3": float(m.bse.get(term, np.nan)),
            "t": float(m.tvalues.get(term, np.nan)),
            "p": float(m.pvalues.get(term, np.nan)),
            "r2": float(m.rsquared),
            "aic": float(m.aic),
        })

    # VIF
    vif_rows = []
    xmat = zuse[[c4_col] + controls].dropna()
    if len(xmat) > len(xmat.columns) + 5:
        Xv = sm.add_constant(xmat)
        for i, col in enumerate(Xv.columns):
            if col == "const":
                continue
            try:
                vif = float(variance_inflation_factor(Xv.values, i))
            except Exception:
                vif = np.nan
            vif_rows.append({
                "model_label": label,
                "term": col,
                "vif": vif,
            })

    return pd.DataFrame(rows), pd.DataFrame(missing), pd.DataFrame(vif_rows)

def bootstrap_c4(df, response, c4_col, controls, B=1000, seed=19):
    rng = np.random.default_rng(seed)
    cols = [response, c4_col] + controls
    use = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(40, len(cols) + 10):
        return np.nan, np.nan

    zuse = pd.DataFrame(index=use.index)
    for c in cols:
        zuse[c] = zscore(use[c])
    zuse = zuse.dropna()
    n = len(zuse)

    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        b = zuse.iloc[idx]
        try:
            X = sm.add_constant(b[[c4_col] + controls])
            m = sm.OLS(b[response], X).fit()
            vals.append(float(m.params.get(c4_col, np.nan)))
        except Exception:
            pass

    vals = pd.Series(vals).dropna()
    if len(vals) < 100:
        return np.nan, np.nan
    return float(vals.quantile(0.025)), float(vals.quantile(0.975))

def crop_mask_check(df, label):
    rows = []

    candidate_cols = []
    for c in df.columns:
        lc = norm(c)
        if any(k in lc for k in ["landcover", "land_cover", "igbp", "lc_", "modis_lc", "crop", "irrig", "management", "plant", "cover", "biome"]):
            candidate_cols.append(c)

    if not candidate_cols:
        return pd.DataFrame([{
            "check_label": label,
            "status": "NO_LANDCOVER_OR_CROP_COLUMNS_FOUND_IN_THIS_TABLE",
            "column": "",
            "n": len(df),
            "crop_flag_n": np.nan,
            "crop_flag_rate": np.nan,
            "non_grass_flag_n": np.nan,
            "non_grass_flag_rate": np.nan,
            "notes": "Need external land-cover/crop-mask evidence if not available elsewhere."
        }])

    for c in candidate_cols:
        s = df[c]
        st = s.astype(str).str.lower()
        crop_flag = st.apply(lambda x: contains_any(x, CROP_WORDS))

        # IGBP numeric cropland classes: 12 = croplands, 14 = cropland/natural mosaic
        nums = pd.to_numeric(s, errors="coerce")
        crop_code = nums.isin([12, 14])
        crop_any = crop_flag | crop_code

        grass_flag = st.apply(lambda x: contains_any(x, GRASS_WORDS))
        # IGBP: 10 grasslands, 8 woody savannas, 9 savannas, 6/7 shrublands can be acceptable sensitivity.
        natural_codes = nums.isin([6, 7, 8, 9, 10])
        known = s.notna()
        non_grass = known & ~(grass_flag | natural_codes)
        # But if column is a crop/irrigation boolean field, non_grass is not meaningful.
        if any(k in norm(c) for k in ["crop", "irrig", "management"]):
            non_grass = pd.Series(False, index=df.index)

        rows.append({
            "check_label": label,
            "status": "CHECKED_COLUMN",
            "column": c,
            "n": int(known.sum()),
            "crop_flag_n": int(crop_any.sum()),
            "crop_flag_rate": float(crop_any.mean()) if len(crop_any) else np.nan,
            "non_grass_flag_n": int(non_grass.sum()),
            "non_grass_flag_rate": float(non_grass.mean()) if len(non_grass) else np.nan,
            "unique_values_preview": "; ".join(st.dropna().astype(str).value_counts().head(15).index.tolist()),
            "notes": "",
        })

    return pd.DataFrame(rows)

def summarize_n_definition(df, model_cols, response, c4_col):
    rows = []
    use = df[model_cols].replace([np.inf, -np.inf], np.nan)
    complete = use.dropna()
    rows.append({
        "item": "n_units_definition",
        "value": f"n is the number of point/pixel units with nonmissing {response}, {c4_col}, and all model control variables in the controlled C4 model."
    })
    rows.append({"item": "n_complete_units", "value": str(len(complete))})
    rows.append({"item": "response_variable", "value": response})
    rows.append({"item": "c4_variable", "value": c4_col})
    rows.append({"item": "model_columns", "value": ", ".join(model_cols)})
    for c in model_cols:
        rows.append({
            "item": f"nonmissing_{c}",
            "value": str(int(pd.to_numeric(df[c], errors="coerce").notna().sum()))
        })
    return pd.DataFrame(rows), complete

def main():
    if not BASE_TABLE.exists():
        raise FileNotFoundError(f"Missing base C4 table: {BASE_TABLE}")

    base = pd.read_csv(BASE_TABLE, low_memory=False)
    base_original_cols = list(base.columns)

    response_col = pick_response_col(base.columns)
    if response_col is None:
        raise RuntimeError("Could not find a real latent_slope_change / uWUE slope-change response in base table.")

    c4_col = pick_col(base.columns, TRAIT_SPECS["c4_fraction"])
    if c4_col is None:
        raise RuntimeError("Could not find C4 fraction column in base table.")

    work = base.copy()

    control_sources = []
    selected_controls = []

    wanted_roles = [
        "rooting_depth",
        "aridity",
        "growing_season_temperature",
        "precipitation",
        "soil_texture_clay",
        "soil_texture_sand",
        "lai_or_productivity",
        "baseline_vpd",
        "baseline_soil_moisture",
    ]

    for role in wanted_roles:
        work, col, source = find_control_in_repo(work, role)
        control_sources.append({
            "required_control": role,
            "selected_column": col if col else "",
            "source": source,
            "nonmissing_n": int(pd.to_numeric(work[col], errors="coerce").notna().sum()) if col else 0,
        })
        if col:
            selected_controls.append(col)

    control_sources_df = pd.DataFrame(control_sources)
    control_sources_df.to_csv(TAB / "Table_PRODUCT03dy_required_control_sources.csv", index=False)

    required_missing = control_sources_df.loc[control_sources_df["selected_column"].eq(""), "required_control"].tolist()

    # Benchmark: C4 + rooting_depth only, to reproduce previous controlled result.
    root_col = control_sources_df.loc[control_sources_df["required_control"].eq("rooting_depth"), "selected_column"].iloc[0]
    benchmark_controls = [root_col] if isinstance(root_col, str) and root_col else []

    benchmark_results, benchmark_missing, benchmark_vif = fit_model(
        work, response_col, c4_col, benchmark_controls, "benchmark_c4_plus_rooting_depth"
    )

    # Full available model: C4 + all found controls.
    full_results, full_missing, full_vif = fit_model(
        work, response_col, c4_col, selected_controls, "full_available_climate_environment_controls"
    )

    model_results = pd.concat([x for x in [benchmark_results, full_results] if x is not None], ignore_index=True) if any(x is not None for x in [benchmark_results, full_results]) else pd.DataFrame()
    missingness = pd.concat([x for x in [benchmark_missing, full_missing] if x is not None], ignore_index=True)
    vif = pd.concat([x for x in [benchmark_vif, full_vif] if x is not None], ignore_index=True) if any(x is not None for x in [benchmark_vif, full_vif]) else pd.DataFrame()

    if len(full_results):
        lo, hi = bootstrap_c4(work, response_col, c4_col, selected_controls)
        model_results.loc[
            (model_results["model_label"].eq("full_available_climate_environment_controls"))
            & (model_results["term"].eq(c4_col)),
            "bootstrap_ci_low"
        ] = lo
        model_results.loc[
            (model_results["model_label"].eq("full_available_climate_environment_controls"))
            & (model_results["term"].eq(c4_col)),
            "bootstrap_ci_high"
        ] = hi

    if len(benchmark_results):
        lo, hi = bootstrap_c4(work, response_col, c4_col, benchmark_controls)
        model_results.loc[
            (model_results["model_label"].eq("benchmark_c4_plus_rooting_depth"))
            & (model_results["term"].eq(c4_col)),
            "bootstrap_ci_low"
        ] = lo
        model_results.loc[
            (model_results["model_label"].eq("benchmark_c4_plus_rooting_depth"))
            & (model_results["term"].eq(c4_col)),
            "bootstrap_ci_high"
        ] = hi

    model_results.to_csv(TAB / "Table_PRODUCT03dz_full_controlled_c4_model_results.csv", index=False)
    missingness.to_csv(TAB / "Table_PRODUCT03ea_model_missingness_n_definition.csv", index=False)
    vif.to_csv(TAB / "Table_PRODUCT03eb_full_control_model_vif.csv", index=False)

    # n definition for full model if possible, otherwise benchmark.
    if selected_controls:
        ndef, complete_units = summarize_n_definition(work, [response_col, c4_col] + selected_controls, response_col, c4_col)
    else:
        ndef, complete_units = summarize_n_definition(work, [response_col, c4_col] + benchmark_controls, response_col, c4_col)

    ndef.to_csv(TAB / "Table_PRODUCT03ec_n112_unit_definition.csv", index=False)

    # Crop checks on model table and complete model units.
    crop_base = crop_mask_check(work, "satellite_point_model_table_all_rows")
    crop_complete = crop_mask_check(complete_units, "satellite_point_complete_model_units")
    crop_checks = pd.concat([crop_base, crop_complete], ignore_index=True)
    crop_checks.to_csv(TAB / "Table_PRODUCT03ed_cropland_c4_crop_mask_check.csv", index=False)

    # Tower land-cover check.
    tower_lc_tables = []
    for p in TOWER_LC_CANDIDATES:
        if p.exists():
            d = pd.read_csv(p, low_memory=False)
            chk = crop_mask_check(d, f"tower_landcover_table__{p.name}")
            chk["source_path"] = str(p)
            tower_lc_tables.append(chk)

    if tower_lc_tables:
        tower_lc = pd.concat(tower_lc_tables, ignore_index=True)
    else:
        tower_lc = pd.DataFrame([{
            "check_label": "tower_landcover_check",
            "status": "NO_TOWER_LANDCOVER_TABLE_FOUND",
            "notes": "Expected Table101/Table108 under results/tower_grassland_spatial_trait_lock/tables."
        }])
    tower_lc.to_csv(TAB / "Table_PRODUCT03ee_tower_site_landcover_check.csv", index=False)

    # Tower final agreement bundle.
    site = pd.read_csv(SITE_STATUS) if SITE_STATUS.exists() else pd.DataFrame()
    strict = pd.read_csv(STRICT_RANK) if STRICT_RANK.exists() else pd.DataFrame()
    sens = pd.read_csv(SENS_RANK) if SENS_RANK.exists() else pd.DataFrame()

    if len(site):
        site.to_csv(TAB / "Table_PRODUCT03ef_final_tower_site_status_for_meeting.csv", index=False)
    if len(strict):
        strict.to_csv(TAB / "Table_PRODUCT03eg_final_strict_tower_et_ranking_for_meeting.csv", index=False)
    if len(sens):
        sens.to_csv(TAB / "Table_PRODUCT03eh_final_sensitivity_tower_et_ranking_for_meeting.csv", index=False)

    # Decision summary
    c4_full = pd.DataFrame()
    if len(model_results):
        c4_full = model_results[
            model_results["model_label"].eq("full_available_climate_environment_controls")
            & model_results["term"].eq(c4_col)
        ].copy()

    c4_bench = pd.DataFrame()
    if len(model_results):
        c4_bench = model_results[
            model_results["model_label"].eq("benchmark_c4_plus_rooting_depth")
            & model_results["term"].eq(c4_col)
        ].copy()

    full_control_complete = len(required_missing) == 0
    c4_holds_full_available = False
    if len(c4_full):
        r = c4_full.iloc[0]
        c4_holds_full_available = bool(
            pd.notna(r.get("p"))
            and r.get("p") <= 0.05
            and pd.notna(r.get("bootstrap_ci_low"))
            and pd.notna(r.get("bootstrap_ci_high"))
            and ((r.get("bootstrap_ci_low") > 0) or (r.get("bootstrap_ci_high") < 0))
        )

    decision = {
        "generated": now(),
        "stage": "1B.6AV_project_followup_checks",
        "response_variable": response_col,
        "c4_variable": c4_col,
        "required_controls_requested_by_project": wanted_roles,
        "controls_found_and_used": control_sources_df.to_dict(orient="records"),
        "missing_requested_controls": required_missing,
        "full_requested_control_set_complete": bool(full_control_complete),
        "c4_holds_in_full_available_control_model": bool(c4_holds_full_available),
        "benchmark_c4_plus_rooting_depth_result": c4_bench.to_dict(orient="records"),
        "full_available_control_result_for_c4": c4_full.to_dict(orient="records"),
        "crop_mask_check_status": crop_checks[["check_label", "status", "column", "crop_flag_n", "crop_flag_rate", "non_grass_flag_n", "non_grass_flag_rate"]].to_dict(orient="records") if len(crop_checks) else [],
        "tower_landcover_check_status": tower_lc.to_dict(orient="records") if len(tower_lc) else [],
        "tower_language_recommended": "Towers provide a limited independent anchor and suggest GLEAM is the better ET product for this specific analysis; do not say towers strongly validate GLEAM.",
    }

    (TAB / "STAGE1B6AV_project_FOLLOWUP_CHECKS_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Meeting note
    lines = []
    lines.append("# project meeting note: follow-up checks")
    lines.append("")
    lines.append("## 1. Full controlled C4 model")
    lines.append("")
    if full_control_complete:
        lines.append("All requested climate/environment controls were found and included: aridity, growing-season temperature, precipitation, soil texture, LAI/productivity, baseline VPD, baseline soil moisture, and rooting depth.")
    else:
        lines.append("Not all requested climate/environment controls were found in mergeable point-level tables.")
        lines.append(f"Missing controls: {', '.join(required_missing) if required_missing else 'none'}")
        lines.append("The current full-available model should therefore be treated as a check, not as the final full-control model project requested.")
    lines.append("")
    lines.append("Selected controls:")
    lines.append(control_sources_df.to_string(index=False))
    lines.append("")
    lines.append("Model results:")
    lines.append(model_results.to_string(index=False) if len(model_results) else "No model results produced.")
    lines.append("")
    lines.append("VIF / collinearity check:")
    lines.append(vif.to_string(index=False) if len(vif) else "No VIF table produced.")
    lines.append("")

    lines.append("## 2. Definition of n units")
    lines.append("")
    lines.append(ndef.to_string(index=False))
    lines.append("")

    lines.append("## 3. Cropland / C4 crop masking check")
    lines.append("")
    lines.append("Crop keywords checked: maize/corn, sorghum, millet, sugarcane, cropland/cultivated/agriculture, irrigation/fertilization/management.")
    lines.append(crop_checks.to_string(index=False) if len(crop_checks) else "No crop check produced.")
    lines.append("")

    lines.append("## 4. Tower-site land-cover check")
    lines.append("")
    lines.append(tower_lc.to_string(index=False) if len(tower_lc) else "No tower land-cover check produced.")
    lines.append("")

    lines.append("## 5. Final tower agreement table")
    lines.append("")
    lines.append("Recommended language: towers provide a limited independent anchor and suggest GLEAM is the better ET product for this specific analysis; do not say towers strongly validate GLEAM.")
    lines.append("")
    lines.append("Strict ET ranking:")
    lines.append(strict.to_string(index=False) if len(strict) else "Strict ranking not found.")
    lines.append("")
    lines.append("Sensitivity ET ranking:")
    lines.append(sens.to_string(index=False) if len(sens) else "Sensitivity ranking not found.")
    lines.append("")
    lines.append("Site status:")
    lines.append(site.to_string(index=False) if len(site) else "Site status not found.")
    lines.append("")

    meeting_note = "\n".join(lines)
    (TXT / "project_MEETING_NOTE_FOLLOWUP_CHECKS.md").write_text(meeting_note, encoding="utf-8")

    # Short email/reply draft if useful
    reply = []
    reply.append("Hi project,")
    reply.append("")
    reply.append("Thanks — that sounds good. I’ll bring a short table/note to the meeting covering exactly those checks: the full controlled C4 model, the definition of the n = 112 units, the cropland/C4 crop masking check, the tower-site land-cover check, and the final tower agreement table.")
    reply.append("")
    reply.append("I’ll also keep the tower language conservative: a limited independent anchor suggesting GLEAM is the better ET product for this analysis, not a strong validation of GLEAM.")
    reply.append("")
    reply.append("See you at 1:30.")
    reply.append("")
    reply.append("Best,")
    reply.append("Akul")
    (TXT / "project_SHORT_REPLY.md").write_text("\n".join(reply), encoding="utf-8")

    print("")
    print("===== DECISION JSON =====")
    print(json.dumps(decision, indent=2))
    print("")
    print("===== MEETING NOTE =====")
    print(meeting_note)
    print("")
    print("===== SHORT REPLY =====")
    print("\n".join(reply))
    print("")
    print("WROTE", TAB / "STAGE1B6AV_project_FOLLOWUP_CHECKS_DECISION.json")
    print("WROTE", TAB / "Table_PRODUCT03dy_required_control_sources.csv")
    print("WROTE", TAB / "Table_PRODUCT03dz_full_controlled_c4_model_results.csv")
    print("WROTE", TAB / "Table_PRODUCT03eb_full_control_model_vif.csv")
    print("WROTE", TAB / "Table_PRODUCT03ec_n112_unit_definition.csv")
    print("WROTE", TAB / "Table_PRODUCT03ed_cropland_c4_crop_mask_check.csv")
    print("WROTE", TAB / "Table_PRODUCT03ee_tower_site_landcover_check.csv")
    print("WROTE", TXT / "project_MEETING_NOTE_FOLLOWUP_CHECKS.md")
    print("WROTE", TXT / "project_SHORT_REPLY.md")

if __name__ == "__main__":
    main()
