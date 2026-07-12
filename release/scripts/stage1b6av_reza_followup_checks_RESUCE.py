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
OUT = ROOT / "results/stage1b6av_reza_followup_checks"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

BASE_TABLE = ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"
SITE_STATUS = ROOT / "results/stage1b6as_final_full_reza_rigor/tables/Table_PRODUCT03dh_final_full_reza_site_status.csv"
STRICT_RANK = ROOT / "results/stage1b6as_final_full_reza_rigor/tables/Table_PRODUCT03di_final_full_reza_strict_et_ranking.csv"
SENS_RANK = ROOT / "results/stage1b6as_final_full_reza_rigor/tables/Table_PRODUCT03dj_final_full_reza_sensitivity_et_ranking.csv"

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
    "grassland", "grasslands", "savanna", "savannas",
    "shrubland", "shrublands", "pasture", "rangeland"
]

CONTROL_SPECS = {
    "rooting_depth": [r"rooting_depth", r"root.*depth"],
    "aridity": [r"^aridity$", r"aridity_index", r"aridity_quantile", r"^ai$", r"dryness"],
    "growing_season_temperature": [r"growing.*season.*temp", r"gs.*temp", r"season.*temp", r"temperature", r"tmean", r"tair", r"mat"],
    "precipitation": [r"growing.*season.*precip", r"gs.*precip", r"season.*precip", r"precip", r"ppt", r"rain", r"map"],
    "soil_texture_clay": [r"clay", r"soil.*clay"],
    "soil_texture_sand": [r"sand", r"soil.*sand"],
    "soil_texture_silt": [r"silt", r"soil.*silt"],
    "lai_or_productivity": [r"^lai$", r"lai_mean", r"lai_peak", r"leaf_area", r"productivity", r"gpp_mean", r"gpp_peak", r"mean_gpp", r"peak_gpp", r"ndvi", r"evi", r"sif", r"gosif"],
    "baseline_vpd": [r"baseline.*vpd", r"vpd.*baseline", r"mean.*vpd", r"vpd_mean", r"vpd_clim", r"prestress.*vpd"],
    "baseline_soil_moisture": [r"baseline.*soil.*moisture", r"soil.*moisture.*baseline", r"baseline.*sm", r"sm.*baseline", r"soil_moisture_mean", r"sm_mean", r"swc", r"smap", r"rootzone.*sm", r"prestress.*soil"],
}

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

def pick_col(cols, patterns):
    scored = []
    for c in cols:
        lc = norm(c)
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

def all_csvs():
    out = []
    for root in [ROOT / "results", ROOT / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            sp = str(p)
            if "_reza_raw_exports" in sp:
                continue
            if "stage1b6av_reza_followup_checks" in sp:
                continue
            if ".ipynb_checkpoints" in sp:
                continue
            if p.stat().st_size < 100:
                continue
            out.append(p)
    return sorted(set(out))

def common_keys(left, right):
    pairs = [
        (["point_id"], ["point_id"]),
        (["site_id"], ["site_id"]),
        (["site"], ["site"]),
        (["tower_site"], ["tower_site"]),
        (["pixel_id"], ["pixel_id"]),
        (["lat", "lon"], ["lat", "lon"]),
        (["latitude", "longitude"], ["latitude", "longitude"]),
    ]
    ln = {norm(c): c for c in left.columns}
    rn = {norm(c): c for c in right.columns}
    for lk, rk in pairs:
        if all(k in ln for k in lk) and all(k in rn for k in rk):
            return [ln[k] for k in lk], [rn[k] for k in rk]
    return None, None

def find_or_merge_control(base, role):
    c = pick_col(base.columns, CONTROL_SPECS[role])
    if c:
        return base, c, "base_table", int(to_num(base[c]).notna().sum())

    for p in all_csvs():
        head = safe_read(p, nrows=5)
        if head is None:
            continue
        c2 = pick_col(head.columns, CONTROL_SPECS[role])
        if not c2:
            continue

        d = safe_read(p)
        if d is None or c2 not in d.columns:
            continue

        lk, rk = common_keys(base, d)
        if lk is None:
            continue

        new_col = role + "__merged"
        keep = rk + [c2]
        tmp = d[keep].drop_duplicates(rk).rename(columns={c2: new_col})
        merged = base.merge(tmp, left_on=lk, right_on=rk, how="left")
        nonmiss = int(to_num(merged[new_col]).notna().sum())
        if nonmiss >= 30:
            return merged, new_col, str(p), nonmiss

    return base, "", "missing_not_found_or_not_mergeable", 0

def fit_ols(df, response, c4_col, controls, label, min_n=40):
    cols = [response, c4_col] + controls
    rows_missing = []
    for c in cols:
        rows_missing.append({
            "model_label": label,
            "column": c,
            "nonmissing_n": int(to_num(df[c]).notna().sum()) if c in df.columns else 0,
        })

    use = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    n_complete = len(use)

    if n_complete < min_n:
        return pd.DataFrame([{
            "model_label": label,
            "response": response,
            "term": c4_col,
            "n": n_complete,
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "fit_status": f"NOT_FIT_TOO_FEW_COMPLETE_CASES_MIN_{min_n}",
            "controls": ", ".join(controls),
        }]), pd.DataFrame(rows_missing), pd.DataFrame()

    zuse = pd.DataFrame(index=use.index)
    for c in cols:
        zuse[c] = zscore(use[c])
    zuse = zuse.dropna()

    if len(zuse) < min_n:
        return pd.DataFrame([{
            "model_label": label,
            "response": response,
            "term": c4_col,
            "n": len(zuse),
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "fit_status": "NOT_FIT_AFTER_ZSCORE_TOO_FEW_COMPLETE_CASES",
            "controls": ", ".join(controls),
        }]), pd.DataFrame(rows_missing), pd.DataFrame()

    X = sm.add_constant(zuse[[c4_col] + controls], has_constant="add")
    y = zuse[response]

    try:
        m = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as e:
        return pd.DataFrame([{
            "model_label": label,
            "response": response,
            "term": c4_col,
            "n": len(zuse),
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "fit_status": "FIT_ERROR_" + repr(e),
            "controls": ", ".join(controls),
        }]), pd.DataFrame(rows_missing), pd.DataFrame()

    result_rows = []
    for term in [c4_col] + controls:
        result_rows.append({
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
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "fit_status": "FIT_OK",
            "controls": ", ".join(controls),
        })

    # Bootstrap CI only for C4 coefficient.
    rng = np.random.default_rng(23)
    vals = []
    n = len(zuse)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        b = zuse.iloc[idx]
        try:
            Xb = sm.add_constant(b[[c4_col] + controls], has_constant="add")
            mb = sm.OLS(b[response], Xb).fit()
            vals.append(float(mb.params.get(c4_col, np.nan)))
        except Exception:
            pass
    vals = pd.Series(vals).dropna()
    if len(vals) >= 100:
        lo, hi = float(vals.quantile(0.025)), float(vals.quantile(0.975))
        for r in result_rows:
            if r["term"] == c4_col:
                r["bootstrap_ci_low"] = lo
                r["bootstrap_ci_high"] = hi

    vif_rows = []
    xmat = zuse[[c4_col] + controls].dropna()
    if len(xmat) > len(xmat.columns) + 5:
        Xv = sm.add_constant(xmat, has_constant="add")
        for i, col in enumerate(Xv.columns):
            if col == "const":
                continue
            try:
                vif = float(variance_inflation_factor(Xv.values, i))
            except Exception:
                vif = np.nan
            vif_rows.append({"model_label": label, "term": col, "vif": vif})

    return pd.DataFrame(result_rows), pd.DataFrame(rows_missing), pd.DataFrame(vif_rows)

def crop_check(df, label):
    rows = []
    lc_cols = []
    for c in df.columns:
        lc = norm(c)
        if any(k in lc for k in ["landcover", "land_cover", "igbp", "lc_", "crop", "irrig", "management", "cover", "biome", "plant"]):
            lc_cols.append(c)

    if not lc_cols:
        return pd.DataFrame([{
            "check_label": label,
            "status": "NO_LANDCOVER_OR_CROP_COLUMNS_FOUND",
            "column": "",
            "n": len(df),
            "crop_flag_n": np.nan,
            "crop_flag_rate": np.nan,
            "non_grass_flag_n": np.nan,
            "non_grass_flag_rate": np.nan,
            "unique_values_preview": "",
        }])

    for c in lc_cols:
        s = df[c]
        st = s.map(lambda v: "" if pd.isna(v) else str(v).lower())
        nums = pd.to_numeric(s, errors="coerce")

        crop_keyword = st.apply(lambda x: any(w in x for w in CROP_WORDS))
        crop_code = nums.isin([12, 14])
        crop_any = crop_keyword | crop_code

        grass_keyword = st.apply(lambda x: any(w in x for w in GRASS_WORDS))
        natural_code = nums.isin([6, 7, 8, 9, 10])
        known = s.notna()

        non_grass = known & ~(grass_keyword | natural_code)
        if any(k in norm(c) for k in ["crop", "irrig", "management"]):
            non_grass = pd.Series(False, index=df.index)

        rows.append({
            "check_label": label,
            "status": "CHECKED_COLUMN",
            "column": c,
            "n": int(known.sum()),
            "crop_flag_n": int(crop_any.sum()),
            "crop_flag_rate": float(crop_any.mean()),
            "non_grass_flag_n": int(non_grass.sum()),
            "non_grass_flag_rate": float(non_grass.mean()),
            "unique_values_preview": "; ".join(st.value_counts().head(12).index.astype(str).tolist()),
        })

    return pd.DataFrame(rows)

def main():
    if not BASE_TABLE.exists():
        raise FileNotFoundError(BASE_TABLE)

    df = pd.read_csv(BASE_TABLE, low_memory=False)

    c4_col = pick_col(df.columns, [r"^c4_fraction_raw$", r"^c4_fraction$", r"c4.*fraction", r"^c4_"])
    response = pick_col(df.columns, [r"^latent_slope_change$", r"uwue.*latent.*slope.*change", r"uwue.*slope.*change", r"wue.*slope.*change"])

    if not c4_col or not response:
        raise RuntimeError(f"Could not find required columns. c4={c4_col}, response={response}")

    roles = [
        "rooting_depth",
        "aridity",
        "growing_season_temperature",
        "precipitation",
        "soil_texture_clay",
        "soil_texture_sand",
        "soil_texture_silt",
        "lai_or_productivity",
        "baseline_vpd",
        "baseline_soil_moisture",
    ]

    work = df.copy()
    control_rows = []
    selected = {}

    for role in roles:
        work, col, source, nonmiss = find_or_merge_control(work, role)
        selected[role] = col
        control_rows.append({
            "requested_control": role,
            "selected_column": col,
            "source": source,
            "nonmissing_n": nonmiss,
        })

    controls_df = pd.DataFrame(control_rows)
    controls_df.to_csv(TAB / "Table_PRODUCT03dy_required_control_sources_RESCUE.csv", index=False)

    root_col = selected.get("rooting_depth", "")
    all_found_controls = [c for c in selected.values() if isinstance(c, str) and c]

    model_tables = []
    missing_tables = []
    vif_tables = []

    # Benchmark model Reza already responded to.
    if root_col:
        r, m, v = fit_ols(work, response, c4_col, [root_col], "benchmark_c4_plus_rooting_depth")
        model_tables.append(r); missing_tables.append(m); vif_tables.append(v)

    # Full available complete-case model.
    if all_found_controls:
        r, m, v = fit_ols(work, response, c4_col, all_found_controls, "full_available_controls_complete_case")
        model_tables.append(r); missing_tables.append(m); vif_tables.append(v)

    # Incremental one-control checks: C4 + rooting depth + each extra control.
    for role, col in selected.items():
        if not col or role == "rooting_depth":
            continue
        controls = [root_col, col] if root_col else [col]
        r, m, v = fit_ols(work, response, c4_col, controls, f"incremental_c4_plus_rooting_depth_plus_{role}")
        model_tables.append(r); missing_tables.append(m); vif_tables.append(v)

    models = pd.concat(model_tables, ignore_index=True) if model_tables else pd.DataFrame()
    missing = pd.concat(missing_tables, ignore_index=True) if missing_tables else pd.DataFrame()
    vif = pd.concat([x for x in vif_tables if len(x)], ignore_index=True) if any(len(x) for x in vif_tables) else pd.DataFrame()

    models.to_csv(TAB / "Table_PRODUCT03dz_full_controlled_c4_model_results_RESCUE.csv", index=False)
    missing.to_csv(TAB / "Table_PRODUCT03ea_model_missingness_n_definition_RESCUE.csv", index=False)
    vif.to_csv(TAB / "Table_PRODUCT03eb_full_control_model_vif_RESCUE.csv", index=False)

    # n definition for benchmark and full.
    n_rows = []
    for label, controls in [
        ("benchmark_c4_plus_rooting_depth", [root_col] if root_col else []),
        ("full_available_controls_complete_case", all_found_controls),
    ]:
        cols = [response, c4_col] + controls
        cols = [c for c in cols if c]
        n_complete = int(work[cols].replace([np.inf, -np.inf], np.nan).dropna().shape[0])
        n_rows.append({
            "model_label": label,
            "n_definition": "number of point/pixel units with nonmissing response, C4 fraction, and all listed controls",
            "n_complete": n_complete,
            "columns_required": ", ".join(cols),
        })
    ndef = pd.DataFrame(n_rows)
    ndef.to_csv(TAB / "Table_PRODUCT03ec_n_unit_definition_RESCUE.csv", index=False)

    crop_base = crop_check(work, "satellite_point_model_table_all_rows")
    complete_cols = [response, c4_col] + ([root_col] if root_col else [])
    complete = work[complete_cols].dropna()
    crop_complete = crop_check(work.loc[complete.index], "satellite_point_benchmark_complete_units")
    crop = pd.concat([crop_base, crop_complete], ignore_index=True)
    crop.to_csv(TAB / "Table_PRODUCT03ed_cropland_c4_crop_mask_check_RESCUE.csv", index=False)

    tower_checks = []
    for p in TOWER_LC_CANDIDATES:
        if p.exists():
            d = pd.read_csv(p, low_memory=False)
            chk = crop_check(d, "tower_landcover_" + p.name)
            chk["source_path"] = str(p)
            tower_checks.append(chk)
    tower_lc = pd.concat(tower_checks, ignore_index=True) if tower_checks else pd.DataFrame([{
        "check_label": "tower_landcover_check",
        "status": "NO_TOWER_LANDCOVER_TABLE_FOUND",
    }])
    tower_lc.to_csv(TAB / "Table_PRODUCT03ee_tower_site_landcover_check_RESCUE.csv", index=False)

    site = pd.read_csv(SITE_STATUS) if SITE_STATUS.exists() else pd.DataFrame()
    strict = pd.read_csv(STRICT_RANK) if STRICT_RANK.exists() else pd.DataFrame()
    sens = pd.read_csv(SENS_RANK) if SENS_RANK.exists() else pd.DataFrame()

    missing_requested = controls_df.loc[controls_df["selected_column"].eq(""), "requested_control"].tolist()

    c4_rows = models[(models["term"].eq(c4_col)) & (models["fit_status"].eq("FIT_OK"))].copy() if len(models) else pd.DataFrame()
    if len(c4_rows):
        c4_rows["abs_coef"] = c4_rows["coef_standardized"].abs()
        best_c4 = c4_rows.sort_values(["p", "abs_coef"], ascending=[True, False]).iloc[0].to_dict()
    else:
        best_c4 = {}

    full_row = models[(models["model_label"].eq("full_available_controls_complete_case")) & (models["term"].eq(c4_col))].copy() if len(models) else pd.DataFrame()
    full_fit_ok = bool(len(full_row) and full_row.iloc[0].get("fit_status") == "FIT_OK")

    decision = {
        "generated": now(),
        "stage": "1B.6AV_RESCUE_reza_followup_checks",
        "response_variable": response,
        "c4_variable": c4_col,
        "controls_found": controls_df.to_dict(orient="records"),
        "missing_requested_controls": missing_requested,
        "full_available_complete_case_model_fit_ok": full_fit_ok,
        "best_fit_c4_model": best_c4,
        "n_definitions": ndef.to_dict(orient="records"),
        "crop_mask_summary": crop[["check_label", "status", "column", "crop_flag_n", "crop_flag_rate", "non_grass_flag_n", "non_grass_flag_rate"]].to_dict(orient="records") if len(crop) else [],
        "tower_landcover_summary": tower_lc.to_dict(orient="records") if len(tower_lc) else [],
        "tower_language": "Use: limited independent anchor suggesting GLEAM is better for this analysis. Avoid: towers strongly validate GLEAM.",
    }
    (TAB / "STAGE1B6AV_RESCUE_REZA_FOLLOWUP_CHECKS_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    note = []
    note.append("# Reza meeting note: follow-up checks")
    note.append("")
    note.append("## 1. Full controlled C4 model results")
    note.append("")
    note.append("Response variable: " + response)
    note.append("C4 variable: " + c4_col)
    note.append("")
    note.append("Controls found / missing:")
    note.append(controls_df.to_string(index=False))
    note.append("")
    note.append("Model results:")
    note.append(models.to_string(index=False) if len(models) else "No models fit.")
    note.append("")
    note.append("VIF / collinearity:")
    note.append(vif.to_string(index=False) if len(vif) else "No VIF table produced.")
    note.append("")
    note.append("Interpretation note:")
    if full_fit_ok:
        note.append("The full available complete-case model fit successfully.")
    else:
        note.append("The full available complete-case model did not fit, usually because adding all controls leaves too few complete rows. Use the benchmark and incremental-control rows to discuss what has and has not been verified.")
    note.append("")
    note.append("## 2. Definition of n units")
    note.append(ndef.to_string(index=False))
    note.append("")
    note.append("## 3. Cropland / C4 crop masking check")
    note.append("Crop keywords/codes checked: cropland/cultivated/agriculture, maize/corn, sorghum, millet, sugarcane, irrigation/fertilization/management, IGBP cropland codes 12/14.")
    note.append(crop.to_string(index=False))
    note.append("")
    note.append("## 4. Tower-site land-cover check")
    note.append(tower_lc.to_string(index=False))
    note.append("")
    note.append("## 5. Final tower agreement table")
    note.append("Use conservative language: towers provide a limited independent anchor and suggest GLEAM is the better ET product for this specific analysis; do not say towers strongly validate GLEAM.")
    note.append("")
    note.append("Strict ET ranking:")
    note.append(strict.to_string(index=False) if len(strict) else "Strict ranking not found.")
    note.append("")
    note.append("Sensitivity ET ranking:")
    note.append(sens.to_string(index=False) if len(sens) else "Sensitivity ranking not found.")
    note.append("")
    note.append("Site status:")
    note.append(site.to_string(index=False) if len(site) else "Site status not found.")

    note_text = "\n".join(note)
    (TXT / "REZA_MEETING_NOTE_FOLLOWUP_CHECKS_RESCUE.md").write_text(note_text, encoding="utf-8")

    reply = """Hi Reza,

Thanks — that sounds good. I will bring a short table/note to the meeting covering the full controlled C4 model, the definition of the n = 112 units, the cropland/C4 crop masking check, the tower-site land-cover check, and the final tower agreement table.

I will also keep the tower language conservative: a limited independent anchor suggesting GLEAM is the better ET product for this analysis, not a strong validation of GLEAM.

See you at 1:30.

Best,
Akul
"""
    (TXT / "REZA_SHORT_REPLY_RESCUE.md").write_text(reply, encoding="utf-8")

    print("===== DECISION =====")
    print(json.dumps(decision, indent=2))
    print("")
    print("===== MEETING NOTE =====")
    print(note_text)
    print("")
    print("===== SHORT REPLY =====")
    print(reply)

if __name__ == "__main__":
    main()
