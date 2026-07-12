from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bb_spatial_full_control_screen"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "results/stage1b6ay_final_reza_audit/tables/Table_PRODUCT03fg_final_audit_dataset.csv"
PRIOR_C4_PATH = ROOT / "results/stage1b6ay_final_reza_audit/tables/Table_PRODUCT03fk_final_c4_rows_only.csv"
CROP_PATH = ROOT / "results/stage1b6ay_final_reza_audit/tables/Table_PRODUCT03fm_final_crop_summary.csv"
TOWER_PATH = ROOT / "results/stage1b6ay_final_reza_audit/tables/Table_PRODUCT03fd_target_tower_landcover_summary.csv"
TOWER_STRICT_PATH = ROOT / "results/stage1b6as_final_full_reza_rigor/tables/Table_PRODUCT03di_final_full_reza_strict_et_ranking.csv"
TOWER_SENS_PATH = ROOT / "results/stage1b6as_final_full_reza_rigor/tables/Table_PRODUCT03dj_final_full_reza_sensitivity_et_ranking.csv"

RESPONSE = "latent_slope_change"
C4 = "c4_fraction_raw"

MIN_N_EXPLORATORY = 35
MIN_N_STRICT = 50
BOOT_B = 1000
SEED = 123

def find_col(df, candidates):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        pat = cand.lower()
        hits = [c for c in cols if pat in c.lower()]
        if hits:
            return hits[0]
    return None

def z(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if x.notna().sum() < 10 or pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd

def fit_c4_full(df, label, group_type, group_value, controls, crop_flag_col=None):
    cols = [RESPONSE, C4] + controls
    missing = [c for c in cols if c not in df.columns]
    out_base = {
        "group_type": group_type,
        "group_value": str(group_value),
        "model_label": label,
        "n_raw_rows": len(df),
        "n_complete": 0,
        "crop_flag_n": np.nan,
        "crop_flag_rate": np.nan,
        "coef_c4_standardized": np.nan,
        "se_hc3": np.nan,
        "t": np.nan,
        "p": np.nan,
        "r2": np.nan,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "ci_excludes_zero": False,
        "nominal_pass_p05_ci": False,
        "strict_n_ge_50": False,
        "fit_status": "",
        "controls": ", ".join(controls),
    }

    if crop_flag_col and crop_flag_col in df.columns:
        crop_vals = df[crop_flag_col].astype(str).str.lower().isin(["true", "1", "yes", "crop", "cro"])
        out_base["crop_flag_n"] = int(crop_vals.sum())
        out_base["crop_flag_rate"] = float(crop_vals.mean()) if len(crop_vals) else np.nan

    if missing:
        out_base["fit_status"] = "MISSING_COLUMNS: " + ", ".join(missing)
        return out_base

    use = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    out_base["n_complete"] = len(use)

    if len(use) < MIN_N_EXPLORATORY:
        out_base["fit_status"] = f"NOT_FIT_TOO_FEW_COMPLETE_ROWS_LT_{MIN_N_EXPLORATORY}"
        return out_base

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()
    out_base["n_complete"] = len(zz)

    if len(zz) < MIN_N_EXPLORATORY:
        out_base["fit_status"] = f"NOT_FIT_TOO_FEW_Z_ROWS_LT_{MIN_N_EXPLORATORY}"
        return out_base

    xvars = [C4] + controls
    X = sm.add_constant(zz[xvars], has_constant="add")
    try:
        model = sm.OLS(zz[RESPONSE], X).fit(cov_type="HC3")
        beta = float(model.params[C4])
        p = float(model.pvalues[C4])
        se = float(model.bse[C4])
        t = float(model.tvalues[C4])
        r2 = float(model.rsquared)

        rng = np.random.default_rng(SEED)
        boots = []
        n = len(zz)
        for _ in range(BOOT_B):
            idx = rng.integers(0, n, n)
            bdf = zz.iloc[idx]
            try:
                bm = sm.OLS(bdf[RESPONSE], sm.add_constant(bdf[xvars], has_constant="add")).fit()
                boots.append(float(bm.params[C4]))
            except Exception:
                continue

        if len(boots) >= 100:
            ci_low = float(pd.Series(boots).quantile(0.025))
            ci_high = float(pd.Series(boots).quantile(0.975))
        else:
            ci_low, ci_high = np.nan, np.nan

        out_base.update({
            "coef_c4_standardized": beta,
            "se_hc3": se,
            "t": t,
            "p": p,
            "r2": r2,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ci_excludes_zero": bool((ci_low > 0) or (ci_high < 0)) if pd.notna(ci_low) and pd.notna(ci_high) else False,
            "nominal_pass_p05_ci": bool(p < 0.05 and ((ci_low > 0) or (ci_high < 0))) if pd.notna(ci_low) and pd.notna(ci_high) else False,
            "strict_n_ge_50": bool(len(zz) >= MIN_N_STRICT),
            "fit_status": "FIT_OK",
        })
    except Exception as e:
        out_base["fit_status"] = "FIT_ERROR: " + repr(e)

    return out_base

def bh_fdr(pvals):
    p = pd.Series(pvals, dtype=float)
    q = pd.Series(np.nan, index=p.index)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return q
    ranked = pd.Series(index=valid.index, dtype=float)
    prev = 1.0
    for i, idx in reversed(list(enumerate(valid.index, start=1))):
        val = min(prev, valid.loc[idx] * m / i)
        ranked.loc[idx] = val
        prev = val
    q.loc[ranked.index] = ranked
    return q

def rough_continent(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return "unknown"
    # Very rough bins for exploratory screening only.
    if -170 <= lon <= -30:
        return "americas"
    if -20 <= lon <= 55 and -35 <= lat <= 40:
        return "africa"
    if 55 < lon <= 180 and -50 <= lat <= 10:
        return "australia_asia_tropics"
    if -20 <= lon <= 180 and lat > 10:
        return "eurasia"
    return "other"

def lat_band(lat):
    if pd.isna(lat):
        return "unknown"
    a = abs(lat)
    if a < 23.5:
        return "tropical_abs_lt_23p5"
    if a < 35:
        return "subtropical_23p5_to_35"
    if a < 55:
        return "temperate_35_to_55"
    return "high_lat_abs_ge_55"

def safe_qcut(s, q, prefix):
    x = pd.to_numeric(s, errors="coerce")
    try:
        cats = pd.qcut(x, q=q, labels=[f"{prefix}_Q{i+1}" for i in range(q)], duplicates="drop")
        return cats.astype(str).replace("nan", np.nan)
    except Exception:
        return pd.Series(np.nan, index=s.index)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing audit dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    selected = {}
    selected["response"] = RESPONSE if RESPONSE in df.columns else None
    selected["c4"] = C4 if C4 in df.columns else None
    selected["rooting_depth"] = find_col(df, ["rooting_depth", "root_depth", "max_rooting_depth"])
    selected["aridity"] = find_col(df, ["aridity", "aridity_index"])
    selected["temperature"] = find_col(df, [
        "growing_season_mean_temperature",
        "growing_season_temperature",
        "mean_growing_season_temperature",
        "gs_mean_temperature",
        "mean_annual_temperature",
        "temperature"
    ])
    selected["precipitation"] = find_col(df, [
        "growing_season_precipitation",
        "mean_growing_season_precipitation",
        "mean_annual_precipitation",
        "precipitation"
    ])
    selected["soil_texture"] = find_col(df, ["soil_texture_pc1", "texture_pc1"])
    selected["lai_or_productivity"] = find_col(df, [
        "growing_season_mean_lai",
        "mean_lai",
        "lai",
        "growing_season_gpp",
        "mean_gpp",
        "gpp",
        "productivity"
    ])
    selected["baseline_vpd"] = find_col(df, ["mean_vpd", "baseline_vpd", "vpd"])
    selected["baseline_soil_moisture"] = find_col(df, ["mean_soil_moisture", "baseline_soil_moisture", "soil_moisture", "sm"])

    controls = [
        selected["rooting_depth"],
        selected["aridity"],
        selected["temperature"],
        selected["precipitation"],
        selected["soil_texture"],
        selected["lai_or_productivity"],
        selected["baseline_vpd"],
        selected["baseline_soil_moisture"],
    ]
    controls = [c for c in controls if c is not None]

    crop_flag_col = find_col(df, ["crop_or_cro_flag", "crop_flag", "is_crop", "cropland_flag"])
    natural_col = find_col(df, ["natural_grassland_like_no_crop_points", "natural_grassland_like", "natural_no_crop"])
    no_crop_col = find_col(df, ["no_crop_flagged_points", "no_crop", "crop_free"])

    # Build base subsets.
    subsets = []
    subsets.append(("all_points", df.copy()))

    if no_crop_col:
        mask = df[no_crop_col].astype(str).str.lower().isin(["true", "1", "yes"])
        subsets.append(("no_crop_flagged_points", df[mask].copy()))

    if natural_col:
        mask = df[natural_col].astype(str).str.lower().isin(["true", "1", "yes"])
        subsets.append(("natural_grassland_like_no_crop_points", df[mask].copy()))

    lat_col = find_col(df, ["lat", "latitude", "point_lat", "site_lat", "y"])
    lon_col = find_col(df, ["lon", "longitude", "point_lon", "site_lon", "x"])

    # Create grouping columns.
    work = df.copy()

    if lat_col and lon_col:
        work["_lat_num"] = pd.to_numeric(work[lat_col], errors="coerce")
        work["_lon_num"] = pd.to_numeric(work[lon_col], errors="coerce")
        work["_lat_band"] = work["_lat_num"].apply(lat_band)
        work["_rough_continent"] = [rough_continent(a, b) for a, b in zip(work["_lat_num"], work["_lon_num"])]
        work["_hemisphere"] = np.where(work["_lat_num"] >= 0, "northern", "southern")
    else:
        work["_lat_band"] = np.nan
        work["_rough_continent"] = np.nan
        work["_hemisphere"] = np.nan

    # Climate/geography bins.
    for role, col in [
        ("aridity", selected["aridity"]),
        ("temperature", selected["temperature"]),
        ("precipitation", selected["precipitation"]),
        ("vpd", selected["baseline_vpd"]),
        ("soil_moisture", selected["baseline_soil_moisture"]),
        ("lai", selected["lai_or_productivity"]),
        ("c4", selected["c4"]),
    ]:
        if col and col in work.columns:
            work[f"_{role}_tercile"] = safe_qcut(work[col], 3, role)
            work[f"_{role}_quartile"] = safe_qcut(work[col], 4, role)

    # Possible existing region columns.
    possible_region_cols = []
    for c in work.columns:
        cl = c.lower()
        if any(k in cl for k in ["continent", "country", "region", "biome", "ecoregion", "realm", "igbp", "landcover", "land_cover", "lc_class"]):
            if work[c].nunique(dropna=True) >= 2 and work[c].nunique(dropna=True) <= 40:
                possible_region_cols.append(c)

    grouping_cols = [
        "_lat_band",
        "_rough_continent",
        "_hemisphere",
        "_aridity_tercile",
        "_temperature_tercile",
        "_precipitation_tercile",
        "_vpd_tercile",
        "_soil_moisture_tercile",
        "_lai_tercile",
        "_c4_tercile",
    ]
    grouping_cols += possible_region_cols[:15]
    grouping_cols = [g for g in grouping_cols if g in work.columns]

    # Replace subset frames with work-indexed frames.
    subset_masks = [("all_points", pd.Series(True, index=work.index))]
    if no_crop_col:
        subset_masks.append(("no_crop_flagged_points", work[no_crop_col].astype(str).str.lower().isin(["true", "1", "yes"])))
    if natural_col:
        subset_masks.append(("natural_grassland_like_no_crop_points", work[natural_col].astype(str).str.lower().isin(["true", "1", "yes"])))

    rows = []

    # First run whole subsets as anchors.
    for subset_name, mask in subset_masks:
        sub = work[mask].copy()
        rows.append(fit_c4_full(
            sub,
            f"{subset_name}__whole_subset_full_controls",
            "whole_subset",
            subset_name,
            controls,
            crop_flag_col=crop_flag_col
        ))

    # Then spatial/climate groups within each subset.
    for subset_name, mask in subset_masks:
        base = work[mask].copy()
        for gcol in grouping_cols:
            if gcol not in base.columns:
                continue
            for gval, gdf in base.groupby(gcol, dropna=True):
                if str(gval).lower() in ["nan", "none", "unknown", ""]:
                    continue
                label = f"{subset_name}__{gcol}__{gval}"
                rows.append(fit_c4_full(
                    gdf,
                    label,
                    f"{subset_name}:{gcol}",
                    gval,
                    controls,
                    crop_flag_col=crop_flag_col
                ))

    res = pd.DataFrame(rows)

    # FDR only over fitted subgroup tests, excluding whole-subset anchors.
    subgroup_mask = res["fit_status"].eq("FIT_OK") & ~res["group_type"].eq("whole_subset")
    res["fdr_q"] = np.nan
    res.loc[subgroup_mask, "fdr_q"] = bh_fdr(res.loc[subgroup_mask, "p"])

    res["crop_clean_pass"] = res["crop_flag_n"].fillna(0).eq(0)
    res["strict_spatial_pass"] = (
        res["fit_status"].eq("FIT_OK")
        & res["strict_n_ge_50"].eq(True)
        & res["crop_clean_pass"].eq(True)
        & res["nominal_pass_p05_ci"].eq(True)
        & res["fdr_q"].fillna(1).le(0.10)
        & ~res["group_type"].eq("whole_subset")
    )
    res["exploratory_spatial_signal"] = (
        res["fit_status"].eq("FIT_OK")
        & res["n_complete"].ge(MIN_N_EXPLORATORY)
        & res["crop_clean_pass"].eq(True)
        & res["nominal_pass_p05_ci"].eq(True)
        & ~res["group_type"].eq("whole_subset")
    )

    res = res.sort_values(["strict_spatial_pass", "exploratory_spatial_signal", "p"], ascending=[False, False, True])
    res.to_csv(TAB / "Table_PRODUCT03fy_spatial_full_control_c4_screen_all_tests.csv", index=False)

    fitted = res[res["fit_status"].eq("FIT_OK")].copy()
    fitted.to_csv(TAB / "Table_PRODUCT03fz_spatial_full_control_c4_screen_fitted_only.csv", index=False)

    strict_pass = res[res["strict_spatial_pass"].eq(True)].copy()
    exploratory = res[res["exploratory_spatial_signal"].eq(True)].copy()
    strict_pass.to_csv(TAB / "Table_PRODUCT03ga_spatial_regions_passing_strict_screen.csv", index=False)
    exploratory.to_csv(TAB / "Table_PRODUCT03gb_spatial_regions_nominal_exploratory_signals.csv", index=False)

    # Load supporting tables if present.
    prior_c4 = pd.read_csv(PRIOR_C4_PATH) if PRIOR_C4_PATH.exists() else pd.DataFrame()
    crop = pd.read_csv(CROP_PATH) if CROP_PATH.exists() else pd.DataFrame()
    tower = pd.read_csv(TOWER_PATH) if TOWER_PATH.exists() else pd.DataFrame()
    tower_strict = pd.read_csv(TOWER_STRICT_PATH) if TOWER_STRICT_PATH.exists() else pd.DataFrame()
    tower_sens = pd.read_csv(TOWER_SENS_PATH) if TOWER_SENS_PATH.exists() else pd.DataFrame()

    decision = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stage": "1B.6BB_spatial_full_control_screen",
        "purpose": "Exploratory screen for predefined spatial/climate subsets where C4 passes Reza-style full controls.",
        "warning": "This is hypothesis-generating. Do not present subgroup hits as the main thesis unless ecologically justified and replicated.",
        "response": RESPONSE,
        "c4_variable": C4,
        "selected_controls": selected,
        "controls_used": controls,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "crop_flag_col": crop_flag_col,
        "no_crop_col": no_crop_col,
        "natural_col": natural_col,
        "n_total_tests": int(len(res)),
        "n_fitted_tests": int(res["fit_status"].eq("FIT_OK").sum()),
        "n_strict_spatial_pass": int(strict_pass.shape[0]),
        "n_exploratory_spatial_signals": int(exploratory.shape[0]),
        "strict_rule": f"FIT_OK, n >= {MIN_N_STRICT}, crop_flag_n = 0, p < 0.05, bootstrap CI excludes zero, FDR q <= 0.10",
        "exploratory_rule": f"FIT_OK, n >= {MIN_N_EXPLORATORY}, crop_flag_n = 0, p < 0.05, bootstrap CI excludes zero",
    }
    (TAB / "STAGE1B6BB_SPATIAL_FULL_CONTROL_SCREEN_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # Google-doc-ready note.
    lines = []
    lines.append("SPATIAL SUBGROUP SCREEN FOR REZA FULL-CONTROL C4 CHECK")
    lines.append("")
    lines.append("Purpose")
    lines.append("This screen asks whether the failed global direct C4 result hides a spatially localized subset where C4 passes the full Reza control set.")
    lines.append("")
    lines.append("Important caveat")
    lines.append("This is exploratory and hypothesis-generating. It should not be presented as proof that the original C4 thesis holds unless a passing region is ecologically coherent, crop-clean, sufficiently powered, and ideally replicated with an independent/product sensitivity check.")
    lines.append("")
    lines.append("Controls used")
    for role, col in selected.items():
        lines.append(f"- {role}: {col}")
    lines.append("")
    lines.append("Strict pass rule")
    lines.append(decision["strict_rule"])
    lines.append("")
    lines.append("Exploratory signal rule")
    lines.append(decision["exploratory_rule"])
    lines.append("")
    lines.append("Summary")
    lines.append(f"- Total tests run: {decision['n_total_tests']}")
    lines.append(f"- Fitted tests: {decision['n_fitted_tests']}")
    lines.append(f"- Strict spatial passes: {decision['n_strict_spatial_pass']}")
    lines.append(f"- Nominal exploratory spatial signals: {decision['n_exploratory_spatial_signals']}")
    lines.append("")

    lines.append("Whole-subset full-control anchors")
    anchors = res[res["group_type"].eq("whole_subset")].copy()
    if len(anchors):
        show = anchors[[
            "group_value", "n_complete", "coef_c4_standardized", "p",
            "bootstrap_ci_low", "bootstrap_ci_high", "nominal_pass_p05_ci", "fit_status"
        ]]
        lines.append(show.to_string(index=False))
    else:
        lines.append("No whole-subset anchors produced.")
    lines.append("")

    lines.append("Strict spatial passes")
    if len(strict_pass):
        show = strict_pass[[
            "group_type", "group_value", "n_complete", "crop_flag_n",
            "coef_c4_standardized", "p", "fdr_q",
            "bootstrap_ci_low", "bootstrap_ci_high", "r2"
        ]].head(30)
        lines.append(show.to_string(index=False))
    else:
        lines.append("No spatial/climate subgroup passed the strict screen.")
    lines.append("")

    lines.append("Nominal exploratory spatial signals")
    if len(exploratory):
        show = exploratory[[
            "group_type", "group_value", "n_complete", "crop_flag_n",
            "coef_c4_standardized", "p", "fdr_q",
            "bootstrap_ci_low", "bootstrap_ci_high", "r2",
            "strict_spatial_pass"
        ]].head(50)
        lines.append(show.to_string(index=False))
    else:
        lines.append("No nominal exploratory spatial signals found.")
    lines.append("")

    lines.append("How to explain this to Reza")
    lines.append("If no strict region passes:")
    lines.append("The spatial subgroup screen did not rescue the independent direct C4 claim. This strengthens the conclusion that the global C4 signal is mainly tied to the VPD/climate gradient rather than a robust independent trait effect.")
    lines.append("")
    lines.append("If a strict region passes:")
    lines.append("There may be a spatially localized C4 effect after full controls, but I would frame it as an exploratory regional signal, not as the main global thesis, until we check ecological coherence and product robustness.")
    lines.append("")
    lines.append("Files written")
    lines.append("results/stage1b6bb_spatial_full_control_screen/tables/STAGE1B6BB_SPATIAL_FULL_CONTROL_SCREEN_DECISION.json")
    lines.append("results/stage1b6bb_spatial_full_control_screen/tables/Table_PRODUCT03fy_spatial_full_control_c4_screen_all_tests.csv")
    lines.append("results/stage1b6bb_spatial_full_control_screen/tables/Table_PRODUCT03fz_spatial_full_control_c4_screen_fitted_only.csv")
    lines.append("results/stage1b6bb_spatial_full_control_screen/tables/Table_PRODUCT03ga_spatial_regions_passing_strict_screen.csv")
    lines.append("results/stage1b6bb_spatial_full_control_screen/tables/Table_PRODUCT03gb_spatial_regions_nominal_exploratory_signals.csv")

    note = "\n".join(lines)
    (TXT / "SPATIAL_FULL_CONTROL_SCREEN_GOOGLE_DOC_READY.txt").write_text(note, encoding="utf-8")

    print("===== SPATIAL FULL-CONTROL SCREEN DECISION =====")
    print(json.dumps(decision, indent=2))
    print("")
    print("===== WHOLE-SUBSET FULL-CONTROL ANCHORS =====")
    if len(anchors):
        print(anchors[[
            "group_value", "n_complete", "coef_c4_standardized", "p",
            "bootstrap_ci_low", "bootstrap_ci_high", "nominal_pass_p05_ci", "fit_status"
        ]].to_string(index=False))
    else:
        print("No anchors.")
    print("")
    print("===== STRICT SPATIAL PASSES =====")
    if len(strict_pass):
        print(strict_pass[[
            "group_type", "group_value", "n_complete", "crop_flag_n",
            "coef_c4_standardized", "p", "fdr_q",
            "bootstrap_ci_low", "bootstrap_ci_high", "r2"
        ]].head(30).to_string(index=False))
    else:
        print("No spatial/climate subgroup passed the strict screen.")
    print("")
    print("===== NOMINAL EXPLORATORY SPATIAL SIGNALS =====")
    if len(exploratory):
        print(exploratory[[
            "group_type", "group_value", "n_complete", "crop_flag_n",
            "coef_c4_standardized", "p", "fdr_q",
            "bootstrap_ci_low", "bootstrap_ci_high", "r2",
            "strict_spatial_pass"
        ]].head(50).to_string(index=False))
    else:
        print("No nominal exploratory spatial signals found.")
    print("")
    print("===== GOOGLE DOC NOTE =====")
    print(note)

if __name__ == "__main__":
    main()
