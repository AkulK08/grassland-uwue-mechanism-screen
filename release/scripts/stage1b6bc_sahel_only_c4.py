from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bc_sahel_only_c4"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "results/stage1b6ay_final_project_audit/tables/Table_PRODUCT03fg_final_audit_dataset.csv"

RESPONSE = "latent_slope_change"
C4 = "c4_fraction_raw"

# project full-control set
CONTROLS = [
    "rooting_depth",
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "soil_texture_pc1",
    "growing_season_mean_lai",
    "mean_vpd",
    "mean_soil_moisture",
]

# Sahel definitions.
# Main definition: broad Sahel belt.
# Core definition: stricter central Sahel belt.
SAHEL_DEFS = {
    "sahel_broad_lat10_20_lon-20_40": {
        "lat_min": 10,
        "lat_max": 20,
        "lon_min": -20,
        "lon_max": 40,
    },
    "sahel_core_lat12_18_lon-17_35": {
        "lat_min": 12,
        "lat_max": 18,
        "lon_min": -17,
        "lon_max": 35,
    },
}

BOOT_B = 2000
SEED = 123

def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        hits = [c for c in df.columns if cand.lower() in c.lower()]
        if hits:
            return hits[0]
    return None

def z(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if x.notna().sum() < 5 or pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd

def bool_mask_from_col(df, col):
    if col is None or col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col].astype(str).str.lower().isin(["true", "1", "yes"])

def crop_mask(df, crop_col):
    if crop_col is None or crop_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[crop_col].astype(str).str.lower().isin(["true", "1", "yes", "crop", "cro"])

def fit_model(df, label, controls):
    cols = [RESPONSE, C4] + controls
    missing = [c for c in cols if c not in df.columns]
    base = {
        "model_label": label,
        "n_raw": len(df),
        "n_complete": 0,
        "coef_c4_standardized": np.nan,
        "se_hc3": np.nan,
        "t": np.nan,
        "p": np.nan,
        "r2": np.nan,
        "bootstrap_ci_low": np.nan,
        "bootstrap_ci_high": np.nan,
        "ci_excludes_zero": False,
        "passes_p05_ci": False,
        "fit_status": "",
        "controls": ", ".join(controls),
    }

    if missing:
        base["fit_status"] = "MISSING_COLUMNS: " + ", ".join(missing)
        return base

    use = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    base["n_complete"] = len(use)

    if len(use) < 10:
        base["fit_status"] = "NOT_FIT_TOO_FEW_COMPLETE_ROWS_LT_10"
        return base

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()
    base["n_complete"] = len(zz)

    if len(zz) < 10:
        base["fit_status"] = "NOT_FIT_TOO_FEW_Z_ROWS_LT_10"
        return base

    # Need more rows than predictors + intercept.
    if len(zz) <= len(controls) + 2:
        base["fit_status"] = "NOT_FIT_UNDERPOWERED_N_LEQ_PARAMETERS"
        return base

    xvars = [C4] + controls
    try:
        X = sm.add_constant(zz[xvars], has_constant="add")
        model = sm.OLS(zz[RESPONSE], X).fit(cov_type="HC3")

        beta = float(model.params.get(C4, np.nan))
        p = float(model.pvalues.get(C4, np.nan))
        se = float(model.bse.get(C4, np.nan))
        t = float(model.tvalues.get(C4, np.nan))
        r2 = float(model.rsquared)

        rng = np.random.default_rng(SEED)
        boots = []
        n = len(zz)
        for _ in range(BOOT_B):
            idx = rng.integers(0, n, n)
            bdf = zz.iloc[idx]
            try:
                bm = sm.OLS(
                    bdf[RESPONSE],
                    sm.add_constant(bdf[xvars], has_constant="add")
                ).fit()
                boots.append(float(bm.params[C4]))
            except Exception:
                pass

        if len(boots) >= 100:
            ci_low = float(pd.Series(boots).quantile(0.025))
            ci_high = float(pd.Series(boots).quantile(0.975))
            ci_excludes = bool((ci_low > 0) or (ci_high < 0))
        else:
            ci_low = np.nan
            ci_high = np.nan
            ci_excludes = False

        base.update({
            "coef_c4_standardized": beta,
            "se_hc3": se,
            "t": t,
            "p": p,
            "r2": r2,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "ci_excludes_zero": ci_excludes,
            "passes_p05_ci": bool(p < 0.05 and ci_excludes),
            "fit_status": "FIT_OK",
        })
    except Exception as e:
        base["fit_status"] = "FIT_ERROR: " + repr(e)

    return base

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    lat_col = find_col(df, ["lat", "latitude", "point_lat", "site_lat", "y"])
    lon_col = find_col(df, ["lon", "longitude", "point_lon", "site_lon", "x"])

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find lat/lon columns. Columns are: {list(df.columns)}")

    df["_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["_lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    crop_col = find_col(df, ["crop_or_cro_flag", "crop_flag", "is_crop", "cropland_flag"])
    no_crop_col = find_col(df, ["no_crop_flagged_points", "no_crop", "crop_free"])
    natural_col = find_col(df, ["natural_grassland_like_no_crop_points", "natural_grassland_like", "natural_no_crop"])

    controls_present = [c for c in CONTROLS if c in df.columns]
    missing_controls = [c for c in CONTROLS if c not in df.columns]

    rows = []
    crop_rows = []

    for sahel_name, bounds in SAHEL_DEFS.items():
        sahel = df[
            (df["_lat"] >= bounds["lat_min"])
            & (df["_lat"] <= bounds["lat_max"])
            & (df["_lon"] >= bounds["lon_min"])
            & (df["_lon"] <= bounds["lon_max"])
        ].copy()

        crop_flags = crop_mask(sahel, crop_col)
        no_crop_mask = bool_mask_from_col(sahel, no_crop_col)
        natural_mask = bool_mask_from_col(sahel, natural_col)

        subsets = {
            f"{sahel_name}__all_sahel_points": sahel,
            f"{sahel_name}__no_crop_flagged": sahel[no_crop_mask].copy(),
            f"{sahel_name}__natural_grassland_like_no_crop": sahel[natural_mask].copy(),
        }

        crop_rows.append({
            "sahel_definition": sahel_name,
            "lat_min": bounds["lat_min"],
            "lat_max": bounds["lat_max"],
            "lon_min": bounds["lon_min"],
            "lon_max": bounds["lon_max"],
            "n_sahel_raw": len(sahel),
            "crop_col": crop_col,
            "crop_flag_n": int(crop_flags.sum()) if len(sahel) else 0,
            "crop_flag_rate": float(crop_flags.mean()) if len(sahel) else np.nan,
            "no_crop_col": no_crop_col,
            "no_crop_n": int(no_crop_mask.sum()) if len(sahel) else 0,
            "natural_col": natural_col,
            "natural_grassland_like_no_crop_n": int(natural_mask.sum()) if len(sahel) else 0,
        })

        for label, sdf in subsets.items():
            res = fit_model(sdf, label, controls_present)
            res["sahel_definition"] = sahel_name
            res["lat_bounds"] = f"{bounds['lat_min']} to {bounds['lat_max']}"
            res["lon_bounds"] = f"{bounds['lon_min']} to {bounds['lon_max']}"
            res["crop_col"] = crop_col
            res["crop_flag_n_in_subset"] = int(crop_mask(sdf, crop_col).sum()) if len(sdf) else 0
            res["crop_clean"] = bool(res["crop_flag_n_in_subset"] == 0)
            rows.append(res)

    model_df = pd.DataFrame(rows)
    crop_df = pd.DataFrame(crop_rows)

    model_df.to_csv(TAB / "Table_PRODUCT03gc_sahel_only_full_control_c4_models.csv", index=False)
    crop_df.to_csv(TAB / "Table_PRODUCT03gd_sahel_only_crop_landcover_check.csv", index=False)

    any_pass = bool(
        len(model_df)
        and model_df["fit_status"].eq("FIT_OK").any()
        and model_df["passes_p05_ci"].eq(True).any()
        and model_df["crop_clean"].eq(True).any()
    )

    decision = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stage": "1B.6BC_sahel_only_c4_full_control",
        "lat_col": lat_col,
        "lon_col": lon_col,
        "crop_col": crop_col,
        "no_crop_col": no_crop_col,
        "natural_col": natural_col,
        "response": RESPONSE,
        "c4": C4,
        "controls_requested": CONTROLS,
        "controls_present_used": controls_present,
        "missing_controls": missing_controls,
        "any_sahel_full_control_direct_c4_pass": any_pass,
        "pass_rule": "FIT_OK, p < 0.05, bootstrap CI excludes zero, crop_clean = True",
        "warning": "Sahel-only is a predefined regional diagnostic. If it passes, treat as regional and confirm product/land-cover robustness before making it a thesis.",
    }

    (TAB / "STAGE1B6BC_SAHEL_ONLY_C4_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = []
    lines.append("SAHEL-ONLY FULL-CONTROL C4 CHECK")
    lines.append("")
    lines.append("Purpose")
    lines.append("Test whether the direct C4 result passes project's full climate/environment control screen specifically within the Sahel.")
    lines.append("")
    lines.append("Sahel definitions tested")
    for name, b in SAHEL_DEFS.items():
        lines.append(f"- {name}: latitude {b['lat_min']} to {b['lat_max']}, longitude {b['lon_min']} to {b['lon_max']}")
    lines.append("")
    lines.append("Full controls used")
    for c in controls_present:
        lines.append(f"- {c}")
    if missing_controls:
        lines.append("")
        lines.append("Missing requested controls")
        for c in missing_controls:
            lines.append(f"- {c}")
    lines.append("")
    lines.append("Pass rule")
    lines.append("FIT_OK, p < 0.05, bootstrap CI excludes zero, and crop_clean = True.")
    lines.append("")
    lines.append("Decision")
    lines.append(f"Any Sahel direct C4 full-control pass: {any_pass}")
    lines.append("")
    lines.append("Crop / land-cover check")
    lines.append(crop_df.to_string(index=False))
    lines.append("")
    lines.append("Sahel full-control direct C4 model results")
    show_cols = [
        "model_label",
        "n_raw",
        "n_complete",
        "crop_flag_n_in_subset",
        "crop_clean",
        "coef_c4_standardized",
        "se_hc3",
        "p",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "ci_excludes_zero",
        "passes_p05_ci",
        "fit_status",
    ]
    lines.append(model_df[show_cols].to_string(index=False))
    lines.append("")
    lines.append("Interpretation template")
    lines.append("If any pass is True: Sahel may be a regional exception where direct C4 survives full controls, but this should be presented as a regional result, not a global result.")
    lines.append("If no pass is True: Sahel does not rescue the direct C4 thesis under project's full controls.")
    lines.append("")
    lines.append("Files")
    lines.append("results/stage1b6bc_sahel_only_c4/tables/STAGE1B6BC_SAHEL_ONLY_C4_DECISION.json")
    lines.append("results/stage1b6bc_sahel_only_c4/tables/Table_PRODUCT03gc_sahel_only_full_control_c4_models.csv")
    lines.append("results/stage1b6bc_sahel_only_c4/tables/Table_PRODUCT03gd_sahel_only_crop_landcover_check.csv")

    note = "\n".join(lines)
    (TXT / "SAHEL_ONLY_GOOGLE_DOC_READY_RESULTS.txt").write_text(note, encoding="utf-8")

    print("===== SAHEL-ONLY C4 FULL-CONTROL DECISION =====")
    print(json.dumps(decision, indent=2))
    print("")
    print("===== SAHEL CROP / LAND-COVER CHECK =====")
    print(crop_df.to_string(index=False))
    print("")
    print("===== SAHEL FULL-CONTROL DIRECT C4 RESULTS =====")
    print(model_df[show_cols].to_string(index=False))
    print("")
    print("===== GOOGLE-DOC-READY NOTE =====")
    print(note)

if __name__ == "__main__":
    main()
