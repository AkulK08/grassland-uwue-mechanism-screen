from pathlib import Path
from datetime import datetime
import json
import re
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6aw_clean_crop_tower_controls"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

BASE = ROOT / "results/stage1b6ai_reza_final_lock_with_c4/tables/Table_PRODUCT03aq_c4_sampled_point_table.csv"

TOWER_TABLES = [
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table101_tower_landcover_spatial_trait_annotation.csv",
    ROOT / "results/tower_grassland_spatial_trait_lock/tables/Table108_satellite_extraction_targets_all_49_tower_response_sites.csv",
]

def norm(x):
    return str(x).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def z(s):
    s = to_num(s)
    sd = s.std()
    if s.notna().sum() < 20 or pd.isna(sd) or sd == 0:
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
        if score:
            scored.append((score, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]

def fit(df, response, c4, controls, label):
    cols = [response, c4] + controls
    use = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 40:
        return [{
            "model_label": label,
            "n": len(use),
            "term": c4,
            "coef_standardized": np.nan,
            "p": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_COMPLETE_ROWS",
            "controls": ", ".join(controls),
        }]

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()
    if len(zz) < 40:
        return [{
            "model_label": label,
            "n": len(zz),
            "term": c4,
            "coef_standardized": np.nan,
            "p": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_AFTER_ZSCORE",
            "controls": ", ".join(controls),
        }]

    X = sm.add_constant(zz[[c4] + controls], has_constant="add")
    y = zz[response]
    m = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for term in [c4] + controls:
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
            "fit_status": "FIT_OK",
            "controls": ", ".join(controls),
        })
    return rows

def is_crop_row(df):
    crop = pd.Series(False, index=df.index)

    if "eco_biome_num" in df.columns:
        nums = to_num(df["eco_biome_num"])
        crop = crop | nums.isin([12, 14])

    for c in df.columns:
        lc = norm(c)
        if any(k in lc for k in ["igbp", "landcover", "land_cover", "crop", "eco_biome", "nearest_sat_eco_biome"]):
            s = df[c].map(lambda v: "" if pd.isna(v) else str(v).lower())
            crop = crop | s.str.contains(r"\bcro\b|crop|cropland|cultivated|maize|corn|sorghum|millet|sugarcane|sugar cane|irrig", regex=True)
    return crop

def is_natural_grassland_like(df):
    ok = pd.Series(False, index=df.index)

    # Accept broad natural grassland/savanna/shrubland classes.
    for c in ["eco_biome", "nearest_sat_eco_biome"]:
        if c in df.columns:
            s = df[c].map(lambda v: "" if pd.isna(v) else str(v).lower())
            ok = ok | s.str.contains("grassland|savanna|shrubland|rangeland|pasture", regex=True)

    # Numeric biome classes inferred from previous output:
    # 7/8/10/11 appear to be grass/savanna/shrub/tundra-like in this table, but avoid crop 12/14.
    if "eco_biome_num" in df.columns:
        nums = to_num(df["eco_biome_num"])
        ok = ok | nums.isin([7, 8, 10, 11])

    return ok

def tower_clean_check():
    rows = []
    for p in TOWER_TABLES:
        if not p.exists():
            rows.append({"source": str(p), "status": "MISSING"})
            continue
        d = pd.read_csv(p, low_memory=False)
        cols = [c for c in ["site_id", "site", "igbp_final", "igbp", "eco_biome", "nearest_sat_eco_biome", "passes_landcover_screen_lenient"] if c in d.columns]
        sub = d[cols].copy()
        crop = is_crop_row(sub)
        grass = is_natural_grassland_like(sub)

        rows.append({
            "source": str(p),
            "n_rows": len(sub),
            "crop_or_cro_flag_n": int(crop.sum()),
            "crop_or_cro_flag_rate": float(crop.mean()) if len(sub) else np.nan,
            "natural_grassland_like_n": int(grass.sum()),
            "natural_grassland_like_rate": float(grass.mean()) if len(sub) else np.nan,
            "columns_checked": ", ".join(cols),
        })

        sub["crop_or_cro_flag"] = crop
        sub["natural_grassland_like"] = grass
        outname = "Table_PRODUCT03ej_clean_tower_landcover_rows_" + p.stem + ".csv"
        sub.to_csv(TAB / outname, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "Table_PRODUCT03ei_clean_tower_landcover_summary.csv", index=False)
    return out

def main():
    d = pd.read_csv(BASE, low_memory=False)

    response = pick_col(d.columns, [r"^latent_slope_change$"])
    c4 = pick_col(d.columns, [r"^c4_fraction_raw$", r"^c4_fraction$", r"c4.*fraction"])
    root = pick_col(d.columns, [r"^rooting_depth$", r"root.*depth"])
    aridity = pick_col(d.columns, [r"^aridity$", r"aridity_index", r"aridity_quantile"])
    temp = pick_col(d.columns, [r"mean_temperature", r"temperature"])
    precip = pick_col(d.columns, [r"mean_precipitation", r"precip"])
    clay = pick_col(d.columns, [r"soil_clay", r"clay"])
    sand = pick_col(d.columns, [r"soil_sand", r"sand"])
    silt = pick_col(d.columns, [r"soil_silt", r"silt"])

    crop = is_crop_row(d)
    natural = is_natural_grassland_like(d)

    d_all = d.copy()
    d_no_crop = d.loc[~crop].copy()
    d_natural_no_crop = d.loc[natural & ~crop].copy()

    for name, frame in [
        ("all_points", d_all),
        ("no_crop_flagged_points", d_no_crop),
        ("natural_grassland_like_no_crop_points", d_natural_no_crop),
    ]:
        frame.to_csv(TAB / f"Table_PRODUCT03ek_{name}_analysis_rows.csv", index=False)

    models = []
    sets = [
        ("all_points", d_all),
        ("no_crop_flagged_points", d_no_crop),
        ("natural_grassland_like_no_crop_points", d_natural_no_crop),
    ]

    control_sets = [
        ("c4_plus_rooting_depth", [root]),
        ("c4_plus_rooting_depth_aridity", [root, aridity]),
        ("c4_plus_rooting_depth_temp_precip", [root, temp, precip]),
        ("c4_plus_rooting_depth_soil_texture", [root, clay, sand, silt]),
        ("c4_plus_rooting_depth_climate_soil", [root, aridity, temp, precip, clay, sand, silt]),
    ]

    for set_name, frame in sets:
        for label, controls in control_sets:
            controls = [c for c in controls if c]
            rows = fit(frame, response, c4, controls, f"{set_name}__{label}")
            models.extend(rows)

    model_df = pd.DataFrame(models)
    model_df.to_csv(TAB / "Table_PRODUCT03el_clean_crop_sensitivity_c4_models.csv", index=False)

    tower_summary = tower_clean_check()

    c4_rows = model_df[(model_df["term"].eq(c4)) & (model_df["fit_status"].eq("FIT_OK"))].copy()
    c4_rows["passes_p05"] = c4_rows["p"] <= 0.05
    c4_rows.to_csv(TAB / "Table_PRODUCT03em_c4_rows_only_clean_crop_sensitivity.csv", index=False)

    natural_core = c4_rows[c4_rows["model_label"].str.contains("natural_grassland_like_no_crop_points", na=False)].copy()
    no_crop_core = c4_rows[c4_rows["model_label"].str.contains("no_crop_flagged_points", na=False)].copy()

    decision = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stage": "1B.6AW_clean_crop_tower_controls",
        "response": response,
        "c4": c4,
        "rows_all": int(len(d_all)),
        "crop_flagged_rows": int(crop.sum()),
        "rows_no_crop": int(len(d_no_crop)),
        "natural_grassland_like_rows_no_crop": int(len(d_natural_no_crop)),
        "c4_holds_after_dropping_crop_flagged_points": bool(len(no_crop_core) and no_crop_core["passes_p05"].any()),
        "c4_holds_in_natural_grassland_like_no_crop_subset": bool(len(natural_core) and natural_core["passes_p05"].any()),
        "best_no_crop_c4": no_crop_core.sort_values("p").head(1).to_dict(orient="records") if len(no_crop_core) else [],
        "best_natural_no_crop_c4": natural_core.sort_values("p").head(1).to_dict(orient="records") if len(natural_core) else [],
        "tower_clean_landcover_summary": tower_summary.to_dict(orient="records"),
    }

    (TAB / "STAGE1B6AW_CLEAN_CROP_TOWER_CONTROLS_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    note = []
    note.append("# Clean crop/tower follow-up")
    note.append("")
    note.append("## Decision")
    note.append("```json")
    note.append(json.dumps(decision, indent=2))
    note.append("```")
    note.append("")
    note.append("## C4 model rows")
    note.append("```text")
    note.append(c4_rows.to_string(index=False))
    note.append("```")
    note.append("")
    note.append("## Tower clean land-cover summary")
    note.append("```text")
    note.append(tower_summary.to_string(index=False))
    note.append("```")

    (TXT / "STAGE1B6AW_CLEAN_CROP_TOWER_CONTROLS_REPORT.md").write_text("\n".join(note), encoding="utf-8")

    print("\n".join(note))

if __name__ == "__main__":
    main()
