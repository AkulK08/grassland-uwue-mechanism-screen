from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path.cwd()
IN = ROOT / "results/stage1b6ay_final_project_audit"
TAB_IN = IN / "tables"

OUT = ROOT / "results/stage1b6az_honest_project_packet"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

DECISION_PATH = TAB_IN / "STAGE1B6AY_FINAL_project_AUDIT_DECISION.json"
C4_ROWS_PATH = TAB_IN / "Table_PRODUCT03fk_final_c4_rows_only.csv"
SELECTED_PATH = TAB_IN / "Table_PRODUCT03fh_final_selected_controls.csv"
DATASET_PATH = TAB_IN / "Table_PRODUCT03fg_final_audit_dataset.csv"
CROP_PATH = TAB_IN / "Table_PRODUCT03fm_final_crop_summary.csv"
TOWER_PATH = TAB_IN / "Table_PRODUCT03fd_target_tower_landcover_summary.csv"

STRICT_RANK = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03di_final_FULL_STRICT_strict_et_ranking.csv"
SENS_RANK = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dj_final_FULL_STRICT_sensitivity_et_ranking.csv"
SITE_STATUS = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dh_final_FULL_STRICT_site_status.csv"

def z(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if x.notna().sum() < 20 or pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd

def fit_std(df, y, xvars, label):
    cols = [y] + xvars
    use = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 40:
        return pd.DataFrame([{
            "model_label": label,
            "n": len(use),
            "response": y,
            "term": "",
            "coef_standardized": np.nan,
            "p": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_ROWS",
            "controls": ", ".join(xvars)
        }])

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()

    if len(zz) < 40:
        return pd.DataFrame([{
            "model_label": label,
            "n": len(zz),
            "response": y,
            "term": "",
            "coef_standardized": np.nan,
            "p": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_Z_ROWS",
            "controls": ", ".join(xvars)
        }])

    X = sm.add_constant(zz[xvars], has_constant="add")
    m = sm.OLS(zz[y], X).fit(cov_type="HC3")

    rows = []
    for term in xvars:
        rows.append({
            "model_label": label,
            "n": int(m.nobs),
            "response": y,
            "term": term,
            "coef_standardized": float(m.params.get(term, np.nan)),
            "se_hc3": float(m.bse.get(term, np.nan)),
            "t": float(m.tvalues.get(term, np.nan)),
            "p": float(m.pvalues.get(term, np.nan)),
            "r2": float(m.rsquared),
            "fit_status": "FIT_OK",
            "controls": ", ".join(xvars)
        })
    return pd.DataFrame(rows)

def main():
    if not DECISION_PATH.exists():
        raise FileNotFoundError(f"Missing final audit decision: {DECISION_PATH}")
    if not C4_ROWS_PATH.exists():
        raise FileNotFoundError(f"Missing C4 rows: {C4_ROWS_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing final audit dataset: {DATASET_PATH}")

    decision = json.loads(DECISION_PATH.read_text())
    c4_rows = pd.read_csv(C4_ROWS_PATH)
    selected = pd.read_csv(SELECTED_PATH) if SELECTED_PATH.exists() else pd.DataFrame()
    data = pd.read_csv(DATASET_PATH, low_memory=False)
    crop = pd.read_csv(CROP_PATH) if CROP_PATH.exists() else pd.DataFrame()
    tower = pd.read_csv(TOWER_PATH) if TOWER_PATH.exists() else pd.DataFrame()
    strict = pd.read_csv(STRICT_RANK) if STRICT_RANK.exists() else pd.DataFrame()
    sens = pd.read_csv(SENS_RANK) if SENS_RANK.exists() else pd.DataFrame()
    site = pd.read_csv(SITE_STATUS) if SITE_STATUS.exists() else pd.DataFrame()

    response = decision["response_variable"]
    c4 = decision["c4_variable"]

    # Pull key rows for simple meeting table.
    wanted_labels = [
        "all_points__benchmark_rooting_depth",
        "all_points__full_with_clean_productivity",
        "all_points__no_vpd",
        "no_crop_flagged_points__benchmark_rooting_depth",
        "no_crop_flagged_points__full_with_clean_productivity",
        "no_crop_flagged_points__no_vpd",
        "natural_grassland_like_no_crop_points__benchmark_rooting_depth",
        "natural_grassland_like_no_crop_points__full_with_clean_productivity",
        "natural_grassland_like_no_crop_points__no_vpd",
    ]

    key = c4_rows[c4_rows["model_label"].isin(wanted_labels)].copy()
    key = key[[
        "model_label", "n", "coef_standardized", "se_hc3", "p",
        "bootstrap_ci_low", "bootstrap_ci_high", "ci_excludes_zero",
        "fit_status", "controls", "passes_p05_ci"
    ]]
    key.to_csv(TAB / "Table_PRODUCT03fr_key_c4_result_summary.csv", index=False)

    # VPD diagnosis.
    required = {
        "rooting_depth": "rooting_depth",
        "aridity": "aridity",
        "temp": "mean_annual_temperature",
        "precip": "mean_annual_precipitation",
        "texture": "soil_texture_pc1",
        "lai": "growing_season_mean_lai",
        "vpd": "mean_vpd",
        "sm": "mean_soil_moisture",
    }

    existing = {k: v for k, v in required.items() if v in data.columns}
    base_controls_no_vpd = [
        existing.get("rooting_depth"),
        existing.get("aridity"),
        existing.get("temp"),
        existing.get("precip"),
        existing.get("texture"),
        existing.get("sm"),
        existing.get("lai"),
    ]
    base_controls_no_vpd = [x for x in base_controls_no_vpd if x]

    diagnostics = []
    for subset_name, subset_df in [
        ("all_points", data),
        ("no_crop_flagged_points", data[data.get("no_crop_flagged_points", True) == True].copy() if "no_crop_flagged_points" in data.columns else data.copy()),
        ("natural_grassland_like_no_crop_points", data[data.get("natural_grassland_like_no_crop_points", True) == True].copy() if "natural_grassland_like_no_crop_points" in data.columns else data.copy()),
    ]:
        if "mean_vpd" in subset_df.columns:
            diagnostics.append(fit_std(
                subset_df,
                "mean_vpd",
                [c4] + [c for c in base_controls_no_vpd if c != "mean_vpd"],
                f"{subset_name}__does_c4_predict_baseline_vpd"
            ))
            diagnostics.append(fit_std(
                subset_df,
                response,
                ["mean_vpd"] + [c for c in base_controls_no_vpd if c != "mean_vpd"],
                f"{subset_name}__does_baseline_vpd_predict_response"
            ))
            diagnostics.append(fit_std(
                subset_df,
                response,
                [c4, "mean_vpd"] + [c for c in base_controls_no_vpd if c != "mean_vpd"],
                f"{subset_name}__c4_after_adding_baseline_vpd"
            ))

    diag = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    diag.to_csv(TAB / "Table_PRODUCT03fs_vpd_killer_diagnostic.csv", index=False)

    # Final concern table.
    concern_rows = [
        {
            "project_concern": "Full climate/environment controls",
            "status": "NOT_SOLVED",
            "evidence": "C4 fails in full model with rooting depth, aridity, temperature, precipitation, soil texture PC1, clean LAI/productivity, baseline VPD, and baseline soil moisture.",
            "meeting_language": "The C4 signal is not independent of the full climate/environment gradient. Baseline VPD appears to absorb the signal."
        },
        {
            "project_concern": "C4 cropland leakage in satellite sample",
            "status": "MOSTLY_SOLVED_FOR_SATELLITE_POINTS",
            "evidence": "Only 2/199 satellite rows were crop/CRO-flagged; after removing crop-flagged points and restricting to natural grassland-like no-crop points, benchmark and no-VPD sensitivity models still pass.",
            "meeting_language": "The satellite C4 result is not driven only by obvious crop-coded points, but the full-control C4 claim still fails."
        },
        {
            "project_concern": "Tower land-cover filtering, especially US-Ne1/2/3",
            "status": "SOLVED_AS_AUDIT_NOT_AS_INCLUSION",
            "evidence": "US-Ne1/2/3 are crop-flagged in the tower land-cover check and should not be used as natural grassland evidence. They were also non-computable in the strict tower validation table.",
            "meeting_language": "US-Ne1/2/3 should be explicitly excluded/caveated; they cannot support the natural grassland tower validation."
        },
        {
            "project_concern": "Tower language",
            "status": "SOLVED",
            "evidence": "Strict and sensitivity rankings modestly favor GLEAM, but agreement is low overall.",
            "meeting_language": "Towers provide a limited independent anchor and suggest GLEAM is the better ET product for this analysis; they do not strongly validate GLEAM."
        },
    ]
    concern = pd.DataFrame(concern_rows)
    concern.to_csv(TAB / "Table_PRODUCT03ft_project_concern_status_table.csv", index=False)

    final_decision = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stage": "1B.6AZ_honest_project_packet",
        "bottom_line": "Do not claim that all project concerns are solved. The clean audit says the C4 effect is sensitivity-dependent and fails the full climate/environment control screen.",
        "main_failed_item": "Full climate/environment controls",
        "why_failed": "Baseline VPD absorbs the C4 signal. C4 reappears when mean_vpd is removed, suggesting the signal is tied to the warm/dry VPD climate gradient rather than an independent C4 mechanism.",
        "safe_scientific_claim": "C4 fraction is associated with uWUE latent slope-change in benchmark, crop-free, natural-grassland, and no-VPD sensitivity models, but this association is not independent of baseline VPD.",
        "unsafe_claim": "C4 photosynthetic composition independently predicts uWUE response after the full climate/environment control set.",
        "recommendation_for_meeting": "Ask project whether baseline VPD should be treated as a confounder or as part of the stress pathway. If confounder, the clean C4 mechanism should be rejected/narrowed. If pathway/mediator, the paper can be reframed as a C4-VPD climate-syndrome result.",
        "all_project_concerns_solved": False,
    }
    (TAB / "STAGE1B6AZ_HONEST_project_PACKET_DECISION.json").write_text(json.dumps(final_decision, indent=2), encoding="utf-8")

    # Meeting note.
    lines = []
    lines.append("# project meeting packet: honest final audit")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append("The C4 result should not be presented as fully locked under project's requested full-control screen. It remains strong in benchmark, crop-free, natural-grassland-like, and no-VPD sensitivity models, but it does not survive the full climate/environment model once baseline VPD is included.")
    lines.append("")
    lines.append("The key interpretation is: C4 appears tied to the warm/dry VPD climate gradient. The remaining decision is whether baseline VPD is a confounder to control away, or part of the stress pathway/mechanism.")
    lines.append("")
    lines.append("## project concern status table")
    lines.append("")
    lines.append("```text")
    lines.append(concern.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Key C4 model rows")
    lines.append("")
    lines.append("```text")
    lines.append(key.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## VPD diagnostic")
    lines.append("")
    lines.append("```text")
    lines.append(diag.to_string(index=False) if len(diag) else "No VPD diagnostic produced.")
    lines.append("```")
    lines.append("")
    lines.append("## Selected controls")
    lines.append("")
    lines.append("```text")
    lines.append(selected.to_string(index=False) if len(selected) else "Selected controls table missing.")
    lines.append("```")
    lines.append("")
    lines.append("## Crop summary")
    lines.append("")
    lines.append("```text")
    lines.append(crop.to_string(index=False) if len(crop) else "Crop summary missing.")
    lines.append("```")
    lines.append("")
    lines.append("## Tower land-cover summary")
    lines.append("")
    lines.append("```text")
    lines.append(tower.to_string(index=False) if len(tower) else "Tower summary missing.")
    lines.append("```")
    lines.append("")
    lines.append("## Tower agreement")
    lines.append("")
    lines.append("Strict ET ranking:")
    lines.append("```text")
    lines.append(strict.to_string(index=False) if len(strict) else "Strict tower ranking missing.")
    lines.append("```")
    lines.append("")
    lines.append("Sensitivity ET ranking:")
    lines.append("```text")
    lines.append(sens.to_string(index=False) if len(sens) else "Sensitivity tower ranking missing.")
    lines.append("```")
    lines.append("")
    lines.append("## What I should say in the meeting")
    lines.append("")
    lines.append("I should say: The C4 result is real in the benchmark and crop-free/natural-grassland sensitivity checks, but it is not independent of the full climate/environment gradient. Baseline VPD is the decisive control. If we treat baseline VPD as a confounder, the clean independent C4 mechanism does not hold. If we treat baseline VPD as part of the compound-stress pathway, the result becomes a C4-VPD climate-syndrome result rather than a pure C4 trait mechanism.")
    lines.append("")
    lines.append("I should not say: All of the checks passed or the C4 mechanism is locked.")

    note = "\n".join(lines)
    (TXT / "project_HONEST_MEETING_PACKET.md").write_text(note, encoding="utf-8")

    reply = """Hi project,

Sounds good — I reran the checks you listed and will bring the short table/note to the meeting.

The clean audit gives a more nuanced answer: the C4 result remains strong in the benchmark and in the crop-free/natural-grassland sensitivity checks, but it is not fully independent of the climate/environment gradient. In particular, baseline VPD appears to absorb the C4 signal in the full-control model.

So I am not going to call the C4 mechanism fully locked yet. I think the key question for the meeting is whether baseline VPD should be treated as a confounder to control away, or as part of the compound-stress pathway. If it is a confounder, the clean independent C4 mechanism should be narrowed; if it is part of the pathway, the result may be better framed as a C4-VPD climate-syndrome result.

I also cleaned the crop/tower checks, including US-Ne1/2/3, and will keep the tower language conservative: a limited independent anchor suggesting GLEAM is the better ET product for this analysis, not a strong validation of GLEAM.

See you at 1:30.

Best,
Akul
"""
    (TXT / "project_REPLY_HONEST_FINAL.md").write_text(reply, encoding="utf-8")

    print("===== FINAL HONEST DECISION =====")
    print(json.dumps(final_decision, indent=2))
    print("")
    print("===== project CONCERN STATUS TABLE =====")
    print(concern.to_string(index=False))
    print("")
    print("===== KEY C4 RESULT SUMMARY =====")
    print(key.to_string(index=False))
    print("")
    print("===== VPD DIAGNOSTIC =====")
    print(diag.to_string(index=False) if len(diag) else "No VPD diagnostic produced.")
    print("")
    print("===== project MEETING PACKET =====")
    print(note)
    print("")
    print("===== project REPLY =====")
    print(reply)
    print("")
    print("WROTE", TAB / "STAGE1B6AZ_HONEST_project_PACKET_DECISION.json")
    print("WROTE", TAB / "Table_PRODUCT03fr_key_c4_result_summary.csv")
    print("WROTE", TAB / "Table_PRODUCT03fs_vpd_killer_diagnostic.csv")
    print("WROTE", TAB / "Table_PRODUCT03ft_project_concern_status_table.csv")
    print("WROTE", TXT / "project_HONEST_MEETING_PACKET.md")
    print("WROTE", TXT / "project_REPLY_HONEST_FINAL.md")

if __name__ == "__main__":
    main()
