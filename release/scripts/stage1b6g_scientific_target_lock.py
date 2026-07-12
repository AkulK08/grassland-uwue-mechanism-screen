from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

OUT = Path("results/stage1b6g_scientific_target_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
REQ = Path("data/raw_local/no_gee_point_requests")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
REQ.mkdir(parents=True, exist_ok=True)

FILES = {
    "primary_main13": Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv"),
    "strict_gra5": Path("results/tower_satellite_extraction_targets_FINAL/SENSITIVITY_strict_GRA_coordinates.csv"),
    "all49_contrast": Path("results/tower_satellite_extraction_targets_FINAL/CONTRAST_all_49_tower_coordinates.csv"),
    "phase18_annotation": Path("results/tower_grassland_spatial_trait_lock/tables/Table101_tower_landcover_spatial_trait_annotation.csv"),
    "phase19_compare": Path("results/tower_centered_phase19_no_gee/tables/Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv"),
    "phase18_summary": Path("results/tower_grassland_spatial_trait_lock/tables/Table102_tower_response_summary_by_validation_scope.csv"),
}

def read_csv_safe(p):
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    for c in df.columns:
        cl = c.lower()
        for name in names:
            if name.lower() in cl:
                return c
    return None

def standardize_points(df, scope):
    if df.empty:
        return pd.DataFrame(columns=["target_id", "latitude", "longitude", "scope"])
    idc = find_col(df, ["target_id", "tower_id", "site_id", "site", "id", "point_id"])
    latc = find_col(df, ["latitude", "lat"])
    lonc = find_col(df, ["longitude", "lon", "long"])
    if latc is None or lonc is None:
        return pd.DataFrame(columns=["target_id", "latitude", "longitude", "scope"])
    if idc is None:
        df = df.copy()
        df["target_id"] = [f"{scope}_{i:03d}" for i in range(len(df))]
        idc = "target_id"
    out = df[[idc, latc, lonc]].rename(columns={idc:"target_id", latc:"latitude", lonc:"longitude"})
    out["scope"] = scope
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).drop_duplicates()
    return out

primary = standardize_points(read_csv_safe(FILES["primary_main13"]), "primary_expanded_grassland_savanna_open")
strict = standardize_points(read_csv_safe(FILES["strict_gra5"]), "sensitivity_strict_GRA")
contrast = standardize_points(read_csv_safe(FILES["all49_contrast"]), "contrast_all49_not_primary")

primary["analysis_role"] = "PRIMARY_DO_NOT_SELECT_ON_EFFECT_SIZE"
strict["analysis_role"] = "SENSITIVITY_STRICT_GRASSLAND"
contrast["analysis_role"] = "CONTRAST_ONLY_NOT_PRIMARY"

# Build strong-effect/mismatch diagnostic set from existing phase19 comparison if possible.
diag_candidates = []

phase19 = read_csv_safe(FILES["phase19_compare"])
all_coords = pd.concat([primary, strict, contrast], ignore_index=True).drop_duplicates("target_id")

if not phase19.empty:
    idc = find_col(phase19, ["target_id", "tower_id", "site_id", "site"])
    if idc is not None:
        phase19 = phase19.rename(columns={idc: "target_id"})
        metric_cols = []
        for c in phase19.columns:
            cl = c.lower()
            if any(k in cl for k in ["slope_change", "post_slope", "agreement", "mismatch", "class", "breakdown", "saturation"]):
                metric_cols.append(c)
        sub = phase19[["target_id"] + metric_cols].copy()
        text = sub.astype(str).agg(" ".join, axis=1).str.lower()

        numeric_score = pd.Series(0.0, index=sub.index)
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce")
            if vals.notna().any():
                numeric_score = np.maximum(numeric_score, vals.abs().fillna(0.0))

        sub["diagnostic_reason"] = ""
        sub.loc[text.str.contains("breakdown", regex=False), "diagnostic_reason"] += "breakdown;"
        sub.loc[text.str.contains("mismatch|disagree|false", regex=True), "diagnostic_reason"] += "product_mismatch;"
        sub.loc[text.str.contains("saturation", regex=False), "diagnostic_reason"] += "saturation;"
        sub.loc[numeric_score >= numeric_score.quantile(0.75), "diagnostic_reason"] += "top_quartile_abs_effect_metric;"

        diag_ids = sub[sub["diagnostic_reason"].ne("")][["target_id", "diagnostic_reason"]].drop_duplicates()
        diag = all_coords.merge(diag_ids, on="target_id", how="inner")
        if len(diag):
            diag_candidates.append(diag)

if diag_candidates:
    diagnostic = pd.concat(diag_candidates, ignore_index=True).drop_duplicates("target_id")
else:
    diagnostic = primary.copy()
    diagnostic["diagnostic_reason"] = "fallback_primary_sites_until_response_table_detected"

diagnostic["scope"] = "diagnostic_strong_effect_or_mismatch"
diagnostic["analysis_role"] = "DIAGNOSTIC_NOT_PRIMARY_INFERENCE"

# Correct strict target hierarchy.
primary_request = primary[["target_id", "latitude", "longitude"]].rename(columns={"target_id":"id"})
strict_request = strict[["target_id", "latitude", "longitude"]].rename(columns={"target_id":"id"})
diagnostic_request = diagnostic[["target_id", "latitude", "longitude"]].rename(columns={"target_id":"id"})

# For product request, use primary + strict + diagnostic, not full all49.
product_request = pd.concat([
    primary_request.assign(request_role="primary"),
    strict_request.assign(request_role="strict_GRA_sensitivity"),
    diagnostic_request.assign(request_role="diagnostic_strong_effect_or_mismatch"),
], ignore_index=True).drop_duplicates(subset=["id", "latitude", "longitude"])

# Save tables.
primary.to_csv(TAB / "Table_PRODUCT02am_PRIMARY_expanded_grassland_savanna_open_points.csv", index=False)
strict.to_csv(TAB / "Table_PRODUCT02an_SENSITIVITY_strict_GRA_points.csv", index=False)
diagnostic.to_csv(TAB / "Table_PRODUCT02ao_DIAGNOSTIC_strong_effect_or_mismatch_points.csv", index=False)
contrast.to_csv(TAB / "Table_PRODUCT02ap_CONTRAST_all49_not_primary_points.csv", index=False)
product_request.to_csv(TAB / "Table_PRODUCT02aq_FINAL_no_gee_product_point_request.csv", index=False)

primary_request.to_csv(REQ / "PRIMARY_expanded_grassland_savanna_open_points_for_appeears.csv", index=False)
strict_request.to_csv(REQ / "SENSITIVITY_strict_GRA_points_for_appeears.csv", index=False)
diagnostic_request.to_csv(REQ / "DIAGNOSTIC_strong_effect_or_mismatch_points_for_appeears.csv", index=False)
product_request[["id", "latitude", "longitude"]].to_csv(REQ / "FINAL_STRICT_no_gee_product_points_for_appeears.csv", index=False)

decision = pd.DataFrame([
    {
        "question": "Should primary requests use all49?",
        "answer": "NO",
        "reason": "all49 contains contrast/forest/non-primary sites and is too broad for grassland inference."
    },
    {
        "question": "Should primary requests use only strong-effect sites?",
        "answer": "NO",
        "reason": "Selecting the primary sample on effect size is circular and would bias the claim."
    },
    {
        "question": "What should be requested for strict no-GEE point extraction?",
        "answer": "primary expanded grassland/savanna/open + strict GRA sensitivity + diagnostic strong-effect/mismatch sites",
        "reason": "This satisfies the grassland question while preserving sensitivity and diagnostic checks."
    },
    {
        "question": "What is all49 used for?",
        "answer": "contrast only",
        "reason": "Useful as a non-primary comparison, not the main product request."
    },
])
decision.to_csv(TAB / "Table_PRODUCT02ar_scientific_target_decision.csv", index=False)

counts = pd.DataFrame([
    {"target_set": "primary_expanded_grassland_savanna_open", "n_points": len(primary_request)},
    {"target_set": "sensitivity_strict_GRA", "n_points": len(strict_request)},
    {"target_set": "diagnostic_strong_effect_or_mismatch", "n_points": len(diagnostic_request)},
    {"target_set": "final_product_request_unique", "n_points": len(product_request)},
    {"target_set": "contrast_all49_not_primary", "n_points": len(contrast)},
])
counts.to_csv(TAB / "Table_PRODUCT02as_target_set_counts.csv", index=False)

report = []
report.append("# Stage 1B.6G scientific target lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("Primary inference should use grassland/savanna/open eligible sites, not all49 and not only strong-effect sites. Strong-effect/mismatch sites are diagnostic/prioritization targets only.")
report.append("")
report.append("## Target set counts")
report.append("")
report.append("```text")
report.append(counts.to_string(index=False))
report.append("```")
report.append("")
report.append("## Scientific decision table")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Final no-GEE product point request")
report.append("")
report.append("```text")
report.append(product_request.to_string(index=False))
report.append("```")
report.append("")
report.append("## Files to use")
report.append("")
report.append("Primary request: data/raw_local/no_gee_point_requests/PRIMARY_expanded_grassland_savanna_open_points_for_appeears.csv")
report.append("")
report.append("Strict GRA sensitivity request: data/raw_local/no_gee_point_requests/SENSITIVITY_strict_GRA_points_for_appeears.csv")
report.append("")
report.append("Diagnostic request: data/raw_local/no_gee_point_requests/DIAGNOSTIC_strong_effect_or_mismatch_points_for_appeears.csv")
report.append("")
report.append("Final combined request: data/raw_local/no_gee_point_requests/FINAL_STRICT_no_gee_product_points_for_appeears.csv")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Use the final combined request for missing MODIS/MCD point extractions. Do not use all49 as the primary request. Do not select the primary sample only by strong effect.")
report.append("")

(TXT / "STAGE1B6G_SCIENTIFIC_TARGET_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6G_scientific_target_lock",
    "status": "complete",
    "n_primary": int(len(primary_request)),
    "n_strict_gra": int(len(strict_request)),
    "n_diagnostic": int(len(diagnostic_request)),
    "n_final_unique_request": int(len(product_request)),
    "outputs": {
        "primary": str(TAB / "Table_PRODUCT02am_PRIMARY_expanded_grassland_savanna_open_points.csv"),
        "strict_gra": str(TAB / "Table_PRODUCT02an_SENSITIVITY_strict_GRA_points.csv"),
        "diagnostic": str(TAB / "Table_PRODUCT02ao_DIAGNOSTIC_strong_effect_or_mismatch_points.csv"),
        "contrast": str(TAB / "Table_PRODUCT02ap_CONTRAST_all49_not_primary_points.csv"),
        "final_request": str(REQ / "FINAL_STRICT_no_gee_product_points_for_appeears.csv"),
        "decision": str(TAB / "Table_PRODUCT02ar_scientific_target_decision.csv"),
        "report": str(TXT / "STAGE1B6G_SCIENTIFIC_TARGET_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6G_SCIENTIFIC_TARGET_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02am_PRIMARY_expanded_grassland_savanna_open_points.csv")
print("WROTE", TAB / "Table_PRODUCT02an_SENSITIVITY_strict_GRA_points.csv")
print("WROTE", TAB / "Table_PRODUCT02ao_DIAGNOSTIC_strong_effect_or_mismatch_points.csv")
print("WROTE", TAB / "Table_PRODUCT02ap_CONTRAST_all49_not_primary_points.csv")
print("WROTE", TAB / "Table_PRODUCT02aq_FINAL_no_gee_product_point_request.csv")
print("WROTE", REQ / "FINAL_STRICT_no_gee_product_points_for_appeears.csv")
print("WROTE", TAB / "Table_PRODUCT02ar_scientific_target_decision.csv")
print("WROTE", TXT / "STAGE1B6G_SCIENTIFIC_TARGET_LOCK_REPORT.md")
