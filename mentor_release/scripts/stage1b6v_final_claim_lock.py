from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6v_final_claim_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PATHS = {
    "product_scope_decision": Path("results/stage1b6o_product_matrix_scope_lock/tables/Table_PRODUCT02cd_product_matrix_scope_decision.csv"),
    "response_table_decision": Path("results/stage1b6p_strict_2x2_response_table/tables/Table_PRODUCT02ch_response_table_decision.csv"),
    "stress_gs_decision": Path("results/stage1b6q2_attach_tower_stress_drivers/tables/Table_PRODUCT02cr_stress_gs_design_decision.csv"),
    "threshold_model_decision": Path("results/stage1b6r_threshold_response_models/tables/Table_PRODUCT02cw_threshold_model_decision.csv"),
    "threshold_class_summary": Path("results/stage1b6r_threshold_response_models/tables/Table_PRODUCT02ct_response_class_summary.csv"),
    "threshold_point_robustness": Path("results/stage1b6r_threshold_response_models/tables/Table_PRODUCT02cv_point_level_response_robustness.csv"),
    "tower_arbitration_decision": Path("results/stage1b6s_tower_arbitration_prep/tables/Table_PRODUCT02cz_tower_arbitration_prep_decision.csv"),
    "tower_arbitration_summary": Path("results/stage1b6s_tower_arbitration_prep/tables/Table_PRODUCT02cy_tower_arbitration_prep_summary.csv"),
    "spatial_decision": Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02dd_spatial_biome_separation_decision.csv"),
    "spatial_site_ranking": Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02dc_site_spatial_limitation_ranking.csv"),
    "spatial_group_scan": Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02da_spatial_biome_group_signal_scan.csv"),
    "spatial_continuous_scan": Path("results/stage1b6t_spatial_biome_separation/tables/Table_PRODUCT02db_spatial_trait_continuous_signal_scan.csv"),
    "trait_spatial_decision": Path("results/stage1b6u_trait_climate_soil_spatial_model/tables/Table_PRODUCT02di_trait_spatial_model_decision.csv"),
    "trait_spatial_model_screen": Path("results/stage1b6u_trait_climate_soil_spatial_model/tables/Table_PRODUCT02df_trait_spatial_model_screen.csv"),
    "trait_spatial_family_summary": Path("results/stage1b6u_trait_climate_soil_spatial_model/tables/Table_PRODUCT02dg_trait_spatial_family_summary.csv"),
}

def read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def first_value(df, col, default=np.nan):
    if len(df) and col in df.columns:
        return df[col].iloc[0]
    return default

loaded = {k: read_csv(p) for k, p in PATHS.items()}

stage_rows = []
for k, p in PATHS.items():
    df = loaded[k]
    row = {
        "artifact": k,
        "path": str(p),
        "exists": p.exists(),
        "n_rows": len(df),
    }
    if "verdict" in df.columns and len(df):
        row["verdict"] = str(df["verdict"].iloc[0])
    if "blocking_next_stage" in df.columns and len(df):
        row["blocking_next_stage"] = bool(df["blocking_next_stage"].iloc[0])
    if "blocking_for_next_analysis" in df.columns and len(df):
        row["blocking_for_next_analysis"] = bool(df["blocking_for_next_analysis"].iloc[0])
    stage_rows.append(row)

stage_status = pd.DataFrame(stage_rows)
stage_status.to_csv(TAB / "Table_PRODUCT02dj_final_stage_status.csv", index=False)

product = loaded["product_scope_decision"]
response = loaded["response_table_decision"]
stress = loaded["stress_gs_decision"]
threshold = loaded["threshold_model_decision"]
tower_sum = loaded["tower_arbitration_summary"]
spatial_dec = loaded["spatial_decision"]
trait_dec = loaded["trait_spatial_decision"]
site_rank = loaded["spatial_site_ranking"]
group_scan = loaded["spatial_group_scan"]
cont_scan = loaded["spatial_continuous_scan"]
top_models = loaded["trait_spatial_model_screen"]

strict_2x2 = bool(first_value(product, "strict_2x2_allowed", False))
pml_3x3_allowed = bool(first_value(product, "strict_3x3_with_pml_allowed", False))
pml_role = str(first_value(product, "pml_role", "UNKNOWN"))

expected_fit_count = int(first_value(threshold, "expected_fit_count", 0)) if len(threshold) else 0
actual_fit_count = int(first_value(threshold, "actual_fit_count", 0)) if len(threshold) else 0
n_ok_fits = int(first_value(threshold, "n_ok_fits", 0)) if len(threshold) else 0

tower_agree = int(first_value(tower_sum, "n_tower_satellite_limitation_agree", 0)) if len(tower_sum) else 0
tower_compared = int(first_value(tower_sum, "n_tower_satellite_limitation_compared", 0)) if len(tower_sum) else 0
tower_agree_frac = float(first_value(tower_sum, "agreement_fraction", np.nan)) if len(tower_sum) else np.nan

best_group_gap = float(first_value(spatial_dec, "best_group_mean_difference", np.nan)) if len(spatial_dec) else np.nan
best_abs_spearman = float(first_value(spatial_dec, "best_abs_spearman", np.nan)) if len(spatial_dec) else np.nan

best_model = str(first_value(trait_dec, "best_model", ""))
best_model_family = str(first_value(trait_dec, "best_model_family", ""))
best_model_loocv = float(first_value(trait_dec, "best_model_loocv_rmse", np.nan)) if len(trait_dec) else np.nan
baseline_loocv = float(first_value(trait_dec, "baseline_loocv_rmse", np.nan)) if len(trait_dec) else np.nan
loocv_gain = float(first_value(trait_dec, "loocv_gain_vs_intercept", np.nan)) if len(trait_dec) else np.nan

top_sites = pd.DataFrame()
if len(site_rank):
    top_sites = site_rank.sort_values("satellite_limitation_mean_fraction", ascending=False).head(8).copy()
    top_sites.to_csv(TAB / "Table_PRODUCT02dm_top_limitation_sites.csv", index=False)

key_numbers = pd.DataFrame([
    {"quantity": "strict_2x2_allowed", "value": strict_2x2, "interpretation": "Primary product matrix is ready."},
    {"quantity": "strict_3x3_with_pml_allowed", "value": pml_3x3_allowed, "interpretation": "False: PML is not strict tower-centered."},
    {"quantity": "pml_role", "value": pml_role, "interpretation": "PML can be coarse sensitivity only."},
    {"quantity": "threshold_expected_fit_count", "value": expected_fit_count, "interpretation": "Total planned strict 2x2 threshold fits."},
    {"quantity": "threshold_actual_fit_count", "value": actual_fit_count, "interpretation": "All planned threshold fits produced."},
    {"quantity": "threshold_ok_fit_count", "value": n_ok_fits, "interpretation": "OK segmented fits."},
    {"quantity": "tower_limitation_agreement", "value": f"{tower_agree}/{tower_compared}", "interpretation": "Tower/satellite agreement is limited, so no universal claim."},
    {"quantity": "tower_limitation_agreement_fraction", "value": tower_agree_frac, "interpretation": "Agreement fraction."},
    {"quantity": "best_spatial_group_gap", "value": best_group_gap, "interpretation": "Exploratory spatial/biome group separation strength."},
    {"quantity": "best_abs_spearman_spatial", "value": best_abs_spearman, "interpretation": "Best continuous spatial association."},
    {"quantity": "best_trait_spatial_model", "value": best_model, "interpretation": "Best tiny-n model."},
    {"quantity": "best_model_family", "value": best_model_family, "interpretation": "Spatial group is best-supported family."},
    {"quantity": "baseline_loocv_rmse", "value": baseline_loocv, "interpretation": "Intercept-only LOOCV RMSE."},
    {"quantity": "best_model_loocv_rmse", "value": best_model_loocv, "interpretation": "Best model LOOCV RMSE."},
    {"quantity": "loocv_gain_vs_intercept", "value": loocv_gain, "interpretation": "Positive gain means model improves over baseline."},
])
key_numbers.to_csv(TAB / "Table_PRODUCT02dk_final_key_numbers.csv", index=False)

safe_claim = (
    "In the final 13-site tower-centered grassland/savanna/open sample, strict 2x2 satellite "
    "product analysis shows heterogeneous WUE/uWUE response to compound atmospheric–soil moisture stress, "
    "not a universal breakdown pattern. Limitation-like responses are concentrated in specific spatial/regional "
    "subsets, especially the US/low-to-mid latitude sites, while several high-latitude or temperate non-US sites "
    "show weak or low limitation. Tower arbitration is mixed rather than fully concordant, so the final claim "
    "should emphasize stratified heterogeneity and product/tower uncertainty. PML is retained only as coarse-grid "
    "sensitivity and is not part of the strict tower-centered product matrix."
)

unsafe_claims = [
    "Do not claim a universal global grassland WUE breakdown.",
    "Do not claim strict 3x3 PML-inclusive product robustness.",
    "Do not claim biome or trait causality from n=13.",
    "Do not claim full tower-satellite agreement; agreement is limited.",
    "Do not use all49 as the primary inference if the final claim is grassland/savanna/open tower-centered.",
]

primary_bullets = [
    "Primary inference: strict 2x2 matrix using MODIS/GOSIF GPP and MODIS/GLEAM ET.",
    "PML role: coarse sensitivity only because spatial mismatch prevents strict tower-centered use.",
    "Response shape: heterogeneous; limitation-like behavior is present but not universal.",
    "Tower arbitration: mixed agreement, requiring uncertainty/disagreement to be reported.",
    "Spatial/biome separation: exploratory signal present; spatial-group terms best explain limitation heterogeneity.",
    "Mechanism framing: spatial/biome stratification and trait/climate/soil context, not causal proof.",
]

claim_lock = pd.DataFrame([
    {"claim_type": "safe_primary_claim", "claim": safe_claim},
    *[{"claim_type": "required_qualifier", "claim": x} for x in primary_bullets],
    *[{"claim_type": "unsafe_claim_to_avoid", "claim": x} for x in unsafe_claims],
])
claim_lock.to_csv(TAB / "Table_PRODUCT02dl_final_claim_lock.csv", index=False)

# Export compact site table.
compact_cols = [
    "point_id",
    "satellite_limitation_mean_fraction",
    "satellite_strength_class_recomputed",
    "tower_response_class",
    "satellite_response_class",
    "tower_satellite_limitation_agreement_strict2x2",
    "broad_region_handbuilt",
    "us_vs_nonus",
    "latitude_band_handbuilt",
    "longitude_sector_handbuilt",
]
compact_cols = [c for c in compact_cols if c in site_rank.columns]
if len(site_rank):
    compact = site_rank[compact_cols].copy().sort_values("satellite_limitation_mean_fraction", ascending=False)
else:
    compact = pd.DataFrame(columns=compact_cols)
compact.to_csv(TAB / "Table_PRODUCT02dn_final_site_ranking_compact.csv", index=False)

# Optional figures.
try:
    import matplotlib.pyplot as plt

    if len(compact):
        fig_df = compact.sort_values("satellite_limitation_mean_fraction", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(fig_df["point_id"], fig_df["satellite_limitation_mean_fraction"])
        plt.xlabel("Satellite limitation-like fraction")
        plt.ylabel("Site")
        plt.title("Strict 2x2 limitation heterogeneity by site")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02a_site_limitation_ranking.png", dpi=200)
        plt.close()

    if len(group_scan):
        g = group_scan.head(6).copy()
        labels = g["group_col"].astype(str)
        vals = pd.to_numeric(g["mean_difference_high_minus_low"], errors="coerce")
        plt.figure(figsize=(8, 4))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Mean limitation fraction gap")
        plt.ylabel("Grouping variable")
        plt.title("Exploratory spatial/biome separation strength")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02b_group_signal_scan.png", dpi=200)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

final_ready = (
    strict_2x2
    and not pml_3x3_allowed
    and actual_fit_count == expected_fit_count
    and expected_fit_count > 0
    and n_ok_fits > 0
    and len(spatial_dec) > 0
    and len(trait_dec) > 0
)

if final_ready:
    verdict = "FINAL_MENTOR_READY_CLAIM_LOCK_COMPLETE"
    blocking_next = False
else:
    verdict = "FINAL_CLAIM_LOCK_NEEDS_REVIEW"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "verdict": verdict,
    "blocking_next_stage": blocking_next,
    "strict_2x2_primary": strict_2x2,
    "pml_strict_3x3_allowed": pml_3x3_allowed,
    "threshold_models_complete": actual_fit_count == expected_fit_count and expected_fit_count > 0,
    "tower_agreement_fraction": tower_agree_frac,
    "spatial_heterogeneity_signal": best_group_gap >= 0.25 if np.isfinite(best_group_gap) else False,
    "best_model_family": best_model_family,
    "figure_status": figure_status,
    "next_stage": "WRITE_FINAL_RESULTS_MEMO_OR_MANUSCRIPT_SECTION",
}])
decision.to_csv(TAB / "Table_PRODUCT02do_final_claim_lock_decision.csv", index=False)

report = []
report.append("# Stage 1B.6V final claim lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Final safe claim")
report.append("")
report.append(safe_claim)
report.append("")
report.append("## Key numbers")
report.append("")
report.append("```text")
report.append(key_numbers.to_string(index=False))
report.append("```")
report.append("")
report.append("## Final site ranking")
report.append("")
report.append("```text")
report.append(compact.to_string(index=False) if len(compact) else "No compact site ranking available.")
report.append("```")
report.append("")
report.append("## Required qualifiers")
report.append("")
for x in primary_bullets:
    report.append(f"- {x}")
report.append("")
report.append("## Claims to avoid")
report.append("")
for x in unsafe_claims:
    report.append(f"- {x}")
report.append("")
report.append("## Top spatial/biome group scan")
report.append("")
report.append("```text")
report.append(group_scan.head(20).to_string(index=False) if len(group_scan) else "No group scan available.")
report.append("```")
report.append("")
report.append("## Top trait/spatial model screen")
report.append("")
report.append("```text")
report.append(top_models.head(15).to_string(index=False) if len(top_models) else "No model screen available.")
report.append("```")
report.append("")
report.append("## Outputs")
report.append("")
report.append("- Final stage status: `results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02dj_final_stage_status.csv`")
report.append("- Final key numbers: `results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02dk_final_key_numbers.csv`")
report.append("- Final claim lock: `results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02dl_final_claim_lock.csv`")
report.append("- Final compact site ranking: `results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02dn_final_site_ranking_compact.csv`")
report.append("- Final decision: `results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02do_final_claim_lock_decision.csv`")
report.append("- Figures directory: `results/stage1b6v_final_claim_lock/figures/`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("This lock supports a mentor-ready heterogeneous-response claim, not a universal global breakdown or causal biome/trait mechanism claim.")
report.append("")

(TXT / "STAGE1B6V_FINAL_CLAIM_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6V_final_claim_lock",
    "status": verdict,
    "blocking_next_stage": bool(blocking_next),
    "safe_claim": safe_claim,
    "outputs": {
        "stage_status": str(TAB / "Table_PRODUCT02dj_final_stage_status.csv"),
        "key_numbers": str(TAB / "Table_PRODUCT02dk_final_key_numbers.csv"),
        "claim_lock": str(TAB / "Table_PRODUCT02dl_final_claim_lock.csv"),
        "site_ranking": str(TAB / "Table_PRODUCT02dn_final_site_ranking_compact.csv"),
        "decision": str(TAB / "Table_PRODUCT02do_final_claim_lock_decision.csv"),
        "report": str(TXT / "STAGE1B6V_FINAL_CLAIM_LOCK_REPORT.md"),
        "figures": str(FIG),
    }
}
(TAB / "STAGE1B6V_FINAL_CLAIM_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02dj_final_stage_status.csv")
print("WROTE", TAB / "Table_PRODUCT02dk_final_key_numbers.csv")
print("WROTE", TAB / "Table_PRODUCT02dl_final_claim_lock.csv")
print("WROTE", TAB / "Table_PRODUCT02dn_final_site_ranking_compact.csv")
print("WROTE", TAB / "Table_PRODUCT02do_final_claim_lock_decision.csv")
print("WROTE", TXT / "STAGE1B6V_FINAL_CLAIM_LOCK_REPORT.md")
print("WROTE figures to", FIG)
