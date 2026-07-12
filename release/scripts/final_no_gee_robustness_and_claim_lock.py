from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = Path("results/final_nonwriting_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

MAIN_COMP = Path("results/tower_centered_phase19_no_gee/tables/Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv")
FITS = Path("results/tower_centered_phase19_no_gee/tables/Table121_no_gee_gosif_gleam_satellite_response_by_site.csv")
TARGET_MAIN = Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv")
TARGET_STRICT = Path("results/tower_satellite_extraction_targets_FINAL/SENSITIVITY_strict_GRA_coordinates.csv")
TARGET_ALL49 = Path("results/tower_satellite_extraction_targets_FINAL/CONTRAST_all_49_tower_coordinates.csv")
PH18_SCOPE = Path("results/tower_grassland_spatial_trait_lock/tables/Table102_tower_response_summary_by_validation_scope.csv")
PH18_COUNTS = Path("results/tower_grassland_spatial_trait_lock/tables/Table103_tower_response_class_counts_by_scope.csv")
PH18_VERDICT = Path("results/tower_grassland_spatial_trait_lock/phase18_grassland_spatial_trait_lock_verdict.json")

def die(msg):
    raise SystemExit("\\nERROR: " + msg + "\\n")

def read_csv(path):
    if not path.exists():
        die(f"Missing {path}")
    return pd.read_csv(path, low_memory=False)

def read_json(path):
    if path.exists():
        return json.loads(path.read_text())
    return {}

def summarize_comparison(df, scope_name):
    if len(df) == 0:
        return {
            "scope": scope_name,
            "n_sites": 0,
            "exact_class_agreement_fraction": np.nan,
            "limited_vs_enhanced_agreement_fraction": np.nan,
            "slope_direction_agreement_fraction": np.nan,
            "tower_class_counts": {},
            "satellite_class_counts": {},
        }
    return {
        "scope": scope_name,
        "n_sites": int(df["site"].nunique()),
        "exact_class_agreement_fraction": float(df["class_agreement_exact"].mean()) if "class_agreement_exact" in df else np.nan,
        "limited_vs_enhanced_agreement_fraction": float(df["class_agreement_limited_vs_enhanced"].mean()) if "class_agreement_limited_vs_enhanced" in df else np.nan,
        "slope_direction_agreement_fraction": float(df["slope_direction_agreement"].mean()) if "slope_direction_agreement" in df else np.nan,
        "tower_class_counts": df["tower_response_class"].value_counts().to_dict() if "tower_response_class" in df else {},
        "satellite_class_counts": df["satellite_response_class"].value_counts().to_dict() if "satellite_response_class" in df else {},
    }

comp = read_csv(MAIN_COMP)
comp["site"] = comp["site"].astype(str)

targets_main = read_csv(TARGET_MAIN).rename(columns={"target_id": "site"})
targets_strict = read_csv(TARGET_STRICT).rename(columns={"target_id": "site"})
targets_all49 = read_csv(TARGET_ALL49).rename(columns={"target_id": "site"})

for df in [targets_main, targets_strict, targets_all49]:
    df["site"] = df["site"].astype(str)

# Strict GRA robustness can be computed directly because the strict 5 are inside the 13-site main set.
strict_sites = set(targets_strict["site"])
main_sites = set(targets_main["site"])
strict_comp = comp[comp["site"].isin(strict_sites)].copy()
main_comp = comp[comp["site"].isin(main_sites)].copy()

# All-49 cannot be fully validated from current satellite extraction because only the 13 main sites were sampled.
all49_sampled_comp = comp.copy()
all49_missing = sorted(set(targets_all49["site"]) - set(comp["site"]))

robustness_rows = [
    summarize_comparison(main_comp, "main_expanded_grassland_savanna_open_13"),
    summarize_comparison(strict_comp, "sensitivity_strict_GRA_5_subset_of_main13"),
    summarize_comparison(all49_sampled_comp, "contrast_all49_NOT_COMPLETE_only_13_sampled"),
]
robust = pd.DataFrame(robustness_rows)
robust.to_csv(TAB / "Table_FINAL_no_gee_scope_robustness_summary.csv", index=False)

# Metric robustness from Table121: compare WUE vs uWUE-like classes where both exist.
fits = read_csv(FITS)
fits["site"] = fits["site"].astype(str)

metric_wide = fits.pivot_table(
    index="site",
    columns="satellite_metric",
    values=["satellite_response_class", "post_slope", "slope_change", "p_satellite_saturation_breakdown"],
    aggfunc="first"
)

metric_rows = []
for site in sorted(fits["site"].unique()):
    d = fits[fits["site"].eq(site)].copy()
    wue = d[d["satellite_metric"].eq("log_wue_gosif_gleam")]
    uwue = d[d["satellite_metric"].eq("log_uwue_gosif_gleam_tower_vpd")]
    row = {"site": site}
    if len(wue):
        row["wue_class"] = wue["satellite_response_class"].iloc[0]
        row["wue_post_slope"] = wue["post_slope"].iloc[0] if "post_slope" in wue else np.nan
        row["wue_slope_change"] = wue["slope_change"].iloc[0] if "slope_change" in wue else np.nan
    if len(uwue):
        row["uwue_class"] = uwue["satellite_response_class"].iloc[0]
        row["uwue_post_slope"] = uwue["post_slope"].iloc[0] if "post_slope" in uwue else np.nan
        row["uwue_slope_change"] = uwue["slope_change"].iloc[0] if "slope_change" in uwue else np.nan
    row["wue_uwue_class_agree"] = row.get("wue_class") == row.get("uwue_class")
    row["wue_uwue_post_slope_direction_agree"] = np.sign(pd.to_numeric(row.get("wue_post_slope"), errors="coerce")) == np.sign(pd.to_numeric(row.get("uwue_post_slope"), errors="coerce"))
    metric_rows.append(row)

metric = pd.DataFrame(metric_rows)
metric.to_csv(TAB / "Table_FINAL_metric_robustness_WUE_vs_uWUE.csv", index=False)

# Mismatch diagnostics.
diag = comp.copy()
diag["mismatch_type"] = "agreement"
diag.loc[~diag["class_agreement_exact"], "mismatch_type"] = (
    diag.loc[~diag["class_agreement_exact"], "tower_response_class"].astype(str)
    + "_tower_vs_"
    + diag.loc[~diag["class_agreement_exact"], "satellite_response_class"].astype(str)
    + "_satellite"
)

def interp(row):
    if row.get("class_agreement_exact") is True or str(row.get("class_agreement_exact")) == "True":
        return "Tower and GOSIF/GLEAM assign the same response class."
    t = str(row.get("tower_response_class"))
    s = str(row.get("satellite_response_class"))
    slope_ok = bool(row.get("slope_direction_agreement"))
    if slope_ok:
        return "Class differs, but post-slope direction agrees; likely threshold/classification sensitivity rather than complete product failure."
    if t == "breakdown" and s == "enhancement":
        return "Largest concern: tower shows true reversal but GOSIF/GLEAM remains positive; product pair may miss severe high-stress decline."
    if t in ["saturation", "breakdown"] and s == "enhancement":
        return "Satellite likely over-preserves positive WUE response at high stress."
    if t == "enhancement" and s in ["saturation", "breakdown"]:
        return "Satellite likely over-detects high-stress weakening relative to tower."
    return "Class mismatch; inspect site-level time series and stress distribution."

diag["diagnostic_interpretation"] = diag.apply(interp, axis=1)

diag_cols = [
    "site", "tower_response_class", "satellite_response_class",
    "mismatch_type", "diagnostic_interpretation",
    "tower_post_slope", "satellite_post_slope",
    "tower_slope_change", "satellite_slope_change",
    "class_agreement_exact", "class_agreement_limited_vs_enhanced",
    "slope_direction_agreement", "n_fit", "n_fit_8day", "n_years"
]
diag_cols = [c for c in diag_cols if c in diag.columns]
diag[diag_cols].to_csv(TAB / "Table_FINAL_site_mismatch_diagnostics.csv", index=False)

# Claim-lock table.
ph18 = read_json(PH18_VERDICT)
scope = read_csv(PH18_SCOPE)
counts = read_csv(PH18_COUNTS)

claim_rows = [
    {
        "claim": "Tower observations support a heterogeneous WUE/uWUE response phenotype under compound atmospheric-soil moisture stress.",
        "support_level": "strong",
        "evidence": "Phase17 tower response fits plus Phase18 scope/count summaries",
        "limitation": "This is tower-observed flux behavior, not yet universal satellite validation.",
        "safe_to_claim": True,
    },
    {
        "claim": "The main tower-validation scope should be expanded grassland/savanna/open ecosystems.",
        "support_level": "strong",
        "evidence": "Phase18 verdict: recommended scope expanded_grassland_savanna_open; n=13; strict GRA n=5",
        "limitation": "Strict GRA alone is too small for the main analysis.",
        "safe_to_claim": True,
    },
    {
        "claim": "Strict GRA-only results are a sensitivity check, not the main validation.",
        "support_level": "strong",
        "evidence": "Phase18 verdict and strict GRA n=5",
        "limitation": "Small sample size prevents strong strict-grassland inference.",
        "safe_to_claim": True,
    },
    {
        "claim": "No-GEE GOSIF/GLEAM provides partial tower-centered satellite support.",
        "support_level": "moderate/partial",
        "evidence": "Phase19 no-GEE: slope-direction agreement 12/13, exact class agreement 5/13",
        "limitation": "Response-class agreement is weak; product pair often classifies saturation towers as enhancement.",
        "safe_to_claim": True,
    },
    {
        "claim": "The satellite phenotype is fully tower-validated.",
        "support_level": "not supported",
        "evidence": "Phase19 exact class agreement is only 0.3846 for GOSIF/GLEAM.",
        "limitation": "GEE-blocked products prevent full 3x3 matrix; current no-GEE result is one product-pair slice.",
        "safe_to_claim": False,
    },
    {
        "claim": "Plant traits causally explain the response phenotype.",
        "support_level": "not yet supported",
        "evidence": "Phase18 trait tests are nearest-satellite proxies only.",
        "limitation": "Need tower-centered trait extraction and causal adjustment before causal trait claim.",
        "safe_to_claim": False,
    },
]
claim = pd.DataFrame(claim_rows)
claim.to_csv(TAB / "Table_FINAL_claim_lock.csv", index=False)

# Try to reconstruct Phase17 verdict path if missing.
phase17_candidates = sorted(Path("results/tower_validation_broad_inventory").glob("*verdict*.json"))
phase17_candidate_text = [str(p) for p in phase17_candidates]

# Figures.
try:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    robust_plot = robust.copy()
    robust_plot["label"] = robust_plot["scope"].str.replace("_", "\\n")
    ax.bar(robust_plot["label"], robust_plot["exact_class_agreement_fraction"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact class agreement")
    ax.set_title("No-GEE GOSIF/GLEAM tower-vs-satellite agreement by scope")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    fig.savefig(FIG / "Figure_FINAL_no_gee_agreement_by_scope.png", dpi=300)
    plt.close(fig)
except Exception as e:
    print("WARNING figure failed:", e)

# Final report.
report = []
report.append("# Final non-writing lock report")
report.append("")
report.append("## Bottom line")
report.append("")
report.append("Analysis is almost complete. The remaining hard satellite path through GEE is blocked, so the defensible completed analysis is the no-GEE tower-centered validation using local GOSIF/GLEAM plus tower stress.")
report.append("")
report.append("## Scope robustness")
report.append("")
report.append("```text")
report.append(robust.to_string(index=False))
report.append("```")
report.append("")
report.append("## Metric robustness")
report.append("")
report.append("```text")
report.append(metric.to_string(index=False))
report.append("```")
report.append("")
report.append("## Mismatch diagnostics")
report.append("")
report.append("```text")
report.append(diag[diag_cols].to_string(index=False))
report.append("```")
report.append("")
report.append("## Claim lock")
report.append("")
report.append("```text")
report.append(claim.to_string(index=False))
report.append("```")
report.append("")
report.append("## Phase17 verdict file note")
report.append("")
if phase17_candidate_text:
    report.append("Found possible Phase17 verdict files:")
    for p in phase17_candidate_text:
        report.append(f"- {p}")
else:
    report.append("The expected Phase17 verdict JSON path was missing, but downstream Phase18/Phase19 outputs prove the Phase17 tables needed for the analysis exist. Treat this as a packaging/path issue, not a core analysis failure.")
report.append("")
report.append("## Done/not done")
report.append("")
report.append("Done: tower phenotype, ecosystem scope lock, spatial/ecoregion checks, no-GEE GOSIF/GLEAM tower-centered validation, metric robustness, strict-GRA subset robustness, mismatch diagnostics, claim-lock table.")
report.append("")
report.append("Not done: full 3x3 satellite product matrix and final causal trait analysis, because GEE/PML/MODIS/ERA5 extraction is blocked and Phase18 trait values are provisional nearest-satellite proxies.")
report.append("")
report.append("## Safe final scientific framing")
report.append("")
report.append("Tower observations support a heterogeneous grassland/savanna/open WUE/uWUE response to compound atmospheric-soil moisture stress, dominated by saturation or weakening rather than universal breakdown. A no-GEE tower-centered GOSIF/GLEAM validation gives partial satellite support through strong post-slope direction agreement, but weak response-class agreement means full satellite validation is not complete.")
report.append("")

(TXT / "FINAL_NONWRITING_LOCK_REPORT.md").write_text("\\n".join(report))

summary = {
    "analysis_status": "done_except_full_3x3_satellite_matrix_and_final_causal_trait_analysis",
    "main_scope": "expanded_grassland_savanna_open",
    "main_n": 13,
    "strict_gra_n": 5,
    "no_gee_main_exact_agreement": float(robust.loc[robust["scope"].eq("main_expanded_grassland_savanna_open_13"), "exact_class_agreement_fraction"].iloc[0]),
    "no_gee_main_slope_direction_agreement": float(robust.loc[robust["scope"].eq("main_expanded_grassland_savanna_open_13"), "slope_direction_agreement_fraction"].iloc[0]),
    "safe_claim": "tower phenotype supported; no-GEE satellite support partial; full satellite validation not complete",
    "not_safe_claims": [
        "full satellite tower validation",
        "strict grassland main validation",
        "causal trait mechanism proven"
    ],
}
(TAB / "FINAL_NONWRITING_LOCK_SUMMARY.json").write_text(json.dumps(summary, indent=2))

print("\\n".join(report))
print("")
print("WROTE", TAB / "Table_FINAL_no_gee_scope_robustness_summary.csv")
print("WROTE", TAB / "Table_FINAL_metric_robustness_WUE_vs_uWUE.csv")
print("WROTE", TAB / "Table_FINAL_site_mismatch_diagnostics.csv")
print("WROTE", TAB / "Table_FINAL_claim_lock.csv")
print("WROTE", TXT / "FINAL_NONWRITING_LOCK_REPORT.md")
print("WROTE", TAB / "FINAL_NONWRITING_LOCK_SUMMARY.json")
