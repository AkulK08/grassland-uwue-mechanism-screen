from pathlib import Path
from datetime import datetime
import json
import re
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

OUT = Path("results/stage1b6ag_integrated_nature_proof_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

ROOT = Path(".")
RESULTS = Path("results")

def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def exists(p):
    return Path(p).exists()

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def num(x):
    return pd.to_numeric(x, errors="coerce")

def bool_pass(x):
    return "PASS" if bool(x) else "NOT_YET"

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def scan_files():
    files = []
    for ext in ["*.csv", "*.json", "*.md", "*.txt", "*.log"]:
        files.extend(RESULTS.rglob(ext))
    return sorted(files)

all_files = scan_files()

# ---------------------------------------------------------------------
# Known important outputs from recent stages.
# ---------------------------------------------------------------------
viability_decision_p = first_existing([
    "results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fv_nature_level_viability_decision.csv",
])

viability_traits_p = first_existing([
    "results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02ft_top_named_regime_trait_results.csv",
])

viability_pillars_p = first_existing([
    "results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fu_prior_evidence_pillars.csv",
])

viability_claims_p = first_existing([
    "results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fw_writing_claim_numbers.csv",
])

named_ad_decision_p = first_existing([
    "results/stage1b6ad_regional_hotspot_trait_atlas_NAMEDPERM500/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv",
    "results/stage1b6ad_regional_hotspot_trait_atlas_NAMED_ALLCONTROLS_PERM5000/tables/Table_PRODUCT02fn_regional_hotspot_trait_decision.csv",
])

point_table_p = first_existing([
    "results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv",
])

# ---------------------------------------------------------------------
# Generic file evidence scan: product robustness, tower validation, bootstrap,
# SMAP/soil moisture validation, trait, regime.
# ---------------------------------------------------------------------
patterns = {
    "product_matrix_or_cross_product": [
        "product_matrix", "cross_product", "gpp", "et_product", "modis", "pml", "gleam", "gosif"
    ],
    "bootstrap_or_permutation": [
        "bootstrap", "boot", "perm", "permutation", "q_spearman", "bh_q"
    ],
    "tower_validation": [
        "tower", "ameriflux", "fluxnet", "icos", "ozflux", "site_validation", "satellite_tower"
    ],
    "smap_soil_moisture_validation": [
        "smap", "soil_moisture_validation", "era5", "matched_points"
    ],
    "trait_mechanism": [
        "trait", "rooting_depth", "p50", "psi50", "isohydricity"
    ],
    "regime_or_threshold": [
        "regime", "threshold", "satbreak", "saturation", "latent_post_slope", "latent_slope_change"
    ],
}

inventory_rows = []
for p in tqdm(all_files, desc="Scanning result files"):
    s = str(p).lower()
    matched = []
    for pillar, pats in patterns.items():
        if any(x in s for x in pats):
            matched.append(pillar)

    if not matched:
        continue

    size_kb = p.stat().st_size / 1024 if p.exists() else np.nan
    inventory_rows.append({
        "file": str(p),
        "name": p.name,
        "size_kb": round(size_kb, 2),
        "matched_pillars": ";".join(matched),
    })

inventory = pd.DataFrame(inventory_rows)
inventory.to_csv(TAB / "Table_PRODUCT02fx_available_evidence_file_inventory.csv", index=False)

# ---------------------------------------------------------------------
# Read viability outputs.
# ---------------------------------------------------------------------
viability_decision = read_csv_safe(viability_decision_p) if viability_decision_p else None
viability_traits = read_csv_safe(viability_traits_p) if viability_traits_p else None
viability_pillars = read_csv_safe(viability_pillars_p) if viability_pillars_p else None
viability_claims = read_csv_safe(viability_claims_p) if viability_claims_p else None
named_ad_decision = read_csv_safe(named_ad_decision_p) if named_ad_decision_p else None
point_table = read_csv_safe(point_table_p) if point_table_p else None

# ---------------------------------------------------------------------
# Extract key numbers.
# ---------------------------------------------------------------------
numbers = {}

if viability_decision is not None and len(viability_decision):
    r = viability_decision.iloc[0]
    for c in viability_decision.columns:
        numbers[f"viability_{c}"] = r[c]

if named_ad_decision is not None and len(named_ad_decision):
    r = named_ad_decision.iloc[0]
    for c in named_ad_decision.columns:
        numbers[f"regional_AD_{c}"] = r[c]

if point_table is not None:
    numbers["point_table_n_points"] = len(point_table)
    for c in ["p_threshold_like", "latent_post_slope", "latent_slope_change"]:
        if c in point_table.columns:
            numbers[f"point_table_{c}_nonmissing"] = int(num(point_table[c]).notna().sum())
            numbers[f"point_table_{c}_median"] = float(num(point_table[c]).median())

if viability_traits is not None and len(viability_traits):
    vt = viability_traits.copy()
    for c in ["n_test", "trait_r2_on_residual", "spearman_r", "perm_p", "bh_q_all_named_tests", "loo_sign_stability"]:
        if c in vt.columns:
            vt[c] = num(vt[c])

    numbers["named_regime_trait_tests_n"] = len(vt)
    if "passes_case_named" in vt.columns:
        numbers["named_regime_case_passes"] = int(vt["passes_case_named"].astype(str).str.lower().eq("true").sum())
    if "passes_main_named" in vt.columns:
        numbers["named_regime_main_passes"] = int(vt["passes_main_named"].astype(str).str.lower().eq("true").sum())
    if "passes_full_control" in vt.columns:
        numbers["named_regime_full_control_passes"] = int(vt["passes_full_control"].astype(str).str.lower().eq("true").sum())

    if "bh_q_all_named_tests" in vt.columns:
        best = vt.sort_values(["bh_q_all_named_tests", "perm_p", "trait_r2_on_residual"], ascending=[True, True, False]).head(1)
        if len(best):
            b = best.iloc[0]
            for c in best.columns:
                numbers[f"best_named_trait_{c}"] = b[c]

# ---------------------------------------------------------------------
# Evidence pillar scoring.
# ---------------------------------------------------------------------
def inventory_has(pillar):
    if inventory.empty:
        return False
    return inventory["matched_pillars"].fillna("").str.contains(pillar, regex=False).any()

n_main = int(safe_float(numbers.get("viability_n_main_named_passes", numbers.get("named_regime_main_passes", 0))) or 0)
n_case = int(safe_float(numbers.get("viability_n_controlled_case_passes", numbers.get("named_regime_case_passes", 0))) or 0)
n_full = int(safe_float(numbers.get("viability_n_full_climate_soil_lai_passes", numbers.get("named_regime_full_control_passes", 0))) or 0)

best_q = safe_float(numbers.get("best_named_trait_bh_q_all_named_tests", np.nan))
best_p = safe_float(numbers.get("best_named_trait_perm_p", np.nan))
best_r2 = safe_float(numbers.get("best_named_trait_trait_r2_on_residual", np.nan))
best_rho = safe_float(numbers.get("best_named_trait_spearman_r", np.nan))
best_n = safe_float(numbers.get("best_named_trait_n_test", np.nan))

product_files = inventory[inventory["matched_pillars"].str.contains("product_matrix_or_cross_product", na=False)] if not inventory.empty else pd.DataFrame()
tower_files = inventory[inventory["matched_pillars"].str.contains("tower_validation", na=False)] if not inventory.empty else pd.DataFrame()
bootstrap_files = inventory[inventory["matched_pillars"].str.contains("bootstrap_or_permutation", na=False)] if not inventory.empty else pd.DataFrame()
smap_files = inventory[inventory["matched_pillars"].str.contains("smap_soil_moisture_validation", na=False)] if not inventory.empty else pd.DataFrame()

pillars = []

pillars.append({
    "pillar": "global_response_regime_evidence",
    "status": bool_pass(point_table is not None and len(point_table) >= 50),
    "strength": "strong" if point_table is not None and len(point_table) >= 100 else "moderate_or_missing",
    "main_number": int(len(point_table)) if point_table is not None else np.nan,
    "evidence_file": str(point_table_p) if point_table_p else "",
    "interpretation": "Point-level response annotation table exists; supports global regime/threshold response analysis." if point_table_p else "Missing point-level response annotation table."
})

pillars.append({
    "pillar": "named_regime_trait_mechanism",
    "status": bool_pass(n_case > 0),
    "strength": "moderate_controlled_case" if n_case > 0 and n_full == 0 else ("strong_full_control" if n_full > 0 else "missing"),
    "main_number": n_case,
    "evidence_file": str(viability_traits_p) if viability_traits_p else "",
    "interpretation": "Controlled named-regime trait cases exist, but full-control universal mechanism is not yet proven." if n_case > 0 and n_full == 0 else ("Full-control passes exist." if n_full > 0 else "No controlled named-regime trait case found.")
})

pillars.append({
    "pillar": "multiple_testing_strength",
    "status": bool_pass(np.isfinite(best_q) and best_q <= 0.10),
    "strength": "strong" if np.isfinite(best_q) and best_q <= 0.10 else ("near_miss" if np.isfinite(best_q) and best_q <= 0.20 else "weak_or_missing"),
    "main_number": best_q,
    "evidence_file": str(viability_traits_p) if viability_traits_p else "",
    "interpretation": "At least one named-regime trait signal passes q<=0.10." if np.isfinite(best_q) and best_q <= 0.10 else "Best named-regime trait q is above 0.10; use cautious language."
})

pillars.append({
    "pillar": "full_climate_soil_lai_mechanism",
    "status": bool_pass(n_full > 0),
    "strength": "strong" if n_full > 0 else "not_yet",
    "main_number": n_full,
    "evidence_file": str(viability_traits_p) if viability_traits_p else "",
    "interpretation": "At least one full climate+soil+LAI trait mechanism passes." if n_full > 0 else "No full climate+soil+LAI trait mechanism pass yet."
})

pillars.append({
    "pillar": "product_matrix_robustness",
    "status": bool_pass(len(product_files) > 0),
    "strength": "inventory_present_not_audited" if len(product_files) > 0 else "missing_or_not_found",
    "main_number": int(len(product_files)),
    "evidence_file": ";".join(product_files["file"].head(5).tolist()) if len(product_files) else "",
    "interpretation": "Product/cross-product files found; still needs explicit pass/fail extraction." if len(product_files) else "No product-matrix evidence found by filename scan."
})

pillars.append({
    "pillar": "tower_validation",
    "status": bool_pass(len(tower_files) > 0),
    "strength": "inventory_present_not_audited" if len(tower_files) > 0 else "missing_or_not_found",
    "main_number": int(len(tower_files)),
    "evidence_file": ";".join(tower_files["file"].head(5).tolist()) if len(tower_files) else "",
    "interpretation": "Tower-related files found; still needs direct regime-aligned validation." if len(tower_files) else "No tower-validation output found by filename scan."
})

pillars.append({
    "pillar": "bootstrap_permutation_robustness",
    "status": bool_pass(len(bootstrap_files) > 0),
    "strength": "present" if len(bootstrap_files) > 0 else "missing_or_not_found",
    "main_number": int(len(bootstrap_files)),
    "evidence_file": ";".join(bootstrap_files["file"].head(5).tolist()) if len(bootstrap_files) else "",
    "interpretation": "Permutation/bootstrap-like files found." if len(bootstrap_files) else "No bootstrap/permutation evidence found by filename scan."
})

pillars.append({
    "pillar": "soil_moisture_validation",
    "status": bool_pass(len(smap_files) > 0),
    "strength": "present" if len(smap_files) > 0 else "missing_or_not_found",
    "main_number": int(len(smap_files)),
    "evidence_file": ";".join(smap_files["file"].head(5).tolist()) if len(smap_files) else "",
    "interpretation": "SMAP/soil-moisture validation-like files found." if len(smap_files) else "No SMAP/soil-moisture validation output found by filename scan."
})

pillar_df = pd.DataFrame(pillars)
pillar_df.to_csv(TAB / "Table_PRODUCT02fy_integrated_proof_pillars.csv", index=False)

# ---------------------------------------------------------------------
# Final claim classification.
# ---------------------------------------------------------------------
has_global = pillar_df.loc[pillar_df["pillar"].eq("global_response_regime_evidence"), "status"].iloc[0] == "PASS"
has_case_trait = n_case > 0
has_main_trait = n_main > 0
has_full_trait = n_full > 0
has_product_inventory = len(product_files) > 0
has_tower_inventory = len(tower_files) > 0
q_pass = np.isfinite(best_q) and best_q <= 0.10
q_near = np.isfinite(best_q) and best_q <= 0.20

if has_global and has_full_trait and has_product_inventory and has_tower_inventory:
    claim_strength = "NATURE_PROOF_STACK_NEAR_COMPLETE"
    verdict = "NATURE_LEVEL_PROOF_PLAUSIBLE_IF_PRODUCT_AND_TOWER_AUDITS_PASS"
    safe_claim = "Compound-stress grassland carbon-water regimes are globally evident and have at least one full-control trait mechanism; product and tower evidence are present and should be audited explicitly."
elif has_global and has_main_trait and q_pass:
    claim_strength = "STRONG_NATURE_STYLE_ASSOCIATIVE_MECHANISM"
    verdict = "NATURE_STYLE_MAIN_RESULT_CANDIDATE_NOT_FULL_CAUSAL_PROOF"
    safe_claim = "Global grassland compound-stress regimes are supported, with named-regime trait associations passing main statistical criteria, but full-control/tower/product proof remains the limiting layer."
elif has_global and has_case_trait and q_near:
    claim_strength = "MODERATE_NATURE_STYLE_CONTROLLED_CASE"
    verdict = "NATURE_DIRECTION_ALIVE_PROOF_STACK_INCOMPLETE"
    safe_claim = "Global grassland compound-stress regimes are supported, and named-regime controlled trait cases are present, but full Nature-level proof remains incomplete without stronger full-control, product, and tower validation."
elif has_global and has_case_trait:
    claim_strength = "WEAK_TO_MODERATE_CASE_EVIDENCE"
    verdict = "HIGH_IMPACT_DIRECTION_ALIVE_MECHANISM_NOT_LOCKED"
    safe_claim = "Global response regimes are supported and trait mechanisms are suggestive, but mechanism strength is not yet proof-grade."
else:
    claim_strength = "EXPLORATORY_ONLY"
    verdict = "NOT_YET_NATURE_STYLE_PROOF"
    safe_claim = "Current outputs do not yet lock a Nature-style proof stack."

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "verdict": verdict,
    "claim_strength": claim_strength,
    "safe_claim": safe_claim,
    "has_global_regime_table": has_global,
    "n_named_controlled_case_passes": n_case,
    "n_named_main_passes": n_main,
    "n_full_climate_soil_lai_passes": n_full,
    "best_named_trait_q": best_q,
    "best_named_trait_p": best_p,
    "best_named_trait_r2": best_r2,
    "best_named_trait_spearman": best_rho,
    "best_named_trait_n": best_n,
    "product_files_found": int(len(product_files)),
    "tower_files_found": int(len(tower_files)),
    "bootstrap_or_permutation_files_found": int(len(bootstrap_files)),
    "smap_validation_files_found": int(len(smap_files)),
    "blocking_next_stage": False,
    "next_stage": "RUN_PRODUCT_TOWER_FULL_CONTROL_PROOF_AUDIT_OR_WRITE_CAUTIONARY_NATURE_STYLE_RESULTS",
}])
decision.to_csv(TAB / "Table_PRODUCT02fz_integrated_nature_proof_decision.csv", index=False)

# ---------------------------------------------------------------------
# Best trait table export.
# ---------------------------------------------------------------------
if viability_traits is not None and len(viability_traits):
    vt = viability_traits.copy()
    for c in ["n_test", "trait_r2_on_residual", "spearman_r", "perm_p", "bh_q_all_named_tests", "loo_sign_stability"]:
        if c in vt.columns:
            vt[c] = num(vt[c])
    sort_cols = [c for c in ["bh_q_all_named_tests", "perm_p", "trait_r2_on_residual"] if c in vt.columns]
    if sort_cols:
        vt = vt.sort_values(sort_cols, ascending=[True, True, False][:len(sort_cols)])
    vt.head(80).to_csv(TAB / "Table_PRODUCT02ga_best_named_trait_mechanism_candidates.csv", index=False)

# ---------------------------------------------------------------------
# Make simple figure.
# ---------------------------------------------------------------------
figure_status = "NO_FIGURE"
try:
    import matplotlib.pyplot as plt
    plot = pillar_df.copy()
    plot["score"] = plot["status"].map({"PASS": 1, "NOT_YET": 0}).fillna(0)
    plt.figure(figsize=(10, 5))
    plt.barh(plot["pillar"][::-1], plot["score"][::-1])
    plt.xlabel("Evidence present/passing")
    plt.title("Integrated Nature proof-lock pillars")
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(FIG / "Figure_PRODUCT02t_integrated_proof_pillars.png", dpi=220)
    plt.close()
    figure_status = "FIGURE_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_FAILED: {repr(e)}"

# ---------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------
report = []
report.append("# Stage 1B.6AG integrated Nature proof-lock")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Final decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Safe claim")
report.append("")
report.append(safe_claim)
report.append("")
report.append("## Evidence pillars")
report.append("")
report.append("```text")
report.append(pillar_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Best named trait mechanism candidates")
report.append("")
if viability_traits is not None and len(viability_traits):
    show = vt.head(30)
    report.append("```text")
    report.append(show.to_string(index=False))
    report.append("```")
else:
    report.append("No named trait table found.")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("- Nature-level direction is alive if global regime evidence exists and named controlled trait cases exist.")
report.append("- Full Nature-level proof requires product robustness and tower validation to be explicitly connected to the same regimes.")
report.append("- Full climate+soil+LAI trait passes would upgrade the mechanism from controlled case evidence to stronger mechanism evidence.")
report.append("- If product/tower files are present only as inventory hits, the next stage should audit their numerical pass/fail status directly.")
report.append("")
report.append(f"Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6AG_INTEGRATED_NATURE_PROOF_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6AG_integrated_nature_proof_lock",
    "status": verdict,
    "claim_strength": claim_strength,
    "safe_claim": safe_claim,
    "outputs": {
        "decision": str(TAB / "Table_PRODUCT02fz_integrated_nature_proof_decision.csv"),
        "pillars": str(TAB / "Table_PRODUCT02fy_integrated_proof_pillars.csv"),
        "inventory": str(TAB / "Table_PRODUCT02fx_available_evidence_file_inventory.csv"),
        "best_traits": str(TAB / "Table_PRODUCT02ga_best_named_trait_mechanism_candidates.csv"),
        "report": str(TXT / "STAGE1B6AG_INTEGRATED_NATURE_PROOF_LOCK_REPORT.md"),
    }
}
(TAB / "STAGE1B6AG_INTEGRATED_NATURE_PROOF_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02fx_available_evidence_file_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT02fy_integrated_proof_pillars.csv")
print("WROTE", TAB / "Table_PRODUCT02fz_integrated_nature_proof_decision.csv")
print("WROTE", TAB / "Table_PRODUCT02ga_best_named_trait_mechanism_candidates.csv")
print("WROTE", TXT / "STAGE1B6AG_INTEGRATED_NATURE_PROOF_LOCK_REPORT.md")
print("WROTE figures to", FIG)
