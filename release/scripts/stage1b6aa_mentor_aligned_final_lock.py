from pathlib import Path
from datetime import datetime
import pandas as pd
import json

OUT = Path("results/stage1b6aa_mentor_aligned_final_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

FILES = {
    "final13_response_lock": Path("results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02do_final_claim_lock_decision.csv"),
    "final13_site_ranking": Path("results/stage1b6v_final_claim_lock/tables/Table_PRODUCT02dn_final_site_ranking_compact.csv"),
    "core_trait_decision": Path("results/stage1b6z_core_plant_trait_mechanism/tables/Table_PRODUCT02ew_core_trait_mechanism_decision.csv"),
    "core_trait_scan": Path("results/stage1b6z_core_plant_trait_mechanism/tables/Table_PRODUCT02et_core_trait_effect_scan.csv"),
    "core_trait_interactions": Path("results/stage1b6z_core_plant_trait_mechanism/tables/Table_PRODUCT02eu_core_trait_environment_interaction_scan.csv"),
    "core_trait_inventory": Path("results/stage1b6z_core_plant_trait_mechanism/tables/Table_PRODUCT02es_core_trait_dataset_inventory.csv"),
}

def read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

loaded = {k: read(v) for k, v in FILES.items()}

status = pd.DataFrame([
    {"artifact": k, "path": str(p), "exists": p.exists(), "n_rows": len(loaded[k])}
    for k, p in FILES.items()
])
status.to_csv(TAB / "Table_PRODUCT02ex_mentor_aligned_artifact_status.csv", index=False)

response_lock = loaded["final13_response_lock"]
trait_decision = loaded["core_trait_decision"]
trait_scan = loaded["core_trait_scan"]
trait_inventory = loaded["core_trait_inventory"]

response_ready = (
    len(response_lock)
    and str(response_lock["verdict"].iloc[0]) == "FINAL_MENTOR_READY_CLAIM_LOCK_COMPLETE"
)

trait_ready = (
    len(trait_decision)
    and str(trait_decision["verdict"].iloc[0]) == "STRONG_CORE_PLANT_TRAIT_EFFECT_FOUND"
)

top_trait = trait_scan.iloc[0].to_dict() if len(trait_scan) else {}

safe_claim = (
    "Grassland ecosystem WUE/uWUE responses to compound atmospheric–soil moisture stress are heterogeneous rather "
    "than universally degrading. In the strict final-13 tower-centered product analysis, response-shape variation is "
    "spatially structured and product-sensitive, with mixed tower-satellite agreement requiring uncertainty to remain "
    "part of the interpretation. In the expanded plant-trait mechanism phase, effective rooting depth emerges as the "
    "strongest core trait predictor of the latent post-stress response slope, especially within north-midlatitude and "
    "temperate grassland-relevant systems. This supports a hydraulic-access mechanism in which rooting-zone storage "
    "modulates whether WUE/uWUE responses are maintained or weakened under compound atmospheric–soil moisture stress. "
    "The trait result is observational and expanded-screen based, not direct causal proof from the strict final-13 towers."
)

mentor_pathway = pd.DataFrame([
    {
        "stage": "Gate 1",
        "claim": "Response-shape classification distinguishes enhancement, saturation, weakening, and breakdown.",
        "status": "complete",
    },
    {
        "stage": "Gate 2",
        "claim": "Strict primary product matrix is MODIS/GOSIF GPP × MODIS/GLEAM ET; PML is coarse sensitivity only.",
        "status": "complete_with_PML_limitation",
    },
    {
        "stage": "Gate 3",
        "claim": "Tower arbitration is mixed/partial, so product uncertainty remains explicit.",
        "status": "complete_with_uncertainty",
    },
    {
        "stage": "Phase 4",
        "claim": "Rooting depth is the strongest core plant-trait predictor of latent post-stress response slope in expanded trait-covered data.",
        "status": "complete_as_observational_trait_mechanism",
    },
])
mentor_pathway.to_csv(TAB / "Table_PRODUCT02ey_mentor_pathway_claims.csv", index=False)

top_trait_table = pd.DataFrame([{
    "dataset": top_trait.get("dataset", ""),
    "trait": top_trait.get("trait", ""),
    "outcome": top_trait.get("outcome", ""),
    "environment": top_trait.get("environment", ""),
    "n": top_trait.get("n", ""),
    "effect_range": top_trait.get("effect_range", ""),
    "spearman_r": top_trait.get("spearman_r", ""),
    "perm_p_spearman": top_trait.get("perm_p_spearman", ""),
    "loo_sign_stability": top_trait.get("loo_sign_stability", ""),
    "claim_strength": top_trait.get("claim_strength", ""),
}])
top_trait_table.to_csv(TAB / "Table_PRODUCT02ez_top_trait_mechanism_effect.csv", index=False)

claims_to_avoid = pd.DataFrame([
    {"avoid": "Do not claim universal global grassland WUE breakdown."},
    {"avoid": "Do not claim strict 3×3 PML robustness."},
    {"avoid": "Do not claim Great Plains trait proof; the clean trait result is north-midlatitude / temperate grassland-relevant rooting depth."},
    {"avoid": "Do not claim isohydricity as a main result; coverage is too weak."},
    {"avoid": "Do not claim direct causality; frame as observational mechanism evidence."},
    {"avoid": "Do not imply the trait result comes from the strict final-13 tower subset."},
])
claims_to_avoid.to_csv(TAB / "Table_PRODUCT02fa_claims_to_avoid.csv", index=False)

verdict = "MENTOR_ALIGNED_FINAL_TRAIT_BASED_STORY_READY" if response_ready and trait_ready else "MENTOR_ALIGNED_LOCK_NEEDS_REVIEW"

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "response_ready": bool(response_ready),
    "trait_ready": bool(trait_ready),
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_RESULTS_DISCUSSION_AND_MENTOR_MEMO",
}])
decision.to_csv(TAB / "Table_PRODUCT02fb_mentor_aligned_final_decision.csv", index=False)

report = []
report.append("# Stage 1B.6AA mentor-aligned final trait-based lock")
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
report.append("## Mentor pathway")
report.append("")
report.append("```text")
report.append(mentor_pathway.to_string(index=False))
report.append("```")
report.append("")
report.append("## Top core plant-trait mechanism effect")
report.append("")
report.append("```text")
report.append(top_trait_table.to_string(index=False))
report.append("```")
report.append("")
report.append("## Claims to avoid")
report.append("")
report.append("```text")
report.append(claims_to_avoid.to_string(index=False))
report.append("```")
report.append("")
report.append("## Manuscript-ready wording")
report.append("")
report.append(
    "We first classified ecosystem WUE/uWUE responses to compound atmospheric–soil moisture stress across a strict "
    "tower-centered satellite product matrix. The final-13 analysis did not support a universal breakdown response; "
    "instead, response shape was heterogeneous, product-sensitive, and only partially aligned with tower-derived response "
    "classes. We therefore treated trait attribution as a separate conditional mechanism phase using the expanded "
    "trait-covered point dataset. In that phase, effective rooting depth was the strongest core plant-trait predictor "
    "of the latent post-stress response slope, with the clearest signal in north-midlatitude and temperate grassland-relevant "
    "systems. These results support a hydraulic-access interpretation of compound-stress WUE response while retaining "
    "the observational and product-uncertainty limitations of the analysis."
)
report.append("")

(TXT / "STAGE1B6AA_MENTOR_ALIGNED_FINAL_TRAIT_LOCK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6AA_mentor_aligned_final_trait_lock",
    "status": verdict,
    "safe_claim": safe_claim,
    "outputs": {
        "artifact_status": str(TAB / "Table_PRODUCT02ex_mentor_aligned_artifact_status.csv"),
        "mentor_pathway": str(TAB / "Table_PRODUCT02ey_mentor_pathway_claims.csv"),
        "top_trait": str(TAB / "Table_PRODUCT02ez_top_trait_mechanism_effect.csv"),
        "claims_to_avoid": str(TAB / "Table_PRODUCT02fa_claims_to_avoid.csv"),
        "decision": str(TAB / "Table_PRODUCT02fb_mentor_aligned_final_decision.csv"),
        "report": str(TXT / "STAGE1B6AA_MENTOR_ALIGNED_FINAL_TRAIT_LOCK_REPORT.md"),
    },
}
(TAB / "STAGE1B6AA_MENTOR_ALIGNED_FINAL_TRAIT_LOCK_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", TAB / "Table_PRODUCT02ex_mentor_aligned_artifact_status.csv")
print("WROTE", TAB / "Table_PRODUCT02ey_mentor_pathway_claims.csv")
print("WROTE", TAB / "Table_PRODUCT02ez_top_trait_mechanism_effect.csv")
print("WROTE", TAB / "Table_PRODUCT02fa_claims_to_avoid.csv")
print("WROTE", TAB / "Table_PRODUCT02fb_mentor_aligned_final_decision.csv")
print("WROTE", TXT / "STAGE1B6AA_MENTOR_ALIGNED_FINAL_TRAIT_LOCK_REPORT.md")
