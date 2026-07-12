from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6s_tower_arbitration_prep")
TAB = OUT / "tables"
TXT = OUT / "text"
DATA = Path("data/processed/stage1b6s")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

FITS = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")

TOWER_CANDIDATES = [
    Path("results/tower_centered_phase19_no_gee/tables/Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv"),
    Path("results/final_nonwriting_lock/files/phase19_tower_satellite_comparison.csv"),
    Path("results/final_nonwriting_lock 2/files/phase19_tower_satellite_comparison.csv"),
]

def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def classify_robust(frac):
    if pd.isna(frac):
        return "unknown"
    if frac >= 0.60:
        return "strong_limitation"
    if frac >= 0.40:
        return "moderate_limitation"
    if frac >= 0.20:
        return "weak_or_mixed_limitation"
    return "low_limitation"

if not FITS.exists():
    raise FileNotFoundError(f"Missing fits table: {FITS}")

fits = pd.read_csv(FITS)
fits["point_id"] = fits["point_id"].astype(str)

fits["is_limitation_like"] = fits["response_class"].isin(["breakdown", "saturation", "weakening"])
fits["is_breakdown"] = fits["response_class"].eq("breakdown")
fits["is_saturation"] = fits["response_class"].eq("saturation")
fits["is_enhancement"] = fits["response_class"].eq("enhancement")
fits["is_ok"] = fits["fit_status"].eq("OK")

point = (
    fits.groupby(["point_id", "metric"], dropna=False)
    .agg(
        n_fits=("response_class", "size"),
        n_ok=("is_ok", "sum"),
        n_limitation_like=("is_limitation_like", "sum"),
        n_breakdown=("is_breakdown", "sum"),
        n_saturation=("is_saturation", "sum"),
        n_enhancement=("is_enhancement", "sum"),
        n_inconclusive=("response_class", lambda s: int((s == "inconclusive").sum())),
        median_pre_slope=("pre_slope", "median"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
    )
    .reset_index()
)
point["limitation_like_fraction"] = point["n_limitation_like"] / point["n_fits"].replace(0, np.nan)
point["breakdown_fraction"] = point["n_breakdown"] / point["n_fits"].replace(0, np.nan)
point["saturation_fraction"] = point["n_saturation"] / point["n_fits"].replace(0, np.nan)
point["enhancement_fraction"] = point["n_enhancement"] / point["n_fits"].replace(0, np.nan)
point["satellite_robust_class"] = point["limitation_like_fraction"].apply(classify_robust)

point.to_csv(TAB / "Table_PRODUCT02cx_satellite_point_robustness_for_arbitration.csv", index=False)

wide = point.pivot(index="point_id", columns="metric", values="limitation_like_fraction").reset_index()
wide.columns = [c if isinstance(c, str) else c for c in wide.columns]
wide = wide.rename(columns={
    "log_wue": "sat_limitation_fraction_log_wue",
    "log_uwue": "sat_limitation_fraction_log_uwue",
})
wide["satellite_limitation_mean_fraction"] = wide[
    [c for c in ["sat_limitation_fraction_log_wue", "sat_limitation_fraction_log_uwue"] if c in wide.columns]
].mean(axis=1)
wide["satellite_overall_class"] = wide["satellite_limitation_mean_fraction"].apply(classify_robust)

tower_path = find_existing(TOWER_CANDIDATES)
tower_status = "MISSING"
comparison = pd.DataFrame()

if tower_path is not None:
    tw = pd.read_csv(tower_path)
    tower_status = "FOUND"

    # Normalize likely site column.
    site_col = None
    for c in ["site", "point_id", "tower_id"]:
        if c in tw.columns:
            site_col = c
            break

    if site_col:
        tw["point_id"] = tw[site_col].astype(str)

        # Keep only useful tower classification columns if present.
        keep = ["point_id"]
        for c in [
            "tower_response_class", "satellite_response_class",
            "tower_post_slope", "satellite_post_slope",
            "tower_slope_change", "satellite_slope_change",
            "class_agreement_exact", "class_agreement_limited_vs_enhanced",
            "slope_direction_agreement", "n_fit", "n_fit_8day", "n_years",
            "tower_metric", "stress_method"
        ]:
            if c in tw.columns:
                keep.append(c)

        tw2 = tw[keep].drop_duplicates()
        comparison = wide.merge(tw2, on="point_id", how="left")

        if "tower_response_class" in comparison.columns:
            comparison["tower_limitation_like"] = comparison["tower_response_class"].astype(str).str.lower().isin([
                "saturation", "breakdown", "weakening", "saturation_or_breakdown"
            ])
            comparison["satellite_limitation_like_from_strict2x2"] = comparison["satellite_overall_class"].isin([
                "strong_limitation", "moderate_limitation"
            ])
            comparison["tower_satellite_limitation_agreement_strict2x2"] = (
                comparison["tower_limitation_like"] == comparison["satellite_limitation_like_from_strict2x2"]
            )
    else:
        comparison = wide.copy()
else:
    comparison = wide.copy()

comparison.to_csv(DATA / "tower_arbitration_prep_strict2x2.csv", index=False)

if len(comparison):
    n_sites = int(comparison["point_id"].nunique())
    n_with_tower = int(comparison["tower_response_class"].notna().sum()) if "tower_response_class" in comparison.columns else 0
    n_agree = int(comparison["tower_satellite_limitation_agreement_strict2x2"].sum()) if "tower_satellite_limitation_agreement_strict2x2" in comparison.columns else 0
    n_agree_denom = int(comparison["tower_satellite_limitation_agreement_strict2x2"].notna().sum()) if "tower_satellite_limitation_agreement_strict2x2" in comparison.columns else 0
else:
    n_sites = 0
    n_with_tower = 0
    n_agree = 0
    n_agree_denom = 0

summary = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "fits_source": str(FITS),
    "tower_source": str(tower_path) if tower_path else "",
    "tower_source_status": tower_status,
    "n_sites": n_sites,
    "n_sites_with_tower_class": n_with_tower,
    "n_tower_satellite_limitation_agree": n_agree,
    "n_tower_satellite_limitation_compared": n_agree_denom,
    "agreement_fraction": n_agree / n_agree_denom if n_agree_denom else np.nan,
}])
summary.to_csv(TAB / "Table_PRODUCT02cy_tower_arbitration_prep_summary.csv", index=False)

if tower_status == "FOUND" and n_sites == 13:
    verdict = "TOWER_ARBITRATION_PREP_READY"
    blocking_next = False
else:
    verdict = "TOWER_ARBITRATION_PREP_PARTIAL_OR_MISSING_TOWER_SOURCE"
    blocking_next = True

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "verdict": verdict,
    "blocking_next_stage": bool(blocking_next),
    "next_stage": "TRAIT_CLIMATE_SOIL_MODEL_PREP" if not blocking_next else "LOCATE_OR_REPAIR_TOWER_ARBITRATION_SOURCE",
}])
decision.to_csv(TAB / "Table_PRODUCT02cz_tower_arbitration_prep_decision.csv", index=False)

report = []
report.append("# Stage 1B.6S tower arbitration prep")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Summary")
report.append("")
report.append("```text")
report.append(summary.to_string(index=False))
report.append("```")
report.append("")
report.append("## Satellite point robustness")
report.append("")
report.append("```text")
report.append(point.to_string(index=False))
report.append("```")
report.append("")
report.append("## Arbitration prep preview")
report.append("")
report.append("```text")
report.append(comparison.head(80).to_string(index=False) if len(comparison) else "No comparison rows.")
report.append("```")
report.append("")
report.append("## Output")
report.append("")
report.append("- Arbitration prep table: `data/processed/stage1b6s/tower_arbitration_prep_strict2x2.csv`")
report.append("")
report.append("## Strict rule")
report.append("")
report.append("Use this as arbitration prep, not final causal proof. Satellite strict 2x2 response and tower response are compared for class/direction agreement; disagreement must be reported.")
report.append("")

(TXT / "STAGE1B6S_TOWER_ARBITRATION_PREP_REPORT.md").write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "tower_arbitration_prep_strict2x2.csv")
print("WROTE", TAB / "Table_PRODUCT02cx_satellite_point_robustness_for_arbitration.csv")
print("WROTE", TAB / "Table_PRODUCT02cy_tower_arbitration_prep_summary.csv")
print("WROTE", TAB / "Table_PRODUCT02cz_tower_arbitration_prep_decision.csv")
print("WROTE", TXT / "STAGE1B6S_TOWER_ARBITRATION_PREP_REPORT.md")
