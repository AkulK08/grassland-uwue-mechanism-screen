from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SRC = ROOT / "results" / "stage1b6bk_strict_plus_closest_mechanism_screen" / "tables"
OUT = ROOT / "results" / "stage1b6bl_INDIVIDUAL_MECHANISM_mechanism_decision"
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for d in [TAB, TXT, FIG]:
    d.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fnum(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def boolish(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in ["true", "1", "yes", "pass"]


def norm_feature_name(x):
    x = str(x)
    aliases = {
        "lai": "LAI / canopy structure",
        "growing_season_mean_lai": "LAI / canopy structure",
        "median_lai": "LAI / canopy structure",
        "mean_lai": "LAI / canopy structure",
        "mat": "temperature regime",
        "mean_annual_temperature": "temperature regime",
        "mean_temperature": "temperature regime",
        "median_temperature": "temperature regime",
        "p10_temperature": "cold-tail temperature regime",
        "p90_temperature": "warm-tail temperature regime",
        "mean_vpd": "VPD regime",
        "median_vpd": "VPD regime",
        "p10_vpd": "low-tail VPD regime",
        "p90_vpd": "high-tail VPD regime",
        "soil_silt": "soil texture",
        "soil_silt_mean": "soil texture",
        "soil_sand": "soil texture",
        "soil_sand_mean": "soil texture",
        "soil_texture_pc1": "soil texture",
        "rooting_depth": "rooting depth",
        "trait_rooting_depth": "rooting depth",
        "rooting_zone_storage_rooting_depth": "rooting depth",
    }
    return aliases.get(x, x)


def mechanism_family(row):
    feature = str(row.get("feature", ""))
    mech = str(row.get("mechanism_type", ""))
    canonical = norm_feature_name(feature)

    if mech in ["low_tail", "hinge_low"]:
        return f"low-threshold / sparse-tail {canonical}"
    if mech in ["high_tail", "hinge_high"]:
        return f"high-threshold / upper-tail {canonical}"
    if mech == "quadratic":
        return f"nonlinear/quadratic {canonical}"
    if mech == "interaction":
        mod = norm_feature_name(row.get("moderator", ""))
        return f"{canonical} × {mod} interaction"
    if mech == "linear":
        return f"linear {canonical}"
    return f"{mech} {canonical}"


def score_for_dedup(row):
    score = 0
    score += 1000 * int(boolish(row.get("FULL_STRICT_STRICT_PASS")))
    score += 500 * int(boolish(row.get("SATELLITE_STRICT_PASS_NO_TOWER")))
    score += 50 * int(fnum(row.get("non_tower_gate_score_0_to_5")) if np.isfinite(fnum(row.get("non_tower_gate_score_0_to_5"))) else 0)
    score += 10 * int(fnum(row.get("full_gate_score_0_to_6")) if np.isfinite(fnum(row.get("full_gate_score_0_to_6"))) else 0)
    q = fnum(row.get("bh_q"))
    p = fnum(row.get("p"))
    score += max(0, 10 - (-np.log10(q) if np.isfinite(q) and q > 0 else 0))
    score += max(0, 5 - (-np.log10(p) if np.isfinite(p) and p > 0 else 0))
    return score


def gate_failure_summary(row):
    fails = []
    if not boolish(row.get("primary_pass")):
        fails.append("primary/FDR")
    if not boolish(row.get("bootstrap_gate")):
        fails.append("bootstrap/LOO")
    if not boolish(row.get("clean_gate")):
        fails.append("clean land-cover")
    if not boolish(row.get("product_all_gate")):
        fails.append("all-product robustness")
    if not boolish(row.get("product_clean_gate")):
        fails.append("clean-product robustness")
    if not boolish(row.get("tower_strict_gate")):
        fails.append("tower")
    return "; ".join(fails) if fails else "PASS"


def main():
    gates = read_csv(SRC / "GATED_AND_CLOSEST_STRICT_ECOLOGICAL_SURVIVORS.csv")
    primary = read_csv(SRC / "PRIMARY_STRICT_ECOLOGICAL_MECHANISM_SCREEN.csv")
    clean = read_csv(SRC / "CLEAN_LANDCOVER_C4_CROP_MASK_AUDIT.csv")
    audit_json = SRC / "PROGRAMMING_AUDIT.json"

    if gates.empty:
        raise SystemExit(
            "Missing B6BK gated table. Run stage1b6bk_strict_plus_closest_mechanism_screen.py first."
        )

    gates = gates.copy()

    # Remove old C4-specific rows from the final decision report only.
    # This does not change the underlying screen; it just stops the report from centering C4.
    if "feature" in gates.columns:
        gates = gates[~gates["feature"].astype(str).str.lower().str.contains("c4", na=False)].copy()

    bool_cols = [
        "FULL_STRICT_STRICT_PASS",
        "SATELLITE_STRICT_PASS_NO_TOWER",
        "primary_pass",
        "bootstrap_gate",
        "clean_gate",
        "product_all_gate",
        "product_clean_gate",
        "tower_strict_gate",
    ]
    for c in bool_cols:
        if c not in gates.columns:
            gates[c] = False
        gates[c] = gates[c].map(boolish)

    gates["mechanism_family_clean"] = gates.apply(mechanism_family, axis=1)
    gates["failed_gates"] = gates.apply(gate_failure_summary, axis=1)
    gates["dedup_score"] = gates.apply(score_for_dedup, axis=1)

    gates = gates.sort_values(
        [
            "FULL_STRICT_STRICT_PASS",
            "SATELLITE_STRICT_PASS_NO_TOWER",
            "non_tower_gate_score_0_to_5",
            "full_gate_score_0_to_6",
            "bh_q",
            "p",
        ],
        ascending=[False, False, False, False, True, True],
    )

    # Keep all variants and also a deduplicated family version.
    gates.to_csv(TAB / "ALL_INDIVIDUAL_MECHANISM_GATE_RESULTS_NO_C4_FOCUS.csv", index=False)

    dedup = (
        gates.sort_values(
            [
                "FULL_STRICT_STRICT_PASS",
                "SATELLITE_STRICT_PASS_NO_TOWER",
                "non_tower_gate_score_0_to_5",
                "full_gate_score_0_to_6",
                "bh_q",
                "p",
            ],
            ascending=[False, False, False, False, True, True],
        )
        .groupby("mechanism_family_clean", as_index=False)
        .head(1)
        .copy()
    )

    dedup = dedup.sort_values(
        [
            "FULL_STRICT_STRICT_PASS",
            "SATELLITE_STRICT_PASS_NO_TOWER",
            "non_tower_gate_score_0_to_5",
            "full_gate_score_0_to_6",
            "bh_q",
            "p",
        ],
        ascending=[False, False, False, False, True, True],
    )

    dedup.to_csv(TAB / "DEDUPLICATED_MECHANISM_FAMILY_DECISION_TABLE.csv", index=False)

    full = gates[gates["FULL_STRICT_STRICT_PASS"]].copy()
    sat = gates[gates["SATELLITE_STRICT_PASS_NO_TOWER"]].copy()
    tier2 = gates[
        (~gates["SATELLITE_STRICT_PASS_NO_TOWER"])
        & (pd.to_numeric(gates["non_tower_gate_score_0_to_5"], errors="coerce") >= 4)
    ].copy()

    full.to_csv(TAB / "FULL_STRICT_STRICT_PASS_NO_C4_FOCUS.csv", index=False)
    sat.to_csv(TAB / "SATELLITE_STRICT_PASS_NO_TOWER_NO_C4_FOCUS.csv", index=False)
    tier2.to_csv(TAB / "ONE_GATE_SHORT_NO_C4_FOCUS.csv", index=False)

    # Programming audit.
    audit = {}
    if audit_json.exists():
        try:
            audit = json.loads(audit_json.read_text())
        except Exception:
            audit = {}

    clean_info = {}
    if not clean.empty:
        clean_info = clean.iloc[0].to_dict()

    summary = {
        "source_table": str(SRC / "GATED_AND_CLOSEST_STRICT_ECOLOGICAL_SURVIVORS.csv"),
        "note": "This is a decision-only report. It does not refit models; it summarizes the strict individual-feature screen.",
        "global_complete_case_filtering": "NO. Each model used its own complete cases for outcome + focal feature + controls.",
        "unrelated_trait_filtering": "NO. A point is not removed because rooting depth or another unrelated feature is missing unless that feature is the focal predictor/model term.",
        "clean_mask_status": clean_info,
        "n_FULL_STRICT_strict_pass_no_c4_focus": int(len(full)),
        "n_SATELLITE_STRICT_pass_no_tower_no_c4_focus": int(len(sat)),
        "n_one_gate_short_no_c4_focus": int(len(tier2)),
        "underlying_audit": audit,
    }
    with open(TAB / "INDIVIDUAL_MECHANISM_DECISION_PROGRAMMING_AUDIT.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("Stage1B6BL individual-feature project mechanism decision")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Reset interpretation")
    lines.append("- This report ignores the old C4 framing.")
    lines.append("- It does not globally filter to points complete for all traits.")
    lines.append("- Each mechanism is tested individually using only the rows needed for that specific model.")
    lines.append("- Missing rooting depth does not remove a point unless rooting depth is the focal variable.")
    lines.append("")
    lines.append("Clean land-cover gate")
    if clean_info:
        lines.append(f"- Status: {clean_info.get('status')}")
        lines.append(f"- Clean n: {clean_info.get('n_clean')} / {clean_info.get('n_total')}")
        lines.append(f"- Used column(s): {clean_info.get('used_columns')}")
    else:
        lines.append("- Clean mask audit not found.")
    lines.append("")
    lines.append("Final counts")
    lines.append(f"- FULL_STRICT_STRICT_PASS including tower: {len(full)}")
    lines.append(f"- SATELLITE_STRICT_PASS_NO_TOWER: {len(sat)}")
    lines.append(f"- ONE_GATE_SHORT non-tower near-passes: {len(tier2)}")
    lines.append("")
    lines.append("Full strict passes")
    if full.empty:
        lines.append("- NONE")
        lines.append("- Reason: tower strict gate is not automatically passed without an independent same-feature tower/uWUE-response test.")
    else:
        cols = [
            "mechanism_family_clean", "mechanism_id", "feature", "mechanism_type",
            "coef", "p", "bh_q", "non_tower_gate_score_0_to_5", "full_gate_score_0_to_6",
            "failed_gates"
        ]
        lines.append(full.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Satellite-side project passes, no tower")
    if sat.empty:
        lines.append("- NONE")
    else:
        cols = [
            "mechanism_family_clean", "mechanism_id", "feature", "story_category",
            "mechanism_type", "coef", "p", "bh_q",
            "non_tower_gate_score_0_to_5", "failed_gates"
        ]
        lines.append(sat.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Deduplicated best mechanism families")
    if dedup.empty:
        lines.append("- NONE")
    else:
        cols = [
            "mechanism_family_clean", "mechanism_id", "feature", "story_category",
            "mechanism_type", "coef", "p", "bh_q",
            "non_tower_gate_score_0_to_5", "failed_gates"
        ]
        lines.append(dedup.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Closest one-gate-short mechanisms")
    if tier2.empty:
        lines.append("- NONE")
    else:
        cols = [
            "mechanism_family_clean", "mechanism_id", "feature", "story_category",
            "mechanism_type", "coef", "p", "bh_q",
            "non_tower_gate_score_0_to_5", "failed_gates",
            "product_all_failure_reason", "product_clean_failure_reason"
        ]
        cols = [c for c in cols if c in tier2.columns]
        lines.append(tier2.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Programming audit")
    lines.append("- No global complete-case table was used for all traits.")
    lines.append("- Each model uses complete cases only for its own outcome, focal term, moderator if any, and controls.")
    lines.append("- Clean land-cover is a gate/sensitivity, not the primary discovery filter.")
    lines.append("- Product robustness is checked on exact product-specific slope-change outcomes.")
    lines.append("- Full tower-inclusive pass remains unavailable unless a true independent tower-derived same-feature test is available.")
    lines.append("")
    lines.append("Important files")
    for p in [
        TAB / "FULL_STRICT_STRICT_PASS_NO_C4_FOCUS.csv",
        TAB / "SATELLITE_STRICT_PASS_NO_TOWER_NO_C4_FOCUS.csv",
        TAB / "DEDUPLICATED_MECHANISM_FAMILY_DECISION_TABLE.csv",
        TAB / "ONE_GATE_SHORT_NO_C4_FOCUS.csv",
        TAB / "ALL_INDIVIDUAL_MECHANISM_GATE_RESULTS_NO_C4_FOCUS.csv",
        TAB / "INDIVIDUAL_MECHANISM_DECISION_PROGRAMMING_AUDIT.json",
    ]:
        lines.append(f"- {p}")

    readme = "\n".join(lines)
    (TXT / "READ_ME_INDIVIDUAL_MECHANISM_mechanism_decision.txt").write_text(readme)

    print("DONE.")
    print(f"Outputs written to: {OUT}")
    print("")
    print("Paste this back:")
    print(f"cat {TXT / 'READ_ME_INDIVIDUAL_MECHANISM_mechanism_decision.txt'}")


if __name__ == "__main__":
    main()
