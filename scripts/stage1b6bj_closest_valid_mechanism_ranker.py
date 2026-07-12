from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path.cwd()
B6BI = ROOT / "results" / "stage1b6bi_strict_ecological_mechanism_screen"
IN_TAB = B6BI / "tables"

OUT = ROOT / "results" / "stage1b6bj_closest_valid_mechanism_ranker"
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for d in [OUT, TAB, TXT, FIG]:
    d.mkdir(parents=True, exist_ok=True)


def safe_read(path):
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def boolify(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in ["true", "1", "yes", "pass"]


def fnum(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def get_clean_mask_status():
    p = IN_TAB / "CLEAN_LANDCOVER_C4_CROP_MASK_AUDIT.csv"
    d = safe_read(p)
    if d.empty:
        return {
            "status": "MISSING_CLEAN_MASK_AUDIT",
            "n_clean": np.nan,
            "n_total": np.nan,
            "used_columns": "",
        }
    r = d.iloc[0].to_dict()
    return {
        "status": str(r.get("status", "")),
        "n_clean": r.get("n_clean", np.nan),
        "n_total": r.get("n_total", np.nan),
        "used_columns": str(r.get("used_columns", "")),
    }


def fail_reason_primary(r):
    reasons = []
    if str(r.get("status", "")) != "FIT_OK":
        reasons.append(f"model status={r.get('status')}")
    if not (fnum(r.get("p")) < 0.05):
        reasons.append(f"p not <0.05: {r.get('p')}")
    if not (fnum(r.get("bh_q")) < 0.05):
        reasons.append(f"BH q not <0.05: {r.get('bh_q')}")
    if not boolify(r.get("ci_excludes_zero")):
        reasons.append("CI includes zero")
    if not (fnum(r.get("delta_r2")) > 0):
        reasons.append(f"ΔR² not positive: {r.get('delta_r2')}")
    if not (fnum(r.get("delta_aic_full_minus_reduced")) < 0):
        reasons.append(f"ΔAIC not improved: {r.get('delta_aic_full_minus_reduced')}")
    if not (fnum(r.get("nested_f_p")) < 0.05):
        reasons.append(f"nested F p not <0.05: {r.get('nested_f_p')}")
    if boolify(r.get("control_self_family_omitted")):
        reasons.append("full-control concern: focal variable is same family as omitted control")
    return "; ".join(reasons) if reasons else "PASS"


def fail_reason_bootstrap(r):
    reasons = []
    if not boolify(r.get("boot_ci_excludes_zero")):
        reasons.append("bootstrap CI includes zero or was not available")
    if not (fnum(r.get("loo_sign_stability")) >= 0.80):
        reasons.append(f"LOO sign stability <0.80: {r.get('loo_sign_stability')}")
    return "; ".join(reasons) if reasons else "PASS"


def fail_reason_clean(r, clean_status):
    reasons = []
    if clean_status["status"] != "CLEAN_MASK_INFERRED":
        reasons.append(f"clean land-cover/C4-crop mask not testable: {clean_status['status']}")
    if str(r.get("clean_status", "")) != "FIT_OK":
        reasons.append(f"clean model status={r.get('clean_status')}")
    if not (fnum(r.get("clean_p")) < 0.05):
        reasons.append(f"clean p not <0.05: {r.get('clean_p')}")
    if not boolify(r.get("clean_ci_excludes_zero")):
        reasons.append("clean CI includes zero")
    if not (fnum(r.get("clean_delta_aic")) < 0):
        reasons.append(f"clean ΔAIC not improved: {r.get('clean_delta_aic')}")
    if np.isfinite(fnum(r.get("coef"))) and np.isfinite(fnum(r.get("clean_coef"))):
        if np.sign(fnum(r.get("coef"))) != np.sign(fnum(r.get("clean_coef"))):
            reasons.append("clean coefficient sign differs from primary")
    return "; ".join(reasons) if reasons else "PASS"


def fail_reason_product(r, prefix):
    gate = str(r.get(f"{prefix}__product_gate", ""))
    reasons = []
    if gate == "PASS":
        return "PASS"

    if gate:
        reasons.append(f"product gate={gate}")
    else:
        reasons.append("product gate missing")

    gg_pass = r.get(f"{prefix}__gosif_gleam_pass", np.nan)
    if not boolify(gg_pass):
        reasons.append(
            "GOSIF×GLEAM did not pass "
            f"(coef={r.get(f'{prefix}__gosif_gleam_coef')}, "
            f"p={r.get(f'{prefix}__gosif_gleam_p')}, "
            f"q={r.get(f'{prefix}__gosif_gleam_bh_q')})"
        )

    sc = fnum(r.get(f"{prefix}__sign_consistency"))
    if not (sc >= 0.75):
        reasons.append(f"product sign consistency <0.75: {r.get(f'{prefix}__sign_consistency')}")

    opp = fnum(r.get(f"{prefix}__n_opposite_significant_products"))
    if np.isfinite(opp) and opp > 0:
        reasons.append(f"{int(opp)} opposite-sign significant product rows")

    if boolify(r.get(f"{prefix}__dependency_flag")):
        reasons.append("effect appears stronger with more algorithmically entangled products")

    return "; ".join(reasons)


def fail_reason_tower(r):
    if boolify(r.get("tower_strict_gate")):
        return "PASS"
    return "not passed automatically; needs independent tower-derived same-feature/uWUE-response test"


def category(feature, mechanism_type):
    f = str(feature).lower()
    t = str(mechanism_type).lower()
    if "c4" in f:
        return "C4/photosynthetic-pathway"
    if "lai" in f or "fpar" in f or "evi" in f or "ndvi" in f:
        return "canopy-structure/productivity"
    if "vpd" in f:
        return "VPD/stress-climate"
    if "soil_moisture" in f or "rootzone" in f:
        return "soil-moisture"
    if "temperature" in f or "temp" in f:
        return "temperature-gradient"
    if "precip" in f or "aridity" in f:
        return "hydroclimate"
    if "sand" in f or "silt" in f or "clay" in f or "soil_texture" in f:
        return "soil-texture"
    if "root" in f:
        return "rooting"
    if "p50" in f or "psi50" in f or "isohydric" in f:
        return "hydraulic-trait"
    if "quadratic" in t or "hinge" in t or "tail" in t:
        return "threshold/nonlinear"
    return "other-ecological"


def main():
    gates_path = IN_TAB / "GATED_STRICT_ECOLOGICAL_SURVIVORS.csv"
    primary_path = IN_TAB / "PRIMARY_STRICT_ECOLOGICAL_MECHANISM_SCREEN.csv"
    c4_path = IN_TAB / "project_REQUIRED_C4_FULL_CONTROL_CHECK.csv"
    audit_path = IN_TAB / "PROGRAMMING_AUDIT.json"

    gates = safe_read(gates_path)
    primary = safe_read(primary_path)
    c4 = safe_read(c4_path)
    clean_status = get_clean_mask_status()

    if gates.empty:
        raise SystemExit(f"No gated survivors found at {gates_path}. Run B6BI first.")

    # Ensure all needed gate columns exist.
    for c in [
        "primary_pass", "bootstrap_gate", "clean_gate",
        "product_all_gate", "product_clean_gate", "tower_strict_gate",
        "FULL_STRICT_STRICT_PASS", "SATELLITE_STRICT_PASS_NO_TOWER",
    ]:
        if c not in gates.columns:
            gates[c] = False

    for c in [
        "primary_pass", "bootstrap_gate", "clean_gate",
        "product_all_gate", "product_clean_gate", "tower_strict_gate",
        "FULL_STRICT_STRICT_PASS", "SATELLITE_STRICT_PASS_NO_TOWER",
    ]:
        gates[c] = gates[c].map(boolify)

    non_tower_gates = [
        "primary_pass",
        "bootstrap_gate",
        "clean_gate",
        "product_all_gate",
        "product_clean_gate",
    ]
    full_gates = non_tower_gates + ["tower_strict_gate"]

    gates["non_tower_gate_score_0_to_5"] = gates[non_tower_gates].sum(axis=1)
    gates["full_gate_score_0_to_6"] = gates[full_gates].sum(axis=1)
    gates["distance_from_satellite_strict"] = 5 - gates["non_tower_gate_score_0_to_5"]
    gates["distance_from_FULL_STRICT_strict"] = 6 - gates["full_gate_score_0_to_6"]

    gates["story_category"] = gates.apply(
        lambda r: category(r.get("feature", ""), r.get("mechanism_type", "")),
        axis=1,
    )

    gates["primary_failure_reason"] = gates.apply(fail_reason_primary, axis=1)
    gates["bootstrap_failure_reason"] = gates.apply(fail_reason_bootstrap, axis=1)
    gates["clean_failure_reason"] = gates.apply(lambda r: fail_reason_clean(r, clean_status), axis=1)
    gates["product_all_failure_reason"] = gates.apply(lambda r: fail_reason_product(r, "product_all"), axis=1)
    gates["product_clean_failure_reason"] = gates.apply(lambda r: fail_reason_product(r, "product_clean"), axis=1)
    gates["tower_failure_reason"] = gates.apply(fail_reason_tower, axis=1)

    def tier(r):
        if boolify(r.get("FULL_STRICT_STRICT_PASS")):
            return "TIER_0_FULL_STRICT_STRICT_PASS"
        if boolify(r.get("SATELLITE_STRICT_PASS_NO_TOWER")):
            return "TIER_1_SATELLITE_STRICT_PASS_TOWER_MISSING"
        if r["non_tower_gate_score_0_to_5"] == 4:
            return "TIER_2_ONE_NON_TOWER_GATE_SHORT"
        if r["non_tower_gate_score_0_to_5"] == 3:
            return "TIER_3_TWO_NON_TOWER_GATES_SHORT"
        if boolify(r.get("primary_pass")):
            return "TIER_4_PRIMARY_SIGNAL_ONLY_OR_WEAK_GATES"
        return "TIER_5_NEAR_DISCOVERY_ONLY"

    gates["closest_tier"] = gates.apply(tier, axis=1)

    # This is the ranking the user actually needs.
    gates = gates.sort_values(
        [
            "FULL_STRICT_STRICT_PASS",
            "SATELLITE_STRICT_PASS_NO_TOWER",
            "non_tower_gate_score_0_to_5",
            "full_gate_score_0_to_6",
            "primary_pass",
            "bootstrap_gate",
            "product_all_gate",
            "product_clean_gate",
            "bh_q",
            "p",
        ],
        ascending=[False, False, False, False, False, False, False, False, True, True],
    )

    keep_cols = [
        "closest_tier",
        "mechanism_id",
        "feature",
        "story_category",
        "feature_family",
        "mechanism_type",
        "moderator",
        "coef",
        "p",
        "bh_q",
        "by_q",
        "holm_p",
        "ci_low",
        "ci_high",
        "delta_r2",
        "delta_aic_full_minus_reduced",
        "nested_f_p",
        "focal_vif",
        "non_tower_gate_score_0_to_5",
        "full_gate_score_0_to_6",
        "distance_from_satellite_strict",
        "distance_from_FULL_STRICT_strict",
        "primary_pass",
        "bootstrap_gate",
        "clean_gate",
        "product_all_gate",
        "product_clean_gate",
        "tower_strict_gate",
        "primary_failure_reason",
        "bootstrap_failure_reason",
        "clean_failure_reason",
        "product_all_failure_reason",
        "product_clean_failure_reason",
        "tower_failure_reason",
    ]
    keep_cols = [c for c in keep_cols if c in gates.columns]

    ranked = gates[keep_cols].copy()
    ranked.to_csv(TAB / "CLOSEST_VALID_MECHANISMS_RANKED.csv", index=False)
    ranked.head(50).to_csv(TAB / "TOP50_CLOSEST_VALID_MECHANISMS.csv", index=False)

    # Best by ecological story category so it is not just a wall of temperature variants.
    best_by_cat = (
        ranked.sort_values(
            ["non_tower_gate_score_0_to_5", "full_gate_score_0_to_6", "bh_q", "p"],
            ascending=[False, False, True, True],
        )
        .groupby("story_category", as_index=False)
        .head(3)
    )
    best_by_cat.to_csv(TAB / "BEST_CLOSEST_MECHANISMS_BY_ECOLOGICAL_CATEGORY.csv", index=False)

    # C4 report.
    c4.to_csv(TAB / "C4_REQUIRED_CHECK_COPY.csv", index=False)

    audit = {}
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text())
        except Exception:
            audit = {}

    full_count = int(gates["FULL_STRICT_STRICT_PASS"].sum())
    sat_count = int(gates["SATELLITE_STRICT_PASS_NO_TOWER"].sum())
    best_score = int(gates["non_tower_gate_score_0_to_5"].max()) if len(gates) else 0

    lines = []
    lines.append("Stage1B6BJ closest valid mechanism ranker")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Purpose")
    lines.append("- Do not loosen the science to manufacture a pass.")
    lines.append("- Rank the closest valid ecological mechanisms after strict B6BI screening.")
    lines.append("- Separate full professor-level pass from satellite-only near-pass and explain each failed gate.")
    lines.append("")
    lines.append("Strict pass counts")
    lines.append(f"- FULL_STRICT_STRICT_PASS: {full_count}")
    lines.append(f"- SATELLITE_STRICT_PASS_NO_TOWER: {sat_count}")
    lines.append(f"- Best non-tower gate score observed: {best_score} / 5")
    lines.append("")
    lines.append("Clean land-cover / C4-crop mask status")
    lines.append(f"- Status: {clean_status['status']}")
    lines.append(f"- Clean n: {clean_status['n_clean']} / {clean_status['n_total']}")
    lines.append(f"- Used columns: {clean_status['used_columns'] if clean_status['used_columns'] else 'NONE'}")
    lines.append("")
    lines.append("Best closest mechanisms overall")
    if ranked.empty:
        lines.append("- None: no gate-tested mechanisms available.")
    else:
        display_cols = [
            "closest_tier",
            "mechanism_id",
            "feature",
            "story_category",
            "mechanism_type",
            "coef",
            "p",
            "bh_q",
            "non_tower_gate_score_0_to_5",
            "full_gate_score_0_to_6",
            "primary_pass",
            "bootstrap_gate",
            "clean_gate",
            "product_all_gate",
            "product_clean_gate",
            "tower_strict_gate",
        ]
        display_cols = [c for c in display_cols if c in ranked.columns]
        lines.append(ranked.head(25)[display_cols].to_string(index=False))
    lines.append("")
    lines.append("Best closest mechanisms by ecological category")
    if best_by_cat.empty:
        lines.append("- None.")
    else:
        display_cols = [
            "closest_tier",
            "mechanism_id",
            "feature",
            "story_category",
            "mechanism_type",
            "p",
            "bh_q",
            "non_tower_gate_score_0_to_5",
            "primary_failure_reason",
            "clean_failure_reason",
            "product_all_failure_reason",
            "product_clean_failure_reason",
        ]
        display_cols = [c for c in display_cols if c in best_by_cat.columns]
        lines.append(best_by_cat.head(40)[display_cols].to_string(index=False))
    lines.append("")
    lines.append("Required C4 check")
    if c4.empty:
        lines.append("- C4 check table is empty: no C4 feature was found or no C4 model fit.")
    else:
        c4cols = [
            "mechanism_id",
            "feature",
            "status",
            "n",
            "coef",
            "p",
            "bh_q",
            "ci_low",
            "ci_high",
            "delta_r2",
            "delta_aic_full_minus_reduced",
            "controls_used",
            "controls_omitted",
        ]
        c4cols = [c for c in c4cols if c in c4.columns]
        lines.append(c4.head(30)[c4cols].to_string(index=False))
    lines.append("")
    lines.append("Interpretation rule")
    lines.append("- Tier 0 means it passes everything, including tower.")
    lines.append("- Tier 1 means satellite/product/statistical gates pass, but tower is missing or not strict.")
    lines.append("- Tier 2 means it is one non-tower gate short; useful for deciding the next targeted fix/test.")
    lines.append("- Lower tiers are exploratory only and should not be sold as the main result.")
    lines.append("")
    lines.append("Important files")
    lines.append(f"- {TAB / 'CLOSEST_VALID_MECHANISMS_RANKED.csv'}")
    lines.append(f"- {TAB / 'TOP50_CLOSEST_VALID_MECHANISMS.csv'}")
    lines.append(f"- {TAB / 'BEST_CLOSEST_MECHANISMS_BY_ECOLOGICAL_CATEGORY.csv'}")
    lines.append(f"- {TAB / 'C4_REQUIRED_CHECK_COPY.csv'}")

    readme = "\n".join(lines)
    (TXT / "READ_ME_closest_valid_mechanisms.txt").write_text(readme)

    print("DONE.")
    print(f"Outputs written to: {OUT}")
    print("")
    print("Paste this back:")
    print(f"cat {TXT / 'READ_ME_closest_valid_mechanisms.txt'}")


if __name__ == "__main__":
    main()
