from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6ba_c4_vpd_pathway"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "results/stage1b6ay_final_project_audit/tables/Table_PRODUCT03fg_final_audit_dataset.csv"

RESPONSE = "latent_slope_change"
C4 = "c4_fraction_raw"
VPD = "mean_vpd"

CONTROLS_NO_VPD = [
    "rooting_depth",
    "aridity",
    "mean_annual_temperature",
    "mean_annual_precipitation",
    "soil_texture_pc1",
    "mean_soil_moisture",
    "growing_season_mean_lai",
]

def z(s):
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std()
    if x.notna().sum() < 20 or pd.isna(sd) or sd == 0:
        return x * np.nan
    return (x - x.mean()) / sd

def numeric_frame(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()

def fit_std(df, y, xvars, label, min_n=35):
    cols = [y] + xvars
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.DataFrame([{
            "model_label": label,
            "response": y,
            "term": "",
            "n": 0,
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "fit_status": "MISSING_COLUMNS: " + ", ".join(missing),
            "controls": ", ".join(xvars),
        }])

    use = numeric_frame(df, cols)
    if len(use) < min_n:
        return pd.DataFrame([{
            "model_label": label,
            "response": y,
            "term": "",
            "n": len(use),
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_ROWS",
            "controls": ", ".join(xvars),
        }])

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()

    if len(zz) < min_n:
        return pd.DataFrame([{
            "model_label": label,
            "response": y,
            "term": "",
            "n": len(zz),
            "coef_standardized": np.nan,
            "se_hc3": np.nan,
            "t": np.nan,
            "p": np.nan,
            "r2": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_Z_ROWS",
            "controls": ", ".join(xvars),
        }])

    X = sm.add_constant(zz[xvars], has_constant="add")
    m = sm.OLS(zz[y], X).fit(cov_type="HC3")

    rows = []
    for term in xvars:
        rows.append({
            "model_label": label,
            "response": y,
            "term": term,
            "n": int(m.nobs),
            "coef_standardized": float(m.params.get(term, np.nan)),
            "se_hc3": float(m.bse.get(term, np.nan)),
            "t": float(m.tvalues.get(term, np.nan)),
            "p": float(m.pvalues.get(term, np.nan)),
            "r2": float(m.rsquared),
            "fit_status": "FIT_OK",
            "controls": ", ".join(xvars),
        })
    return pd.DataFrame(rows)

def bootstrap_pathway(df, subset_name, B=2000, seed=123):
    controls = [c for c in CONTROLS_NO_VPD if c in df.columns]
    cols = [RESPONSE, C4, VPD] + controls
    use = numeric_frame(df, cols)

    if len(use) < 45:
        return {
            "subset": subset_name,
            "n": len(use),
            "status": "NOT_ENOUGH_ROWS",
        }

    zz = pd.DataFrame(index=use.index)
    for c in cols:
        zz[c] = z(use[c])
    zz = zz.dropna()

    n = len(zz)
    rng = np.random.default_rng(seed)

    indirect_vals = []
    direct_vals = []
    total_vals = []
    a_vals = []
    b_vals = []

    x_a = [C4] + controls
    x_b = [VPD, C4] + controls
    x_total = [C4] + controls

    for _ in range(B):
        idx = rng.integers(0, n, n)
        bdf = zz.iloc[idx]

        try:
            ma = sm.OLS(bdf[VPD], sm.add_constant(bdf[x_a], has_constant="add")).fit()
            a = float(ma.params[C4])

            mb = sm.OLS(bdf[RESPONSE], sm.add_constant(bdf[x_b], has_constant="add")).fit()
            b = float(mb.params[VPD])
            direct = float(mb.params[C4])

            mt = sm.OLS(bdf[RESPONSE], sm.add_constant(bdf[x_total], has_constant="add")).fit()
            total = float(mt.params[C4])

            a_vals.append(a)
            b_vals.append(b)
            indirect_vals.append(a * b)
            direct_vals.append(direct)
            total_vals.append(total)
        except Exception:
            pass

    def q(vals, prob):
        vals = pd.Series(vals).dropna()
        if len(vals) < 100:
            return np.nan
        return float(vals.quantile(prob))

    indirect_low = q(indirect_vals, 0.025)
    indirect_high = q(indirect_vals, 0.975)

    return {
        "subset": subset_name,
        "n": n,
        "status": "OK",
        "a_c4_to_vpd_mean": float(np.nanmean(a_vals)),
        "b_vpd_to_response_mean": float(np.nanmean(b_vals)),
        "indirect_ab_mean": float(np.nanmean(indirect_vals)),
        "indirect_ab_ci_low": indirect_low,
        "indirect_ab_ci_high": indirect_high,
        "indirect_ci_excludes_zero": bool((indirect_low > 0) or (indirect_high < 0)),
        "direct_c4_after_vpd_mean": float(np.nanmean(direct_vals)),
        "direct_c4_ci_low": q(direct_vals, 0.025),
        "direct_c4_ci_high": q(direct_vals, 0.975),
        "total_c4_no_vpd_mean": float(np.nanmean(total_vals)),
        "total_c4_ci_low": q(total_vals, 0.025),
        "total_c4_ci_high": q(total_vals, 0.975),
    }

def vpd_strata_models(df, subset_name):
    if VPD not in df.columns:
        return pd.DataFrame()

    d = df.copy()
    d[VPD] = pd.to_numeric(d[VPD], errors="coerce")
    d = d.dropna(subset=[VPD])

    try:
        d["vpd_tercile"] = pd.qcut(d[VPD], q=3, labels=["low_vpd", "mid_vpd", "high_vpd"], duplicates="drop")
    except Exception:
        return pd.DataFrame()

    rows = []
    controls = [c for c in CONTROLS_NO_VPD if c in d.columns]
    # Remove variables that become too costly in small strata, but keep core climate/soil.
    controls_stratum = [c for c in controls if c not in ["mean_annual_temperature", "mean_annual_precipitation"]]

    for tercile, g in d.groupby("vpd_tercile", observed=False):
        res = fit_std(
            g,
            RESPONSE,
            [C4] + controls_stratum,
            f"{subset_name}__within_{tercile}",
            min_n=25,
        )
        rows.append(res)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def interaction_model(df, subset_name):
    controls = [c for c in CONTROLS_NO_VPD if c in df.columns]
    d = df.copy()

    needed = [RESPONSE, C4, VPD] + controls
    use = numeric_frame(d, needed)
    if len(use) < 45:
        return pd.DataFrame([{
            "model_label": f"{subset_name}__c4_x_vpd_interaction",
            "response": RESPONSE,
            "term": "",
            "n": len(use),
            "coef_standardized": np.nan,
            "p": np.nan,
            "fit_status": "NOT_FIT_TOO_FEW_ROWS",
            "controls": "",
        }])

    use["c4_x_vpd"] = z(use[C4]) * z(use[VPD])
    xvars = [C4, VPD, "c4_x_vpd"] + controls
    return fit_std(use, RESPONSE, xvars, f"{subset_name}__c4_x_vpd_interaction", min_n=45)

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing input dataset: {DATA}")

    df = pd.read_csv(DATA, low_memory=False)

    subsets = {
        "all_points": df,
        "no_crop_flagged_points": df[df["no_crop_flagged_points"].astype(bool)].copy() if "no_crop_flagged_points" in df.columns else df.copy(),
        "natural_grassland_like_no_crop_points": df[df["natural_grassland_like_no_crop_points"].astype(bool)].copy() if "natural_grassland_like_no_crop_points" in df.columns else df.copy(),
    }

    model_rows = []
    pathway_rows = []
    strata_rows = []
    interaction_rows = []

    for name, sub in subsets.items():
        controls = [c for c in CONTROLS_NO_VPD if c in sub.columns]

        model_rows.append(fit_std(
            sub,
            VPD,
            [C4] + controls,
            f"{name}__path_a_c4_to_vpd",
            min_n=45,
        ))

        model_rows.append(fit_std(
            sub,
            RESPONSE,
            [VPD, C4] + controls,
            f"{name}__path_b_and_direct_response_on_vpd_c4",
            min_n=45,
        ))

        model_rows.append(fit_std(
            sub,
            RESPONSE,
            [C4] + controls,
            f"{name}__total_effect_c4_no_vpd",
            min_n=45,
        ))

        pathway_rows.append(bootstrap_pathway(sub, name))
        strata_rows.append(vpd_strata_models(sub, name))
        interaction_rows.append(interaction_model(sub, name))

    model_df = pd.concat(model_rows, ignore_index=True)
    pathway_df = pd.DataFrame(pathway_rows)
    strata_df = pd.concat([x for x in strata_rows if len(x)], ignore_index=True) if any(len(x) for x in strata_rows) else pd.DataFrame()
    interaction_df = pd.concat(interaction_rows, ignore_index=True)

    model_df.to_csv(TAB / "Table_PRODUCT03fu_c4_vpd_path_models.csv", index=False)
    pathway_df.to_csv(TAB / "Table_PRODUCT03fv_bootstrap_c4_vpd_indirect_pathway.csv", index=False)
    strata_df.to_csv(TAB / "Table_PRODUCT03fw_vpd_stratified_c4_models.csv", index=False)
    interaction_df.to_csv(TAB / "Table_PRODUCT03fx_c4_vpd_interaction_models.csv", index=False)

    # Extract main logic.
    def term_p(table, label_contains, term):
        r = table[table["model_label"].str.contains(label_contains, na=False) & table["term"].eq(term)]
        if len(r) == 0:
            return np.nan
        return float(r.iloc[0]["p"])

    def term_coef(table, label_contains, term):
        r = table[table["model_label"].str.contains(label_contains, na=False) & table["term"].eq(term)]
        if len(r) == 0:
            return np.nan
        return float(r.iloc[0]["coef_standardized"])

    nat_pathway = pathway_df[pathway_df["subset"].eq("natural_grassland_like_no_crop_points")]
    nat_indirect_ok = bool(len(nat_pathway) and nat_pathway.iloc[0].get("indirect_ci_excludes_zero", False))

    nat_c4_to_vpd_p = term_p(model_df, "natural_grassland_like_no_crop_points__path_a", C4)
    nat_vpd_to_response_p = term_p(model_df, "natural_grassland_like_no_crop_points__path_b", VPD)
    nat_direct_c4_p = term_p(model_df, "natural_grassland_like_no_crop_points__path_b", C4)
    nat_total_c4_p = term_p(model_df, "natural_grassland_like_no_crop_points__total_effect", C4)

    interaction_p = term_p(interaction_df, "natural_grassland_like_no_crop_points__c4_x_vpd_interaction", "c4_x_vpd")

    pathway_supported = bool(
        pd.notna(nat_c4_to_vpd_p) and nat_c4_to_vpd_p <= 0.05
        and pd.notna(nat_vpd_to_response_p) and nat_vpd_to_response_p <= 0.05
        and pd.notna(nat_total_c4_p) and nat_total_c4_p <= 0.05
        and pd.notna(nat_direct_c4_p) and nat_direct_c4_p > 0.05
        and nat_indirect_ok
    )

    interaction_supported = bool(pd.notna(interaction_p) and interaction_p <= 0.05)

    if pathway_supported:
        thesis = "C4 composition is best framed as organizing a VPD-linked climate/stress pathway to uWUE response, not as an independent direct effect after baseline VPD is controlled away."
    elif interaction_supported:
        thesis = "C4 composition may modify how VPD relates to uWUE response, suggesting a C4 × VPD interaction thesis rather than an independent C4 main-effect thesis."
    else:
        thesis = "The independent C4 trait thesis is not supported; C4 should be treated as a sensitivity/secondary result unless another response variable gives a cleaner mechanism."

    decision = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stage": "1B.6BA_c4_vpd_pathway_reframe",
        "pathway_supported_in_natural_grassland_no_crop_subset": pathway_supported,
        "interaction_supported_in_natural_grassland_no_crop_subset": interaction_supported,
        "natural_subset_c4_to_vpd_p": nat_c4_to_vpd_p,
        "natural_subset_vpd_to_response_p": nat_vpd_to_response_p,
        "natural_subset_total_c4_no_vpd_p": nat_total_c4_p,
        "natural_subset_direct_c4_after_vpd_p": nat_direct_c4_p,
        "natural_subset_interaction_p": interaction_p,
        "recommended_thesis": thesis,
        "plain_english": "If VPD is treated as a confounder, C4 is not an independent mechanism. If VPD is treated as a pathway/mediator or climate syndrome, the C4 result can still support a trait-to-flux story, but it must be written as C4–VPD-linked organization rather than direct C4 causation.",
    }

    (TAB / "STAGE1B6BA_C4_VPD_PATHWAY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    report = []
    report.append("# C4–VPD pathway reframe")
    report.append("")
    report.append("## Decision")
    report.append("```json")
    report.append(json.dumps(decision, indent=2))
    report.append("```")
    report.append("")
    report.append("## Path models")
    report.append("```text")
    report.append(model_df.to_string(index=False))
    report.append("```")
    report.append("")
    report.append("## Bootstrap indirect pathway")
    report.append("```text")
    report.append(pathway_df.to_string(index=False))
    report.append("```")
    report.append("")
    report.append("## VPD-stratified C4 models")
    report.append("```text")
    report.append(strata_df.to_string(index=False) if len(strata_df) else "No strata models produced.")
    report.append("```")
    report.append("")
    report.append("## C4 × VPD interaction models")
    report.append("```text")
    report.append(interaction_df.to_string(index=False))
    report.append("```")
    report.append("")
    report.append("## Meeting language")
    report.append("")
    report.append("I should say: The C4 result is not a clean independent direct effect after controlling for baseline VPD. Instead, the evidence points to a C4–VPD pathway or climate-syndrome structure: C4 composition is associated with higher baseline VPD regimes, and baseline VPD strongly predicts the uWUE latent slope-change response. The main decision is whether project sees baseline VPD as a confounder, which would narrow/reject the C4 mechanism, or as part of the stress pathway, which would let us frame the thesis as a C4–VPD-linked trait-to-flux mechanism.")
    report.append("")
    report.append("I should not say: C4 independently causes the ecosystem flux response after full climate controls.")

    report_text = "\n".join(report)
    (TXT / "C4_VPD_PATHWAY_REFRAME_REPORT.md").write_text(report_text, encoding="utf-8")

    print("===== C4-VPD PATHWAY DECISION =====")
    print(json.dumps(decision, indent=2))
    print("")
    print("===== PATHWAY BOOTSTRAP =====")
    print(pathway_df.to_string(index=False))
    print("")
    print("===== KEY PATH MODELS =====")
    key = model_df[
        model_df["model_label"].str.contains("natural_grassland_like_no_crop_points", na=False)
        & model_df["term"].isin([C4, VPD])
    ].copy()
    print(key.to_string(index=False))
    print("")
    print("===== VPD-STRATIFIED MODELS =====")
    if len(strata_df):
        print(strata_df[strata_df["term"].eq(C4)].to_string(index=False))
    else:
        print("No strata models produced.")
    print("")
    print("===== INTERACTION MODELS =====")
    print(interaction_df[
        interaction_df["model_label"].str.contains("natural_grassland_like_no_crop_points", na=False)
        & interaction_df["term"].isin([C4, VPD, "c4_x_vpd"])
    ].to_string(index=False))
    print("")
    print("WROTE", TAB / "STAGE1B6BA_C4_VPD_PATHWAY_DECISION.json")
    print("WROTE", TAB / "Table_PRODUCT03fu_c4_vpd_path_models.csv")
    print("WROTE", TAB / "Table_PRODUCT03fv_bootstrap_c4_vpd_indirect_pathway.csv")
    print("WROTE", TAB / "Table_PRODUCT03fw_vpd_stratified_c4_models.csv")
    print("WROTE", TAB / "Table_PRODUCT03fx_c4_vpd_interaction_models.csv")
    print("WROTE", TXT / "C4_VPD_PATHWAY_REFRAME_REPORT.md")

if __name__ == "__main__":
    main()
