from pathlib import Path
from datetime import datetime
import json, re, warnings, subprocess, sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

for pkg in ["statsmodels", "sklearn"]:
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6au_strict_real_trait_flux"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

project_DECISION = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/STAGE1B6AS_FINAL_FULL_STRICT_RIGOR_DECISION.json"
SITE_STATUS = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dh_final_FULL_STRICT_site_status.csv"

BAD_RESPONSE_WORDS = [
    "uncertainty", "range", "posterior_sd", "sd", "stderr", "std", "se_",
    "ci_", "ci95", "quantile", "p_value", "pvalue", "q_value", "qvalue",
    "agreement", "disagreement", "correlation", "r2", "rmse", "mae",
    "product", "metric_uncertainty", "product_uncertainty"
]

GOOD_RESPONSE_PATTERNS = {
    "gpp_response": [
        r"^gpp_.*slope_change$",
        r"^gpp_.*response.*slope$",
        r"^gpp_.*stress.*slope$",
        r"^gpp_.*latent.*slope",
        r"^gpp_.*beta",
    ],
    "et_response": [
        r"^et_.*slope_change$",
        r"^et_.*response.*slope$",
        r"^et_.*stress.*slope$",
        r"^et_.*latent.*slope",
        r"^et_.*beta",
        r"^le_.*slope_change$",
    ],
    "uwue_response": [
        r"^uwue_.*slope_change$",
        r"^uwue_.*response.*slope$",
        r"^uwue_.*stress.*slope$",
        r"^uwue_.*latent.*slope",
        r"^wue_.*slope_change$",
        r"^wue_.*response.*slope$",
        r"^latent_slope_change$",
        r"^latent_post_slope$",
        r"^slope_change$",
        r"^response_slope$",
    ],
    "response_class": [
        r"^latent_response_class$",
        r"^response_class$",
        r"^class$",
        r"^satbreak_class$",
        r"^saturation_breakdown_class$",
    ],
}

TRAIT_PATTERNS = {
    "c4_fraction": [r"^c4_fraction", r"^c4_", r"c4.*fraction", r"photosynthetic.*c4"],
    "rooting_depth": [r"rooting_depth", r"root.*depth"],
    "aridity": [r"^aridity$", r"aridity_index", r"aridity_quantile", r"^ai$"],
}

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(c):
    return str(c).strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

def bad_response_col(c):
    lc = norm(c)
    return any(bad in lc for bad in BAD_RESPONSE_WORDS)

def safe_read(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return None

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def z(s):
    s = to_num(s)
    if s.notna().sum() < 8:
        return s * np.nan
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean()) / sd

def find_trait_col(cols, trait):
    pats = TRAIT_PATTERNS[trait]
    best = None
    best_score = -1
    for c in cols:
        lc = norm(c)
        score = 0
        for pat in pats:
            if re.search(pat, lc):
                score += 10
        if score > best_score:
            best_score = score
            best = c
    return best if best_score > 0 else None

def find_response_cols(cols):
    out = {}
    for response, pats in GOOD_RESPONSE_PATTERNS.items():
        matches = []
        for c in cols:
            lc = norm(c)
            if response != "response_class" and bad_response_col(c):
                continue
            for pat in pats:
                if re.search(pat, lc):
                    matches.append(c)
                    break
        out[response] = matches
    return out

def all_candidate_csvs():
    files = []
    for root in [ROOT / "results", ROOT / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            sp = str(p)
            if "_project_raw_exports" in sp:
                continue
            if "stage1b6at_trait_flux_mechanism" in sp:
                continue
            if ".ipynb_checkpoints" in sp:
                continue
            if p.stat().st_size < 50:
                continue
            files.append(p)
    return sorted(set(files))

def inventory_tables():
    rows = []
    for p in all_candidate_csvs():
        d = safe_read(p, nrows=20)
        if d is None or len(d.columns) < 2:
            continue
        cols = list(d.columns)

        traits = {t: find_trait_col(cols, t) for t in TRAIT_PATTERNS}
        responses = find_response_cols(cols)

        n_traits = sum(v is not None for v in traits.values())
        n_responses = sum(len(v) for k, v in responses.items() if k != "response_class")
        n_classes = len(responses.get("response_class", []))

        score = n_traits * 50 + n_responses * 40 + n_classes * 20
        lower_path = str(p).lower()
        if "c4" in lower_path:
            score += 20
        if "trait" in lower_path:
            score += 20
        if "response" in lower_path or "slope" in lower_path:
            score += 20
        if "uncertainty" in lower_path:
            score -= 20

        rows.append({
            "path": str(p),
            "score": score,
            "n_cols": len(cols),
            "n_traits": n_traits,
            "n_real_continuous_responses": n_responses,
            "n_response_class_cols": n_classes,
            "c4_col": traits["c4_fraction"],
            "rooting_depth_col": traits["rooting_depth"],
            "aridity_col": traits["aridity"],
            "gpp_response_cols": ";".join(responses["gpp_response"]),
            "et_response_cols": ";".join(responses["et_response"]),
            "uwue_response_cols": ";".join(responses["uwue_response"]),
            "response_class_cols": ";".join(responses["response_class"]),
            "columns": ";".join(cols[:120]),
        })

    inv = pd.DataFrame(rows).sort_values("score", ascending=False)
    inv.to_csv(TAB / "Table_PRODUCT03dv_strict_real_response_table_inventory.csv", index=False)
    return inv

def bh(p):
    p = pd.Series(pd.to_numeric(p, errors="coerce"))
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.sum():
        q.loc[ok] = multipletests(p.loc[ok], method="fdr_bh")[1]
    return q

def fit_ols(df, y, xvars, label):
    use = df[[y] + xvars].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(30, len(xvars) + 10):
        return []
    for c in [y] + xvars:
        use[c] = z(use[c])
    use = use.dropna()
    if len(use) < max(30, len(xvars) + 10):
        return []

    X = sm.add_constant(use[xvars])
    try:
        m = sm.OLS(use[y], X).fit(cov_type="HC3")
    except Exception:
        return []

    rows = []
    for x in xvars:
        rows.append({
            "model_label": label,
            "response": y,
            "term": x,
            "n": int(m.nobs),
            "coef_standardized": float(m.params.get(x, np.nan)),
            "se_hc3": float(m.bse.get(x, np.nan)),
            "t": float(m.tvalues.get(x, np.nan)),
            "p": float(m.pvalues.get(x, np.nan)),
            "r2": float(m.rsquared),
            "aic": float(m.aic),
        })
    return rows

def bootstrap_ci(df, y, xvars, term, B=1000, seed=11):
    rng = np.random.default_rng(seed)
    use = df[[y] + xvars].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(30, len(xvars) + 10):
        return np.nan, np.nan
    for c in [y] + xvars:
        use[c] = z(use[c])
    use = use.dropna()
    n = len(use)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        b = use.iloc[idx]
        try:
            X = sm.add_constant(b[xvars])
            m = sm.OLS(b[y], X).fit()
            vals.append(float(m.params.get(term, np.nan)))
        except Exception:
            pass
    vals = pd.Series(vals).dropna()
    if len(vals) < 100:
        return np.nan, np.nan
    return float(vals.quantile(0.025)), float(vals.quantile(0.975))

def build_analysis_dataset(path, row):
    d = safe_read(path)
    if d is None or len(d) < 10:
        return None, None

    cols = list(d.columns)
    out = pd.DataFrame(index=d.index)

    # IDs/products if available.
    id_col = None
    for pat in [r"^site_id$", r"^point_id$", r"^pixel_id$", r"id$"]:
        for c in cols:
            if re.search(pat, norm(c)):
                id_col = c
                break
        if id_col:
            break
    out["unit_id"] = d[id_col].astype(str) if id_col else [f"row_{i}" for i in range(len(d))]

    prod_col = None
    for c in cols:
        if "product" in norm(c):
            prod_col = c
            break
    out["product"] = d[prod_col].astype(str) if prod_col else "unknown"

    trait_cols = {
        "c4_fraction": row.get("c4_col"),
        "rooting_depth": row.get("rooting_depth_col"),
        "aridity": row.get("aridity_col"),
    }
    for k, c in trait_cols.items():
        if isinstance(c, str) and c:
            out[k] = to_num(d[c])

    # Response cols: choose all real ones.
    response_map = {}
    for response_type, csv_col_name in [
        ("gpp_response", "gpp_response_cols"),
        ("et_response", "et_response_cols"),
        ("uwue_response", "uwue_response_cols"),
    ]:
        raw = row.get(csv_col_name, "")
        if not isinstance(raw, str) or not raw:
            continue
        for c in raw.split(";"):
            if c and c in d.columns and not bad_response_col(c):
                clean_name = f"{response_type}__{norm(c)}"
                out[clean_name] = to_num(d[c])
                response_map[clean_name] = c

    class_raw = row.get("response_class_cols", "")
    if isinstance(class_raw, str) and class_raw:
        c = class_raw.split(";")[0]
        if c in d.columns:
            out["response_class"] = d[c].astype(str).str.lower()
            out["breakdown_binary"] = out["response_class"].str.contains("breakdown|break").astype(float)
            out["satbreak_binary"] = out["response_class"].str.contains("saturation|breakdown|sat|break").astype(float)

    meta = {
        "source_path": str(path),
        "response_map": response_map,
        "trait_cols": trait_cols,
        "n_rows": len(out),
    }
    return out, meta

def run_tests(df):
    traits = [c for c in ["c4_fraction", "rooting_depth", "aridity"] if c in df.columns and df[c].notna().sum() >= 30]
    responses = [c for c in df.columns if c.startswith(("gpp_response__", "et_response__", "uwue_response__")) and df[c].notna().sum() >= 30]

    rows = []

    # Univariate trait -> real response.
    for y in responses:
        for x in traits:
            rows += fit_ols(df, y, [x], f"univariate_{x}_to_{y}")

    # Controlled C4 mechanism.
    if "c4_fraction" in traits:
        controls = [x for x in ["c4_fraction", "aridity", "rooting_depth"] if x in traits]
        if len(controls) >= 2:
            for y in responses:
                rows += fit_ols(df, y, controls, f"controlled_c4_plus_covariates_to_{y}")

    # Controlled rooting depth.
    if "rooting_depth" in traits:
        controls = [x for x in ["rooting_depth", "aridity", "c4_fraction"] if x in traits]
        if len(controls) >= 2:
            for y in responses:
                rows += fit_ols(df, y, controls, f"controlled_rooting_depth_plus_covariates_to_{y}")

    test = pd.DataFrame(rows)
    if len(test):
        test["q_bh"] = bh(test["p"])
        los, his = [], []
        for _, r in test.iterrows():
            label = str(r["model_label"])
            y = r["response"]
            if label.startswith("univariate"):
                xvars = [r["term"]]
            elif "c4_plus" in label:
                xvars = [x for x in ["c4_fraction", "aridity", "rooting_depth"] if x in traits]
            elif "rooting_depth_plus" in label:
                xvars = [x for x in ["rooting_depth", "aridity", "c4_fraction"] if x in traits]
            else:
                xvars = [r["term"]]
            lo, hi = bootstrap_ci(df, y, xvars, r["term"])
            los.append(lo)
            his.append(hi)
        test["boot_ci_low"] = los
        test["boot_ci_high"] = his
        test["ci_excludes_zero"] = (
            test["boot_ci_low"].notna()
            & test["boot_ci_high"].notna()
            & ((test["boot_ci_low"] > 0) | (test["boot_ci_high"] < 0))
        )
        test["passes_primary_trait_flux_screen"] = (
            (test["q_bh"] <= 0.10)
            & test["ci_excludes_zero"]
            & (test["n"] >= 50)
        )

    return test

def choose_best(test):
    if not len(test):
        return {}, pd.DataFrame()

    passed = test[test["passes_primary_trait_flux_screen"].astype(bool)].copy()
    if not len(passed):
        return {}, passed

    passed["abs_coef"] = passed["coef_standardized"].abs()

    # Prefer controlled models over univariate, C4 over rooting if both pass, then q/coefficient.
    passed["controlled_priority"] = passed["model_label"].astype(str).str.contains("controlled").astype(int)
    passed["trait_priority"] = passed["term"].map({"c4_fraction": 3, "rooting_depth": 2, "aridity": 1}).fillna(0)
    passed = passed.sort_values(
        ["controlled_priority", "trait_priority", "q_bh", "abs_coef"],
        ascending=[False, False, True, False]
    )

    best = passed.iloc[0].to_dict()
    return best, passed

def main():
    project = {}
    if project_DECISION.exists():
        project = json.loads(project_DECISION.read_text())

    inv = inventory_tables()

    datasets = []
    all_tests = []

    for _, row in inv.head(60).iterrows():
        if row["score"] <= 0:
            continue
        path = Path(row["path"])
        df, meta = build_analysis_dataset(path, row)
        if df is None:
            continue
        tests = run_tests(df)
        if len(tests):
            tests["source_path"] = str(path)
            all_tests.append(tests)
            datasets.append((df, meta, tests))

    if all_tests:
        tests = pd.concat(all_tests, ignore_index=True)
    else:
        tests = pd.DataFrame()

    tests.to_csv(TAB / "Table_PRODUCT03dw_strict_real_trait_to_flux_tests.csv", index=False)

    best, passed = choose_best(tests)
    passed.to_csv(TAB / "Table_PRODUCT03dx_strict_real_candidate_trait_flux_theses.csv", index=False)

    c4_controlled = False
    if len(tests):
        c4_controlled = bool(
            tests["term"].eq("c4_fraction")
            & tests["model_label"].astype(str).str.contains("controlled")
            & tests["passes_primary_trait_flux_screen"].astype(bool)
        ).any()

    if best:
        response_clean = str(best["response"])
        if response_clean.startswith("gpp_response"):
            flux = "GPP response slope"
        elif response_clean.startswith("et_response"):
            flux = "ET response slope"
        elif response_clean.startswith("uwue_response"):
            flux = "uWUE response slope"
        else:
            flux = response_clean

        trait_label = {
            "c4_fraction": "C4 fraction",
            "rooting_depth": "rooting depth",
            "aridity": "aridity",
        }.get(best["term"], best["term"])

        thesis_sentence = f"{trait_label} predicts {flux} under compound VPD–soil-moisture stress."
    else:
        thesis_sentence = "No strict real-response trait→flux thesis passed after excluding uncertainty/range/SD response columns."

    decision = {
        "generated": now(),
        "stage": "1B.6AU_strict_real_trait_flux",
        "strict_exclusion_rule": "response columns containing uncertainty/range/posterior_sd/sd/se/ci/agreement/product_uncertainty/metric_uncertainty were excluded",
        "n_candidate_tables_scanned": int(len(inv)),
        "n_trait_flux_tests": int(len(tests)),
        "n_passing_candidate_theses": int(len(passed)),
        "best_strict_trait_flux_thesis": best,
        "thesis_sentence": thesis_sentence,
        "can_claim_trait_predicts_real_flux_response": bool(best),
        "can_claim_c4_predicts_real_flux_response_after_controls": bool(c4_controlled),
        "project_context": {
            "strict_quality_sites_n": project.get("strict_quality_sites_n"),
            "sensitivity_quality_sites_n": project.get("sensitivity_quality_sites_n"),
            "strict_top_et_product": project.get("strict_top_et_product"),
            "sensitivity_top_et_product": project.get("sensitivity_top_et_product"),
            "can_send_project_as_full_quality_filtered_rigor": project.get("can_send_project_as_full_quality_filtered_rigor"),
        }
    }

    (TAB / "STAGE1B6AU_STRICT_REAL_TRAIT_FLUX_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    email = []
    email.append("Hi project,")
    email.append("")
    email.append("I reran the trait-to-flux mechanism screen with a stricter response-variable rule. I excluded any columns that were uncertainty/range/posterior-SD/standard-error/product-disagreement quantities, so the tested responses are actual flux-response variables rather than uncertainty artifacts.")
    email.append("")
    if best:
        email.append("The strongest strict trait-to-flux result is:")
        email.append("")
        email.append(thesis_sentence)
        email.append("")
        email.append(f"Model: {best.get('model_label')}")
        email.append(f"n = {int(best.get('n'))}")
        email.append(f"standardized coefficient = {float(best.get('coef_standardized')):.3f}")
        email.append(f"BH q = {float(best.get('q_bh')):.4g}")
        email.append(f"bootstrap 95% CI = [{float(best.get('boot_ci_low')):.3f}, {float(best.get('boot_ci_high')):.3f}]")
    else:
        email.append("No trait passed the strict real-response mechanism screen after excluding uncertainty/range/SD columns. This means I should not claim a trait causes a flux response from the current outputs without recomputing the actual response-slope table.")
    email.append("")
    if c4_controlled:
        email.append("C4 fraction also survives at least one controlled real-response model, so a C4-centered thesis may be defensible depending on which response variable you think is most biologically central.")
    else:
        email.append("C4 fraction does not survive the strict controlled real-response screen in the current outputs, so I would avoid a C4-causal thesis unless we recompute the response metrics directly.")
    email.append("")
    email.append("Best,")
    email.append("Akul")

    (TXT / "project_STRICT_REAL_TRAIT_FLUX_EMAIL.md").write_text("\n".join(email), encoding="utf-8")

    report = []
    report.append("# Stage 1B.6AU strict real trait-flux screen")
    report.append("")
    report.append("## Decision")
    report.append("```json")
    report.append(json.dumps(decision, indent=2))
    report.append("```")
    report.append("")
    report.append("## Passing candidate theses")
    report.append("```text")
    report.append(passed.to_string(index=False) if len(passed) else "No passing strict real-response trait-flux thesis.")
    report.append("```")
    report.append("")
    report.append("## All strict real-response tests")
    report.append("```text")
    if len(tests):
        report.append(tests.sort_values("q_bh").head(120).to_string(index=False))
    else:
        report.append("No strict real-response tests were possible.")
    report.append("```")
    report.append("")
    report.append("## Table inventory")
    report.append("```text")
    report.append(inv.head(40).to_string(index=False))
    report.append("```")
    report.append("")
    report.append("## Email draft")
    report.append("```text")
    report.append("\n".join(email))
    report.append("```")

    (TXT / "STAGE1B6AU_STRICT_REAL_TRAIT_FLUX_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("WROTE", TAB / "STAGE1B6AU_STRICT_REAL_TRAIT_FLUX_DECISION.json")
    print("WROTE", TAB / "Table_PRODUCT03dw_strict_real_trait_to_flux_tests.csv")
    print("WROTE", TAB / "Table_PRODUCT03dx_strict_real_candidate_trait_flux_theses.csv")
    print("WROTE", TXT / "project_STRICT_REAL_TRAIT_FLUX_EMAIL.md")

if __name__ == "__main__":
    main()
