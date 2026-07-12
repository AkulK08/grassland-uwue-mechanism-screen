from pathlib import Path
from datetime import datetime
import json
import re
import warnings
import subprocess
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Lightweight dependency check
for pkg in ["statsmodels", "sklearn"]:
    try:
        __import__(pkg)
    except Exception:
        print(f"Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6at_trait_flux_mechanism"
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

FINAL_project = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/STAGE1B6AS_FINAL_FULL_STRICT_RIGOR_DECISION.json"
FINAL_SITE = ROOT / "results/stage1b6as_final_FULL_STRICT_rigor/tables/Table_PRODUCT03dh_final_FULL_STRICT_site_status.csv"
C4_SUMMARY = ROOT / "results/stage1b6ak_project_complete_resolution_packet/tables/Table_PRODUCT03bm_c4_project_decision_by_model.csv"

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(c):
    return (
        str(c).strip().lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
    )

def safe_read_csv(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
        except Exception:
            return None

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def bh_q(pvals):
    p = pd.to_numeric(pd.Series(pvals), errors="coerce")
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    if ok.sum():
        q.loc[ok] = multipletests(p.loc[ok], method="fdr_bh")[1]
    return q

def zscore(s):
    s = to_num(s)
    if s.notna().sum() < 3:
        return s * np.nan
    sd = s.std()
    if sd == 0 or pd.isna(sd):
        return s * np.nan
    return (s - s.mean()) / sd

def find_col(cols, patterns, avoid=None):
    avoid = avoid or []
    ncols = {norm(c): c for c in cols}
    scored = []
    for c in cols:
        lc = norm(c)
        if any(a in lc for a in avoid):
            continue
        score = 0
        for pat, w in patterns:
            if re.search(pat, lc):
                score += w
        if score > 0:
            scored.append((score, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]

def all_csvs():
    roots = [ROOT / "results", ROOT / "data"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.csv"))
    # Skip massive raw extracted tower files and hidden temp.
    files = [
        p for p in files
        if "_project_raw_exports" not in str(p)
        and ".ipynb_checkpoints" not in str(p)
        and p.stat().st_size > 10
    ]
    return sorted(set(files))

def column_inventory():
    rows = []
    for p in all_csvs():
        d = safe_read_csv(p, nrows=5)
        if d is None or len(d.columns) == 0:
            continue
        cols = list(d.columns)
        joined = " ".join(norm(c) for c in cols)
        rows.append({
            "path": str(p),
            "n_cols": len(cols),
            "has_c4": bool(re.search(r"c4|photosynthetic", joined)),
            "has_root": bool(re.search(r"root|rooting", joined)),
            "has_aridity": bool(re.search(r"aridity|arid|ai_", joined)),
            "has_gpp": bool(re.search(r"gpp", joined)),
            "has_et": bool(re.search(r"(^|_)et($|_)|evap|le_", joined)),
            "has_wue": bool(re.search(r"wue|uwue|water_use", joined)),
            "has_slope": bool(re.search(r"slope|beta|coef|response|latent", joined)),
            "has_class": bool(re.search(r"class|saturation|breakdown|enhancement", joined)),
            "columns_preview": ";".join(cols[:80]),
        })
    inv = pd.DataFrame(rows)
    inv.to_csv(TAB / "Table_PRODUCT03dl_trait_flux_column_inventory.csv", index=False)
    return inv

def score_candidate_table(row):
    score = 0
    score += 10 if row["has_c4"] else 0
    score += 6 if row["has_root"] else 0
    score += 6 if row["has_aridity"] else 0
    score += 5 if row["has_gpp"] else 0
    score += 5 if row["has_et"] else 0
    score += 5 if row["has_wue"] else 0
    score += 7 if row["has_slope"] else 0
    score += 4 if row["has_class"] else 0
    path = row["path"].lower()
    if "c4" in path:
        score += 6
    if "trait" in path:
        score += 6
    if "response" in path or "slope" in path:
        score += 6
    if "point" in path or "site" in path:
        score += 2
    return score

def load_best_trait_response_table(inv):
    inv = inv.copy()
    inv["candidate_score"] = inv.apply(score_candidate_table, axis=1)
    inv = inv.sort_values("candidate_score", ascending=False)
    inv.to_csv(TAB / "Table_PRODUCT03dm_trait_flux_candidate_tables_ranked.csv", index=False)

    loaded = []
    for _, r in inv.head(80).iterrows():
        p = Path(r["path"])
        d = safe_read_csv(p)
        if d is None or len(d) < 5:
            continue
        loaded.append((p, d, r["candidate_score"]))

    # Prefer a single table that already has traits + response metrics.
    best = None
    best_score = -1
    best_info = None

    for p, d, base_score in loaded:
        cols = list(d.columns)

        c4 = find_col(cols, [(r"^c4(_|$)", 20), (r"c4_fraction", 30), (r"photosynthetic.*c4", 20)])
        root = find_col(cols, [(r"rooting_depth", 30), (r"root.*depth", 25), (r"root", 10)])
        arid = find_col(cols, [(r"aridity", 30), (r"aridity_index", 30), (r"^ai($|_)", 12), (r"arid", 10)])

        gpp_resp = find_col(cols, [
            (r"gpp.*slope", 30), (r"slope.*gpp", 30),
            (r"gpp.*response", 20), (r"response.*gpp", 20),
            (r"latent.*gpp", 12),
        ])
        et_resp = find_col(cols, [
            (r"et.*slope", 25), (r"slope.*et", 25),
            (r"et.*response", 18), (r"response.*et", 18),
            (r"le.*response", 12), (r"latent.*et", 12),
        ], avoid=["gpp", "wue", "uwue"])
        uwue_resp = find_col(cols, [
            (r"uwue.*slope", 30), (r"slope.*uwue", 30),
            (r"uwue.*response", 25), (r"response.*uwue", 25),
            (r"wue.*slope", 15), (r"slope.*wue", 15),
            (r"latent.*slope", 8), (r"latent_slope_change", 30)
        ])

        response_class = find_col(cols, [
            (r"response_class", 30),
            (r"satellite.*class", 20),
            (r"class", 8),
            (r"saturation", 5),
            (r"breakdown", 5),
        ])

        trait_count = sum(x is not None for x in [c4, root, arid])
        resp_count = sum(x is not None for x in [gpp_resp, et_resp, uwue_resp, response_class])
        score = base_score + trait_count * 20 + resp_count * 20 + min(len(d), 300) / 10

        if trait_count >= 1 and resp_count >= 1 and score > best_score:
            best = d.copy()
            best_score = score
            best_info = {
                "path": str(p),
                "score": score,
                "c4_col": c4,
                "root_col": root,
                "aridity_col": arid,
                "gpp_response_col": gpp_resp,
                "et_response_col": et_resp,
                "uwue_response_col": uwue_resp,
                "response_class_col": response_class,
                "n_rows": len(d),
            }

    return best, best_info, loaded

def build_from_summary_c4_only():
    # Last-resort fallback: use existing C4 summary, not ideal for new GPP/ET/uWUE tests.
    if not C4_SUMMARY.exists():
        return None, None
    c4 = pd.read_csv(C4_SUMMARY)
    return c4, {
        "path": str(C4_SUMMARY),
        "fallback": "summary_only_existing_c4_results",
    }

def standardize_analysis_table(d, info):
    df = d.copy()
    cols = list(df.columns)

    id_col = find_col(cols, [
        (r"^site_id$", 40), (r"site", 8), (r"point_id", 20), (r"pixel_id", 15), (r"id$", 3)
    ])

    product_col = find_col(cols, [
        (r"product", 20), (r"product_combo", 20), (r"et_product", 15)
    ])

    c4_col = info.get("c4_col") or find_col(cols, [(r"c4_fraction", 30), (r"^c4(_|$)", 20), (r"c4", 10)])
    root_col = info.get("root_col") or find_col(cols, [(r"rooting_depth", 30), (r"root.*depth", 25), (r"root", 10)])
    arid_col = info.get("aridity_col") or find_col(cols, [(r"aridity", 30), (r"aridity_index", 30), (r"arid", 10), (r"^ai($|_)", 10)])

    gpp_col = info.get("gpp_response_col")
    et_col = info.get("et_response_col")
    uwue_col = info.get("uwue_response_col")
    class_col = info.get("response_class_col")

    out = pd.DataFrame(index=df.index)
    if id_col:
        out["unit_id"] = df[id_col].astype(str)
    else:
        out["unit_id"] = [f"row_{i}" for i in range(len(df))]

    if product_col:
        out["product"] = df[product_col].astype(str)
    else:
        out["product"] = "unknown"

    if c4_col:
        out["c4_fraction"] = to_num(df[c4_col])
    if root_col:
        out["rooting_depth"] = to_num(df[root_col])
    if arid_col:
        out["aridity"] = to_num(df[arid_col])

    if gpp_col:
        out["gpp_response_slope"] = to_num(df[gpp_col])
    if et_col:
        out["et_response_slope"] = to_num(df[et_col])
    if uwue_col:
        out["uwue_response_slope"] = to_num(df[uwue_col])

    if class_col:
        out["response_class"] = df[class_col].astype(str).str.lower()
        out["saturation_or_breakdown"] = out["response_class"].str.contains("saturation|breakdown|sat|break", regex=True).astype(float)
        out["breakdown_class"] = out["response_class"].str.contains("breakdown|break", regex=True).astype(float)

    return out

def merge_tower_quality_filter(df):
    if not FINAL_SITE.exists():
        df["tower_quality_subset"] = "all"
        return df
    sites = pd.read_csv(FINAL_SITE)
    if "site_id" not in sites.columns:
        df["tower_quality_subset"] = "all"
        return df
    strict = set(sites.loc[sites["final_status"].eq("PASS_STRICT_FILTER"), "site_id"].astype(str))
    sens = set(sites.loc[sites["final_status"].isin(["PASS_STRICT_FILTER", "PASS_SENSITIVITY_ONLY"]), "site_id"].astype(str))

    def q(u):
        if u in strict:
            return "strict"
        if u in sens:
            return "sensitivity_only"
        return "not_quality_passing_or_not_tower"

    df["tower_quality_subset"] = df["unit_id"].map(q)
    return df

def robust_ols(df, y, xvars, label, response_family="continuous"):
    use = df[[y] + xvars].copy()
    use = use.replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(8, len(xvars) + 4):
        return None

    # Standardize continuous predictors and response for comparable coefficients.
    for c in [y] + xvars:
        use[c] = zscore(use[c])

    use = use.dropna()
    if len(use) < max(8, len(xvars) + 4):
        return None

    X = sm.add_constant(use[xvars])
    model = sm.OLS(use[y], X).fit(cov_type="HC3")

    rows = []
    for x in xvars:
        rows.append({
            "model_label": label,
            "response": y,
            "term": x,
            "n": int(model.nobs),
            "coef_standardized": float(model.params.get(x, np.nan)),
            "se_hc3": float(model.bse.get(x, np.nan)),
            "t_or_z": float(model.tvalues.get(x, np.nan)),
            "p": float(model.pvalues.get(x, np.nan)),
            "r2": float(model.rsquared),
            "aic": float(model.aic),
        })
    return rows

def logistic(df, y, xvars, label):
    use = df[[y] + xvars].copy()
    use = use.replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(20, len(xvars) + 8):
        return None
    if use[y].nunique() < 2:
        return None

    for c in xvars:
        use[c] = zscore(use[c])
    use = use.dropna()
    if len(use) < max(20, len(xvars) + 8) or use[y].nunique() < 2:
        return None

    X = sm.add_constant(use[xvars])
    try:
        model = sm.Logit(use[y], X).fit(disp=False, maxiter=200)
    except Exception:
        return None

    rows = []
    for x in xvars:
        rows.append({
            "model_label": label,
            "response": y,
            "term": x,
            "n": int(model.nobs),
            "coef_logit_standardized": float(model.params.get(x, np.nan)),
            "se": float(model.bse.get(x, np.nan)),
            "t_or_z": float(model.tvalues.get(x, np.nan)),
            "p": float(model.pvalues.get(x, np.nan)),
            "pseudo_r2": float(model.prsquared),
            "aic": float(model.aic),
        })
    return rows

def bootstrap_ci(df, y, xvars, term, n_boot=1000, seed=7):
    rng = np.random.default_rng(seed)
    use = df[[y] + xvars].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < max(10, len(xvars) + 5):
        return np.nan, np.nan
    for c in [y] + xvars:
        use[c] = zscore(use[c])
    use = use.dropna()
    if len(use) < max(10, len(xvars) + 5):
        return np.nan, np.nan

    vals = []
    n = len(use)
    for _ in range(n_boot):
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

def run_trait_flux_tests(df):
    traits = [c for c in ["c4_fraction", "rooting_depth", "aridity"] if c in df.columns and df[c].notna().sum() >= 10]
    responses = [c for c in ["gpp_response_slope", "et_response_slope", "uwue_response_slope"] if c in df.columns and df[c].notna().sum() >= 10]
    class_responses = [c for c in ["saturation_or_breakdown", "breakdown_class"] if c in df.columns and df[c].notna().sum() >= 10]

    rows = []

    # Main univariate trait -> flux response
    for y in responses:
        for x in traits:
            r = robust_ols(df, y, [x], f"univariate_{x}_to_{y}")
            if r:
                rows.extend(r)

    # Controlled mechanism: C4 + aridity + rooting depth
    if "c4_fraction" in traits:
        controls = ["c4_fraction"]
        for c in ["aridity", "rooting_depth"]:
            if c in traits:
                controls.append(c)
        if len(controls) >= 2:
            for y in responses:
                r = robust_ols(df, y, controls, f"controlled_c4_plus_covariates_to_{y}")
                if r:
                    rows.extend(r)

    # Aridity/rooting-depth controls separately.
    for focal in ["rooting_depth", "aridity"]:
        if focal in traits:
            controls = [focal]
            for c in ["c4_fraction", "aridity", "rooting_depth"]:
                if c in traits and c != focal:
                    controls.append(c)
            if len(controls) >= 2:
                for y in responses:
                    r = robust_ols(df, y, controls, f"controlled_{focal}_plus_covariates_to_{y}")
                    if r:
                        rows.extend(r)

    test = pd.DataFrame(rows)
    if len(test):
        test["q_bh"] = bh_q(test["p"])
        ci_low = []
        ci_high = []
        for _, row in test.iterrows():
            xvars = []
            label = str(row["model_label"])
            y = row["response"]
            if label.startswith("univariate"):
                xvars = [row["term"]]
            elif "controlled" in label:
                # parse model type by available traits
                if "c4_plus" in label:
                    xvars = [x for x in ["c4_fraction", "aridity", "rooting_depth"] if x in traits]
                elif "rooting_depth" in label:
                    xvars = [x for x in ["rooting_depth", "c4_fraction", "aridity"] if x in traits]
                elif "aridity" in label:
                    xvars = [x for x in ["aridity", "c4_fraction", "rooting_depth"] if x in traits]
                else:
                    xvars = [row["term"]]
            lo, hi = bootstrap_ci(df, y, xvars, row["term"], n_boot=1000)
            ci_low.append(lo)
            ci_high.append(hi)
        test["boot_ci_low"] = ci_low
        test["boot_ci_high"] = ci_high
        test["ci_excludes_zero"] = (
            (test["boot_ci_low"].notna())
            & (test["boot_ci_high"].notna())
            & ((test["boot_ci_low"] > 0) | (test["boot_ci_high"] < 0))
        )
        test["passes_trait_flux_mechanism_screen"] = (
            (test["q_bh"] <= 0.10)
            & (test["ci_excludes_zero"])
            & (test["n"] >= 20)
        )

    # Class models
    class_rows = []
    for y in class_responses:
        for x in traits:
            r = logistic(df, y, [x], f"logit_univariate_{x}_to_{y}")
            if r:
                class_rows.extend(r)

        if "c4_fraction" in traits:
            controls = [x for x in ["c4_fraction", "aridity", "rooting_depth"] if x in traits]
            if len(controls) >= 2:
                r = logistic(df, y, controls, f"logit_controlled_c4_plus_covariates_to_{y}")
                if r:
                    class_rows.extend(r)

    class_test = pd.DataFrame(class_rows)
    if len(class_test):
        class_test["q_bh"] = bh_q(class_test["p"])
        class_test["passes_class_screen"] = (class_test["q_bh"] <= 0.10) & (class_test["n"] >= 20)

    return test, class_test

def restrict_to_tower_ranked_product(df):
    # Product column is messy in existing outputs; keep GLEAM if product labels exist, otherwise keep all.
    if "product" not in df.columns:
        return df.copy(), "no_product_column_keep_all"

    lab = df["product"].astype(str).str.upper()
    if lab.str.contains("GLEAM").any():
        return df[lab.str.contains("GLEAM")].copy(), "restricted_to_GLEAM_tower_ranked_ET"
    return df.copy(), "product_column_present_but_no_GLEAM_label_keep_all"

def choose_thesis(test, class_test):
    candidates = []

    if len(test):
        sig = test[test["passes_trait_flux_mechanism_screen"].astype(bool)].copy()
        if len(sig):
            sig["abs_coef"] = sig["coef_standardized"].abs()
            sig = sig.sort_values(["q_bh", "abs_coef"], ascending=[True, False])
            for _, r in sig.iterrows():
                trait = r["term"]
                response = r["response"]
                candidates.append({
                    "trait": trait,
                    "response": response,
                    "model_label": r["model_label"],
                    "coef": r["coef_standardized"],
                    "q": r["q_bh"],
                    "n": r["n"],
                    "sentence": f"{trait} organizes {response} under compound VPD-soil moisture stress.",
                    "strength": "primary_candidate_continuous_trait_flux",
                })

    if len(class_test):
        sigc = class_test[class_test["passes_class_screen"].astype(bool)].copy()
        if len(sigc):
            sigc["abs_coef"] = sigc["coef_logit_standardized"].abs()
            sigc = sigc.sort_values(["q_bh", "abs_coef"], ascending=[True, False])
            for _, r in sigc.iterrows():
                trait = r["term"]
                response = r["response"]
                candidates.append({
                    "trait": trait,
                    "response": response,
                    "model_label": r["model_label"],
                    "coef": r["coef_logit_standardized"],
                    "q": r["q_bh"],
                    "n": r["n"],
                    "sentence": f"{trait} predicts {response} under compound VPD-soil moisture stress.",
                    "strength": "primary_candidate_class_trait_flux",
                })

    cand = pd.DataFrame(candidates)
    if len(cand):
        cand = cand.sort_values(["q", "n"], ascending=[True, False])
        thesis = cand.iloc[0].to_dict()
        return cand, thesis

    # fallback if no robust primary result
    thesis = {
        "sentence": "No trait currently survives the full mechanism screen as a primary causal thesis. The closest defensible framing is that C4 fraction shows an exploratory trait-flux signal, but not a controlled primary mechanism.",
        "strength": "no_primary_mechanism_yet",
    }
    return cand, thesis

def main():
    decision_project = {}
    if FINAL_project.exists():
        decision_project = json.loads(FINAL_project.read_text())

    inv = column_inventory()
    data, info, loaded = load_best_trait_response_table(inv)

    if data is None:
        data, info = build_from_summary_c4_only()

    if data is None:
        raise RuntimeError("Could not find any trait/response table. Check Table_PRODUCT03dl_trait_flux_column_inventory.csv.")

    standardized = standardize_analysis_table(data, info)
    standardized = merge_tower_quality_filter(standardized)
    standardized.to_csv(TAB / "Table_PRODUCT03dn_trait_flux_analysis_dataset.csv", index=False)

    gleam_df, product_filter_note = restrict_to_tower_ranked_product(standardized)
    gleam_df.to_csv(TAB / "Table_PRODUCT03do_trait_flux_analysis_dataset_tower_ranked_product.csv", index=False)

    # Run on tower-ranked product if possible.
    tests, class_tests = run_trait_flux_tests(gleam_df)
    tests.to_csv(TAB / "Table_PRODUCT03dp_trait_to_flux_response_tests.csv", index=False)
    class_tests.to_csv(TAB / "Table_PRODUCT03dq_trait_to_response_class_tests.csv", index=False)

    candidates, thesis = choose_thesis(tests, class_tests)
    candidates.to_csv(TAB / "Table_PRODUCT03dr_candidate_trait_flux_theses.csv", index=False)

    # Also run strict tower-only version if site IDs overlap.
    strict_df = gleam_df[gleam_df["tower_quality_subset"].isin(["strict", "sensitivity_only"])].copy()
    if len(strict_df) >= 10:
        tests_q, class_q = run_trait_flux_tests(strict_df)
    else:
        tests_q, class_q = pd.DataFrame(), pd.DataFrame()
    tests_q.to_csv(TAB / "Table_PRODUCT03ds_trait_to_flux_tests_quality_subset_only.csv", index=False)
    class_q.to_csv(TAB / "Table_PRODUCT03dt_trait_to_class_tests_quality_subset_only.csv", index=False)

    # Existing C4 summary context
    c4_context = pd.DataFrame()
    if C4_SUMMARY.exists():
        c4_context = pd.read_csv(C4_SUMMARY)
        c4_context.to_csv(TAB / "Table_PRODUCT03du_existing_c4_pre_specified_results_context.csv", index=False)

    # Decision
    mechanism_pass = len(candidates) > 0
    c4_controlled_survives = False
    if len(tests):
        c4_rows = tests[
            tests["term"].eq("c4_fraction")
            & tests["model_label"].astype(str).str.contains("controlled")
            & tests["passes_trait_flux_mechanism_screen"].astype(bool)
        ]
        c4_controlled_survives = len(c4_rows) > 0

    decision = {
        "generated": now(),
        "stage": "1B.6AT_trait_flux_mechanism_test",
        "analysis_table_used": info,
        "product_filter_note": product_filter_note,
        "rows_in_analysis_dataset": int(len(standardized)),
        "rows_after_tower_ranked_product_filter": int(len(gleam_df)),
        "available_traits": [c for c in ["c4_fraction", "rooting_depth", "aridity"] if c in gleam_df.columns and gleam_df[c].notna().sum() >= 10],
        "available_flux_responses": [c for c in ["gpp_response_slope", "et_response_slope", "uwue_response_slope"] if c in gleam_df.columns and gleam_df[c].notna().sum() >= 10],
        "available_class_responses": [c for c in ["saturation_or_breakdown", "breakdown_class"] if c in gleam_df.columns and gleam_df[c].notna().sum() >= 10],
        "n_trait_flux_tests": int(len(tests)),
        "n_class_tests": int(len(class_tests)),
        "n_candidate_trait_flux_theses": int(len(candidates)),
        "c4_controlled_survives_aridity_rooting_depth": bool(c4_controlled_survives),
        "best_trait_flux_thesis": thesis,
        "can_claim_trait_causes_flux_response": bool(mechanism_pass),
        "can_claim_c4_trait_causes_flux_response_after_controls": bool(c4_controlled_survives),
        "project_quality_context": {
            "strict_quality_sites_n": decision_project.get("strict_quality_sites_n"),
            "sensitivity_quality_sites_n": decision_project.get("sensitivity_quality_sites_n"),
            "strict_top_et_product": decision_project.get("strict_top_et_product"),
            "sensitivity_top_et_product": decision_project.get("sensitivity_top_et_product"),
        },
    }

    (TAB / "STAGE1B6AT_TRAIT_FLUX_MECHANISM_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # project addendum
    lines = []
    lines.append("Hi project,")
    lines.append("")
    lines.append("I agree that the manuscript thesis should not be simply that products differ. I treated product uncertainty as the guardrail, not the biological claim, and ran an explicit trait-to-flux mechanism screen on the tower-ranked product framing.")
    lines.append("")
    lines.append("The tested structure was: trait X → ecosystem flux response Y under compound VPD–soil-moisture stress Z. I tested C4 fraction, rooting depth, and aridity against GPP response slope, ET response slope, uWUE response slope, and the saturation/breakdown response class where those responses were available.")
    lines.append("")
    lines.append(f"The table used for the mechanism screen was: {info.get('path') if isinstance(info, dict) else info}.")
    lines.append(f"The product/tower guardrail was: {product_filter_note}.")
    lines.append("")
    if mechanism_pass:
        lines.append("The strongest candidate thesis from the screen is:")
        lines.append("")
        lines.append(f"{thesis.get('sentence')}")
        lines.append("")
        lines.append(f"Model: {thesis.get('model_label')}; n = {thesis.get('n')}; standardized coefficient = {thesis.get('coef'):.3f}; BH q = {thesis.get('q'):.4g}.")
    else:
        lines.append("The mechanism screen did not produce a primary controlled trait-to-flux result strong enough to claim a causal ecological mechanism yet.")
        lines.append("")
        lines.append("The closest biological result remains the exploratory C4 signal, but I would not frame it as a proven mechanism unless you think the exploratory result is sufficient for a narrower hypothesis paper.")
    lines.append("")
    if c4_controlled_survives:
        lines.append("Importantly, C4 fraction does survive the controlled screen including aridity/rooting-depth controls in at least one flux-response model.")
    else:
        lines.append("Importantly, C4 fraction does not currently survive as a controlled primary mechanism across aridity/rooting-depth controls. That means I would avoid the claim that C4 definitively causes the flux response, unless we narrow the response variable to the specific exploratory signal.")
    lines.append("")
    lines.append("Best,")
    lines.append("Akul")
    (TXT / "project_TRAIT_FLUX_MECHANISM_ADDENDUM.md").write_text("\n".join(lines), encoding="utf-8")

    report = []
    report.append("# Stage 1B.6AT trait-to-flux mechanism test")
    report.append("")
    report.append("## Decision")
    report.append("```json")
    report.append(json.dumps(decision, indent=2))
    report.append("```")
    report.append("")
    report.append("## Candidate trait-flux theses")
    report.append("```text")
    report.append(candidates.to_string(index=False) if len(candidates) else "No primary candidate thesis passed the mechanism screen.")
    report.append("```")
    report.append("")
    report.append("## Trait to continuous flux-response tests")
    report.append("```text")
    report.append(tests.sort_values('q_bh').head(80).to_string(index=False) if len(tests) else "No continuous flux-response tests were possible from the discovered table.")
    report.append("```")
    report.append("")
    report.append("## Trait to response-class tests")
    report.append("```text")
    report.append(class_tests.sort_values('q_bh').head(80).to_string(index=False) if len(class_tests) else "No response-class tests were possible from the discovered table.")
    report.append("```")
    report.append("")
    report.append("## project addendum")
    report.append("```text")
    report.append("\n".join(lines))
    report.append("```")
    (TXT / "STAGE1B6AT_TRAIT_FLUX_MECHANISM_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("")
    print("WROTE", TAB / "STAGE1B6AT_TRAIT_FLUX_MECHANISM_DECISION.json")
    print("WROTE", TAB / "Table_PRODUCT03dn_trait_flux_analysis_dataset.csv")
    print("WROTE", TAB / "Table_PRODUCT03dp_trait_to_flux_response_tests.csv")
    print("WROTE", TAB / "Table_PRODUCT03dq_trait_to_response_class_tests.csv")
    print("WROTE", TAB / "Table_PRODUCT03dr_candidate_trait_flux_theses.csv")
    print("WROTE", TXT / "project_TRAIT_FLUX_MECHANISM_ADDENDUM.md")

if __name__ == "__main__":
    main()
