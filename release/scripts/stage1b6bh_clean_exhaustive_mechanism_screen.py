from pathlib import Path
import os
import re
import json
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# Stage1B6BH: clean exhaustive ecological-mechanism screen
#
# Purpose:
#   Exhaustively screen all eligible ecological / trait / structure / climate
#   numeric features in the current point-level dataset and test whether any
#   relationship passes Reza-style rigor:
#
#   1. Full climate/environment controls.
#   2. Discovery-wide FDR.
#   3. Clean cropland / C4-crop / managed-system sensitivity.
#   4. Exact product-combination robustness.
#   5. Least-directly-LAI-dependent product pair check: GOSIF × GLEAM.
#   6. Product-dependency/circularity diagnosis.
#   7. Tower-directional anchor where tower data contain the same feature.
#   8. Programming audit: what was discovered, skipped, tested, and filtered.
#
# Design choices:
#   - No manual cherry-pick filtering.
#   - Main tests use all available complete cases for the focal test.
#   - Clean land-cover masks are run as sensitivity / gate checks, not as the
#     only discovery sample.
#   - Controls are not used to filter rows unless required by the model.
#   - If the focal predictor is itself a control-family variable, that control
#     family is omitted to avoid controlling the predictor by itself.
#   - Product tests use exact GPP × ET outcomes when available, not product
#     averages.
# =============================================================================

ROOT = Path.cwd()
OUT = ROOT / "results" / "stage1b6bh_clean_exhaustive_mechanism_screen"
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for d in [OUT, TAB, TXT, FIG]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = int(os.environ.get("SEED", "123"))
N_BOOT = int(os.environ.get("N_BOOT", "300"))
MIN_N_PRIMARY = int(os.environ.get("MIN_N_PRIMARY", "60"))
MIN_N_PRODUCT = int(os.environ.get("MIN_N_PRODUCT", "40"))
MIN_N_CLEAN = int(os.environ.get("MIN_N_CLEAN", "40"))
MIN_N_TOWER = int(os.environ.get("MIN_N_TOWER", "5"))

# To avoid insane runtime on accidental giant tables, but still be broad.
MAX_CSV_SIZE_MB = float(os.environ.get("MAX_CSV_SIZE_MB", "250"))
RUN_PRODUCT_FOR_ALL_DISCOVERY_SURVIVORS = True

rng = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# Utilities
# =============================================================================

def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def finite_numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore_arr(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - m) / sd


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def sign(x):
    x = safe_float(x)
    if not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def bh_qvalues(pvals):
    p = np.array([safe_float(v) for v in pvals], dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], alpha=0.05, method="fdr_bh")[1]
    return q


def by_qvalues(pvals):
    p = np.array([safe_float(v) for v in pvals], dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], alpha=0.05, method="fdr_by")[1]
    return q


def holm_pvalues(pvals):
    p = np.array([safe_float(v) for v in pvals], dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], alpha=0.05, method="holm")[1]
    return q


def contains_any(nm, patterns):
    return any(re.search(p, nm) for p in patterns)


def first_existing_numeric(df, candidates):
    for c in candidates:
        if c in df.columns:
            s = finite_numeric(df[c])
            if s.notna().sum() >= MIN_N_PRIMARY and s.nunique(dropna=True) > 2:
                return c
    return None


# =============================================================================
# Column families
# =============================================================================

CONTROL_FAMILIES = {
    "aridity": [
        r"(^|_)aridity($|_)", r"(^|_)arid($|_)", r"(^|_)ai($|_)",
        r"aridity_index"
    ],
    "temperature": [
        r"growing_season.*temp", r"temperature", r"(^|_)temp($|_)",
        r"mean_annual_temperature", r"(^|_)mat($|_)", r"gst"
    ],
    "precipitation": [
        r"precip", r"mean_annual_precipitation", r"(^|_)map($|_)",
        r"rainfall", r"rain"
    ],
    "soil_texture": [
        r"soil_texture_pc1", r"texture_pc1", r"soil.*texture",
        r"(^|_)sand($|_)", r"(^|_)clay($|_)", r"(^|_)silt($|_)"
    ],
    "lai_productivity": [
        r"growing_season_mean_lai", r"(^|_)lai($|_)", r"lai",
        r"fpar", r"evi", r"ndvi", r"productivity", r"baseline_gpp",
        r"mean_gpp"
    ],
    "baseline_vpd": [
        r"(^|_)vpd($|_)", r"vapor_pressure_deficit", r"vapour_pressure_deficit",
        r"baseline_vpd", r"mean_vpd"
    ],
    "baseline_soil_moisture": [
        r"soil_moisture", r"soilmoisture", r"rootzone_sm", r"root_zone",
        r"sm_root", r"(^|_)sm($|_)", r"mean_sm"
    ],
}

SPATIAL_PATTERNS = {
    "lat": [r"(^|_)lat($|_)", r"latitude"],
    "lon": [r"(^|_)lon($|_)", r"longitude", r"(^|_)lng($|_)"],
}

BAD_CANDIDATE_PATTERNS = [
    r"(^|_)id($|_)", r"point_id", r"site_id", r"grid_id", r"pixel_id",
    r"row", r"col", r"index", r"year", r"month", r"day", r"date", r"time",
    r"lat", r"lon", r"latitude", r"longitude",
    r"flag", r"qc", r"quality", r"mask", r"valid", r"weight",
    r"crop", r"cropland", r"maize", r"sorghum", r"millet", r"sugarcane",
    r"managed", r"irrig", r"tower", r"site", r"product", r"source",
    r"formula", r"p_value", r"q_value", r"ci_", r"aic", r"r2",
    r"coef", r"stderr", r"se_", r"bootstrap", r"status",
]

OUTCOME_PATTERNS = [
    r"latent_slope_change",
    r"slope_change",
    r"uwue.*response",
    r"wue.*response",
    r"stress.*response",
    r"response.*stress",
    r"breakdown",
    r"threshold_response",
    r"latent_y",
]


PRODUCT_GPP = ["GOSIF", "MODIS", "PML"]
PRODUCT_ET = ["GLEAM", "MODIS", "PML"]

GPP_DEP_RANK = {
    "GOSIF": 1,   # SIF-based but reconstruction may use optical proxies
    "MODIS": 2,   # MOD17/FPAR vegetation dependence
    "PML": 3,     # PML coupled conductance/photosynthesis/LAI dependence
}
ET_DEP_RANK = {
    "GLEAM": 0,   # least direct optical LAI entanglement
    "MODIS": 2,   # MOD16 vegetation dynamics / LAI-FPAR dependence
    "PML": 3,     # PML LAI/conductance dependence
}


# =============================================================================
# Discover candidate input table
# =============================================================================

def read_header(path):
    try:
        return list(pd.read_csv(path, nrows=5).columns)
    except Exception:
        return []


def score_csv_for_point_table(path):
    cols = read_header(path)
    if not cols:
        return -999, {}
    ncols = [norm_name(c) for c in cols]
    score = 0
    details = {}

    outcome_hits = [c for c, n in zip(cols, ncols) if contains_any(n, OUTCOME_PATTERNS)]
    control_hits = []
    for fam, pats in CONTROL_FAMILIES.items():
        hits = [c for c, n in zip(cols, ncols) if contains_any(n, pats)]
        if hits:
            score += 8
            control_hits.extend(hits)
    product_hits = []
    for c, n in zip(cols, ncols):
        up = c.upper()
        if any(g in up for g in PRODUCT_GPP) and any(e in up for e in PRODUCT_ET) and contains_any(n, OUTCOME_PATTERNS):
            product_hits.append(c)

    if outcome_hits:
        score += 50 + 4 * len(outcome_hits)
    if product_hits:
        score += 30 + 2 * len(product_hits)
    if any("point" in norm_name(c) for c in cols):
        score += 10
    if any(contains_any(norm_name(c), SPATIAL_PATTERNS["lat"]) for c in cols):
        score += 4
    if any(contains_any(norm_name(c), SPATIAL_PATTERNS["lon"]) for c in cols):
        score += 4

    ptxt = str(path).lower()
    if "point" in ptxt:
        score += 12
    if "table" in ptxt:
        score += 1
    if "exact_3x3_product_matrix" in ptxt:
        score -= 60
    if "corrected_fdr" in ptxt or "diagnosis" in ptxt:
        score -= 50
    if "tower" in ptxt:
        score -= 15
    if "read_me" in ptxt:
        score -= 100

    details = {
        "n_columns": len(cols),
        "outcome_hits": ";".join(map(str, outcome_hits[:20])),
        "control_hit_count": len(set(control_hits)),
        "product_outcome_hit_count": len(product_hits),
    }
    return score, details


def find_candidate_csvs():
    roots = []
    for rel in ["results", "data/processed", "data/raw", "data"]:
        p = ROOT / rel
        if p.exists():
            roots.append(p)
    paths = []
    for r in roots:
        for p in r.rglob("*.csv"):
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                continue
            if size_mb <= MAX_CSV_SIZE_MB:
                paths.append(p)
    scored = []
    for p in paths:
        score, details = score_csv_for_point_table(p)
        if score > 0:
            d = {"path": str(p), "score": score, **details}
            scored.append(d)
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored


def choose_point_table():
    scored = find_candidate_csvs()
    pd.DataFrame(scored).to_csv(TAB / "INPUT_CSV_DISCOVERY_SCORES.csv", index=False)

    tried = []
    for rec in scored[:80]:
        p = Path(rec["path"])
        try:
            df = pd.read_csv(p)
        except Exception as e:
            tried.append({**rec, "load_status": f"FAILED_READ: {e}"})
            continue

        n_numeric = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        outcome_cols = identify_outcome_columns(df)
        exact_cols = identify_exact_product_outcomes(df)
        ok = (len(df) >= 50) and (n_numeric >= 8) and (len(outcome_cols) >= 1)

        tried.append({
            **rec,
            "rows": len(df),
            "numeric_cols": n_numeric,
            "identified_outcomes": ";".join(outcome_cols[:20]),
            "identified_exact_product_outcomes": len(exact_cols),
            "chosen_candidate_ok": ok,
        })

        if ok:
            pd.DataFrame(tried).to_csv(TAB / "INPUT_TABLE_TRIED.csv", index=False)
            return p, df

    pd.DataFrame(tried).to_csv(TAB / "INPUT_TABLE_TRIED.csv", index=False)
    raise RuntimeError(
        "Could not identify a usable point-level table. See INPUT_CSV_DISCOVERY_SCORES.csv "
        "and INPUT_TABLE_TRIED.csv."
    )


def identify_outcome_columns(df):
    out = []
    for c in df.columns:
        n = norm_name(c)
        if contains_any(n, OUTCOME_PATTERNS):
            s = finite_numeric(df[c])
            if s.notna().sum() >= MIN_N_PRIMARY and s.nunique(dropna=True) > 2:
                out.append(c)
    return out


def choose_latent_outcome(df):
    outcome_cols = identify_outcome_columns(df)
    if not outcome_cols:
        raise RuntimeError("No latent/generic outcome column identified.")

    priorities = [
        "latent_slope_change",
        "latent_y",
        "mean_slope_change",
        "slope_change",
        "uwue_response",
        "wue_response",
        "stress_response",
    ]

    def score(c):
        n = norm_name(c)
        sc = 0
        for i, p in enumerate(priorities):
            if p in n:
                sc += 100 - i * 5
        up = c.upper()
        if any(g in up for g in PRODUCT_GPP) and any(e in up for e in PRODUCT_ET):
            sc -= 30
        if "product" in n:
            sc -= 10
        s = finite_numeric(df[c])
        sc += min(20, int(s.notna().sum() / 10))
        return sc

    outcome_cols = sorted(outcome_cols, key=score, reverse=True)
    return outcome_cols[0], outcome_cols


def identify_exact_product_outcomes(df):
    rows = []
    for c in df.columns:
        n = norm_name(c)
        if not contains_any(n, OUTCOME_PATTERNS):
            continue
        s = finite_numeric(df[c])
        if s.notna().sum() < MIN_N_PRODUCT or s.nunique(dropna=True) <= 2:
            continue
        up = c.upper()
        found = []
        for g in PRODUCT_GPP:
            for e in PRODUCT_ET:
                if g in up and e in up:
                    # MODIS×MODIS must have clear double MODIS or explicit labels.
                    if g == "MODIS" and e == "MODIS":
                        if up.count("MODIS") < 2 and "MODIS_MODIS" not in up and "MODIS__X__MODIS" not in up:
                            continue
                    found.append((g, e))
        # Avoid duplicate ambiguous pairs from MODIS appearing once.
        for g, e in found:
            rows.append({
                "gpp_product": g,
                "et_product": e,
                "outcome_col": c,
                "gpp_dependency_rank": GPP_DEP_RANK.get(g, np.nan),
                "et_dependency_rank": ET_DEP_RANK.get(e, np.nan),
                "combo_dependency_rank_sum": GPP_DEP_RANK.get(g, np.nan) + ET_DEP_RANK.get(e, np.nan),
                "is_gosif_gleam": (g == "GOSIF" and e == "GLEAM"),
            })

    # De-duplicate exact same col/pair.
    if not rows:
        return []
    tmp = pd.DataFrame(rows).drop_duplicates(["gpp_product", "et_product", "outcome_col"])
    return tmp.to_dict("records")


# =============================================================================
# Controls and candidates
# =============================================================================

def find_family_columns(df):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or finite_numeric(df[c]).notna().sum() > 0]
    out = {}
    for fam, pats in CONTROL_FAMILIES.items():
        hits = []
        for c in numeric_cols:
            n = norm_name(c)
            if contains_any(n, pats):
                s = finite_numeric(df[c])
                if s.notna().sum() >= MIN_N_PRIMARY and s.nunique(dropna=True) > 2:
                    hits.append(c)

        def pref_score(c):
            n = norm_name(c)
            sc = finite_numeric(df[c]).notna().sum()
            if "growing_season" in n:
                sc += 200
            if fam == "soil_texture" and "pc1" in n:
                sc += 300
            if fam == "lai_productivity" and "growing_season_mean_lai" in n:
                sc += 400
            if fam == "temperature" and ("mat" in n or "mean_annual_temperature" in n):
                sc += 200
            if fam == "precipitation" and ("map" in n or "mean_annual_precipitation" in n):
                sc += 200
            if fam == "baseline_vpd" and ("mean_vpd" in n or n.endswith("vpd")):
                sc += 200
            if fam == "baseline_soil_moisture" and ("mean_soil_moisture" in n or "soil_moisture" in n):
                sc += 200
            if n.endswith("_z"):
                sc -= 20
            return sc

        hits = sorted(set(hits), key=pref_score, reverse=True)
        out[fam] = hits

    spatial = {}
    for fam, pats in SPATIAL_PATTERNS.items():
        hits = []
        for c in numeric_cols:
            n = norm_name(c)
            if contains_any(n, pats):
                s = finite_numeric(df[c])
                if s.notna().sum() >= MIN_N_PRIMARY and s.nunique(dropna=True) > 2:
                    hits.append(c)
        spatial[fam] = sorted(set(hits), key=lambda c: finite_numeric(df[c]).notna().sum(), reverse=True)

    return out, spatial


def col_families(col):
    n = norm_name(col)
    fams = []
    for fam, pats in CONTROL_FAMILIES.items():
        if contains_any(n, pats):
            fams.append(fam)
    for fam, pats in SPATIAL_PATTERNS.items():
        if contains_any(n, pats):
            fams.append(fam)
    return fams


def choose_controls_for_model(df, family_cols, spatial_cols, focal_base_col, moderator_col=None):
    exclude_fams = set(col_families(focal_base_col))
    if moderator_col is not None:
        exclude_fams.update(col_families(moderator_col))

    controls = []
    omitted = []

    for fam in [
        "aridity",
        "temperature",
        "precipitation",
        "soil_texture",
        "lai_productivity",
        "baseline_vpd",
        "baseline_soil_moisture",
    ]:
        if fam in exclude_fams:
            omitted.append(f"{fam}:omitted_self_or_moderator")
            continue
        hits = family_cols.get(fam, [])
        if hits:
            controls.append(hits[0])
        else:
            omitted.append(f"{fam}:not_found")

    # Spatial controls are included if available. This is not a row filter.
    for fam in ["lat", "lon"]:
        hits = spatial_cols.get(fam, [])
        if hits and fam not in exclude_fams:
            controls.append(hits[0])
        elif not hits:
            omitted.append(f"{fam}:not_found")
        else:
            omitted.append(f"{fam}:omitted_self_or_moderator")

    # Remove duplicates while preserving order.
    seen = set()
    controls2 = []
    for c in controls:
        if c not in seen and c != focal_base_col and c != moderator_col:
            controls2.append(c)
            seen.add(c)

    return controls2, omitted


def is_bad_candidate_col(c, outcome_cols, exact_product_outcomes):
    n = norm_name(c)
    if c in outcome_cols:
        return True
    if c in [r["outcome_col"] for r in exact_product_outcomes]:
        return True
    if contains_any(n, BAD_CANDIDATE_PATTERNS):
        # Keep actual ecological C4 fraction despite c4_crop exclusions.
        if "c4" in n and "crop" not in n and "maize" not in n and "sorghum" not in n and "millet" not in n and "sugarcane" not in n:
            return False
        return True
    return False


def discover_base_candidates(df, outcome_cols, exact_product_outcomes):
    rows = []
    candidates = []
    for c in df.columns:
        if is_bad_candidate_col(c, outcome_cols, exact_product_outcomes):
            rows.append({"column": c, "eligible": False, "reason": "bad_name_or_outcome"})
            continue

        s = finite_numeric(df[c])
        n_nonmiss = int(s.notna().sum())
        n_unique = int(s.nunique(dropna=True))
        missing_frac = float(1 - n_nonmiss / len(df)) if len(df) else 1

        if n_nonmiss < MIN_N_PRIMARY:
            rows.append({"column": c, "eligible": False, "reason": f"too_few_nonmissing_{n_nonmiss}"})
            continue
        if n_unique <= 2:
            rows.append({"column": c, "eligible": False, "reason": f"too_few_unique_{n_unique}"})
            continue
        if missing_frac > 0.70:
            rows.append({"column": c, "eligible": False, "reason": f"missing_frac_gt_0.70_{missing_frac:.3f}"})
            continue
        if not np.isfinite(np.nanstd(s.values)) or np.nanstd(s.values) == 0:
            rows.append({"column": c, "eligible": False, "reason": "zero_variance"})
            continue

        candidates.append(c)
        rows.append({
            "column": c,
            "eligible": True,
            "reason": "eligible",
            "n_nonmissing": n_nonmiss,
            "n_unique": n_unique,
            "missing_frac": missing_frac,
            "families": ";".join(col_families(c)),
        })

    # Avoid duplicate raw/z columns: since every predictor is standardized anyway,
    # prefer the non-z raw column when both exist.
    norm_to_cols = defaultdict(list)
    for c in candidates:
        n = norm_name(c)
        n2 = re.sub(r"_z$", "", n)
        norm_to_cols[n2].append(c)

    final = []
    removed_dupes = set()
    for key, cols in norm_to_cols.items():
        if len(cols) == 1:
            final.append(cols[0])
        else:
            raw = [c for c in cols if not norm_name(c).endswith("_z")]
            keep = raw[0] if raw else cols[0]
            final.append(keep)
            for c in cols:
                if c != keep:
                    removed_dupes.add(c)

    for r in rows:
        if r["column"] in removed_dupes:
            r["eligible"] = False
            r["reason"] = "duplicate_z_or_scaled_version_removed"

    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(TAB / "BASE_FEATURE_CANDIDATE_AUDIT.csv", index=False)

    return final


# =============================================================================
# Mechanism construction
# =============================================================================

def build_mechanism_list(df, base_candidates, family_cols):
    mechanisms = []

    # Potential moderators for ecological interaction tests.
    moderator_candidates = []
    for fam in ["temperature", "baseline_vpd", "baseline_soil_moisture", "aridity", "precipitation"]:
        hits = family_cols.get(fam, [])
        if hits:
            moderator_candidates.append((fam, hits[0]))

    for base in base_candidates:
        x = finite_numeric(df[base])
        if x.notna().sum() < MIN_N_PRIMARY:
            continue

        mechanisms.append({
            "mechanism_id": f"linear__{norm_name(base)}",
            "base_feature": base,
            "mechanism_type": "linear_main_effect",
            "params": {},
            "moderator": None,
        })

        # Quantile thresholds / tails.
        xz = pd.Series(zscore_arr(x.values), index=df.index)
        vals = xz.dropna().values
        if len(vals) >= MIN_N_PRIMARY:
            qs = {}
            for q in [0.10, 0.25, 0.75, 0.90]:
                try:
                    qs[q] = float(np.nanquantile(vals, q))
                except Exception:
                    pass

            for q in [0.75, 0.90]:
                if q in qs:
                    mechanisms.append({
                        "mechanism_id": f"high_tail_q{int(q*100)}__{norm_name(base)}",
                        "base_feature": base,
                        "mechanism_type": "high_tail_indicator",
                        "params": {"q": q, "cut_z": qs[q]},
                        "moderator": None,
                    })
                    mechanisms.append({
                        "mechanism_id": f"hinge_high_q{int(q*100)}__{norm_name(base)}",
                        "base_feature": base,
                        "mechanism_type": "high_hinge_threshold",
                        "params": {"q": q, "cut_z": qs[q]},
                        "moderator": None,
                    })

            for q in [0.10, 0.25]:
                if q in qs:
                    mechanisms.append({
                        "mechanism_id": f"low_tail_q{int(q*100)}__{norm_name(base)}",
                        "base_feature": base,
                        "mechanism_type": "low_tail_indicator",
                        "params": {"q": q, "cut_z": qs[q]},
                        "moderator": None,
                    })
                    mechanisms.append({
                        "mechanism_id": f"hinge_low_q{int(q*100)}__{norm_name(base)}",
                        "base_feature": base,
                        "mechanism_type": "low_hinge_threshold",
                        "params": {"q": q, "cut_z": qs[q]},
                        "moderator": None,
                    })

            mechanisms.append({
                "mechanism_id": f"quadratic__{norm_name(base)}",
                "base_feature": base,
                "mechanism_type": "quadratic_nonlinearity",
                "params": {},
                "moderator": None,
            })

        # Interactions with major ecological/climate axes.
        for modfam, modcol in moderator_candidates:
            if modcol == base:
                continue
            mechanisms.append({
                "mechanism_id": f"interaction_with_{norm_name(modcol)}__{norm_name(base)}",
                "base_feature": base,
                "mechanism_type": f"interaction_with_{modfam}",
                "params": {},
                "moderator": modcol,
            })

    pd.DataFrame(mechanisms).to_csv(TAB / "MECHANISM_LIBRARY.csv", index=False)
    return mechanisms


def construct_terms(df, mech):
    base = mech["base_feature"]
    kind = mech["mechanism_type"]
    x = finite_numeric(df[base])
    xz = pd.Series(zscore_arr(x.values), index=df.index)

    full_terms = {}
    reduced_terms = {}
    focal_name = "focal"

    if kind == "linear_main_effect":
        full_terms[focal_name] = xz

    elif kind == "high_tail_indicator":
        cut = mech["params"]["cut_z"]
        full_terms[focal_name] = (xz >= cut).astype(float)

    elif kind == "low_tail_indicator":
        cut = mech["params"]["cut_z"]
        full_terms[focal_name] = (xz <= cut).astype(float)

    elif kind == "high_hinge_threshold":
        cut = mech["params"]["cut_z"]
        full_terms["base_linear"] = xz
        reduced_terms["base_linear"] = xz
        full_terms[focal_name] = np.maximum(0, xz - cut)

    elif kind == "low_hinge_threshold":
        cut = mech["params"]["cut_z"]
        full_terms["base_linear"] = xz
        reduced_terms["base_linear"] = xz
        full_terms[focal_name] = np.maximum(0, cut - xz)

    elif kind == "quadratic_nonlinearity":
        full_terms["base_linear"] = xz
        reduced_terms["base_linear"] = xz
        full_terms[focal_name] = xz ** 2

    elif kind.startswith("interaction_with_"):
        mod = mech["moderator"]
        mz = pd.Series(zscore_arr(finite_numeric(df[mod]).values), index=df.index)
        full_terms["base_linear"] = xz
        full_terms["moderator_linear"] = mz
        reduced_terms["base_linear"] = xz
        reduced_terms["moderator_linear"] = mz
        full_terms[focal_name] = xz * mz

    else:
        raise ValueError(f"Unknown mechanism_type: {kind}")

    return full_terms, reduced_terms, focal_name


def build_design(df, y_col, mech, family_cols, spatial_cols, subset_mask=None):
    if subset_mask is None:
        subset_mask = pd.Series(True, index=df.index)
    else:
        subset_mask = pd.Series(subset_mask, index=df.index).fillna(False).astype(bool)

    full_terms, reduced_terms, focal_name = construct_terms(df, mech)

    controls, omitted_controls = choose_controls_for_model(
        df,
        family_cols,
        spatial_cols,
        focal_base_col=mech["base_feature"],
        moderator_col=mech.get("moderator"),
    )

    for c in controls:
        z = pd.Series(zscore_arr(finite_numeric(df[c]).values), index=df.index)
        nm = f"control__{norm_name(c)}"
        full_terms[nm] = z
        reduced_terms[nm] = z

    data = pd.DataFrame({"y": finite_numeric(df[y_col])}, index=df.index)
    for k, v in full_terms.items():
        data[k] = pd.Series(v, index=df.index)
    for k, v in reduced_terms.items():
        if k not in data.columns:
            data[k] = pd.Series(v, index=df.index)

    data = data.loc[subset_mask].replace([np.inf, -np.inf], np.nan).dropna()

    # Standardize y and all terms after complete-case selection.
    if len(data) > 0:
        data["y"] = zscore_arr(data["y"].values)
        for c in [x for x in data.columns if x != "y"]:
            data[c] = zscore_arr(data[c].values)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

    return data, list(full_terms.keys()), list(reduced_terms.keys()), focal_name, controls, omitted_controls


def fit_from_design(data, full_cols, reduced_cols, focal_name, min_n):
    if data is None or len(data) < min_n:
        return {
            "status": "TOO_FEW_COMPLETE_CASES",
            "n": 0 if data is None else len(data),
        }

    if focal_name not in full_cols:
        return {"status": "FOCAL_NOT_IN_MODEL", "n": len(data)}

    if data[focal_name].nunique(dropna=True) <= 1:
        return {"status": "FOCAL_ZERO_VARIANCE_AFTER_COMPLETE_CASE", "n": len(data)}

    # Remove accidental duplicate reduced/full terms.
    full_cols = list(dict.fromkeys(full_cols))
    reduced_cols = list(dict.fromkeys(reduced_cols))

    try:
        X_full = sm.add_constant(data[full_cols], has_constant="add")
        X_red = sm.add_constant(data[reduced_cols], has_constant="add") if reduced_cols else sm.add_constant(
            pd.DataFrame(index=data.index), has_constant="add"
        )

        full = sm.OLS(data["y"], X_full).fit()
        red = sm.OLS(data["y"], X_red).fit()
        rob = full.get_robustcov_results(cov_type="HC3")

        names = list(full.model.exog_names)
        idx = names.index(focal_name)

        coef = float(full.params[focal_name])
        se = float(rob.bse[idx])
        p = float(rob.pvalues[idx])
        ci_low, ci_high = map(float, rob.conf_int()[idx])

        try:
            nested_p = float(full.compare_f_test(red)[1])
        except Exception:
            nested_p = np.nan

        delta_r2 = float(full.rsquared - red.rsquared)
        delta_aic = float(full.aic - red.aic)

        # Focal VIF.
        focal_vif = np.nan
        try:
            if len(full_cols) >= 2:
                Xv = data[full_cols].copy()
                Xv = Xv.loc[:, Xv.std(axis=0) > 0]
                if focal_name in Xv.columns and Xv.shape[1] >= 2:
                    focal_idx = list(Xv.columns).index(focal_name)
                    focal_vif = float(variance_inflation_factor(Xv.values, focal_idx))
        except Exception:
            focal_vif = np.nan

        return {
            "status": "FIT_OK",
            "n": int(len(data)),
            "coef": coef,
            "se_hc3": se,
            "p": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_excludes_zero": bool((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)),
            "full_r2": float(full.rsquared),
            "reduced_r2": float(red.rsquared),
            "delta_r2": delta_r2,
            "full_aic": float(full.aic),
            "reduced_aic": float(red.aic),
            "delta_aic_full_minus_reduced": delta_aic,
            "nested_f_p": nested_p,
            "focal_vif": focal_vif,
        }

    except Exception as e:
        return {
            "status": f"FIT_FAILED: {type(e).__name__}: {e}",
            "n": int(len(data)),
        }


def fit_mechanism(df, y_col, mech, family_cols, spatial_cols, subset_mask=None, min_n=MIN_N_PRIMARY):
    data, full_cols, reduced_cols, focal_name, controls, omitted_controls = build_design(
        df, y_col, mech, family_cols, spatial_cols, subset_mask=subset_mask
    )
    res = fit_from_design(data, full_cols, reduced_cols, focal_name, min_n=min_n)
    res.update({
        "mechanism_id": mech["mechanism_id"],
        "base_feature": mech["base_feature"],
        "mechanism_type": mech["mechanism_type"],
        "moderator": mech.get("moderator"),
        "outcome_col": y_col,
        "controls_used": ";".join(controls),
        "controls_omitted": ";".join(omitted_controls),
    })
    return res


def bootstrap_and_loo(df, y_col, mech, family_cols, spatial_cols, subset_mask=None):
    data, full_cols, reduced_cols, focal_name, controls, omitted_controls = build_design(
        df, y_col, mech, family_cols, spatial_cols, subset_mask=subset_mask
    )
    if data is None or len(data) < MIN_N_PRIMARY:
        return {
            "boot_status": "TOO_FEW_COMPLETE_CASES",
            "boot_n": 0 if data is None else len(data),
        }

    coefs = []
    n = len(data)

    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        bd = data.iloc[idx].copy()
        fit = fit_from_design(bd, full_cols, reduced_cols, focal_name, min_n=max(20, int(0.5 * MIN_N_PRIMARY)))
        if fit.get("status") == "FIT_OK" and np.isfinite(fit.get("coef", np.nan)):
            coefs.append(fit["coef"])

    if len(coefs) >= max(30, int(0.1 * N_BOOT)):
        ci = np.nanquantile(coefs, [0.025, 0.975])
        boot_median = float(np.nanmedian(coefs))
        boot_sign_frac = float(np.mean(np.sign(coefs) == np.sign(boot_median))) if boot_median != 0 else np.nan
        boot_ci_excludes_zero = bool((ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0))
    else:
        ci = [np.nan, np.nan]
        boot_median = np.nan
        boot_sign_frac = np.nan
        boot_ci_excludes_zero = False

    # Leave-one-out sign stability.
    loo_coefs = []
    if n <= 500:
        for i in range(n):
            ld = data.drop(data.index[i])
            fit = fit_from_design(ld, full_cols, reduced_cols, focal_name, min_n=max(20, MIN_N_PRIMARY - 1))
            if fit.get("status") == "FIT_OK" and np.isfinite(fit.get("coef", np.nan)):
                loo_coefs.append(fit["coef"])

    if loo_coefs:
        main_sign = sign(np.nanmedian(loo_coefs))
        loo_sign_stability = float(np.mean([sign(v) == main_sign for v in loo_coefs])) if main_sign != 0 else np.nan
    else:
        loo_sign_stability = np.nan

    return {
        "boot_status": "FIT_OK" if len(coefs) else "NO_VALID_BOOTSTRAPS",
        "boot_n": int(n),
        "n_boot_requested": int(N_BOOT),
        "n_boot_valid": int(len(coefs)),
        "boot_median_coef": boot_median,
        "boot_ci_low": float(ci[0]),
        "boot_ci_high": float(ci[1]),
        "boot_ci_excludes_zero": boot_ci_excludes_zero,
        "boot_sign_fraction": boot_sign_frac,
        "loo_sign_stability": loo_sign_stability,
    }


# =============================================================================
# Land-cover / C4-crop clean mask
# =============================================================================

def infer_clean_landcover_mask(df):
    mask = pd.Series(True, index=df.index)
    used_cols = []
    notes = []

    bad_patterns = [
        "crop", "cropland", "managed", "irrig", "maize", "corn",
        "sorghum", "millet", "sugarcane", "sugar_cane", "c4_crop"
    ]
    good_patterns = ["natural_grass", "grassland_natural", "is_natural"]

    for c in df.columns:
        n = norm_name(c)
        if any(g in n for g in good_patterns):
            s = df[c]
            used_cols.append(c)
            if pd.api.types.is_bool_dtype(s):
                mask &= s.fillna(False).astype(bool)
            elif pd.api.types.is_numeric_dtype(s):
                mask &= finite_numeric(s).fillna(0) > 0
            else:
                mask &= s.astype(str).str.lower().str.contains("true|yes|natural|grass", regex=True, na=False)
            notes.append(f"{c}:required_true")

    for c in df.columns:
        n = norm_name(c)
        if any(b in n for b in bad_patterns):
            # Do not treat real c4 fraction as crop flag.
            if "c4" in n and "crop" not in n and not any(k in n for k in ["maize", "sorghum", "millet", "sugarcane", "corn"]):
                continue

            s = df[c]
            used_cols.append(c)
            if pd.api.types.is_bool_dtype(s):
                bad = s.fillna(False).astype(bool)
            elif pd.api.types.is_numeric_dtype(s):
                bad = finite_numeric(s).fillna(0) > 0
            else:
                bad = s.astype(str).str.lower().str.contains(
                    "true|yes|crop|cropland|managed|irrig|maize|corn|sorghum|millet|sugar",
                    regex=True,
                    na=False,
                )
            mask &= ~bad
            notes.append(f"{c}:required_false")

    if not used_cols:
        return pd.Series(True, index=df.index), {
            "landcover_status": "NO_LANDCOVER_OR_CROP_FLAG_COLUMNS_FOUND",
            "landcover_columns_used": "",
            "n_clean": int(len(df)),
            "n_total": int(len(df)),
            "notes": "",
        }

    return mask.fillna(False).astype(bool), {
        "landcover_status": "CLEAN_MASK_INFERRED",
        "landcover_columns_used": ";".join(sorted(set(used_cols))),
        "n_clean": int(mask.sum()),
        "n_total": int(len(df)),
        "notes": ";".join(notes),
    }


# =============================================================================
# Product robustness
# =============================================================================

def run_product_tests(df, exact_outcomes, mechanisms, family_cols, spatial_cols, clean_mask):
    rows = []
    for i, mech in enumerate(mechanisms, start=1):
        if i % 25 == 0:
            print(f"  product tests for survivor {i}/{len(mechanisms)}: {mech['mechanism_id']}", flush=True)

        for sample_mode, mask in [
            ("all_available_complete_case", None),
            ("cropland_clean_complete_case", clean_mask),
        ]:
            for rec in exact_outcomes:
                ycol = rec["outcome_col"]
                res = fit_mechanism(
                    df, ycol, mech, family_cols, spatial_cols,
                    subset_mask=mask,
                    min_n=MIN_N_PRODUCT,
                )
                res.update({
                    "sample_mode": sample_mode,
                    "gpp_product": rec["gpp_product"],
                    "et_product": rec["et_product"],
                    "gpp_dependency_rank": rec["gpp_dependency_rank"],
                    "et_dependency_rank": rec["et_dependency_rank"],
                    "combo_dependency_rank_sum": rec["combo_dependency_rank_sum"],
                    "is_gosif_gleam": rec["is_gosif_gleam"],
                })
                rows.append(res)

    prod = pd.DataFrame(rows)
    if prod.empty:
        return prod, pd.DataFrame()

    prod["product_bh_q_within_mechanism_sample"] = np.nan
    prod["product_by_q_within_mechanism_sample"] = np.nan
    for (mid, smode), idx in prod.groupby(["mechanism_id", "sample_mode"]).groups.items():
        p = prod.loc[idx, "p"].values
        prod.loc[idx, "product_bh_q_within_mechanism_sample"] = bh_qvalues(p)
        prod.loc[idx, "product_by_q_within_mechanism_sample"] = by_qvalues(p)

    summaries = []
    for (mid, smode), g in prod.groupby(["mechanism_id", "sample_mode"]):
        fit = g[g["status"] == "FIT_OK"].copy()
        if fit.empty:
            summaries.append({
                "mechanism_id": mid,
                "sample_mode": smode,
                "product_gate_status": "NO_FIT_OK_PRODUCT_ROWS",
            })
            continue

        # Primary sign is median product sign if discovery sign not merged yet.
        med_coef = float(np.nanmedian(fit["coef"]))
        expected_sign = sign(med_coef)

        sig_same = fit[(fit["p"] < 0.05) & (fit["ci_excludes_zero"] == True) & (np.sign(fit["coef"]) == expected_sign)]
        sig_opp = fit[(fit["p"] < 0.05) & (fit["ci_excludes_zero"] == True) & (np.sign(fit["coef"]) == -expected_sign)]

        least = fit[fit["is_gosif_gleam"] == True]
        least_pass = False
        least_p = np.nan
        least_q = np.nan
        least_coef = np.nan
        least_status = "GOSIF_GLEAM_NOT_FOUND"
        if not least.empty:
            # If multiple GOSIF×GLEAM columns somehow exist, use strongest complete-case result.
            least2 = least.sort_values("p", na_position="last").head(1)
            r = least2.iloc[0]
            least_p = safe_float(r.get("p"))
            least_q = safe_float(r.get("product_bh_q_within_mechanism_sample"))
            least_coef = safe_float(r.get("coef"))
            least_pass = (
                r.get("status") == "FIT_OK"
                and np.isfinite(least_p)
                and least_p < 0.05
                and np.isfinite(least_q)
                and least_q < 0.10
                and bool(r.get("ci_excludes_zero")) is True
                and sign(least_coef) == expected_sign
                and safe_float(r.get("delta_aic_full_minus_reduced")) < 0
            )
            least_status = "PASS" if least_pass else "FAIL"

        sign_consistency = float(np.mean(np.sign(fit["coef"]) == expected_sign)) if expected_sign != 0 else np.nan
        all_same_sign = bool(sign_consistency == 1.0) if np.isfinite(sign_consistency) else False

        rho_abs, rho_neg = np.nan, np.nan
        try:
            rho_abs = float(stats.spearmanr(
                fit["combo_dependency_rank_sum"],
                np.abs(fit["coef"]),
                nan_policy="omit",
            ).correlation)
        except Exception:
            pass

        try:
            # If expected sign is negative, stronger negative = -coef.
            # If expected sign is positive, stronger positive = coef.
            strength = expected_sign * fit["coef"]
            rho_neg = float(stats.spearmanr(
                fit["combo_dependency_rank_sum"],
                strength,
                nan_policy="omit",
            ).correlation)
        except Exception:
            pass

        dependency_flag = bool(np.isfinite(rho_abs) and rho_abs > 0.50 and not least_pass)

        product_gate_pass = bool(
            least_pass
            and sign_consistency >= 0.75
            and len(sig_opp) == 0
            and not dependency_flag
        )

        strict_all_combo_gate_pass = bool(
            least_pass
            and all_same_sign
            and len(sig_opp) == 0
            and not dependency_flag
        )

        summaries.append({
            "mechanism_id": mid,
            "sample_mode": smode,
            "product_gate_status": "PASS" if product_gate_pass else "FAIL",
            "strict_all_product_sign_gate_status": "PASS" if strict_all_combo_gate_pass else "FAIL",
            "n_product_rows_fit_ok": int(len(fit)),
            "n_product_rows_total": int(len(g)),
            "expected_product_sign": expected_sign,
            "product_median_coef": med_coef,
            "product_sign_consistency": sign_consistency,
            "n_product_sig_same_direction_p_lt_0p05": int(len(sig_same)),
            "n_product_sig_opposite_direction_p_lt_0p05": int(len(sig_opp)),
            "gosif_gleam_status": least_status,
            "gosif_gleam_coef": least_coef,
            "gosif_gleam_p": least_p,
            "gosif_gleam_bh_q": least_q,
            "spearman_dependency_rank_vs_abs_coef": rho_abs,
            "spearman_dependency_rank_vs_directional_strength": rho_neg,
            "dependency_flag_stronger_when_more_algorithmically_entangled": dependency_flag,
        })

    return prod, pd.DataFrame(summaries)


# =============================================================================
# Tower inventory and directional anchor
# =============================================================================

def find_tower_tables():
    rows = []
    for p in list((ROOT / "results").rglob("*.csv")) + list((ROOT / "data").rglob("*.csv")):
        if "tower" not in str(p).lower() and "flux" not in str(p).lower() and "ameriflux" not in str(p).lower():
            continue
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except Exception:
            continue
        if size_mb > MAX_CSV_SIZE_MB:
            continue
        try:
            cols = list(pd.read_csv(p, nrows=5).columns)
        except Exception:
            continue
        ncols = [norm_name(c) for c in cols]
        outcome_hits = [c for c, n in zip(cols, ncols) if contains_any(n, OUTCOME_PATTERNS) or ("uwue" in n and "response" in n)]
        site_hits = [c for c, n in zip(cols, ncols) if "site" in n or "tower" in n]
        product_hits = [c for c, n in zip(cols, ncols) if "product" in n or any(x.lower() in n for x in ["gleam", "modis", "pml", "gosif"])]
        if outcome_hits or site_hits or product_hits:
            rows.append({
                "path": str(p),
                "n_columns": len(cols),
                "outcome_hits": ";".join(outcome_hits[:20]),
                "site_hits": ";".join(site_hits[:20]),
                "product_hits": ";".join(product_hits[:20]),
            })
    inv = pd.DataFrame(rows).drop_duplicates()
    inv.to_csv(TAB / "TOWER_TABLE_INVENTORY.csv", index=False)
    return inv


def read_possible_tower_dfs(inv):
    dfs = []
    if inv.empty:
        return dfs
    for _, r in inv.iterrows():
        p = Path(r["path"])
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if len(d) < MIN_N_TOWER:
            continue
        dfs.append((p, d))
    return dfs


def choose_tower_y_col(tdf):
    candidates = []
    for c in tdf.columns:
        n = norm_name(c)
        s = finite_numeric(tdf[c])
        if s.notna().sum() < MIN_N_TOWER or s.nunique(dropna=True) <= 2:
            continue
        sc = 0
        if "tower" in n and "uwue" in n:
            sc += 100
        if "uwue" in n and "response" in n:
            sc += 80
        if "slope_change" in n:
            sc += 70
        if "latent" in n:
            sc += 20
        if sc > 0:
            candidates.append((sc, c))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def find_feature_in_tower(tdf, base_feature):
    target = norm_name(base_feature)
    candidates = []
    for c in tdf.columns:
        n = norm_name(c)
        s = finite_numeric(tdf[c])
        if s.notna().sum() < MIN_N_TOWER or s.nunique(dropna=True) <= 2:
            continue
        sc = 0
        if n == target:
            sc += 100
        if target in n or n in target:
            sc += 50
        if "nearest" in n or "sat" in n:
            sc += 5
        if sc > 0:
            candidates.append((sc, c))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def tower_directional_tests(mechanisms, primary_df, tower_dfs):
    rows = []

    # Only tower-test mechanisms that passed discovery at least partially.
    mids = set(primary_df.loc[primary_df["discovery_survivor_for_gate_testing"] == True, "mechanism_id"])
    mechs = [m for m in mechanisms if m["mechanism_id"] in mids]

    for mech in mechs:
        best = None
        for p, tdf in tower_dfs:
            ycol = choose_tower_y_col(tdf)
            xcol = find_feature_in_tower(tdf, mech["base_feature"])
            if ycol is None or xcol is None:
                continue

            # Simple directional tower anchor. Do not pretend full tower validation.
            x = finite_numeric(tdf[xcol])
            y = finite_numeric(tdf[ycol])
            tmp = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(tmp) < MIN_N_TOWER or tmp["x"].nunique() <= 2 or tmp["y"].nunique() <= 2:
                continue

            rho, sp = stats.spearmanr(tmp["x"], tmp["y"], nan_policy="omit")
            X = sm.add_constant(pd.DataFrame({"x": zscore_arr(tmp["x"].values)}), has_constant="add")
            yy = zscore_arr(tmp["y"].values)
            try:
                fit = sm.OLS(yy, X).fit()
                coef = float(fit.params["x"])
                pval = float(fit.pvalues["x"])
            except Exception:
                coef, pval = np.nan, np.nan

            rec = {
                "mechanism_id": mech["mechanism_id"],
                "base_feature": mech["base_feature"],
                "tower_table": str(p),
                "tower_y_col": ycol,
                "tower_x_col": xcol,
                "n_tower": int(len(tmp)),
                "tower_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "tower_spearman_p": float(sp) if np.isfinite(sp) else np.nan,
                "tower_ols_coef": coef,
                "tower_ols_p": pval,
            }

            if best is None:
                best = rec
            else:
                old_score = -safe_float(best.get("tower_ols_p", np.inf))
                new_score = -safe_float(rec.get("tower_ols_p", np.inf))
                if new_score > old_score:
                    best = rec

        if best is None:
            rows.append({
                "mechanism_id": mech["mechanism_id"],
                "base_feature": mech["base_feature"],
                "tower_gate_status": "NOT_TESTABLE_NO_MATCHING_TOWER_FEATURE_AND_RESPONSE",
            })
        else:
            rows.append(best)

    t = pd.DataFrame(rows)
    if t.empty:
        return t

    # Merge expected sign from primary.
    exp = primary_df[["mechanism_id", "coef"]].rename(columns={"coef": "primary_coef"})
    t = t.merge(exp, on="mechanism_id", how="left")
    t["same_direction_as_primary"] = t.apply(
        lambda r: sign(r.get("tower_ols_coef")) == sign(r.get("primary_coef"))
        if np.isfinite(safe_float(r.get("tower_ols_coef"))) and np.isfinite(safe_float(r.get("primary_coef")))
        else False,
        axis=1,
    )
    t["tower_gate_status"] = t.apply(
        lambda r: "PASS_DIRECTIONAL_P_LT_0P10"
        if r.get("same_direction_as_primary") is True and safe_float(r.get("tower_ols_p")) < 0.10
        else (
            "PASS_DIRECTION_ONLY_UNDERPOWERED"
            if r.get("same_direction_as_primary") is True
            else r.get("tower_gate_status", "FAIL_DIRECTION_MISMATCH_OR_NOT_SIGNIFICANT")
        ),
        axis=1,
    )
    return t


# =============================================================================
# Main
# =============================================================================

def main():
    print("Stage1B6BH clean exhaustive ecological-mechanism screen", flush=True)
    print(f"Root: {ROOT}", flush=True)
    print(f"Output: {OUT}", flush=True)
    print(f"N_BOOT={N_BOOT}", flush=True)

    input_path, df = choose_point_table()
    print(f"Chosen point table: {input_path}", flush=True)
    print(f"Rows={len(df)}, columns={len(df.columns)}", flush=True)

    latent_y, all_outcome_cols = choose_latent_outcome(df)
    exact_outcomes = identify_exact_product_outcomes(df)
    family_cols, spatial_cols = find_family_columns(df)

    pd.DataFrame({
        "control_family": list(family_cols.keys()),
        "candidate_columns_ranked": [";".join(v) for v in family_cols.values()],
    }).to_csv(TAB / "CONTROL_FAMILY_COLUMN_CANDIDATES.csv", index=False)

    pd.DataFrame(exact_outcomes).to_csv(TAB / "EXACT_PRODUCT_OUTCOME_COLUMNS.csv", index=False)

    clean_mask, clean_info = infer_clean_landcover_mask(df)
    pd.DataFrame([clean_info]).to_csv(TAB / "LANDCOVER_C4_CROP_CLEAN_MASK_AUDIT.csv", index=False)

    base_candidates = discover_base_candidates(df, all_outcome_cols, exact_outcomes)
    mechanisms = build_mechanism_list(df, base_candidates, family_cols)

    print(f"Latent outcome: {latent_y}", flush=True)
    print(f"Base feature candidates: {len(base_candidates)}", flush=True)
    print(f"Mechanisms to test: {len(mechanisms)}", flush=True)
    print(f"Exact product outcomes found: {len(exact_outcomes)}", flush=True)
    print(f"Clean land-cover n: {int(clean_mask.sum())}/{len(df)}", flush=True)

    # -------------------------------------------------------------------------
    # Discovery screen on latent/global outcome.
    # -------------------------------------------------------------------------
    rows = []
    for i, mech in enumerate(mechanisms, start=1):
        if i % 100 == 0:
            print(f"  primary discovery model {i}/{len(mechanisms)}", flush=True)
        res = fit_mechanism(df, latent_y, mech, family_cols, spatial_cols, subset_mask=None, min_n=MIN_N_PRIMARY)
        rows.append(res)

    primary = pd.DataFrame(rows)
    primary["bh_q"] = bh_qvalues(primary["p"].values if "p" in primary.columns else [])
    primary["by_q"] = by_qvalues(primary["p"].values if "p" in primary.columns else [])
    primary["holm_p"] = holm_pvalues(primary["p"].values if "p" in primary.columns else [])

    primary["primary_basic_pass"] = (
        (primary["status"] == "FIT_OK")
        & (primary["p"] < 0.05)
        & (primary["bh_q"] < 0.05)
        & (primary["ci_excludes_zero"] == True)
        & (primary["delta_r2"] > 0)
        & (primary["delta_aic_full_minus_reduced"] < 0)
        & (primary["nested_f_p"] < 0.05)
    )

    # Slightly wider survivor set for expensive gate testing so we can inspect
    # near misses without accepting them.
    primary["discovery_survivor_for_gate_testing"] = (
        (primary["status"] == "FIT_OK")
        & (primary["p"] < 0.05)
        & (primary["bh_q"] < 0.10)
        & (primary["ci_excludes_zero"] == True)
        & (primary["delta_aic_full_minus_reduced"] < 0)
    )

    primary = primary.sort_values(["primary_basic_pass", "bh_q", "p"], ascending=[False, True, True])
    primary.to_csv(TAB / "PRIMARY_EXHAUSTIVE_DISCOVERY_ALL_MECHANISMS.csv", index=False)

    survivor_ids = primary.loc[primary["discovery_survivor_for_gate_testing"] == True, "mechanism_id"].tolist()
    survivor_mechs = [m for m in mechanisms if m["mechanism_id"] in set(survivor_ids)]
    print(f"Discovery survivors for gate testing: {len(survivor_mechs)}", flush=True)

    # -------------------------------------------------------------------------
    # Bootstrap and leave-one-out stability only for discovery survivors.
    # -------------------------------------------------------------------------
    boot_rows = []
    for i, mech in enumerate(survivor_mechs, start=1):
        print(f"  bootstrap/LOO {i}/{len(survivor_mechs)}: {mech['mechanism_id']}", flush=True)
        b = bootstrap_and_loo(df, latent_y, mech, family_cols, spatial_cols, subset_mask=None)
        b.update({
            "mechanism_id": mech["mechanism_id"],
            "base_feature": mech["base_feature"],
            "mechanism_type": mech["mechanism_type"],
        })
        boot_rows.append(b)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TAB / "BOOTSTRAP_LOO_FOR_DISCOVERY_SURVIVORS.csv", index=False)

    # -------------------------------------------------------------------------
    # Clean land-cover / C4 crop sensitivity.
    # -------------------------------------------------------------------------
    clean_rows = []
    for mech in survivor_mechs:
        res = fit_mechanism(df, latent_y, mech, family_cols, spatial_cols, subset_mask=clean_mask, min_n=MIN_N_CLEAN)
        clean_rows.append(res)
    clean = pd.DataFrame(clean_rows)
    if not clean.empty:
        clean["clean_bh_q"] = bh_qvalues(clean["p"].values)
    clean.to_csv(TAB / "CROPLAND_C4_CROP_CLEAN_SENSITIVITY.csv", index=False)

    # -------------------------------------------------------------------------
    # Exact product-combination robustness.
    # -------------------------------------------------------------------------
    if exact_outcomes and survivor_mechs:
        prod, prod_summary = run_product_tests(df, exact_outcomes, survivor_mechs, family_cols, spatial_cols, clean_mask)
    else:
        prod, prod_summary = pd.DataFrame(), pd.DataFrame()

    prod.to_csv(TAB / "EXACT_PRODUCT_TESTS_FOR_DISCOVERY_SURVIVORS.csv", index=False)
    prod_summary.to_csv(TAB / "PRODUCT_DEPENDENCY_GATE_SUMMARY.csv", index=False)

    # -------------------------------------------------------------------------
    # Tower inventory and directional anchor.
    # -------------------------------------------------------------------------
    tower_inv = find_tower_tables()
    tower_dfs = read_possible_tower_dfs(tower_inv)
    tower = tower_directional_tests(mechanisms, primary, tower_dfs)
    tower.to_csv(TAB / "TOWER_DIRECTIONAL_ANCHOR_FOR_DISCOVERY_SURVIVORS.csv", index=False)

    # -------------------------------------------------------------------------
    # Required C4 model table.
    # -------------------------------------------------------------------------
    c4_ids = [m["mechanism_id"] for m in mechanisms if "c4" in norm_name(m["base_feature"]) and m["mechanism_type"] == "linear_main_effect"]
    c4_required = primary[primary["mechanism_id"].isin(c4_ids)].copy()
    c4_required.to_csv(TAB / "REZA_REQUIRED_C4_FULL_CONTROL_MODEL_RESULTS.csv", index=False)

    # -------------------------------------------------------------------------
    # Merge gates.
    # -------------------------------------------------------------------------
    gates = primary[primary["discovery_survivor_for_gate_testing"] == True].copy()

    if not boot.empty:
        gates = gates.merge(
            boot[[
                "mechanism_id", "boot_status", "boot_ci_low", "boot_ci_high",
                "boot_ci_excludes_zero", "boot_sign_fraction", "loo_sign_stability",
                "n_boot_valid"
            ]],
            on="mechanism_id",
            how="left",
        )
    else:
        gates["boot_status"] = "NOT_RUN_NO_DISCOVERY_SURVIVORS"

    if not clean.empty:
        clean_small = clean[[
            "mechanism_id", "status", "n", "coef", "p", "ci_excludes_zero",
            "delta_r2", "delta_aic_full_minus_reduced", "clean_bh_q"
        ]].rename(columns={
            "status": "clean_status",
            "n": "clean_n",
            "coef": "clean_coef",
            "p": "clean_p",
            "ci_excludes_zero": "clean_ci_excludes_zero",
            "delta_r2": "clean_delta_r2",
            "delta_aic_full_minus_reduced": "clean_delta_aic_full_minus_reduced",
        })
        gates = gates.merge(clean_small, on="mechanism_id", how="left")
    else:
        gates["clean_status"] = "NOT_RUN_NO_DISCOVERY_SURVIVORS"

    if not prod_summary.empty:
        ps_all = prod_summary[prod_summary["sample_mode"] == "all_available_complete_case"].copy()
        ps_all = ps_all.add_prefix("product_all__")
        ps_all = ps_all.rename(columns={"product_all__mechanism_id": "mechanism_id"})
        gates = gates.merge(ps_all, on="mechanism_id", how="left")

        ps_clean = prod_summary[prod_summary["sample_mode"] == "cropland_clean_complete_case"].copy()
        ps_clean = ps_clean.add_prefix("product_clean__")
        ps_clean = ps_clean.rename(columns={"product_clean__mechanism_id": "mechanism_id"})
        gates = gates.merge(ps_clean, on="mechanism_id", how="left")
    else:
        gates["product_all__product_gate_status"] = "NOT_RUN_NO_EXACT_PRODUCT_OUTCOMES"
        gates["product_clean__product_gate_status"] = "NOT_RUN_NO_EXACT_PRODUCT_OUTCOMES"

    if not tower.empty:
        tw_small = tower[[
            "mechanism_id", "tower_gate_status", "n_tower",
            "tower_ols_coef", "tower_ols_p", "tower_spearman_rho", "tower_spearman_p",
            "tower_table", "tower_y_col", "tower_x_col"
        ]].copy()
        gates = gates.merge(tw_small, on="mechanism_id", how="left")
    else:
        gates["tower_gate_status"] = "NOT_RUN_NO_TOWER_TABLES_OR_NO_DISCOVERY_SURVIVORS"

    # Gate definitions.
    gates["bootstrap_gate_pass"] = (
        (gates.get("boot_ci_excludes_zero", False) == True)
        & (gates.get("loo_sign_stability", 0).fillna(0) >= 0.80)
    )

    gates["clean_landcover_gate_pass"] = (
        (clean_info["landcover_status"] == "CLEAN_MASK_INFERRED")
        & (gates.get("clean_status", "") == "FIT_OK")
        & (gates.get("clean_p", np.nan) < 0.05)
        & (gates.get("clean_ci_excludes_zero", False) == True)
        & (np.sign(gates.get("clean_coef", np.nan)) == np.sign(gates.get("coef", np.nan)))
        & (gates.get("clean_delta_aic_full_minus_reduced", np.nan) < 0)
    )

    gates["product_all_gate_pass"] = gates.get("product_all__product_gate_status", "") == "PASS"
    gates["product_clean_gate_pass"] = gates.get("product_clean__product_gate_status", "") == "PASS"

    # Tower gate: strict pass requires p<0.10 same direction. Direction-only is
    # reported separately but does not count as full-prof pass.
    gates["tower_strict_gate_pass"] = gates.get("tower_gate_status", "").astype(str).str.contains("PASS_DIRECTIONAL_P_LT_0P10", regex=False)

    gates["FULL_PROF_PASS_STRICT"] = (
        (gates["primary_basic_pass"] == True)
        & (gates["bootstrap_gate_pass"] == True)
        & (gates["clean_landcover_gate_pass"] == True)
        & (gates["product_all_gate_pass"] == True)
        & (gates["product_clean_gate_pass"] == True)
        & (gates["tower_strict_gate_pass"] == True)
    )

    gates["SATELLITE_ONLY_STRICT_PASS_NO_TOWER"] = (
        (gates["primary_basic_pass"] == True)
        & (gates["bootstrap_gate_pass"] == True)
        & (gates["clean_landcover_gate_pass"] == True)
        & (gates["product_all_gate_pass"] == True)
        & (gates["product_clean_gate_pass"] == True)
    )

    # Simple gate score for ranking partial mechanisms.
    gate_cols = [
        "primary_basic_pass",
        "bootstrap_gate_pass",
        "clean_landcover_gate_pass",
        "product_all_gate_pass",
        "product_clean_gate_pass",
        "tower_strict_gate_pass",
    ]
    gates["gate_score_0_to_6"] = gates[gate_cols].sum(axis=1)

    gates = gates.sort_values(
        ["FULL_PROF_PASS_STRICT", "SATELLITE_ONLY_STRICT_PASS_NO_TOWER", "gate_score_0_to_6", "bh_q", "p"],
        ascending=[False, False, False, True, True],
    )
    gates.to_csv(TAB / "GATED_DISCOVERY_SURVIVORS_ALL_TESTS.csv", index=False)

    full_pass = gates[gates["FULL_PROF_PASS_STRICT"] == True].copy()
    sat_pass = gates[gates["SATELLITE_ONLY_STRICT_PASS_NO_TOWER"] == True].copy()

    full_pass.to_csv(TAB / "FULL_PROF_PASSING_MECHANISMS_STRICT.csv", index=False)
    sat_pass.to_csv(TAB / "SATELLITE_ONLY_STRICT_PASSING_MECHANISMS_NO_TOWER.csv", index=False)

    # -------------------------------------------------------------------------
    # Programming audit.
    # -------------------------------------------------------------------------
    audit = {
        "stage": "stage1b6bh_clean_exhaustive_mechanism_screen",
        "root": str(ROOT),
        "output_dir": str(OUT),
        "input_point_table": str(input_path),
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "latent_primary_outcome": latent_y,
        "all_outcome_columns_identified": all_outcome_cols,
        "exact_product_outcome_count": int(len(exact_outcomes)),
        "exact_product_outcomes": exact_outcomes,
        "base_feature_candidate_count": int(len(base_candidates)),
        "mechanism_count_tested": int(len(mechanisms)),
        "n_primary_fit_ok": int((primary["status"] == "FIT_OK").sum()),
        "n_primary_basic_pass": int(primary["primary_basic_pass"].sum()),
        "n_discovery_survivors_for_gate_testing": int(len(survivor_mechs)),
        "n_full_prof_strict_pass": int(len(full_pass)),
        "n_satellite_only_strict_pass_no_tower": int(len(sat_pass)),
        "landcover_clean_mask": clean_info,
        "n_tower_tables_inventory": int(len(tower_inv)),
        "n_tower_tables_loaded": int(len(tower_dfs)),
        "random_seed": RANDOM_SEED,
        "n_boot": N_BOOT,
        "min_n_primary": MIN_N_PRIMARY,
        "min_n_product": MIN_N_PRODUCT,
        "min_n_clean": MIN_N_CLEAN,
        "min_n_tower": MIN_N_TOWER,
        "programming_principles": [
            "No hand-selected focal variables.",
            "No row filtering in the primary discovery screen except model-specific complete cases.",
            "Clean land-cover/C4-crop filtering is used only as a sensitivity/gate.",
            "Exact GPP×ET product outcomes are used when available.",
            "GOSIF×GLEAM is treated as the least directly LAI-dependent product pair.",
            "FDR is applied across the full mechanism discovery family.",
            "Controls are omitted only when they are the focal variable family or moderator family.",
            "All skipped base columns are logged in BASE_FEATURE_CANDIDATE_AUDIT.csv.",
        ],
    }
    with open(TAB / "PROGRAMMING_AUDIT.json", "w") as f:
        json.dump(audit, f, indent=2)

    # -------------------------------------------------------------------------
    # Human-readable README.
    # -------------------------------------------------------------------------
    lines = []
    lines.append("Stage1B6BH clean exhaustive ecological-mechanism screen")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Purpose")
    lines.append("- Exhaustively screen eligible ecological / trait / structure / climate features.")
    lines.append("- Apply Reza-style gates: full controls, FDR, land-cover/C4-crop cleaning, exact product robustness, GOSIF×GLEAM least-dependent check, tower anchor where possible.")
    lines.append("")
    lines.append("Input")
    lines.append(f"- Selected point table: {input_path}")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append(f"- Primary latent/global outcome: {latent_y}")
    lines.append("")
    lines.append("Controls")
    for fam, hits in family_cols.items():
        lines.append(f"- {fam}: {hits[0] if hits else 'NOT FOUND'}")
    for fam, hits in spatial_cols.items():
        lines.append(f"- spatial_{fam}: {hits[0] if hits else 'NOT FOUND'}")
    lines.append("")
    lines.append("Land-cover / C4-crop clean mask")
    lines.append(f"- Status: {clean_info['landcover_status']}")
    lines.append(f"- Clean n: {clean_info['n_clean']} / {clean_info['n_total']}")
    lines.append(f"- Columns used: {clean_info['landcover_columns_used'] if clean_info['landcover_columns_used'] else 'NONE'}")
    lines.append("")
    lines.append("Search size")
    lines.append(f"- Eligible base features: {len(base_candidates)}")
    lines.append(f"- Mechanism tests constructed: {len(mechanisms)}")
    lines.append(f"- Primary FIT_OK models: {(primary['status'] == 'FIT_OK').sum()}")
    lines.append(f"- Primary discovery basic passes: {primary['primary_basic_pass'].sum()}")
    lines.append(f"- Discovery survivors sent to gates: {len(survivor_mechs)}")
    lines.append(f"- Exact product outcome columns found: {len(exact_outcomes)}")
    lines.append(f"- Tower tables inventoried: {len(tower_inv)}")
    lines.append("")
    lines.append("Strict final answer")
    lines.append("- FULL_PROF_PASS_STRICT requires:")
    lines.append("  1. primary p < 0.05 and BH q < 0.05;")
    lines.append("  2. CI excludes zero, ΔR² > 0, ΔAIC < 0, nested F p < 0.05;")
    lines.append("  3. bootstrap CI excludes zero and LOO sign stability >= 0.80;")
    lines.append("  4. cropland/C4-crop clean sample same-direction p < 0.05;")
    lines.append("  5. exact-product gate passes in all-available and clean samples;")
    lines.append("  6. GOSIF×GLEAM passes as least-directly-LAI-dependent pair;")
    lines.append("  7. tower directional anchor same direction with p < 0.10 where testable.")
    lines.append("")
    if len(full_pass) == 0:
        lines.append("FULL_PROF_PASS_STRICT result: NONE")
        lines.append("- No mechanism passed every single strict professor-level gate.")
    else:
        lines.append(f"FULL_PROF_PASS_STRICT result: {len(full_pass)} mechanism(s)")
        show = full_pass.head(20)[[
            "mechanism_id", "base_feature", "mechanism_type", "coef", "p", "bh_q",
            "gate_score_0_to_6"
        ]]
        lines.append(show.to_string(index=False))
    lines.append("")
    if len(sat_pass) == 0:
        lines.append("SATELLITE_ONLY_STRICT_PASS_NO_TOWER result: NONE")
    else:
        lines.append(f"SATELLITE_ONLY_STRICT_PASS_NO_TOWER result: {len(sat_pass)} mechanism(s)")
        show = sat_pass.head(20)[[
            "mechanism_id", "base_feature", "mechanism_type", "coef", "p", "bh_q",
            "gate_score_0_to_6",
            "product_all__gosif_gleam_status",
            "product_clean__gosif_gleam_status",
        ]]
        lines.append(show.to_string(index=False))
    lines.append("")
    lines.append("Top partial / near-pass mechanisms")
    if gates.empty:
        lines.append("- No discovery survivors reached the gate-testing stage.")
    else:
        cols = [
            "mechanism_id", "base_feature", "mechanism_type", "coef", "p", "bh_q",
            "gate_score_0_to_6", "primary_basic_pass", "bootstrap_gate_pass",
            "clean_landcover_gate_pass", "product_all_gate_pass",
            "product_clean_gate_pass",
            "tower_strict_gate_pass",
        ]
        existing = [c for c in cols if c in gates.columns]
        lines.append(gates.head(30)[existing].to_string(index=False))
    lines.append("")
    lines.append("Required C4 full-control model")
    if c4_required.empty:
        lines.append("- No eligible C4 fraction feature was identified.")
    else:
        show = c4_required.head(20)[[
            "mechanism_id", "base_feature", "status", "n", "coef", "p", "bh_q",
            "ci_low", "ci_high", "delta_r2", "delta_aic_full_minus_reduced",
            "controls_used"
        ]]
        lines.append(show.to_string(index=False))
    lines.append("")
    lines.append("Programming audit")
    lines.append("- The primary screen did not apply manual crop/warm/rooting-depth/product filters.")
    lines.append("- Each model used its own complete-case rows for outcome + focal term + required controls.")
    lines.append("- Clean land-cover/C4-crop filtering is reported separately as a sensitivity/gate.")
    lines.append("- Exact product tests are not averaged before gate diagnosis.")
    lines.append("- GOSIF×GLEAM is checked explicitly, not replaced by a GLEAM aggregate.")
    lines.append("- All candidate/skipped feature decisions are written to BASE_FEATURE_CANDIDATE_AUDIT.csv.")
    lines.append("")
    lines.append("Important files")
    for p in [
        TAB / "FULL_PROF_PASSING_MECHANISMS_STRICT.csv",
        TAB / "SATELLITE_ONLY_STRICT_PASSING_MECHANISMS_NO_TOWER.csv",
        TAB / "GATED_DISCOVERY_SURVIVORS_ALL_TESTS.csv",
        TAB / "PRIMARY_EXHAUSTIVE_DISCOVERY_ALL_MECHANISMS.csv",
        TAB / "EXACT_PRODUCT_TESTS_FOR_DISCOVERY_SURVIVORS.csv",
        TAB / "PRODUCT_DEPENDENCY_GATE_SUMMARY.csv",
        TAB / "CROPLAND_C4_CROP_CLEAN_SENSITIVITY.csv",
        TAB / "REZA_REQUIRED_C4_FULL_CONTROL_MODEL_RESULTS.csv",
        TAB / "TOWER_DIRECTIONAL_ANCHOR_FOR_DISCOVERY_SURVIVORS.csv",
        TAB / "PROGRAMMING_AUDIT.json",
    ]:
        lines.append(f"- {p}")

    readme = "\n".join(lines)
    (TXT / "READ_ME_clean_exhaustive_mechanism_screen.txt").write_text(readme)

    print("")
    print("DONE.")
    print(f"Outputs written to: {OUT}")
    print("")
    print("Paste this back:")
    print(f"cat {TXT / 'READ_ME_clean_exhaustive_mechanism_screen.txt'}")


if __name__ == "__main__":
    main()
