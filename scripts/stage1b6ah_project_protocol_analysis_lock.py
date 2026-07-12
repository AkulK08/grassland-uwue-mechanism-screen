from pathlib import Path
from datetime import datetime
import json
import math
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(".").resolve()
OUT = Path("results/stage1b6ah_project_protocol_analysis_lock")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6ah")
for p in [TAB, TXT, FIG, DATA]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260701
rng = np.random.default_rng(SEED)

PRIMARY_WUE_METRIC = "uWUE"
PRIMARY_RESPONSE_METRIC = "latent_post_slope"
PRIMARY_PRODUCT_PAIR_PREFERENCE = "tower_ranked_ET_then_GOSIF_GLEAM"
N_BOOT = 250

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

def read_csv_safe(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", nrows=nrows)
        except Exception:
            return None

def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def first_col(df, names):
    lut = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lut:
            return lut[norm(n)]
    return None

def contains_any(s, words):
    s = str(s).lower()
    return any(w.lower() in s for w in words)

def clean_class(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    aliases = {
        "saturation_breakdown_probability": "saturation/breakdown",
        "satbreak": "saturation/breakdown",
        "threshold_like": "threshold-like",
        "threshold": "threshold-like",
        "breakdown": "breakdown/reversal",
        "reversal": "breakdown/reversal",
        "enhancement": "enhancement",
        "saturation": "saturation",
        "inconclusive": "inconclusive",
        "weak": "weak/inconclusive",
        "weak_mixed": "weak/inconclusive",
        "low": "weak/inconclusive",
        "limitation": "limitation-like",
        "limitation_like": "limitation-like",
    }
    for k, v in aliases.items():
        if k in s:
            return v
    return s

def infer_et_product(text):
    s = str(text).lower()
    if "gleam" in s:
        return "GLEAM"
    if "mod16" in s or "modis_et" in s or "modis-et" in s or "modis et" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    if "et_product" in s:
        return str(text)
    return "UNKNOWN_ET"

def infer_gpp_product(text):
    s = str(text).lower()
    if "gosif" in s:
        return "GOSIF"
    if "mod17" in s or "modis_gpp" in s or "modis-gpp" in s or "modis gpp" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    return "UNKNOWN_GPP"

def product_combo_label(row):
    vals = " ".join([str(v) for v in row.values if pd.notna(v)])
    gpp = infer_gpp_product(vals)
    et = infer_et_product(vals)
    if gpp == "UNKNOWN_GPP" and et == "UNKNOWN_ET":
        return "UNKNOWN_PRODUCT"
    return f"{gpp}x{et}"

def standardize_lat_lon(df):
    lat = first_col(df, ["lat", "latitude", "LATITUDE", "site_lat", "tower_lat", "y"])
    lon = first_col(df, ["lon", "longitude", "LONGITUDE", "site_lon", "tower_lon", "x"])
    out = df.copy()
    if lat and "lat" not in out.columns:
        out["lat"] = to_num(out[lat])
    if lon and "lon" not in out.columns:
        out["lon"] = to_num(out[lon])
    return out

def normal_p_from_t(t):
    # two-sided normal approximation.
    if pd.isna(t):
        return np.nan
    z = abs(float(t))
    # normal survival = 0.5 * erfc(z/sqrt(2))
    return float(math.erfc(z / math.sqrt(2)))

def ols_fit(df, y_col, x_cols):
    d = df[[y_col] + x_cols].copy()
    for c in d.columns:
        d[c] = to_num(d[c])
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < max(8, len(x_cols) + 3):
        return None

    y = d[y_col].to_numpy(float)
    X_parts = [np.ones(len(d))]
    kept = []
    means = {}
    sds = {}

    for c in x_cols:
        x = d[c].to_numpy(float)
        sd = np.std(x)
        if not np.isfinite(sd) or sd == 0:
            continue
        means[c] = float(np.mean(x))
        sds[c] = float(sd)
        X_parts.append((x - means[c]) / sds[c])
        kept.append(c)

    if len(kept) == 0:
        return None

    X = np.column_stack(X_parts)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return None

    pred = X @ beta
    resid = y - pred
    n = len(y)
    k = X.shape[1]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / max(1, n - k) if np.isfinite(r2) else np.nan

    sigma2 = rss / max(1, n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)

    rows = []
    names = ["intercept"] + kept
    for name, b, s in zip(names, beta, se):
        t = b / s if np.isfinite(s) and s != 0 else np.nan
        p = normal_p_from_t(t)
        rows.append({
            "term": name,
            "coef_standardized": float(b),
            "se": float(s) if np.isfinite(s) else np.nan,
            "t_normal_approx": float(t) if np.isfinite(t) else np.nan,
            "p_normal_approx": p,
        })

    return {
        "n": n,
        "r2": float(r2),
        "adj_r2": float(adj),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coef_table": pd.DataFrame(rows),
        "residuals": resid,
        "pred": pred,
        "fit_index": d.index,
        "predictors_used": kept,
    }

def spearman(x, y):
    d = pd.DataFrame({"x": to_num(x), "y": to_num(y)}).dropna()
    if len(d) < 5 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan
    return float(d["x"].corr(d["y"], method="spearman"))

def bh_qvalues(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order]
    m = len(ranked)
    qv = ranked * m / np.arange(1, m + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    q[order] = np.minimum(qv, 1.0)
    return q

def find_files():
    patterns = ["*.csv", "*.tsv", "*.txt", "*.nc", "*.nc4", "*.tif", "*.tiff", "*.parquet"]
    roots = [Path("data"), Path("results"), Path("configs"), Path("scripts")]
    files = []
    for r in roots:
        if not r.exists():
            continue
        for pat in patterns:
            for p in r.rglob(pat):
                try:
                    if p.is_file() and p.stat().st_size > 0 and p.stat().st_size < 750_000_000:
                        files.append(p)
                except Exception:
                    pass
    return sorted(set(files))

all_files = find_files()

inventory_rows = []
for p in all_files:
    name = str(p).lower()
    tags = []
    if any(w in name for w in ["tower", "fluxnet", "ameriflux", "icos", "ozflux"]):
        tags.append("tower_candidate")
    if any(w in name for w in ["c4", "c3", "photosynthetic", "photosynthesis", "luo"]):
        tags.append("c3c4_candidate")
    if any(w in name for w in ["screen", "filter", "agreement", "product"]):
        tags.append("screening_candidate")
    if any(w in name for w in ["strict_2x2", "stage1b6r", "threshold_response"]):
        tags.append("strict_response_candidate")
    if tags:
        inventory_rows.append({
            "path": str(p),
            "size_mb": round(p.stat().st_size / 1e6, 3),
            "tags": ";".join(tags),
        })

inventory = pd.DataFrame(inventory_rows).sort_values("path") if inventory_rows else pd.DataFrame(columns=["path", "size_mb", "tags"])
inventory.to_csv(TAB / "Table_PRODUCT03aa_input_inventory.csv", index=False)

# -----------------------------------------------------------------------------
# 1. Product-screening audit
# -----------------------------------------------------------------------------
audit_keywords = {
    "agreement_filter_suspicious": [
        "product_agree", "product_agreement", "agreement_filter", "keep_agree", "only_agree",
        "where products agree", "products agree", "consensus_only", "consensus filter",
        "filter.*agreement", "screen.*agreement"
    ],
    "quality_filter_ok": [
        "qc", "quality", "cloud", "land_cover", "land-cover", "igbp", "stable",
        "mcd12", "mcd64", "burn", "gap", "closure", "good", "other quality"
    ],
    "product_screening_language": [
        "product-screened", "product screened", "strict 2x2", "2x2", "product matrix"
    ],
}

audit_rows = []
text_suffixes = {".py", ".R", ".r", ".sh", ".md", ".txt", ".yaml", ".yml", ".json"}
text_files = [p for p in all_files if p.suffix in text_suffixes and p.stat().st_size < 5_000_000]
iterator = tqdm(text_files, desc="Auditing screening logic", unit="file") if tqdm else text_files

for p in iterator:
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        continue
    for i, line in enumerate(txt, start=1):
        low = line.lower()
        for category, kws in audit_keywords.items():
            for kw in kws:
                hit = False
                if ".*" in kw:
                    try:
                        hit = re.search(kw, low) is not None
                    except Exception:
                        hit = False
                else:
                    hit = kw.lower() in low
                if hit:
                    audit_rows.append({
                        "file": str(p),
                        "line": i,
                        "category": category,
                        "keyword": kw,
                        "excerpt": line.strip()[:300],
                    })

product_audit = pd.DataFrame(audit_rows)
if len(product_audit):
    product_audit = product_audit.sort_values(["category", "file", "line"])
else:
    product_audit = pd.DataFrame(columns=["file", "line", "category", "keyword", "excerpt"])
product_audit.to_csv(TAB / "Table_PRODUCT03ab_product_screening_audit.csv", index=False)

agreement_suspicion_count = int((product_audit["category"] == "agreement_filter_suspicious").sum()) if len(product_audit) else 0
quality_filter_count = int((product_audit["category"] == "quality_filter_ok").sum()) if len(product_audit) else 0

# -----------------------------------------------------------------------------
# 2. Tower-validation table + ET product ranking
# -----------------------------------------------------------------------------
tower_files = [p for p in all_files if contains_any(str(p), ["tower", "fluxnet", "ameriflux", "icos", "ozflux"])]
tower_header_rows = []
tower_candidate_frames = []

for p in tqdm(tower_files, desc="Scanning tower files", unit="file") if tqdm else tower_files:
    if p.suffix.lower() not in [".csv", ".tsv", ".txt"]:
        continue
    d0 = read_csv_safe(p, nrows=5)
    if d0 is None:
        continue
    cols = list(d0.columns)
    header = " ".join(cols).lower()
    relevance = 0
    for kw in ["site", "tower", "igbp", "closure", "gap", "response", "class", "product", "satellite", "agreement", "fluxnet", "ameriflux"]:
        if kw in header or kw in str(p).lower():
            relevance += 1
    tower_header_rows.append({
        "path": str(p),
        "n_preview_cols": len(cols),
        "columns_preview": ";".join(cols[:60]),
        "relevance_score": relevance,
    })
    if relevance >= 2:
        d = read_csv_safe(p)
        if d is not None and len(d):
            d["__source_path"] = str(p)
            d["__source_name"] = p.name
            tower_candidate_frames.append(d)

tower_file_inventory = pd.DataFrame(tower_header_rows).sort_values("relevance_score", ascending=False) if tower_header_rows else pd.DataFrame()
tower_file_inventory.to_csv(TAB / "Table_PRODUCT03ab2_tower_file_inventory.csv", index=False)

def standardize_tower_frame(d):
    site = first_col(d, ["site_id", "SITE_ID", "site", "tower_id", "tower", "site_name", "id"])
    igbp = first_col(d, ["igbp", "IGBP", "land_cover", "landcover", "class", "site_class"])
    years = first_col(d, ["site_years", "n_site_years", "years", "n_years", "usable_years"])
    closure = first_col(d, ["closure_ratio", "energy_balance_closure", "ebc", "closure", "mean_closure_ratio"])
    gap = first_col(d, ["gapfill_fraction", "gap_fill_fraction", "gap_fraction", "gapfill", "gap_filling_fraction"])
    tower_class = first_col(d, ["tower_response_class", "tower_class", "response_class_tower", "tower_shape_class", "tower_limitation_class"])
    sat_class = first_col(d, ["satellite_response_class", "sat_class", "response_class_satellite", "satellite_class", "sat_shape_class"])
    product_combo = first_col(d, ["product_combo", "product_combination", "combo", "gpp_et_combo", "pair"])
    et = first_col(d, ["et_product", "ET_product", "et", "ET"])
    gpp = first_col(d, ["gpp_product", "GPP_product", "gpp", "GPP"])
    agreement = first_col(d, ["agreement", "agree", "agreement_flag", "exact_agreement", "class_agreement"])
    slope_agreement = first_col(d, ["slope_direction_agreement", "slope_agree", "direction_agreement"])

    if site is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["source_path"] = d.get("__source_path", "")
    out["site_id"] = d[site].astype(str)
    out["igbp_class"] = d[igbp].astype(str) if igbp else np.nan
    out["site_years"] = to_num(d[years]) if years else np.nan
    out["closure_ratio"] = to_num(d[closure]) if closure else np.nan
    out["gapfill_fraction"] = to_num(d[gap]) if gap else np.nan
    out["tower_response_class"] = d[tower_class].map(clean_class) if tower_class else np.nan
    out["satellite_response_class"] = d[sat_class].map(clean_class) if sat_class else np.nan

    if product_combo:
        out["product_combo"] = d[product_combo].astype(str)
    else:
        out["product_combo"] = d.apply(product_combo_label, axis=1)

    if et:
        out["et_product"] = d[et].astype(str).map(infer_et_product)
    else:
        out["et_product"] = out["product_combo"].map(infer_et_product)

    if gpp:
        out["gpp_product"] = d[gpp].astype(str).map(infer_gpp_product)
    else:
        out["gpp_product"] = out["product_combo"].map(infer_gpp_product)

    if agreement:
        out["exact_agreement"] = d[agreement].astype(str).str.lower().isin(["true", "1", "yes", "agree", "matched", "match"])
    else:
        out["exact_agreement"] = (
            out["tower_response_class"].notna()
            & out["satellite_response_class"].notna()
            & (out["tower_response_class"].astype(str) == out["satellite_response_class"].astype(str))
        )

    if slope_agreement:
        out["slope_direction_agreement"] = d[slope_agreement].astype(str).str.lower().isin(["true", "1", "yes", "agree", "matched", "match"])
    else:
        out["slope_direction_agreement"] = np.nan

    return out

tower_std_frames = []
for d in tower_candidate_frames:
    st = standardize_tower_frame(d)
    if len(st):
        tower_std_frames.append(st)

if tower_std_frames:
    tower_validation = pd.concat(tower_std_frames, ignore_index=True)
    tower_validation = tower_validation.drop_duplicates()
else:
    tower_validation = pd.DataFrame(columns=[
        "source_path", "site_id", "igbp_class", "site_years", "closure_ratio", "gapfill_fraction",
        "tower_response_class", "satellite_response_class", "product_combo", "gpp_product", "et_product",
        "exact_agreement", "slope_direction_agreement"
    ])

required_tower_cols = [
    "site_id", "igbp_class", "site_years", "closure_ratio", "gapfill_fraction",
    "tower_response_class", "satellite_response_class", "product_combo", "et_product", "exact_agreement"
]

missing_rows = []
if len(tower_validation):
    for col in required_tower_cols:
        n_missing = int(tower_validation[col].isna().sum()) if col in tower_validation.columns else len(tower_validation)
        missing_rows.append({
            "field": col,
            "n_missing": n_missing,
            "n_total_rows": int(len(tower_validation)),
            "satisfied": bool(n_missing < len(tower_validation)),
        })
else:
    for col in required_tower_cols:
        missing_rows.append({
            "field": col,
            "n_missing": np.nan,
            "n_total_rows": 0,
            "satisfied": False,
        })

missing_tower = pd.DataFrame(missing_rows)

if len(tower_validation):
    tower_validation["usable_for_exact_ranking"] = (
        tower_validation["site_id"].notna()
        & tower_validation["tower_response_class"].notna()
        & tower_validation["satellite_response_class"].notna()
        & tower_validation["et_product"].notna()
    )
else:
    tower_validation["usable_for_exact_ranking"] = []

tower_validation.to_csv(TAB / "Table_PRODUCT03ac_tower_validation_deliverable.csv", index=False)
missing_tower.to_csv(TAB / "Table_PRODUCT03ae_tower_validation_missing_fields.csv", index=False)

if len(tower_validation) and tower_validation["usable_for_exact_ranking"].any():
    rank = (
        tower_validation[tower_validation["usable_for_exact_ranking"]]
        .groupby("et_product", dropna=False)
        .agg(
            n_site_product_rows=("site_id", "size"),
            n_unique_sites=("site_id", "nunique"),
            exact_agreement_rate=("exact_agreement", "mean"),
            slope_direction_agreement_rate=("slope_direction_agreement", lambda x: np.nanmean(pd.to_numeric(x, errors="coerce")) if pd.to_numeric(x, errors="coerce").notna().any() else np.nan),
        )
        .reset_index()
        .sort_values(["exact_agreement_rate", "n_unique_sites"], ascending=[False, False])
    )
else:
    rank = pd.DataFrame(columns=["et_product", "n_site_product_rows", "n_unique_sites", "exact_agreement_rate", "slope_direction_agreement_rate"])

rank.to_csv(TAB / "Table_PRODUCT03ad_et_product_tower_agreement_ranking.csv", index=False)

tower_ranked_et = None
if len(rank):
    tower_ranked_et = str(rank.iloc[0]["et_product"])

# -----------------------------------------------------------------------------
# 3. Strict foundation and product confidence filter
# -----------------------------------------------------------------------------
strict_candidates = [
    Path("data/processed/stage1b6p/strict_2x2_response_table_final13.csv"),
    Path("data/processed/stage1b6q2/analysis_design_strict_2x2_with_tower_stress_and_gs.csv"),
    Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv"),
    Path("results/stage1b6r/threshold_response_fits_strict_2x2.csv"),
]
strict_existing = [p for p in strict_candidates if p.exists()]
strict_foundation_rows = []
for p in strict_existing:
    d = read_csv_safe(p)
    if d is None:
        continue
    row = {
        "path": str(p),
        "n_rows": int(len(d)),
        "n_cols": int(len(d.columns)),
        "columns": ";".join(list(d.columns)[:80]),
    }
    for key, names in {
        "n_sites": ["site_id", "SITE_ID", "site", "tower_id"],
        "n_dates": ["date", "time", "datetime", "timestamp"],
        "n_product_combos": ["product_combo", "product_combination", "combo"],
        "n_gpp_products": ["gpp_product", "GPP_product"],
        "n_et_products": ["et_product", "ET_product"],
    }.items():
        c = first_col(d, names)
        row[key] = int(d[c].nunique(dropna=True)) if c else np.nan
    strict_foundation_rows.append(row)

strict_foundation = pd.DataFrame(strict_foundation_rows)
strict_foundation.to_csv(TAB / "Table_PRODUCT03ab3_strict_response_foundation_inventory.csv", index=False)

confidence_rows = []
strict_pairs = ["MODISxMODIS", "MODISxGLEAM", "GOSIFxMODIS", "GOSIFxGLEAM"]
for pair in strict_pairs:
    et = infer_et_product(pair)
    confidence_rows.append({
        "product_pair": pair,
        "et_product": et,
        "is_independent_pair": pair == "GOSIFxGLEAM",
        "uses_tower_ranked_et": bool(tower_ranked_et and et == tower_ranked_et),
        "confidence_role": (
            "primary if tower-ranked ET" if tower_ranked_et and et == tower_ranked_et else
            "independent product check" if pair == "GOSIFxGLEAM" else
            "community-standard / comparison product"
        ),
    })
confidence = pd.DataFrame(confidence_rows)
confidence.to_csv(TAB / "Table_PRODUCT03aj_product_confidence_filter.csv", index=False)

# -----------------------------------------------------------------------------
# 4. 2D VPD x soil-moisture surface metric if data support it
# -----------------------------------------------------------------------------
def compute_uwue_table(d):
    out = d.copy()
    gpp_col = first_col(out, ["gpp", "GPP", "gpp_value", "GPP_value", "gpp_mean", "GPP_NT_VUT_REF"])
    et_col = first_col(out, ["et", "ET", "et_value", "ET_value", "et_mean", "LE_F_MDS_ET", "et_mm"])
    vpd_col = first_col(out, ["vpd", "VPD", "vpd_mean", "mean_vpd", "VPD_F", "vapor_pressure_deficit"])
    sm_col = first_col(out, ["soil_moisture", "sm", "SM", "rootzone_sm", "swvl", "swvl1", "swvl2", "mean_soil_moisture"])
    if not all([gpp_col, et_col, vpd_col, sm_col]):
        return None, {
            "gpp_col": gpp_col, "et_col": et_col, "vpd_col": vpd_col, "sm_col": sm_col,
            "status": "missing_required_columns"
        }

    out["GPP_for_uWUE"] = to_num(out[gpp_col])
    out["ET_for_uWUE"] = to_num(out[et_col])
    out["VPD_for_uWUE"] = to_num(out[vpd_col])
    out["SM_for_surface"] = to_num(out[sm_col])
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[(out["GPP_for_uWUE"] > 0) & (out["ET_for_uWUE"] > 0) & (out["VPD_for_uWUE"] > 0)]
    if len(out) == 0:
        return None, {"status": "no_positive_gpp_et_vpd"}

    out["uWUE"] = out["GPP_for_uWUE"] * np.sqrt(out["VPD_for_uWUE"]) / out["ET_for_uWUE"]
    return out, {
        "gpp_col": gpp_col, "et_col": et_col, "vpd_col": vpd_col, "sm_col": sm_col,
        "status": "ready"
    }

surface_rows = []
surface_status_rows = []
surface_source = None
for p in strict_existing:
    d = read_csv_safe(p)
    if d is None or len(d) < 50:
        continue
    uw, status = compute_uwue_table(d)
    status["path"] = str(p)
    surface_status_rows.append(status)
    if uw is None:
        continue

    site_col = first_col(uw, ["site_id", "SITE_ID", "site", "tower_id", "point_id", "pixel_id"])
    combo_col = first_col(uw, ["product_combo", "product_combination", "combo", "gpp_et_combo"])
    gpp_col = first_col(uw, ["gpp_product", "GPP_product"])
    et_col = first_col(uw, ["et_product", "ET_product"])

    if site_col is None:
        uw["site_id_inferred"] = "all_points"
        site_col = "site_id_inferred"
    if combo_col is None:
        if gpp_col or et_col:
            uw["product_combo_inferred"] = uw.apply(product_combo_label, axis=1)
        else:
            uw["product_combo_inferred"] = "UNKNOWN_PRODUCT"
        combo_col = "product_combo_inferred"

    groups = list(uw.groupby([site_col, combo_col], dropna=False))
    for (site, combo), sub in tqdm(groups, desc="Fitting 2D VPDxSM surfaces", unit="fit") if tqdm else groups:
        sub = sub[["uWUE", "VPD_for_uWUE", "SM_for_surface"]].dropna().copy()
        if len(sub) < 25:
            continue
        sub["vpd_pct"] = sub["VPD_for_uWUE"].rank(pct=True)
        sub["sm_pct"] = sub["SM_for_surface"].rank(pct=True)
        sub["dryness_pct"] = 1 - sub["sm_pct"]
        sub["interaction"] = sub["vpd_pct"] * sub["dryness_pct"]
        sub["log_uWUE"] = np.log(sub["uWUE"].clip(lower=1e-9))
        fit = ols_fit(sub, "log_uWUE", ["vpd_pct", "dryness_pct", "interaction"])
        if fit is None:
            continue
        ct = fit["coef_table"]
        getcoef = lambda term: float(ct.loc[ct["term"] == term, "coef_standardized"].iloc[0]) if (ct["term"] == term).any() else np.nan
        getp = lambda term: float(ct.loc[ct["term"] == term, "p_normal_approx"].iloc[0]) if (ct["term"] == term).any() else np.nan
        surface_rows.append({
            "source_path": str(p),
            "site_id": site,
            "product_combo": combo,
            "n_obs": fit["n"],
            "response_metric": "log_uWUE_2D_surface",
            "coef_vpd_percentile": getcoef("vpd_pct"),
            "coef_dryness_percentile": getcoef("dryness_pct"),
            "coef_vpd_x_dryness": getcoef("interaction"),
            "p_vpd_x_dryness": getp("interaction"),
            "r2": fit["r2"],
            "adj_r2": fit["adj_r2"],
        })
    surface_source = str(p)
    break

surface = pd.DataFrame(surface_rows)
surface_status = pd.DataFrame(surface_status_rows)
surface.to_csv(TAB / "Table_PRODUCT03ai_2d_surface_response_metrics.csv", index=False)
surface_status.to_csv(TAB / "Table_PRODUCT03ai2_2d_surface_input_status.csv", index=False)

# -----------------------------------------------------------------------------
# 5. C3/C4 inventory + pre-specified C4 test if local data exist
# -----------------------------------------------------------------------------
c4_files = [p for p in all_files if contains_any(str(p), ["c4", "c3", "photosynthetic", "photosynthesis", "luo"])]
c4_inventory_rows = []
c4_candidate_frames = []

for p in tqdm(c4_files, desc="Scanning C3/C4 files", unit="file") if tqdm else c4_files:
    row = {"path": str(p), "size_mb": round(p.stat().st_size / 1e6, 3), "suffix": p.suffix}
    if p.suffix.lower() in [".csv", ".tsv", ".txt"]:
        d0 = read_csv_safe(p, nrows=20)
        if d0 is not None:
            cols = list(d0.columns)
            row["columns_preview"] = ";".join(cols[:80])
            coltext = " ".join(cols).lower()
            row["has_c4_like_column"] = any(w in coltext for w in ["c4", "c_4", "c4_fraction", "c4frac", "photosynthetic"])
            row["has_latlon"] = any(w in coltext for w in ["lat", "latitude"]) and any(w in coltext for w in ["lon", "longitude"])
            if row["has_c4_like_column"]:
                d = read_csv_safe(p)
                if d is not None:
                    d["__source_path"] = str(p)
                    c4_candidate_frames.append(d)
        else:
            row["columns_preview"] = ""
            row["has_c4_like_column"] = False
            row["has_latlon"] = False
    else:
        row["columns_preview"] = ""
        row["has_c4_like_column"] = True
        row["has_latlon"] = "raster_or_netcdf_needs_sampling"
    c4_inventory_rows.append(row)

c4_inventory = pd.DataFrame(c4_inventory_rows).sort_values(["has_c4_like_column", "path"], ascending=[False, True]) if c4_inventory_rows else pd.DataFrame(columns=["path", "size_mb", "suffix", "columns_preview", "has_c4_like_column", "has_latlon"])
c4_inventory.to_csv(TAB / "Table_PRODUCT03af_c3c4_data_inventory.csv", index=False)

# Load response point table.
point_candidates = [
    Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv"),
    Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
    Path("results/stage1b6af_nature_level_viability_lock/tables/Table_PRODUCT02fs_named_regime_trait_tests.csv"),
]
point_df = None
point_path = None
for p in point_candidates:
    if p.exists():
        d = read_csv_safe(p)
        if d is not None and len(d):
            point_df = standardize_lat_lon(d)
            point_path = p
            break

c4_joined = pd.DataFrame()
c4_model_results = pd.DataFrame()
c4_status = {
    "point_table_found": str(point_path) if point_path else None,
    "c4_table_found": None,
    "c4_test_ran": False,
    "reason_if_not_run": None,
}

def identify_c4_col(df):
    candidates = []
    for c in df.columns:
        lc = norm(c)
        if any(w in lc for w in ["c4_fraction", "c4_frac", "c4_percent", "c4_pct", "c4_cover", "c4grass", "c4_grass", "c4"]):
            candidates.append(c)
    # avoid columns like c4_status maybe if nonnumeric.
    scored = []
    for c in candidates:
        vals = to_num(df[c])
        usable = vals.notna().sum()
        scored.append((usable, c))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]

def nearest_join_latlon(points, covars, value_col, max_distance_deg=0.5):
    p = points.copy()
    c = covars.copy()
    p = standardize_lat_lon(p)
    c = standardize_lat_lon(c)
    if "lat" not in p.columns or "lon" not in p.columns or "lat" not in c.columns or "lon" not in c.columns:
        return None

    p["lat"] = to_num(p["lat"])
    p["lon"] = to_num(p["lon"])
    c["lat"] = to_num(c["lat"])
    c["lon"] = to_num(c["lon"])
    c[value_col] = to_num(c[value_col])
    p = p.dropna(subset=["lat", "lon"])
    c = c.dropna(subset=["lat", "lon", value_col])

    if len(p) == 0 or len(c) == 0:
        return None

    out_vals = []
    out_dist = []
    c_lat = c["lat"].to_numpy(float)
    c_lon = c["lon"].to_numpy(float)
    c_val = c[value_col].to_numpy(float)

    for _, r in tqdm(p.iterrows(), total=len(p), desc="Nearest joining C4 fraction", unit="point") if tqdm else p.iterrows():
        lat = float(r["lat"])
        lon = float(r["lon"])
        dlat = c_lat - lat
        dlon = (c_lon - lon) * math.cos(math.radians(lat))
        dist = np.sqrt(dlat ** 2 + dlon ** 2)
        j = int(np.argmin(dist))
        if dist[j] <= max_distance_deg:
            out_vals.append(c_val[j])
            out_dist.append(float(dist[j]))
        else:
            out_vals.append(np.nan)
            out_dist.append(float(dist[j]))

    joined = p.copy()
    joined["c4_fraction"] = out_vals
    joined["c4_nearest_distance_deg"] = out_dist
    joined["c4_source_column"] = value_col
    return joined

if point_df is None:
    c4_status["reason_if_not_run"] = "No point-level response table found."
else:
    point_df = standardize_lat_lon(point_df)

    existing_c4_col = identify_c4_col(point_df)
    if existing_c4_col:
        c4_joined = point_df.copy()
        c4_joined["c4_fraction"] = to_num(c4_joined[existing_c4_col])
        c4_status["c4_table_found"] = f"already_in_point_table:{existing_c4_col}"
    else:
        for cand in c4_candidate_frames:
            ccol = identify_c4_col(cand)
            if ccol is None:
                continue
            joined = nearest_join_latlon(point_df, cand, ccol)
            if joined is not None and joined["c4_fraction"].notna().sum() >= 20:
                c4_joined = joined
                c4_status["c4_table_found"] = cand.get("__source_path", pd.Series(["unknown"])).iloc[0] + f"::{ccol}"
                break

    if c4_joined is None or len(c4_joined) == 0 or "c4_fraction" not in c4_joined.columns or c4_joined["c4_fraction"].notna().sum() < 20:
        c4_status["reason_if_not_run"] = "No usable local C4 fraction table with lat/lon coverage was found. Add Luo 2024 C4 fraction raster/table to data/raw/traits/c4_fraction/ and rerun."
    else:
        # Choose response metric and controls.
        if PRIMARY_RESPONSE_METRIC not in c4_joined.columns:
            # fallback to response metrics if primary absent.
            for alt in ["latent_post_slope", "p_threshold_like", "latent_slope_change", "p_satbreak", "latent_satbreak_probability"]:
                if alt in c4_joined.columns:
                    c4_status["primary_response_metric_fallback"] = alt
                    response_col = alt
                    break
            else:
                response_col = None
        else:
            response_col = PRIMARY_RESPONSE_METRIC

        if response_col is None:
            c4_status["reason_if_not_run"] = "C4 data joined, but no response metric column was found."
        else:
            # Standardize control names if present.
            control_candidates = [
                "aridity", "aridity_index",
                "soil_sand", "soil_clay", "soil_silt",
                "mean_annual_temperature", "mean_temperature", "temperature",
                "mean_annual_precipitation", "mean_precipitation", "precipitation",
                "mean_lai", "growing_season_mean_lai",
                "mean_vpd", "baseline_vpd",
                "mean_soil_moisture", "baseline_soil_moisture",
                "rooting_depth",
            ]
            controls = []
            for c in control_candidates:
                cc = first_col(c4_joined, [c])
                if cc and cc not in controls:
                    controls.append(cc)

            # Require at least aridity/climate/soil-ish if available, but run with whatever is present.
            model_df = c4_joined.copy()
            model_df["c4_fraction"] = to_num(model_df["c4_fraction"])
            model_df[response_col] = to_num(model_df[response_col])

            # If C4 appears as 0-100, convert to 0-1 for interpretability but model standardizes anyway.
            if model_df["c4_fraction"].quantile(0.95) > 1.5:
                model_df["c4_fraction"] = model_df["c4_fraction"] / 100.0

            # If product columns exist, create a product filter summary, but do not silently filter unless tower-ranked ET is present.
            if tower_ranked_et:
                product_cols = [c for c in model_df.columns if "product" in norm(c) or "combo" in norm(c) or norm(c) in ["et", "gpp"]]
                if product_cols:
                    prod_text = model_df[product_cols].astype(str).agg(" ".join, axis=1)
                    filt = prod_text.str.lower().str.contains(tower_ranked_et.lower())
                    if filt.sum() >= 20:
                        model_df = model_df[filt].copy()
                        c4_status["filtered_to_tower_ranked_et"] = tower_ranked_et

            predictors = ["c4_fraction"] + [c for c in controls if c != response_col and c != "c4_fraction"]
            predictors = [c for c in predictors if c in model_df.columns and to_num(model_df[c]).notna().sum() >= 20 and to_num(model_df[c]).nunique(dropna=True) > 2]

            fit = ols_fit(model_df, response_col, predictors)
            if fit is None or "c4_fraction" not in fit["predictors_used"]:
                c4_status["reason_if_not_run"] = f"C4 model could not be fit with enough complete rows. Response={response_col}; predictors={predictors}"
            else:
                c4_status["c4_test_ran"] = True
                c4_status["response_col"] = response_col
                c4_status["predictors_used"] = fit["predictors_used"]
                c4_status["n_model"] = fit["n"]

                coef = fit["coef_table"].copy()
                coef["model"] = "primary_c4_fraction_partial_effect"
                coef["response"] = response_col
                coef["n"] = fit["n"]
                coef["r2"] = fit["r2"]
                coef["adj_r2"] = fit["adj_r2"]

                # Spatial blocks for bootstrap and leave-one-block-out.
                dfit = model_df.loc[fit["fit_index"]].copy()
                dfit["model_residual"] = fit["residuals"]
                dfit = standardize_lat_lon(dfit)

                if "eco_biome" in dfit.columns:
                    block_col = "eco_biome"
                elif "eco_realm" in dfit.columns:
                    block_col = "eco_realm"
                elif "region" in dfit.columns:
                    block_col = "region"
                elif "lat" in dfit.columns and "lon" in dfit.columns:
                    dfit["spatial_block"] = (
                        np.floor(to_num(dfit["lat"]) / 10).astype("Int64").astype(str)
                        + "_"
                        + np.floor(to_num(dfit["lon"]) / 10).astype("Int64").astype(str)
                    )
                    block_col = "spatial_block"
                else:
                    dfit["spatial_block"] = "all"
                    block_col = "spatial_block"

                blocks = [b for b in dfit[block_col].dropna().unique()]
                boot_coefs = []
                if len(blocks) >= 3:
                    for _ in tqdm(range(N_BOOT), desc="Spatial block bootstrap C4 effect", unit="boot") if tqdm else range(N_BOOT):
                        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
                        boot = pd.concat([dfit[dfit[block_col] == b] for b in sampled_blocks], ignore_index=True)
                        bf = ols_fit(boot, response_col, fit["predictors_used"])
                        if bf is not None:
                            ct = bf["coef_table"]
                            if (ct["term"] == "c4_fraction").any():
                                boot_coefs.append(float(ct.loc[ct["term"] == "c4_fraction", "coef_standardized"].iloc[0]))

                loo_rows = []
                if len(blocks) >= 3:
                    for b in blocks:
                        train = dfit[dfit[block_col] != b].copy()
                        if len(train) < 20:
                            continue
                        lf = ols_fit(train, response_col, fit["predictors_used"])
                        if lf is None:
                            continue
                        ct = lf["coef_table"]
                        if (ct["term"] == "c4_fraction").any():
                            c4b = float(ct.loc[ct["term"] == "c4_fraction", "coef_standardized"].iloc[0])
                            loo_rows.append({
                                "left_out_block": str(b),
                                "n_train": lf["n"],
                                "c4_coef_standardized": c4b,
                            })

                # Moran's I approximation on residuals using inverse-distance nearest neighbors.
                moran_i = np.nan
                if "lat" in dfit.columns and "lon" in dfit.columns and len(dfit) >= 10:
                    coords = dfit[["lat", "lon"]].apply(to_num).dropna()
                    if len(coords) == len(dfit):
                        vals = dfit["model_residual"].to_numpy(float)
                        vals = vals - vals.mean()
                        latv = coords["lat"].to_numpy(float)
                        lonv = coords["lon"].to_numpy(float)
                        Wsum = 0.0
                        num = 0.0
                        for i in range(len(vals)):
                            dlat = latv - latv[i]
                            dlon = (lonv - lonv[i]) * math.cos(math.radians(latv[i]))
                            dist = np.sqrt(dlat**2 + dlon**2)
                            order = np.argsort(dist)
                            neigh = [j for j in order[1:7] if dist[j] > 0]
                            for j in neigh:
                                w = 1.0 / dist[j]
                                Wsum += w
                                num += w * vals[i] * vals[j]
                        den = np.sum(vals**2)
                        if Wsum > 0 and den > 0:
                            moran_i = float((len(vals) / Wsum) * (num / den))

                c4_extra_rows = []
                c4_extra_rows.append({
                    "model": "spatial_block_bootstrap",
                    "response": response_col,
                    "term": "c4_fraction",
                    "n": fit["n"],
                    "block_col": block_col,
                    "n_blocks": len(blocks),
                    "bootstrap_n": len(boot_coefs),
                    "bootstrap_coef_median": float(np.median(boot_coefs)) if boot_coefs else np.nan,
                    "bootstrap_coef_p025": float(np.quantile(boot_coefs, 0.025)) if boot_coefs else np.nan,
                    "bootstrap_coef_p975": float(np.quantile(boot_coefs, 0.975)) if boot_coefs else np.nan,
                    "loo_blocks_n": len(loo_rows),
                    "loo_sign_stability": float(np.mean(np.sign([r["c4_coef_standardized"] for r in loo_rows]) == np.sign(coef.loc[coef["term"] == "c4_fraction", "coef_standardized"].iloc[0]))) if loo_rows and (coef["term"] == "c4_fraction").any() else np.nan,
                    "moran_i_residual_knn6": moran_i,
                })

                extra = pd.DataFrame(c4_extra_rows)
                loo_df = pd.DataFrame(loo_rows)
                c4_model_results = pd.concat([coef, extra], ignore_index=True, sort=False)
                loo_df.to_csv(TAB / "Table_PRODUCT03ah2_c3c4_leave_one_block_out.csv", index=False)

                # Add q-values for coefficient p-values.
                if "p_normal_approx" in c4_model_results.columns:
                    c4_model_results["bh_q_normal_approx"] = bh_qvalues(c4_model_results["p_normal_approx"].to_numpy(float))

c4_joined.to_csv(TAB / "Table_PRODUCT03ag_c3c4_joined_model_data.csv", index=False)
c4_model_results.to_csv(TAB / "Table_PRODUCT03ah_c3c4_model_results.csv", index=False)
pd.DataFrame([c4_status]).to_csv(TAB / "Table_PRODUCT03ah0_c3c4_test_status.csv", index=False)

# -----------------------------------------------------------------------------
# 6. Pre-registration / protocol satisfaction checklist
# -----------------------------------------------------------------------------
prereg_lines = []
prereg_lines.append("# Pre-registered C3/C4 analysis plan")
prereg_lines.append("")
prereg_lines.append(f"Generated: {now()}")
prereg_lines.append("")
prereg_lines.append("## Primary hypothesis")
prereg_lines.append("")
prereg_lines.append("In global grasslands and savannas, C4 photosynthetic fraction is associated with the uWUE response phenotype to compound high-VPD / low-soil-moisture stress after controlling for aridity, soil texture, mean climate, LAI, baseline conditions, and rooting-zone storage where available.")
prereg_lines.append("")
prereg_lines.append("## Primary metric locked before running")
prereg_lines.append("")
prereg_lines.append(f"- Primary WUE metric: `{PRIMARY_WUE_METRIC}`")
prereg_lines.append(f"- Primary response metric: `{PRIMARY_RESPONSE_METRIC}`")
prereg_lines.append("- Sensitivity response metrics: `p_threshold_like`, `p_satbreak`, `latent_satbreak_probability`, `latent_slope_change` where available.")
prereg_lines.append("")
prereg_lines.append("## Product rule")
prereg_lines.append("")
prereg_lines.append("- Use the tower-ranked ET product if the tower-validation table can rank ET products.")
prereg_lines.append("- Always compare against the independent `GOSIFxGLEAM` pair when available.")
prereg_lines.append("- Multi-product agreement is a confidence layer, not the primary scientific claim.")
prereg_lines.append("")
prereg_lines.append("## Model form")
prereg_lines.append("")
prereg_lines.append("- Primary test: partial effect of C4 fraction in a single pooled model with controls.")
prereg_lines.append("- Preferred final model after this lock: hierarchical or mixed-effects model with ecoregion random effects.")
prereg_lines.append("- This script runs the fast lock-screening version: standardized OLS with spatial block bootstrap and leave-one-block-out stability.")
prereg_lines.append("")
prereg_lines.append("## Spatial robustness")
prereg_lines.append("")
prereg_lines.append("- Spatial block bootstrap by ecoregion/realm/10-degree lat-lon block.")
prereg_lines.append("- Leave-one-ecoregion/block-out sign stability.")
prereg_lines.append("- Moran's I approximation on residuals.")
prereg_lines.append("")
prereg_lines.append("## Interpretation rule")
prereg_lines.append("")
prereg_lines.append("- If C4 fraction survives controls on the tower-ranked product and the independent product pair, write the ecological mechanism paper.")
prereg_lines.append("- If C4 fraction fails, write the product-identifiability / satellite WUE uncertainty paper.")
prereg_lines.append("- Do not claim global causal proof from this observational analysis.")
(TXT / "PREREGISTERED_C3C4_ANALYSIS_PLAN.md").write_text("\n".join(prereg_lines), encoding="utf-8")

checklist = []

def add_check(item, satisfied, details, output_file=None):
    checklist.append({
        "protocol_item": item,
        "satisfied": bool(satisfied),
        "details": details,
        "output_file": output_file or "",
    })

add_check(
    "Product-screening definition audited",
    True,
    f"Found {agreement_suspicion_count} agreement-filter keyword hits and {quality_filter_count} quality-filter keyword hits. Review audit table to determine whether product-screening meant agreement filtering or QC only.",
    "Table_PRODUCT03ab_product_screening_audit.csv"
)
add_check(
    "Tower-validation deliverable table produced",
    len(tower_validation) > 0,
    f"Tower validation rows produced: {len(tower_validation)}. Required-field completeness is in missing-fields table.",
    "Table_PRODUCT03ac_tower_validation_deliverable.csv"
)
add_check(
    "Per-ET-product tower agreement ranking produced",
    len(rank) > 0,
    f"ET ranking rows: {len(rank)}. Tower-ranked ET product: {tower_ranked_et}.",
    "Table_PRODUCT03ad_et_product_tower_agreement_ranking.csv"
)
add_check(
    "Tower required field audit produced",
    True,
    "Missing fields table lists site ID, IGBP, site-years, closure, gap-fill, tower class, satellite class, product combo, ET product, and agreement availability.",
    "Table_PRODUCT03ae_tower_validation_missing_fields.csv"
)
add_check(
    "Strict 2x2 product foundation inventoried",
    len(strict_foundation) > 0,
    f"Strict foundation files found: {len(strict_foundation)}.",
    "Table_PRODUCT03ab3_strict_response_foundation_inventory.csv"
)
add_check(
    "2D VPD x soil-moisture surface attempted",
    len(surface) > 0,
    f"2D surface rows: {len(surface)}. If zero, status table explains missing columns.",
    "Table_PRODUCT03ai_2d_surface_response_metrics.csv"
)
add_check(
    "C3/C4 data inventory produced",
    True,
    f"C3/C4 candidate files found: {len(c4_inventory)}.",
    "Table_PRODUCT03af_c3c4_data_inventory.csv"
)
add_check(
    "C3/C4 model test run if data available",
    bool(c4_status.get("c4_test_ran")),
    c4_status.get("reason_if_not_run") or f"C4 model ran with n={c4_status.get('n_model')}, response={c4_status.get('response_col')}.",
    "Table_PRODUCT03ah_c3c4_model_results.csv"
)
add_check(
    "Pre-registration file written",
    True,
    "Pre-registered C3/C4 analysis plan written before manuscript drafting.",
    "PREREGISTERED_C3C4_ANALYSIS_PLAN.md"
)
add_check(
    "Product families treated as confidence filter",
    True,
    "Product confidence table marks tower-ranked ET and independent GOSIFxGLEAM pair.",
    "Table_PRODUCT03aj_product_confidence_filter.csv"
)
add_check(
    "Observed water covariate preference checked",
    any("soil" in c.lower() or "sm" in c.lower() for c in (list(point_df.columns) if point_df is not None else [])),
    "Script searches for mean_soil_moisture/baseline_soil_moisture/root-zone soil moisture controls in point table.",
    "Table_PRODUCT03ah_c3c4_model_results.csv"
)

checklist_df = pd.DataFrame(checklist)
checklist_df.to_csv(TAB / "Table_PRODUCT03ak_protocol_satisfaction_checklist.csv", index=False)

# -----------------------------------------------------------------------------
# 7. Figures
# -----------------------------------------------------------------------------
figure_status = []
if plt is not None:
    try:
        if len(rank):
            plt.figure(figsize=(7, 4))
            plot = rank.copy()
            plt.bar(plot["et_product"].astype(str), plot["exact_agreement_rate"].astype(float))
            plt.ylabel("Tower exact-class agreement rate")
            plt.xlabel("ET product")
            plt.title("Tower-ranked ET products")
            plt.tight_layout()
            plt.savefig(FIG / "Figure_PRODUCT03a_tower_et_product_ranking.png", dpi=220)
            plt.close()
            figure_status.append("tower_ranking")
    except Exception as e:
        figure_status.append(f"tower_ranking_failed:{e}")

    try:
        if len(c4_joined) and "c4_fraction" in c4_joined.columns and PRIMARY_RESPONSE_METRIC in c4_joined.columns:
            pd2 = c4_joined[["c4_fraction", PRIMARY_RESPONSE_METRIC]].copy()
            pd2["c4_fraction"] = to_num(pd2["c4_fraction"])
            pd2[PRIMARY_RESPONSE_METRIC] = to_num(pd2[PRIMARY_RESPONSE_METRIC])
            pd2 = pd2.dropna()
            if len(pd2) >= 20:
                plt.figure(figsize=(6, 4.5))
                plt.scatter(pd2["c4_fraction"], pd2[PRIMARY_RESPONSE_METRIC], alpha=0.7)
                coef = np.polyfit(pd2["c4_fraction"], pd2[PRIMARY_RESPONSE_METRIC], 1)
                xs = np.linspace(pd2["c4_fraction"].min(), pd2["c4_fraction"].max(), 100)
                plt.plot(xs, coef[0] * xs + coef[1], linestyle="--")
                plt.xlabel("C4 fraction")
                plt.ylabel(PRIMARY_RESPONSE_METRIC)
                plt.title("C4 fraction vs response phenotype")
                plt.tight_layout()
                plt.savefig(FIG / "Figure_PRODUCT03b_c4_fraction_response_scatter.png", dpi=220)
                plt.close()
                figure_status.append("c4_scatter")
    except Exception as e:
        figure_status.append(f"c4_scatter_failed:{e}")

# -----------------------------------------------------------------------------
# 8. Final report and decision
# -----------------------------------------------------------------------------
if len(rank):
    tower_ranking_text = rank.to_string(index=False)
else:
    tower_ranking_text = "No usable tower product-ranking rows found. Review tower file inventory and missing-fields table."

if len(c4_model_results):
    c4_model_text = c4_model_results.to_string(index=False)
else:
    c4_model_text = f"C4 model did not run. Reason: {c4_status.get('reason_if_not_run')}"

if len(product_audit):
    audit_summary = product_audit.groupby("category").size().reset_index(name="n_hits").to_string(index=False)
else:
    audit_summary = "No product-screening keyword hits found."

if len(surface):
    surface_summary = surface.head(30).to_string(index=False)
else:
    surface_summary = "No 2D VPD×soil-moisture surface rows were produced. See input-status table."

if len(tower_validation):
    tower_summary = tower_validation.head(80).to_string(index=False)
else:
    tower_summary = "No tower-validation rows were assembled. See tower-file inventory and missing-fields table."

final_decision = {
    "generated": now(),
    "stage": "1B.6AH_project_protocol_analysis_lock",
    "product_screening_audit": {
        "agreement_filter_keyword_hits": agreement_suspicion_count,
        "quality_filter_keyword_hits": quality_filter_count,
        "interpretation": "If agreement-filter hits correspond to actual data filtering, rerun downstream analyses without product-agreement selection. If hits are only reporting language, product-screened means QC/product-confidence layer."
    },
    "tower_validation": {
        "n_rows": int(len(tower_validation)),
        "n_sites": int(tower_validation["site_id"].nunique()) if len(tower_validation) and "site_id" in tower_validation.columns else 0,
        "tower_ranked_et_product": tower_ranked_et,
        "ranking_available": bool(len(rank)),
    },
    "strict_response_foundation": {
        "files_found": int(len(strict_foundation)),
        "inventory": strict_foundation.to_dict(orient="records") if len(strict_foundation) <= 5 else strict_foundation.head(5).to_dict(orient="records"),
    },
    "surface_2d": {
        "n_surface_rows": int(len(surface)),
        "source": surface_source,
    },
    "c3c4": c4_status,
    "protocol_checklist": {
        "n_items": int(len(checklist_df)),
        "n_satisfied": int(checklist_df["satisfied"].sum()),
        "n_not_satisfied": int((~checklist_df["satisfied"]).sum()),
    },
    "recommended_next_step": (
        "If tower ranking and C4 data are both available, review Table_PRODUCT03ah_c3c4_model_results.csv and decide ecology vs methods paper. "
        "If tower ranking is missing, assemble the tower response-class table first. If C4 data are missing, add Luo 2024 C4 fraction data locally and rerun."
    )
}

(TAB / "STAGE1B6AH_project_PROTOCOL_ANALYSIS_LOCK_DECISION.json").write_text(json.dumps(final_decision, indent=2), encoding="utf-8")

report = []
report.append("# Stage 1B.6AH project protocol analysis lock")
report.append("")
report.append(f"Generated: {final_decision['generated']}")
report.append("")
report.append("## Final decision JSON")
report.append("")
report.append("```json")
report.append(json.dumps(final_decision, indent=2))
report.append("```")
report.append("")
report.append("## What this stage satisfies")
report.append("")
report.append("```text")
report.append(checklist_df.to_string(index=False))
report.append("```")
report.append("")
report.append("## Product-screening audit")
report.append("")
report.append("This answers whether product-screened meant product-agreement selection or ordinary QC/confidence filtering. Review the audit table lines manually; the script flags suspicious agreement-filter language but does not assume it was actually used for filtering.")
report.append("")
report.append("```text")
report.append(audit_summary)
report.append("```")
report.append("")
report.append("## Tower-validation deliverable")
report.append("")
report.append("Purpose: rank ET products against tower WUE/uWUE, not prove a threshold.")
report.append("")
report.append("```text")
report.append(tower_summary)
report.append("```")
report.append("")
report.append("## Per-ET-product tower ranking")
report.append("")
report.append("```text")
report.append(tower_ranking_text)
report.append("```")
report.append("")
report.append("## Missing tower fields")
report.append("")
report.append("```text")
report.append(missing_tower.to_string(index=False))
report.append("```")
report.append("")
report.append("## Strict 2x2 response foundation")
report.append("")
report.append("```text")
report.append(strict_foundation.to_string(index=False) if len(strict_foundation) else "No strict foundation files found.")
report.append("```")
report.append("")
report.append("## 2D VPD × soil-moisture surface response")
report.append("")
report.append("```text")
report.append(surface_summary)
report.append("```")
report.append("")
report.append("## C3/C4 data inventory")
report.append("")
report.append("```text")
report.append(c4_inventory.head(80).to_string(index=False) if len(c4_inventory) else "No C3/C4 candidate files found.")
report.append("```")
report.append("")
report.append("## C3/C4 pre-specified model result")
report.append("")
report.append("```text")
report.append(c4_model_text)
report.append("```")
report.append("")
report.append("## Product confidence filter")
report.append("")
report.append("```text")
report.append(confidence.to_string(index=False))
report.append("```")
report.append("")
report.append("## Pre-registration")
report.append("")
report.append("A dated pre-registration file was written to `results/stage1b6ah_project_protocol_analysis_lock/text/PREREGISTERED_C3C4_ANALYSIS_PLAN.md`.")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("Do not write the abstract or introduction yet. The next decision depends on whether the tower table can rank ET products and whether C4 fraction can be attached to the response points. If C4 fraction survives the pre-specified controlled model on the tower-ranked product, the paper becomes an ecological mechanism paper. If not, the paper becomes a satellite-WUE identifiability and product-ranking paper.")
report.append("")
report.append("## Figures")
report.append("")
report.append("; ".join(figure_status) if figure_status else "No figures produced.")

report_text = "\n".join(report)
(TXT / "STAGE1B6AH_project_PROTOCOL_ANALYSIS_LOCK_REPORT.md").write_text(report_text, encoding="utf-8")

print(report_text)
print("")
print("WROTE", TAB / "Table_PRODUCT03aa_input_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT03ab_product_screening_audit.csv")
print("WROTE", TAB / "Table_PRODUCT03ab2_tower_file_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT03ab3_strict_response_foundation_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT03ac_tower_validation_deliverable.csv")
print("WROTE", TAB / "Table_PRODUCT03ad_et_product_tower_agreement_ranking.csv")
print("WROTE", TAB / "Table_PRODUCT03ae_tower_validation_missing_fields.csv")
print("WROTE", TAB / "Table_PRODUCT03af_c3c4_data_inventory.csv")
print("WROTE", TAB / "Table_PRODUCT03ag_c3c4_joined_model_data.csv")
print("WROTE", TAB / "Table_PRODUCT03ah_c3c4_model_results.csv")
print("WROTE", TAB / "Table_PRODUCT03ah0_c3c4_test_status.csv")
print("WROTE", TAB / "Table_PRODUCT03ah2_c3c4_leave_one_block_out.csv")
print("WROTE", TAB / "Table_PRODUCT03ai_2d_surface_response_metrics.csv")
print("WROTE", TAB / "Table_PRODUCT03ai2_2d_surface_input_status.csv")
print("WROTE", TAB / "Table_PRODUCT03aj_product_confidence_filter.csv")
print("WROTE", TAB / "Table_PRODUCT03ak_protocol_satisfaction_checklist.csv")
print("WROTE", TAB / "STAGE1B6AH_project_PROTOCOL_ANALYSIS_LOCK_DECISION.json")
print("WROTE", TXT / "PREREGISTERED_C3C4_ANALYSIS_PLAN.md")
print("WROTE", TXT / "STAGE1B6AH_project_PROTOCOL_ANALYSIS_LOCK_REPORT.md")
print("WROTE figures to", FIG)
