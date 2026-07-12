from pathlib import Path
from datetime import datetime
import json
import math
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    sm = None
    smf = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

ROOT = Path(".")
OUT = Path("results/stage1b6ai_project_final_lock_with_c4")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260704
rng = np.random.default_rng(SEED)
N_BOOT = 250
PRIMARY_RESPONSE = "latent_post_slope"
PRIMARY_METRIC = "uWUE"

def now():
    return datetime.now().isoformat(timespec="seconds")

def norm(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")

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

def first_col(df, candidates):
    lut = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    return None

def contains_any(s, words):
    s = str(s).lower()
    return any(w.lower() in s for w in words)

def clean_bool(x):
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "agree", "match", "matched"}

def clean_class(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    if s in {"", "nan", "none"}:
        return np.nan
    if "threshold" in s:
        return "threshold-like"
    if "sat" in s or "break" in s:
        return "saturation/breakdown"
    if "reversal" in s:
        return "reversal"
    if "enhance" in s:
        return "enhancement"
    if "limit" in s:
        return "limitation-like"
    if "weak" in s or "inconclusive" in s or "mixed" in s:
        return "weak/inconclusive"
    return s

def infer_et_product(text):
    s = str(text).lower()
    if "gleam" in s:
        return "GLEAM"
    if "modis" in s or "mod16" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    return "UNKNOWN_ET"

def infer_gpp_product(text):
    s = str(text).lower()
    if "gosif" in s:
        return "GOSIF"
    if "modis" in s or "mod17" in s:
        return "MODIS"
    if "pml" in s:
        return "PML"
    return "UNKNOWN_GPP"

def product_combo_from_row(row):
    txt = " ".join([str(v) for v in row.values if pd.notna(v)])
    gpp = infer_gpp_product(txt)
    et = infer_et_product(txt)
    if gpp == "UNKNOWN_GPP" and et == "UNKNOWN_ET":
        return "UNKNOWN_PRODUCT"
    return f"{gpp}x{et}"

def standardize_lat_lon(df):
    out = df.copy()
    lat = first_col(out, ["lat", "latitude", "LATITUDE", "site_lat", "tower_lat", "y"])
    lon = first_col(out, ["lon", "longitude", "LONGITUDE", "site_lon", "tower_lon", "x"])
    if lat and "lat" not in out.columns:
        out["lat"] = to_num(out[lat])
    if lon and "lon" not in out.columns:
        out["lon"] = to_num(out[lon])
    return out

def zscore(s):
    s = to_num(s)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def normal_p_from_t(t):
    if not np.isfinite(t):
        return np.nan
    return float(math.erfc(abs(t) / math.sqrt(2)))

def ols_standardized(df, y_col, x_cols):
    d = df[[y_col] + x_cols].copy()
    for c in d.columns:
        d[c] = to_num(d[c])
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    if len(d) < max(20, len(x_cols) + 5):
        return None

    y = d[y_col].to_numpy(float)
    X_parts = [np.ones(len(d))]
    kept = []
    for c in x_cols:
        z = zscore(d[c]).to_numpy(float)
        if np.isfinite(z).all() and np.nanstd(z) > 0:
            X_parts.append(z)
            kept.append(c)

    if "c4_fraction" not in kept:
        return None

    X = np.column_stack(X_parts)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    n = len(y)
    k = X.shape[1]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1 - (1-r2)*(n-1)/max(1, n-k) if np.isfinite(r2) else np.nan

    sigma2 = rss / max(1, n-k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)

    rows = []
    for name, b, s in zip(["intercept"] + kept, beta, se):
        t = b / s if np.isfinite(s) and s != 0 else np.nan
        rows.append({
            "term": name,
            "coef_standardized": float(b),
            "se": float(s) if np.isfinite(s) else np.nan,
            "t_normal_approx": float(t) if np.isfinite(t) else np.nan,
            "p_normal_approx": normal_p_from_t(t),
        })

    return {
        "n": n,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coef_table": pd.DataFrame(rows),
        "residuals": resid,
        "pred": pred,
        "fit_index": d.index,
        "predictors_used": kept,
    }

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

def list_files():
    roots = [Path("data"), Path("results")]
    suffixes = {".csv", ".txt", ".tsv", ".nc", ".nc4", ".parquet"}
    out = []
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in suffixes and p.stat().st_size > 0:
                    out.append(p)
            except Exception:
                pass
    return sorted(set(out))

all_files = list_files()

# -----------------------------------------------------------------------------
# A. Clean product-screening answer
# -----------------------------------------------------------------------------
screen_audit_path = Path("results/stage1b6ah_project_protocol_analysis_lock/tables/Table_PRODUCT03ab_product_screening_audit.csv")
screening_answer = {
    "agreement_filter_keyword_hits": np.nan,
    "quality_filter_keyword_hits": np.nan,
    "answer_for_project": "unknown; prior audit file not found",
}
if screen_audit_path.exists():
    audit = read_csv_safe(screen_audit_path)
    if audit is not None and len(audit):
        agreement_hits = int((audit["category"] == "agreement_filter_suspicious").sum()) if "category" in audit.columns else 0
        quality_hits = int((audit["category"] == "quality_filter_ok").sum()) if "category" in audit.columns else 0
        screening_answer = {
            "agreement_filter_keyword_hits": agreement_hits,
            "quality_filter_keyword_hits": quality_hits,
            "answer_for_project": (
                "No evidence of product-agreement filtering was found by keyword audit; product-screened appears to mean QC/product-confidence screening, not keeping only product-agreeing pixels."
                if agreement_hits == 0 else
                "Potential product-agreement filtering language was found; review line-level audit before making downstream claims."
            ),
        }
    else:
        screening_answer = {
            "agreement_filter_keyword_hits": 0,
            "quality_filter_keyword_hits": 0,
            "answer_for_project": "Audit file existed but was empty."
        }

pd.DataFrame([screening_answer]).to_csv(TAB / "Table_PRODUCT03al_product_screening_answer_for_project.csv", index=False)

# -----------------------------------------------------------------------------
# B. Clean tower validation table from prior deliverable
# -----------------------------------------------------------------------------
prior_tower = Path("results/stage1b6ah_project_protocol_analysis_lock/tables/Table_PRODUCT03ac_tower_validation_deliverable.csv")
if not prior_tower.exists():
    raise FileNotFoundError("Prior tower deliverable missing. Run stage1b6ah first.")

tw = read_csv_safe(prior_tower)
if tw is None:
    raise ValueError("Could not read prior tower validation deliverable.")

for c in ["tower_response_class", "satellite_response_class"]:
    if c in tw.columns:
        tw[c] = tw[c].map(clean_class)

if "exact_agreement" in tw.columns:
    tw["exact_agreement"] = tw["exact_agreement"].map(clean_bool)
else:
    tw["exact_agreement"] = tw["tower_response_class"].astype(str).eq(tw["satellite_response_class"].astype(str))

if "slope_direction_agreement" in tw.columns:
    tw["slope_direction_agreement"] = tw["slope_direction_agreement"].map(lambda x: np.nan if pd.isna(x) else clean_bool(x))
else:
    tw["slope_direction_agreement"] = np.nan

# Strictly exclude demo, raw point-only metadata, unknown products, and rows lacking tower/satellite classes.
source = tw["source_path"].astype(str).str.lower() if "source_path" in tw.columns else pd.Series("", index=tw.index)
site = tw["site_id"].astype(str) if "site_id" in tw.columns else pd.Series("", index=tw.index)
et = tw["et_product"].astype(str) if "et_product" in tw.columns else pd.Series("UNKNOWN_ET", index=tw.index)
combo = tw["product_combo"].astype(str) if "product_combo" in tw.columns else pd.Series("UNKNOWN_PRODUCT", index=tw.index)

clean_mask = (
    (~source.str.contains("data/demo", na=False))
    & (~site.str.startswith("DEMO", na=False))
    & (~combo.str.contains("UNKNOWN_PRODUCT", na=False))
    & (~et.str.contains("UNKNOWN_ET", na=False))
    & (tw["tower_response_class"].notna())
    & (tw["satellite_response_class"].notna())
)

clean_tower = tw[clean_mask].copy()

# Deduplicate to one row per site-product-towerclass-satclass where possible.
dedupe_cols = [c for c in ["site_id", "igbp_class", "product_combo", "gpp_product", "et_product", "tower_response_class", "satellite_response_class"] if c in clean_tower.columns]
if dedupe_cols:
    clean_tower = clean_tower.sort_values(["site_id", "product_combo"]).drop_duplicates(dedupe_cols, keep="first")

# -----------------------------------------------------------------------------
# C. Try to compute closure/gap-fill from raw flux files
# -----------------------------------------------------------------------------
def infer_site_id_from_file_or_df(path, df):
    for c in ["site_id", "SITE_ID", "site", "Site", "tower_id"]:
        if c in df.columns:
            vals = df[c].dropna().astype(str).unique()
            if len(vals):
                return vals[0]
    name = path.name
    m = re.search(r"([A-Z]{2}-[A-Za-z0-9]{2,3})", name)
    if m:
        return m.group(1)
    return None

def infer_year_col(df):
    for c in ["year", "YEAR", "yr"]:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = norm(c)
        if "timestamp" in lc or "date" in lc:
            return c
    return None

def extract_year(series):
    s = series.astype(str)
    y = s.str.extract(r"(\d{4})")[0]
    return to_num(y)

def compute_flux_quality(path):
    d0 = read_csv_safe(path, nrows=5)
    if d0 is None:
        return None
    cols_low = " ".join([norm(c) for c in d0.columns])
    if not any(k in cols_low for k in ["le_f_mds", "h_f_mds", "netrad", "sw_in", "g_f_mds", "gpp_nt", "nee_vut"]):
        return None

    d = read_csv_safe(path)
    if d is None or len(d) < 20:
        return None

    site_id = infer_site_id_from_file_or_df(path, d)
    if site_id is None:
        return None

    le_col = first_col(d, ["LE_F_MDS", "LE_CORR", "LE", "LE_F_MDS_CORR", "LE_PI_F"])
    h_col = first_col(d, ["H_F_MDS", "H_CORR", "H", "H_F_MDS_CORR", "H_PI_F"])
    rn_col = first_col(d, ["NETRAD", "NETRAD_F", "Rn", "RNET", "NETRAD_PI_F"])
    g_col = first_col(d, ["G_F_MDS", "G", "G_1_1_1", "G_PI_F", "G_F"])
    year_col = infer_year_col(d)

    if year_col:
        years = extract_year(d[year_col])
    else:
        years = pd.Series(np.nan, index=d.index)

    out = {
        "site_id": site_id,
        "source_flux_file": str(path),
        "n_flux_rows": int(len(d)),
        "year_min": float(years.min()) if years.notna().any() else np.nan,
        "year_max": float(years.max()) if years.notna().any() else np.nan,
        "site_years_flux_file": int(years.nunique(dropna=True)) if years.notna().any() else np.nan,
    }

    if le_col and h_col and rn_col:
        le = to_num(d[le_col])
        h = to_num(d[h_col])
        rn = to_num(d[rn_col])
        if g_col:
            g = to_num(d[g_col])
        else:
            g = pd.Series(0.0, index=d.index)

        valid = le.notna() & h.notna() & rn.notna() & g.notna() & ((rn - g).abs() > 1e-9)
        if valid.sum() >= 20:
            closure = (h[valid] + le[valid]).sum() / (rn[valid] - g[valid]).sum()
            out["closure_ratio"] = float(closure)
            out["closure_pass_0p7_1p3"] = bool(0.7 <= closure <= 1.3)
        else:
            out["closure_ratio"] = np.nan
            out["closure_pass_0p7_1p3"] = False
    else:
        out["closure_ratio"] = np.nan
        out["closure_pass_0p7_1p3"] = False

    qc_cols = [c for c in d.columns if norm(c).endswith("_qc") or "qc" in norm(c)]
    flux_qc_cols = [c for c in qc_cols if any(k in norm(c) for k in ["le", "h_", "gpp", "nee", "vpd"])]
    gap_fracs = []
    for c in flux_qc_cols:
        q = to_num(d[c])
        if q.notna().sum() >= 20:
            # FLUXNET-style QC: 0 observed/high quality; >0 gapfilled/lower quality.
            gap_fracs.append(float((q > 0).mean()))
    out["gapfill_fraction_estimated"] = float(np.nanmean(gap_fracs)) if gap_fracs else np.nan
    out["gapfill_pass_0p3"] = bool(pd.notna(out["gapfill_fraction_estimated"]) and out["gapfill_fraction_estimated"] <= 0.3)
    out["qc_columns_used"] = ";".join(flux_qc_cols[:20])
    return out

flux_files = [
    p for p in all_files
    if p.suffix.lower() in {".csv", ".txt", ".tsv"}
    and contains_any(str(p), ["fluxnet", "ameriflux", "icos", "ozflux", "tower"])
    and not contains_any(str(p), ["tower_validation_deliverable", "tower_file_inventory"])
]

flux_quality_rows = []
iterator = tqdm(flux_files, desc="Computing tower closure/gap-fill if raw fluxes exist", unit="file") if tqdm else flux_files
for p in iterator:
    r = compute_flux_quality(p)
    if r is not None:
        flux_quality_rows.append(r)

flux_quality = pd.DataFrame(flux_quality_rows)
if len(flux_quality):
    flux_quality = flux_quality.sort_values(["site_id", "source_flux_file"])
else:
    flux_quality = pd.DataFrame(columns=[
        "site_id", "source_flux_file", "n_flux_rows", "year_min", "year_max", "site_years_flux_file",
        "closure_ratio", "closure_pass_0p7_1p3", "gapfill_fraction_estimated", "gapfill_pass_0p3",
        "qc_columns_used"
    ])

flux_quality.to_csv(TAB / "Table_PRODUCT03am_tower_flux_quality_closure_gapfill.csv", index=False)

# Merge flux quality onto clean tower table when available.
if len(clean_tower) and len(flux_quality):
    fq = (
        flux_quality
        .sort_values(["closure_pass_0p7_1p3", "gapfill_pass_0p3", "n_flux_rows"], ascending=[False, False, False])
        .drop_duplicates("site_id", keep="first")
    )
    clean_tower = clean_tower.merge(fq, on="site_id", how="left", suffixes=("", "_computed"))
    if "closure_ratio_computed" in clean_tower.columns:
        clean_tower["closure_ratio_final"] = clean_tower.get("closure_ratio", np.nan)
        clean_tower["closure_ratio_final"] = to_num(clean_tower["closure_ratio_final"]).fillna(to_num(clean_tower["closure_ratio_computed"]))
    else:
        clean_tower["closure_ratio_final"] = clean_tower.get("closure_ratio", np.nan)
    if "gapfill_fraction_estimated" in clean_tower.columns:
        clean_tower["gapfill_fraction_final"] = clean_tower.get("gapfill_fraction", np.nan)
        clean_tower["gapfill_fraction_final"] = to_num(clean_tower["gapfill_fraction_final"]).fillna(to_num(clean_tower["gapfill_fraction_estimated"]))
    else:
        clean_tower["gapfill_fraction_final"] = clean_tower.get("gapfill_fraction", np.nan)
else:
    if len(clean_tower):
        clean_tower["closure_ratio_final"] = clean_tower.get("closure_ratio", np.nan)
        clean_tower["gapfill_fraction_final"] = clean_tower.get("gapfill_fraction", np.nan)

if len(clean_tower):
    clean_tower["passes_project_quality_fields"] = (
        clean_tower["site_id"].notna()
        & clean_tower["tower_response_class"].notna()
        & clean_tower["satellite_response_class"].notna()
        & clean_tower["et_product"].notna()
        & (~clean_tower["et_product"].astype(str).str.contains("UNKNOWN", na=False))
    )
else:
    clean_tower["passes_project_quality_fields"] = []

clean_tower.to_csv(TAB / "Table_PRODUCT03an_project_clean_tower_validation_table.csv", index=False)

if len(clean_tower):
    et_rank = (
        clean_tower[clean_tower["passes_project_quality_fields"]]
        .groupby("et_product", dropna=False)
        .agg(
            n_site_product_rows=("site_id", "size"),
            n_unique_sites=("site_id", "nunique"),
            exact_agreement_rate=("exact_agreement", "mean"),
            slope_direction_agreement_rate=("slope_direction_agreement", lambda x: np.nanmean(to_num(x)) if to_num(x).notna().any() else np.nan),
            n_with_closure=("closure_ratio_final", lambda x: int(to_num(x).notna().sum())),
            n_with_gapfill=("gapfill_fraction_final", lambda x: int(to_num(x).notna().sum())),
        )
        .reset_index()
        .sort_values(["exact_agreement_rate", "n_unique_sites"], ascending=[False, False])
    )
else:
    et_rank = pd.DataFrame(columns=["et_product", "n_site_product_rows", "n_unique_sites", "exact_agreement_rate", "slope_direction_agreement_rate", "n_with_closure", "n_with_gapfill"])

et_rank.to_csv(TAB / "Table_PRODUCT03ao_project_clean_et_product_ranking.csv", index=False)
tower_ranked_et = str(et_rank.iloc[0]["et_product"]) if len(et_rank) else None

# -----------------------------------------------------------------------------
# D. C4 sampling from Luo NetCDF
# -----------------------------------------------------------------------------
c4_nc = Path("data/raw/traits/c4_fraction/C4_distribution_NUS_v2.2.nc")
point_candidates = [
    Path("results/paper_point_geography_thesis_lock/tables/Table70_point_level_geography_response_annotation.csv"),
    Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv"),
]
point_path = None
point = None
for p in point_candidates:
    if p.exists():
        d = read_csv_safe(p)
        if d is not None and len(d):
            d = standardize_lat_lon(d)
            if "lat" in d.columns and "lon" in d.columns:
                point_path = p
                point = d
                break

c4_status = {
    "c4_nc_path": str(c4_nc),
    "c4_nc_exists": c4_nc.exists(),
    "point_table": str(point_path) if point_path else None,
    "c4_variable_selected": None,
    "c4_sampling_ran": False,
    "c4_model_ran": False,
    "reason_if_not": None,
}

c4_var_inventory = []
joined = pd.DataFrame()

if not c4_nc.exists():
    c4_status["reason_if_not"] = "C4 NetCDF missing after download step."
elif xr is None:
    c4_status["reason_if_not"] = "xarray is not available."
elif point is None:
    c4_status["reason_if_not"] = "No point table with lat/lon found."
else:
    ds = xr.open_dataset(c4_nc)

    lat_name = None
    lon_name = None
    for c in list(ds.coords) + list(ds.dims):
        lc = norm(c)
        if lc in {"lat", "latitude", "y"}:
            lat_name = c
        if lc in {"lon", "longitude", "x"}:
            lon_name = c

    if lat_name is None or lon_name is None:
        c4_status["reason_if_not"] = f"Could not infer lat/lon dimensions from NetCDF. coords={list(ds.coords)}, dims={list(ds.dims)}"
    else:
        for v in ds.data_vars:
            da = ds[v]
            dims = set(da.dims)
            name = norm(v)
            score = 0
            if "c4" in name:
                score += 10
            if "grass" in name or "natural" in name:
                score += 8
            if "crop" in name:
                score -= 5
            if "uncert" in name or "std" in name or "sd" in name or "error" in name:
                score -= 6
            if lat_name in dims and lon_name in dims:
                score += 5
            if len(da.shape) >= 2:
                score += 2
            c4_var_inventory.append({
                "variable": v,
                "dims": ";".join(map(str, da.dims)),
                "shape": ";".join(map(str, da.shape)),
                "score": score,
                "attrs": str(da.attrs)[:300],
            })

        var_inv = pd.DataFrame(c4_var_inventory).sort_values("score", ascending=False)
        var_inv.to_csv(TAB / "Table_PRODUCT03ap_c4_netcdf_variable_inventory.csv", index=False)

        if len(var_inv) == 0 or var_inv.iloc[0]["score"] < 10:
            c4_status["reason_if_not"] = "No plausible C4 fraction variable found in NetCDF."
        else:
            c4_var = str(var_inv.iloc[0]["variable"])
            c4_status["c4_variable_selected"] = c4_var
            da = ds[c4_var]

            # Average over time/year if present.
            time_dims = [d for d in da.dims if norm(d) in {"time", "year", "years"}]
            if time_dims:
                da2 = da.mean(dim=time_dims, skipna=True)
                c4_status["time_handling"] = f"mean_over_{';'.join(time_dims)}"
            else:
                da2 = da
                c4_status["time_handling"] = "static_no_time_dim"

            # Convert longitudes if needed.
            lon_vals = ds[lon_name].values
            point_lons = to_num(point["lon"]).copy()
            if np.nanmin(lon_vals) >= 0 and point_lons.min() < 0:
                sample_lons = point_lons % 360
            else:
                sample_lons = point_lons

            # Xarray nearest sampling.
            sample_vals = []
            sample_dist = []
            lat_vals = np.asarray(ds[lat_name].values, dtype=float)
            lon_vals2 = np.asarray(ds[lon_name].values, dtype=float)

            iterator = tqdm(point.iterrows(), total=len(point), desc="Sampling Luo C4 fraction to response points", unit="point") if tqdm else point.iterrows()
            for idx, r in iterator:
                lat = float(r["lat"]) if pd.notna(r["lat"]) else np.nan
                lon = float(sample_lons.loc[idx]) if pd.notna(sample_lons.loc[idx]) else np.nan
                if not np.isfinite(lat) or not np.isfinite(lon):
                    sample_vals.append(np.nan)
                    sample_dist.append(np.nan)
                    continue
                try:
                    val = da2.sel({lat_name: lat, lon_name: lon}, method="nearest").values
                    val = float(np.asarray(val).squeeze())
                except Exception:
                    val = np.nan
                # rough nearest distance for audit
                jlat = np.nanargmin(np.abs(lat_vals - lat))
                jlon = np.nanargmin(np.abs(lon_vals2 - lon))
                dist = math.sqrt((lat_vals[jlat] - lat)**2 + ((lon_vals2[jlon] - lon) * math.cos(math.radians(lat)))**2)
                sample_vals.append(val)
                sample_dist.append(float(dist))

            joined = point.copy()
            joined["c4_fraction_raw"] = sample_vals
            joined["c4_sample_distance_deg"] = sample_dist
            joined["c4_variable"] = c4_var

            # Heuristic scale: if values are percentages, convert to 0-1.
            vals = to_num(joined["c4_fraction_raw"])
            if vals.quantile(0.95) > 1.5:
                joined["c4_fraction"] = vals / 100.0
                c4_status["scale_conversion"] = "divided_by_100"
            else:
                joined["c4_fraction"] = vals
                c4_status["scale_conversion"] = "no_conversion"

            c4_status["c4_sampling_ran"] = True
            c4_status["n_points"] = int(len(joined))
            c4_status["n_points_with_c4"] = int(joined["c4_fraction"].notna().sum())
            c4_status["c4_mean"] = float(joined["c4_fraction"].mean(skipna=True))
            c4_status["c4_min"] = float(joined["c4_fraction"].min(skipna=True))
            c4_status["c4_max"] = float(joined["c4_fraction"].max(skipna=True))

joined.to_csv(TAB / "Table_PRODUCT03aq_c4_sampled_point_table.csv", index=False)
pd.DataFrame([c4_status]).to_csv(TAB / "Table_PRODUCT03ar_c4_sampling_status.csv", index=False)

# -----------------------------------------------------------------------------
# E. Pre-specified C4 model
# -----------------------------------------------------------------------------
def choose_response_col(df):
    for c in [PRIMARY_RESPONSE, "p_threshold_like", "latent_slope_change", "p_satbreak", "latent_satbreak_probability"]:
        if c in df.columns:
            return c
    return None

def choose_controls(df):
    candidates = [
        "aridity", "aridity_index",
        "soil_sand", "soil_clay", "soil_silt",
        "mean_annual_temperature", "mean_temperature",
        "mean_annual_precipitation", "mean_precipitation",
        "mean_lai", "growing_season_mean_lai",
        "mean_vpd", "baseline_vpd",
        "mean_soil_moisture", "baseline_soil_moisture",
        "rooting_depth",
    ]
    controls = []
    for c in candidates:
        cc = first_col(df, [c])
        if cc and cc not in controls and cc != "c4_fraction":
            s = to_num(df[cc])
            if s.notna().sum() >= 20 and s.nunique(dropna=True) > 2:
                controls.append(cc)
    return controls

c4_model_rows = []
c4_boot_rows = []
c4_block_rows = []
c4_decision = {
    "c4_test_ran": False,
    "reason_if_not": c4_status.get("reason_if_not"),
    "tower_ranked_et": tower_ranked_et,
}

if len(joined) and "c4_fraction" in joined.columns and joined["c4_fraction"].notna().sum() >= 20:
    response_col = choose_response_col(joined)
    if response_col is None:
        c4_decision["reason_if_not"] = "No response metric found in joined C4 table."
    else:
        model_df = joined.copy()

        # If product columns are present and tower-ranked ET is available, run both all-points and tower-ranked subset.
        product_cols = [c for c in model_df.columns if any(k in norm(c) for k in ["product", "combo", "et_product", "gpp_product", "matrix_role"])]
        if product_cols and tower_ranked_et:
            prod_text = model_df[product_cols].astype(str).agg(" ".join, axis=1)
            model_df["uses_tower_ranked_et"] = prod_text.str.lower().str.contains(str(tower_ranked_et).lower())
        else:
            model_df["uses_tower_ranked_et"] = True

        model_variants = [
            ("all_available_points", model_df.copy()),
            ("tower_ranked_et_points", model_df[model_df["uses_tower_ranked_et"]].copy()),
        ]

        for model_name, md in model_variants:
            if len(md) < 20:
                continue
            controls = choose_controls(md)
            predictors = ["c4_fraction"] + controls
            fit = ols_standardized(md, response_col, predictors)
            if fit is None:
                continue

            coef = fit["coef_table"].copy()
            coef["model_name"] = model_name
            coef["response"] = response_col
            coef["n"] = fit["n"]
            coef["r2"] = fit["r2"]
            coef["adj_r2"] = fit["adj_r2"]
            coef["predictors_used"] = ";".join(fit["predictors_used"])
            c4_model_rows.append(coef)

            fit_df = md.loc[fit["fit_index"]].copy()
            fit_df["residual"] = fit["residuals"]

            # Blocks: prefer eco_biome, then eco_realm, then lat-lon 10 degree.
            if "eco_biome" in fit_df.columns:
                block_col = "eco_biome"
            elif "eco_realm" in fit_df.columns:
                block_col = "eco_realm"
            elif "lat" in fit_df.columns and "lon" in fit_df.columns:
                fit_df["spatial_block_10deg"] = (
                    np.floor(to_num(fit_df["lat"]) / 10).astype("Int64").astype(str)
                    + "_"
                    + np.floor(to_num(fit_df["lon"]) / 10).astype("Int64").astype(str)
                )
                block_col = "spatial_block_10deg"
            else:
                fit_df["all_block"] = "all"
                block_col = "all_block"

            blocks = list(fit_df[block_col].dropna().unique())
            c4_coef = float(coef.loc[coef["term"] == "c4_fraction", "coef_standardized"].iloc[0])
            c4_p = float(coef.loc[coef["term"] == "c4_fraction", "p_normal_approx"].iloc[0])

            # Spatial block bootstrap.
            boot_coefs = []
            if len(blocks) >= 3:
                for _ in tqdm(range(N_BOOT), desc=f"Block bootstrap {model_name}", unit="boot") if tqdm else range(N_BOOT):
                    sampled = rng.choice(blocks, size=len(blocks), replace=True)
                    bd = pd.concat([fit_df[fit_df[block_col] == b] for b in sampled], ignore_index=True)
                    bf = ols_standardized(bd, response_col, fit["predictors_used"])
                    if bf is not None:
                        bt = bf["coef_table"]
                        if (bt["term"] == "c4_fraction").any():
                            boot_coefs.append(float(bt.loc[bt["term"] == "c4_fraction", "coef_standardized"].iloc[0]))

            # Leave-one-block-out.
            loo_coefs = []
            for b in blocks:
                ld = fit_df[fit_df[block_col] != b].copy()
                if len(ld) < 20:
                    continue
                lf = ols_standardized(ld, response_col, fit["predictors_used"])
                if lf is None:
                    continue
                lt = lf["coef_table"]
                if (lt["term"] == "c4_fraction").any():
                    val = float(lt.loc[lt["term"] == "c4_fraction", "coef_standardized"].iloc[0])
                    loo_coefs.append(val)
                    c4_block_rows.append({
                        "model_name": model_name,
                        "left_out_block": str(b),
                        "n_train": lf["n"],
                        "c4_coef_standardized": val,
                    })

            c4_boot_rows.append({
                "model_name": model_name,
                "response": response_col,
                "n": fit["n"],
                "block_col": block_col,
                "n_blocks": len(blocks),
                "c4_coef_standardized": c4_coef,
                "c4_p_normal_approx": c4_p,
                "bootstrap_n": len(boot_coefs),
                "bootstrap_median": float(np.median(boot_coefs)) if boot_coefs else np.nan,
                "bootstrap_p025": float(np.quantile(boot_coefs, 0.025)) if boot_coefs else np.nan,
                "bootstrap_p975": float(np.quantile(boot_coefs, 0.975)) if boot_coefs else np.nan,
                "loo_n": len(loo_coefs),
                "loo_sign_stability": float(np.mean(np.sign(loo_coefs) == np.sign(c4_coef))) if loo_coefs else np.nan,
            })

        if c4_model_rows:
            c4_decision["c4_test_ran"] = True
            c4_decision["reason_if_not"] = None
        else:
            c4_decision["reason_if_not"] = "C4 table sampled, but model did not have enough complete rows after controls."

c4_model = pd.concat(c4_model_rows, ignore_index=True) if c4_model_rows else pd.DataFrame()
if len(c4_model) and "p_normal_approx" in c4_model.columns:
    c4_model["bh_q_normal_approx"] = bh_qvalues(c4_model["p_normal_approx"].to_numpy(float))

c4_boot = pd.DataFrame(c4_boot_rows)
c4_blocks = pd.DataFrame(c4_block_rows)

c4_model.to_csv(TAB / "Table_PRODUCT03as_c4_partial_effect_model.csv", index=False)
c4_boot.to_csv(TAB / "Table_PRODUCT03at_c4_spatial_block_bootstrap.csv", index=False)
c4_blocks.to_csv(TAB / "Table_PRODUCT03au_c4_leave_one_block_out.csv", index=False)

# Decision interpretation.
if len(c4_model):
    focal = c4_model[c4_model["term"] == "c4_fraction"].copy()
    if len(focal):
        focal = focal.sort_values(["model_name", "p_normal_approx"])
        best = focal.iloc[0]
        boot_match = c4_boot[c4_boot["model_name"] == best["model_name"]]
        loo_stab = float(boot_match["loo_sign_stability"].iloc[0]) if len(boot_match) and pd.notna(boot_match["loo_sign_stability"].iloc[0]) else np.nan
        ci_low = float(boot_match["bootstrap_p025"].iloc[0]) if len(boot_match) and pd.notna(boot_match["bootstrap_p025"].iloc[0]) else np.nan
        ci_high = float(boot_match["bootstrap_p975"].iloc[0]) if len(boot_match) and pd.notna(boot_match["bootstrap_p975"].iloc[0]) else np.nan

        passes = (
            pd.notna(best["p_normal_approx"])
            and best["p_normal_approx"] <= 0.05
            and (pd.isna(loo_stab) or loo_stab >= 0.80)
            and (pd.isna(ci_low) or pd.isna(ci_high) or ci_low * ci_high > 0)
        )

        c4_decision.update({
            "best_model_name": str(best["model_name"]),
            "response": str(best["response"]),
            "n": int(best["n"]),
            "c4_coef_standardized": float(best["coef_standardized"]),
            "c4_p_normal_approx": float(best["p_normal_approx"]),
            "c4_bh_q_normal_approx": float(best["bh_q_normal_approx"]) if "bh_q_normal_approx" in best and pd.notna(best["bh_q_normal_approx"]) else np.nan,
            "spatial_bootstrap_ci": [ci_low, ci_high],
            "loo_sign_stability": loo_stab,
            "passes_fast_pre_specified_screen": bool(passes),
        })
    else:
        c4_decision["reason_if_not"] = "C4 model ran but c4_fraction term missing."

pd.DataFrame([c4_decision]).to_csv(TAB / "Table_PRODUCT03av_c4_test_decision.csv", index=False)

# -----------------------------------------------------------------------------
# F. project-ready final packet
# -----------------------------------------------------------------------------
tower_summary = {
    "clean_tower_rows": int(len(clean_tower)),
    "clean_tower_sites": int(clean_tower["site_id"].nunique()) if len(clean_tower) else 0,
    "et_ranking_rows": int(len(et_rank)),
    "tower_ranked_et": tower_ranked_et,
    "has_closure_for_any_clean_rows": bool(len(clean_tower) and to_num(clean_tower.get("closure_ratio_final", pd.Series([]))).notna().any()),
    "has_gapfill_for_any_clean_rows": bool(len(clean_tower) and to_num(clean_tower.get("gapfill_fraction_final", pd.Series([]))).notna().any()),
}

final_decision = {
    "generated": datetime.now().isoformat(timespec="seconds"),
    "stage": "1B.6AI_project_final_lock_with_C4",
    "product_screening_answer": screening_answer,
    "tower_validation_clean": tower_summary,
    "c4_sampling_status": c4_status,
    "c4_test_decision": c4_decision,
    "paper_fork": (
        "ecological_C3C4_mechanism_paper"
        if c4_decision.get("passes_fast_pre_specified_screen") else
        "methods_or_identifiability_paper_unless_C4_improves_after_hierarchical_model"
    ),
    "what_is_still_missing_for_project": [
        "Closure ratio remains missing if no raw tower flux files with H/LE/NETRAD/G columns were found.",
        "Gap-fill fraction remains missing if no FLUXNET/AmeriFlux QC columns were found.",
        "If C4 passes only in OLS, upgrade to full mixed/hierarchical model before manuscript claims.",
    ],
}

(TAB / "STAGE1B6AI_project_FINAL_LOCK_DECISION.json").write_text(json.dumps(final_decision, indent=2), encoding="utf-8")

# Figures.
fig_status = []
if plt is not None:
    try:
        if len(et_rank):
            plt.figure(figsize=(7,4))
            plt.bar(et_rank["et_product"].astype(str), et_rank["exact_agreement_rate"].astype(float))
            plt.ylabel("Exact tower/satellite class agreement")
            plt.xlabel("ET product")
            plt.title("Clean tower-ranked ET products")
            plt.tight_layout()
            plt.savefig(FIG / "Figure_PRODUCT03c_clean_tower_et_ranking.png", dpi=220)
            plt.close()
            fig_status.append("clean_tower_ranking")
    except Exception as e:
        fig_status.append(f"tower_fig_failed:{e}")

    try:
        if len(joined) and "c4_fraction" in joined.columns and PRIMARY_RESPONSE in joined.columns:
            dd = joined[["c4_fraction", PRIMARY_RESPONSE]].copy()
            dd["c4_fraction"] = to_num(dd["c4_fraction"])
            dd[PRIMARY_RESPONSE] = to_num(dd[PRIMARY_RESPONSE])
            dd = dd.dropna()
            if len(dd) >= 20:
                plt.figure(figsize=(6,4.5))
                plt.scatter(dd["c4_fraction"], dd[PRIMARY_RESPONSE], alpha=0.75)
                coef = np.polyfit(dd["c4_fraction"], dd[PRIMARY_RESPONSE], 1)
                xs = np.linspace(dd["c4_fraction"].min(), dd["c4_fraction"].max(), 100)
                plt.plot(xs, coef[0]*xs + coef[1], linestyle="--")
                plt.xlabel("C4 fraction")
                plt.ylabel(PRIMARY_RESPONSE)
                plt.title("C4 fraction vs response phenotype")
                plt.tight_layout()
                plt.savefig(FIG / "Figure_PRODUCT03d_c4_response_scatter.png", dpi=220)
                plt.close()
                fig_status.append("c4_scatter")
    except Exception as e:
        fig_status.append(f"c4_fig_failed:{e}")

# Report.
report = []
report.append("# Stage 1B.6AI project final lock with C4")
report.append("")
report.append(f"Generated: {final_decision['generated']}")
report.append("")
report.append("## Final decision")
report.append("")
report.append("```json")
report.append(json.dumps(final_decision, indent=2))
report.append("```")
report.append("")
report.append("## 1. Product-screening answer")
report.append("")
report.append(screening_answer["answer_for_project"])
report.append("")
report.append("```text")
report.append(pd.DataFrame([screening_answer]).to_string(index=False))
report.append("```")
report.append("")
report.append("## 2. Clean tower-validation table")
report.append("")
report.append("This table excludes demo rows, raw coordinate-only exports, unknown products, and rows without both tower and satellite response classes.")
report.append("")
report.append("```text")
show_tower_cols = [c for c in [
    "site_id", "igbp_class", "site_years", "closure_ratio_final", "gapfill_fraction_final",
    "tower_response_class", "satellite_response_class", "product_combo", "gpp_product", "et_product",
    "exact_agreement", "slope_direction_agreement", "source_path"
] if c in clean_tower.columns]
report.append(clean_tower[show_tower_cols].head(120).to_string(index=False) if len(clean_tower) else "No clean tower rows found.")
report.append("```")
report.append("")
report.append("## 3. Clean ET product ranking")
report.append("")
report.append("```text")
report.append(et_rank.to_string(index=False) if len(et_rank) else "No clean ET ranking available.")
report.append("```")
report.append("")
report.append("## 4. Tower closure and gap-fill audit")
report.append("")
report.append("```text")
report.append(flux_quality.head(80).to_string(index=False) if len(flux_quality) else "No raw flux files with enough H/LE/NETRAD/G and QC columns were found for closure/gap-fill calculation.")
report.append("```")
report.append("")
report.append("## 5. C4 NetCDF sampling status")
report.append("")
report.append("```text")
report.append(pd.DataFrame([c4_status]).to_string(index=False))
report.append("```")
report.append("")
report.append("## 6. C4 partial-effect model")
report.append("")
report.append("```text")
report.append(c4_model.to_string(index=False) if len(c4_model) else "C4 model did not run.")
report.append("```")
report.append("")
report.append("## 7. C4 spatial block bootstrap")
report.append("")
report.append("```text")
report.append(c4_boot.to_string(index=False) if len(c4_boot) else "No C4 bootstrap rows.")
report.append("```")
report.append("")
report.append("## 8. C4 leave-one-block-out")
report.append("")
report.append("```text")
report.append(c4_blocks.head(80).to_string(index=False) if len(c4_blocks) else "No leave-one-block-out rows.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
if c4_decision.get("passes_fast_pre_specified_screen"):
    report.append("C4 fraction passes the fast pre-specified screen. The next step is to upgrade this to a full mixed/hierarchical model with ecoregion random effects, then write the ecological mechanism paper around C3/C4 composition organizing uWUE response phenotypes.")
else:
    report.append("C4 fraction does not yet pass the fast pre-specified screen, or the model could not run. Unless the full hierarchical model changes this, the safer paper is a product-identifiability / satellite WUE uncertainty paper with tower-ranked ET products and C4 as a tested but unsupported mechanism.")
report.append("")
report.append("## Files written")
report.append("")
for p in [
    "Table_PRODUCT03al_product_screening_answer_for_project.csv",
    "Table_PRODUCT03am_tower_flux_quality_closure_gapfill.csv",
    "Table_PRODUCT03an_project_clean_tower_validation_table.csv",
    "Table_PRODUCT03ao_project_clean_et_product_ranking.csv",
    "Table_PRODUCT03ap_c4_netcdf_variable_inventory.csv",
    "Table_PRODUCT03aq_c4_sampled_point_table.csv",
    "Table_PRODUCT03ar_c4_sampling_status.csv",
    "Table_PRODUCT03as_c4_partial_effect_model.csv",
    "Table_PRODUCT03at_c4_spatial_block_bootstrap.csv",
    "Table_PRODUCT03au_c4_leave_one_block_out.csv",
    "Table_PRODUCT03av_c4_test_decision.csv",
    "STAGE1B6AI_project_FINAL_LOCK_DECISION.json",
]:
    report.append(f"- `{TAB / p}`")
report.append("")
report.append("## Figures")
report.append("; ".join(fig_status) if fig_status else "No figures written.")

report_text = "\n".join(report)
(TXT / "STAGE1B6AI_project_FINAL_LOCK_REPORT.md").write_text(report_text, encoding="utf-8")

print(report_text)
print("")
print("WROTE", TXT / "STAGE1B6AI_project_FINAL_LOCK_REPORT.md")
print("WROTE", TAB / "STAGE1B6AI_project_FINAL_LOCK_DECISION.json")
print("WROTE tables to", TAB)
print("WROTE figures to", FIG)
