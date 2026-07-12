from pathlib import Path
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results" / "stage1b6bk_strict_plus_closest_mechanism_screen"
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for d in [TAB, TXT, FIG]:
    d.mkdir(parents=True, exist_ok=True)

SEED = int(os.environ.get("SEED", "123"))
N_BOOT = int(os.environ.get("N_BOOT", "300"))
MIN_N = int(os.environ.get("MIN_N", "60"))
MIN_N_PRODUCT = int(os.environ.get("MIN_N_PRODUCT", "40"))
MAX_CSV_MB = float(os.environ.get("MAX_CSV_MB", "250"))
rng = np.random.default_rng(SEED)

GPP_PRODUCTS = ["GOSIF", "MODIS", "PML"]
ET_PRODUCTS = ["GLEAM", "MODIS", "PML"]
GPP_DEP_RANK = {"GOSIF": 1, "MODIS": 2, "PML": 3}
ET_DEP_RANK = {"GLEAM": 0, "MODIS": 2, "PML": 3}


def norm(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(s):
    x = np.asarray(s, dtype=float)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return np.full(len(x), np.nan)
    return (x - np.nanmean(x)) / sd


def fnum(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def sgn(x):
    x = fnum(x)
    if not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def qvals(p, method):
    p = np.asarray(pd.to_numeric(pd.Series(p), errors="coerce"), dtype=float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], method=method)[1]
    return q


def read_csv_safe(p, nrows=None):
    try:
        return pd.read_csv(p, nrows=nrows)
    except Exception:
        return None


def size_mb(p):
    try:
        return p.stat().st_size / 1024 / 1024
    except Exception:
        return 999999


def find_lat_lon(df):
    lat = None
    lon = None
    for c in df.columns:
        n = norm(c)
        if lat is None and n in ["lat", "latitude"]:
            lat = c
        if lon is None and n in ["lon", "longitude", "lng"]:
            lon = c
    return lat, lon


def id_cols(df):
    out = []
    for c in df.columns:
        n = norm(c)
        if n in ["point_id", "grid_id", "pixel_id", "site_id", "id"] or n.endswith("_point_id"):
            out.append(c)
    return out


def choose_base_dataset():
    preferred = [
        ROOT / "results" / "trait_framework" / "trait_model_dataset.csv",
        ROOT / "results" / "stage1b6be_FULL_STRICT_lai_artifact_screen" / "tables" / "POINT_LEVEL_MODEL_DATASET.csv",
        ROOT / "results" / "stage1b6be_FULL_STRICT_lai_artifact_screen" / "tables" / "point_level_model_dataset.csv",
    ]
    rows = []
    for p in preferred:
        if p.exists():
            d = read_csv_safe(p)
            if d is not None and len(d) >= 50:
                rows.append({"path": str(p), "status": "CHOSEN_PREFERRED", "rows": len(d), "cols": len(d.columns)})
                pd.DataFrame(rows).to_csv(TAB / "BASE_DATASET_SELECTION.csv", index=False)
                return p, d
            rows.append({"path": str(p), "status": "FAILED_READ_OR_TOO_SMALL"})

    scored = []
    for root in [ROOT / "results", ROOT / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if size_mb(p) > MAX_CSV_MB:
                continue
            h = read_csv_safe(p, nrows=5)
            if h is None:
                continue
            ns = [norm(c) for c in h.columns]
            score = 0
            if any("consensus_slope_change" in n or "latent_slope_change" in n or n == "slope_change" for n in ns):
                score += 100
            if any("lai" in n for n in ns):
                score += 10
            if any("c4" in n for n in ns):
                score += 15
            if any(n in ["lat", "latitude"] for n in ns):
                score += 5
            if any(n in ["lon", "longitude"] for n in ns):
                score += 5
            if "trait" in str(p).lower():
                score += 10
            if score > 0:
                scored.append({"path": str(p), "score": score, "ncols": len(h.columns)})
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    for r in scored:
        d = read_csv_safe(Path(r["path"]))
        if d is not None and len(d) >= 50:
            r["status"] = "CHOSEN_SCANNED"
            pd.DataFrame(scored).to_csv(TAB / "BASE_DATASET_SELECTION.csv", index=False)
            return Path(r["path"]), d
    pd.DataFrame(scored).to_csv(TAB / "BASE_DATASET_SELECTION.csv", index=False)
    raise RuntimeError("No usable base dataset found.")


def aux_wanted(c):
    n = norm(c)
    wanted = [
        "c4", "c3", "grass", "natural", "crop", "cropland", "maize", "corn", "sorghum", "millet",
        "sugarcane", "sugar_cane", "managed", "irrig",
        "rooting", "root_depth", "rooting_depth", "p50", "psi50", "isohydric",
        "soil_texture", "sand", "clay", "silt",
        "lai", "fpar", "evi", "ndvi",
        "aridity", "mean_annual_temperature", "mean_annual_precipitation", "mat", "map",
        "mean_vpd", "median_vpd", "p90_vpd", "p10_vpd",
        "soil_moisture", "rootzone", "root_zone", "sm_root"
    ]
    return any(w in n for w in wanted)


def merge_auxiliary(base):
    base = base.copy()
    added = []
    skipped = []
    blat, blon = find_lat_lon(base)
    bids = id_cols(base)

    if blat and blon:
        base["_lat_round4"] = num(base[blat]).round(4)
        base["_lon_round4"] = num(base[blon]).round(4)

    csvs = []
    for root in [ROOT / "results", ROOT / "data"]:
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            if size_mb(p) <= MAX_CSV_MB:
                csvs.append(p)

    for p in csvs:
        h = read_csv_safe(p, nrows=5)
        if h is None:
            continue
        wanted_cols = [c for c in h.columns if aux_wanted(c) and c not in base.columns]
        if not wanted_cols:
            continue
        d = read_csv_safe(p)
        if d is None or len(d) < 20:
            continue
        if len(d) > max(5000, len(base) * 25):
            skipped.append({"path": str(p), "reason": "too_many_rows_for_point_merge", "rows": len(d)})
            continue

        dids = id_cols(d)
        dlat, dlon = find_lat_lon(d)
        merged = False

        for bid in bids:
            for did in dids:
                if bid in base.columns and did in d.columns:
                    tmp = d[[did] + wanted_cols].drop_duplicates(did)
                    before = set(base.columns)
                    base = base.merge(tmp, left_on=bid, right_on=did, how="left", suffixes=("", "_aux"))
                    if did != bid:
                        base = base.drop(columns=[did], errors="ignore")
                    new_cols = [c for c in base.columns if c not in before]
                    for c in new_cols:
                        added.append({"path": str(p), "merge": f"id:{bid}->{did}", "column_added": c})
                    merged = len(new_cols) > 0
                    break
            if merged:
                break

        if (not merged) and blat and blon and dlat and dlon:
            dd = d.copy()
            dd["_lat_round4"] = num(dd[dlat]).round(4)
            dd["_lon_round4"] = num(dd[dlon]).round(4)
            tmp = dd[["_lat_round4", "_lon_round4"] + wanted_cols].drop_duplicates(["_lat_round4", "_lon_round4"])
            before = set(base.columns)
            base = base.merge(tmp, on=["_lat_round4", "_lon_round4"], how="left", suffixes=("", "_aux"))
            new_cols = [c for c in base.columns if c not in before]
            for c in new_cols:
                added.append({"path": str(p), "merge": "latlon_round4", "column_added": c})
            merged = len(new_cols) > 0

        if not merged:
            skipped.append({"path": str(p), "reason": "wanted_columns_but_no_merge_key", "wanted_columns": ";".join(wanted_cols[:40])})

    base = base.drop(columns=["_lat_round4", "_lon_round4"], errors="ignore")
    pd.DataFrame(added).to_csv(TAB / "AUXILIARY_COLUMNS_MERGED.csv", index=False)
    pd.DataFrame(skipped).to_csv(TAB / "AUXILIARY_COLUMNS_SKIPPED.csv", index=False)
    return base, added


def compute_soil_pc1(df):
    if any(norm(c) == "soil_texture_pc1" for c in df.columns):
        return df, None
    sand = next((c for c in df.columns if norm(c) in ["soil_sand", "sand"]), None)
    clay = next((c for c in df.columns if norm(c) in ["soil_clay", "clay"]), None)
    silt = next((c for c in df.columns if norm(c) in ["soil_silt", "silt"]), None)
    if not (sand and clay and silt):
        return df, None
    X = pd.DataFrame({"sand": num(df[sand]), "clay": num(df[clay]), "silt": num(df[silt])}).dropna()
    if len(X) < MIN_N:
        return df, None
    Xz = (X - X.mean()) / X.std(ddof=0)
    u, s, vt = np.linalg.svd(Xz.values, full_matrices=False)
    pc = Xz.values @ vt[0]
    if np.corrcoef(pc, X["clay"].values)[0, 1] < 0:
        pc = -pc
    out = np.full(len(df), np.nan)
    out[X.index] = pc
    df["computed_soil_texture_pc1"] = out
    return df, "computed_soil_texture_pc1"


def choose_outcome(df):
    for target in ["consensus_slope_change_all", "latent_slope_change", "mean_slope_change", "slope_change"]:
        for c in df.columns:
            if norm(c) == target:
                s = num(df[c])
                if s.notna().sum() >= MIN_N and s.nunique(dropna=True) > 3:
                    return c
    candidates = []
    for c in df.columns:
        n = norm(c)
        if ("slope_change" in n or "latent" in n) and not any(k in n for k in ["gosif", "gleam", "modis", "pml", "fraction", "post_slope", "pre_slope"]):
            s = num(df[c])
            if s.notna().sum() >= MIN_N and s.nunique(dropna=True) > 3:
                candidates.append((s.notna().sum(), c))
    if not candidates:
        raise RuntimeError("No clean primary outcome found.")
    return sorted(candidates, reverse=True)[0][1]


def exact_product_outcomes(df):
    rows = []
    for c in df.columns:
        n = norm(c)
        if "slope_change" not in n:
            continue
        if any(bad in n for bad in [
            "consensus", "negative_slope_fraction", "positive_slope_fraction", "post_slope", "pre_slope",
            "satbreak", "stability", "partial_effect", "sensitivity", "interaction"
        ]):
            continue
        s = num(df[c])
        if s.notna().sum() < MIN_N_PRODUCT or s.nunique(dropna=True) <= 3:
            continue
        up = c.upper()
        for g in GPP_PRODUCTS:
            for e in ET_PRODUCTS:
                if g in up and e in up:
                    if g == "MODIS" and e == "MODIS" and up.count("MODIS") < 2:
                        continue
                    rows.append({
                        "outcome_col": c,
                        "gpp_product": g,
                        "et_product": e,
                        "gpp_dependency_rank": GPP_DEP_RANK[g],
                        "et_dependency_rank": ET_DEP_RANK[e],
                        "combo_dependency_rank_sum": GPP_DEP_RANK[g] + ET_DEP_RANK[e],
                        "is_gosif_gleam": bool(g == "GOSIF" and e == "GLEAM"),
                    })
    out = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()
    out.to_csv(TAB / "EXACT_PRODUCT_SLOPE_OUTCOME_COLUMNS.csv", index=False)
    return out.to_dict("records")


CONTROL_PRIORITY = {
    "aridity": ["aridity", "z_aridity"],
    "temperature": ["mean_annual_temperature", "z_mean_annual_temperature", "mean_temperature", "z_mean_temperature", "mat"],
    "precipitation": ["mean_annual_precipitation", "z_mean_annual_precipitation", "mean_precipitation", "z_mean_precipitation", "map"],
    "soil_texture": ["computed_soil_texture_pc1", "soil_texture_pc1", "z_soil_texture_pc1", "soil_silt", "z_soil_silt", "soil_clay", "soil_sand"],
    "lai_productivity": ["growing_season_mean_lai", "z_growing_season_mean_lai", "median_lai", "mean_lai"],
    "baseline_vpd": ["mean_vpd", "z_mean_vpd", "median_vpd", "z_median_vpd", "p90_vpd", "p10_vpd"],
    "baseline_soil_moisture": ["mean_soil_moisture", "z_mean_soil_moisture", "median_soil_moisture", "z_median_soil_moisture"],
}


def pick_control(df, family):
    for target in CONTROL_PRIORITY[family]:
        for c in df.columns:
            if norm(c) == norm(target):
                s = num(df[c])
                if s.notna().sum() >= MIN_N and s.nunique(dropna=True) > 2:
                    return c
    return None


def feature_family(c):
    n = norm(c)
    if "c4" in n and not any(k in n for k in ["crop", "maize", "corn", "sorghum", "millet", "sugar"]):
        return "c4_photosynthetic_pathway"
    if "lai" in n or "fpar" in n or "evi" in n or "ndvi" in n:
        return "canopy_structure"
    if "root" in n:
        return "rooting"
    if "p50" in n or "psi50" in n or "isohyd" in n:
        return "hydraulic_trait"
    if "soil_texture" in n or "sand" in n or "clay" in n or "silt" in n:
        return "soil_texture"
    if "aridity" in n:
        return "aridity"
    if "temp" in n or "temperature" in n or n == "mat":
        return "temperature"
    if "precip" in n or n == "map":
        return "precipitation"
    if "vpd" in n:
        return "baseline_vpd"
    if "soil_moisture" in n or "rootzone" in n or "root_zone" in n or "sm_root" in n:
        return "baseline_soil_moisture"
    return "other"


def controls_for(df, feature, moderator=None):
    fam = feature_family(feature)
    mfam = feature_family(moderator) if moderator else None
    controls = []
    omitted = []
    for cfam in CONTROL_PRIORITY:
        c = pick_control(df, cfam)
        if c is None:
            omitted.append(f"{cfam}:not_found")
        elif cfam == fam or cfam == mfam:
            omitted.append(f"{cfam}:omitted_same_family_as_focal_or_moderator")
        else:
            controls.append(c)
    lat, lon = find_lat_lon(df)
    if lat and lat not in [feature, moderator]:
        controls.append(lat)
    else:
        omitted.append("lat:not_found_or_focal")
    if lon and lon not in [feature, moderator]:
        controls.append(lon)
    else:
        omitted.append("lon:not_found_or_focal")
    seen = set()
    out = []
    for c in controls:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out, omitted


FORBIDDEN = [
    "consensus", "slope", "response", "break", "satbreak", "stability", "positive_slope", "negative_slope",
    "post_slope", "pre_slope", "partial_effect", "surface_sensitivity", "sensitivity", "agreement",
    "product", "tower", "uwue", "wue", "gpp", "raw_vs_co2", "co2_stability",
    "pml", "gosif", "gleam", "modis", "independent",
    "sample_distance", "missing", "has_c4", "region_", "_region"
]
ALLOWED = [
    "c4", "c3", "lai", "fpar", "evi", "ndvi", "root", "p50", "psi50", "isohyd",
    "soil_texture", "sand", "clay", "silt", "aridity", "temperature", "temp", "mat",
    "precip", "map", "vpd", "soil_moisture", "rootzone", "root_zone", "sm_root"
]


def candidate_features(df, y_col, exacts):
    exact_cols = set([r["outcome_col"] for r in exacts])
    rows = []
    keep_pre = []
    for c in df.columns:
        n = norm(c)
        reason = "eligible"
        ok = True
        if c == y_col or c in exact_cols:
            ok, reason = False, "outcome_column"
        elif any(k in n for k in FORBIDDEN):
            ok, reason = False, "response_or_product_derived_name"
        elif not any(k in n for k in ALLOWED):
            ok, reason = False, "not_ecological_trait_condition_name"
        elif any(k in n for k in ["flag", "mask", "valid", "qc", "quality", "id", "year", "month", "day"]):
            if not ("c4" in n and "crop" not in n):
                ok, reason = False, "metadata_flag_or_time_column"
        s = num(df[c])
        nmiss = int(s.notna().sum())
        nunq = int(s.nunique(dropna=True))
        if ok and nmiss < MIN_N:
            ok, reason = False, f"too_few_nonmissing_{nmiss}"
        if ok and nunq < 2:
            ok, reason = False, f"too_few_unique_{nunq}"
        if ok and np.nanstd(s) == 0:
            ok, reason = False, "zero_variance"
        rows.append({"column": c, "eligible_pre_dedup": ok, "reason": reason, "family": feature_family(c), "n_nonmissing": nmiss, "n_unique": nunq})
        if ok:
            keep_pre.append(c)

    chosen_by_base = {}
    for c in keep_pre:
        base = re.sub(r"^z_", "", norm(c))
        base = re.sub(r"_z$", "", base)
        old = chosen_by_base.get(base)
        if old is None:
            chosen_by_base[base] = c
        elif norm(old).startswith("z_") and not norm(c).startswith("z_"):
            chosen_by_base[base] = c

    keep = set(chosen_by_base.values())
    for r in rows:
        if r["eligible_pre_dedup"] and r["column"] not in keep:
            r["eligible_final"] = False
            r["reason"] = "duplicate_scaled_version_removed"
        else:
            r["eligible_final"] = bool(r["column"] in keep)
    pd.DataFrame(rows).to_csv(TAB / "STRICT_FEATURE_CANDIDATE_AUDIT.csv", index=False)
    return sorted(keep, key=lambda c: (feature_family(c), norm(c)))


def construct_mechanisms(df, candidates):
    moderators = []
    for fam in ["temperature", "baseline_vpd", "baseline_soil_moisture", "aridity", "precipitation"]:
        c = pick_control(df, fam)
        if c:
            moderators.append((fam, c))
    mechs = []
    for c in candidates:
        x = num(df[c])
        vals = pd.Series(zscore(x), index=df.index).dropna()
        nunq = x.nunique(dropna=True)
        mechs.append({"id": f"linear__{norm(c)}", "feature": c, "type": "linear", "moderator": None, "params": {}})
        if nunq > 3:
            mechs.append({"id": f"quadratic__{norm(c)}", "feature": c, "type": "quadratic", "moderator": None, "params": {}})
            for q in [0.10, 0.25, 0.75, 0.90]:
                cut = float(vals.quantile(q))
                if q < 0.5:
                    mechs.append({"id": f"low_tail_q{int(q*100)}__{norm(c)}", "feature": c, "type": "low_tail", "moderator": None, "params": {"q": q, "cut": cut}})
                    mechs.append({"id": f"hinge_low_q{int(q*100)}__{norm(c)}", "feature": c, "type": "hinge_low", "moderator": None, "params": {"q": q, "cut": cut}})
                else:
                    mechs.append({"id": f"high_tail_q{int(q*100)}__{norm(c)}", "feature": c, "type": "high_tail", "moderator": None, "params": {"q": q, "cut": cut}})
                    mechs.append({"id": f"hinge_high_q{int(q*100)}__{norm(c)}", "feature": c, "type": "hinge_high", "moderator": None, "params": {"q": q, "cut": cut}})
            for mfam, m in moderators:
                if m != c and feature_family(m) != feature_family(c):
                    mechs.append({"id": f"interaction_with_{norm(m)}__{norm(c)}", "feature": c, "type": "interaction", "moderator": m, "params": {"moderator_family": mfam}})
    pd.DataFrame(mechs).to_csv(TAB / "STRICT_MECHANISM_LIBRARY.csv", index=False)
    return mechs


def design(df, y_col, mech, subset=None):
    mask = pd.Series(True, index=df.index) if subset is None else pd.Series(subset, index=df.index).fillna(False).astype(bool)
    x = pd.Series(zscore(num(df[mech["feature"]])), index=df.index)
    full = {}
    red = {}
    t = mech["type"]
    if t == "linear":
        full["focal"] = x
    elif t == "quadratic":
        full["base_linear"] = x
        red["base_linear"] = x
        full["focal"] = x ** 2
    elif t == "low_tail":
        full["focal"] = (x <= mech["params"]["cut"]).astype(float)
    elif t == "high_tail":
        full["focal"] = (x >= mech["params"]["cut"]).astype(float)
    elif t == "hinge_low":
        full["base_linear"] = x
        red["base_linear"] = x
        full["focal"] = np.maximum(0, mech["params"]["cut"] - x)
    elif t == "hinge_high":
        full["base_linear"] = x
        red["base_linear"] = x
        full["focal"] = np.maximum(0, x - mech["params"]["cut"])
    elif t == "interaction":
        m = pd.Series(zscore(num(df[mech["moderator"]])), index=df.index)
        full["base_linear"] = x
        full["moderator_linear"] = m
        red["base_linear"] = x
        red["moderator_linear"] = m
        full["focal"] = x * m
    controls, omitted = controls_for(df, mech["feature"], mech.get("moderator"))
    for c in controls:
        nm = "control__" + norm(c)
        v = pd.Series(zscore(num(df[c])), index=df.index)
        full[nm] = v
        red[nm] = v
    d = pd.DataFrame({"y": num(df[y_col])}, index=df.index)
    for k, v in full.items():
        d[k] = v
    for k, v in red.items():
        if k not in d.columns:
            d[k] = v
    d = d.loc[mask].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d):
        d["y"] = zscore(d["y"])
        for c in d.columns:
            if c != "y":
                d[c] = zscore(d[c])
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
    return d, list(full.keys()), list(red.keys()), controls, omitted


def fit_design(d, full_cols, red_cols, min_n):
    if len(d) < min_n:
        return {"status": "TOO_FEW_COMPLETE_CASES", "n": len(d)}
    if "focal" not in full_cols:
        return {"status": "NO_FOCAL", "n": len(d)}
    if d["focal"].nunique(dropna=True) <= 1:
        return {"status": "FOCAL_ZERO_VARIANCE", "n": len(d)}
    try:
        Xf = sm.add_constant(d[full_cols], has_constant="add")
        Xr = sm.add_constant(d[red_cols], has_constant="add") if red_cols else sm.add_constant(pd.DataFrame(index=d.index), has_constant="add")
        mf = sm.OLS(d["y"], Xf).fit()
        mr = sm.OLS(d["y"], Xr).fit()
        rob = mf.get_robustcov_results(cov_type="HC3")
        idx = list(mf.model.exog_names).index("focal")
        ci = rob.conf_int()[idx]
        try:
            nested_p = float(mf.compare_f_test(mr)[1])
        except Exception:
            nested_p = np.nan
        vif = np.nan
        try:
            if len(full_cols) >= 2:
                Xv = d[full_cols].loc[:, d[full_cols].std() > 0]
                if "focal" in Xv.columns and Xv.shape[1] >= 2:
                    vif = float(variance_inflation_factor(Xv.values, list(Xv.columns).index("focal")))
        except Exception:
            pass
        return {
            "status": "FIT_OK",
            "n": int(len(d)),
            "coef": float(mf.params["focal"]),
            "se_hc3": float(rob.bse[idx]),
            "p": float(rob.pvalues[idx]),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "ci_excludes_zero": bool((ci[0] > 0 and ci[1] > 0) or (ci[0] < 0 and ci[1] < 0)),
            "full_r2": float(mf.rsquared),
            "reduced_r2": float(mr.rsquared),
            "delta_r2": float(mf.rsquared - mr.rsquared),
            "full_aic": float(mf.aic),
            "reduced_aic": float(mr.aic),
            "delta_aic_full_minus_reduced": float(mf.aic - mr.aic),
            "nested_f_p": nested_p,
            "focal_vif": vif,
        }
    except Exception as e:
        return {"status": f"FIT_FAILED:{type(e).__name__}:{e}", "n": int(len(d))}


def fit_mech(df, y_col, mech, subset=None, min_n=MIN_N):
    d, fc, rc, controls, omitted = design(df, y_col, mech, subset=subset)
    res = fit_design(d, fc, rc, min_n)
    res.update({
        "mechanism_id": mech["id"],
        "feature": mech["feature"],
        "feature_family": feature_family(mech["feature"]),
        "mechanism_type": mech["type"],
        "moderator": mech.get("moderator"),
        "controls_used": ";".join(controls),
        "controls_omitted": ";".join(omitted),
    })
    return res



def truthy_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|natural|grass|clean|no_crop", regex=True, na=False)


def falsy_or_bad_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|crop|managed|irrig|maize|corn|sorghum|millet|sugar", regex=True, na=False)



def truthy_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|natural|grass|clean|no_crop", regex=True, na=False)


def falsy_or_bad_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|crop|managed|irrig|maize|corn|sorghum|millet|sugar", regex=True, na=False)



def truthy_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|natural|grass|clean|no_crop", regex=True, na=False)


def falsy_or_bad_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return num(s).fillna(0) > 0
    return s.astype(str).str.lower().str.contains("true|yes|1|crop|managed|irrig|maize|corn|sorghum|millet|sugar", regex=True, na=False)


def clean_landcover_mask(df):
    total = len(df)
    audit_rows = []

    def usable_mask_column(c):
        n = norm(c)
        if n.endswith("_z"):
            return False
        if n.startswith("n_"):
            return False
        if "count" in n or "_n_" in n:
            return False
        if "num_" in n:
            return False
        return True

    direct_clean_cols = []
    natural_cols = []
    bad_cols = []

    for c in df.columns:
        n = norm(c)
        if not usable_mask_column(c):
            continue

        is_direct_clean = (
            "analysis_natural_grassland_like_no_crop" in n
            or "natural_grassland_like_no_crop" in n
            or "natural_grassland_like_no_crop_points" in n
            or "no_crop_flagged_points" in n
        )

        is_natural = (
            "natural_grassland_like" in n
            or "any_natural_grassland_indicator" in n
            or n in ["natural_grassland", "natural_grassland_indicator"]
        )

        is_bad = (
            "any_cropland_managed_irrigation_flag" in n
            or "cropland_managed_irrigation_flag" in n
            or "irrigation_or_agri_mask" in n
            or "crop_flag" in n
            or "cropland" in n
            or "managed" in n
            or "irrig" in n
            or "maize" in n
            or "corn" in n
            or "sorghum" in n
            or "millet" in n
            or "sugarcane" in n
            or "sugar_cane" in n
            or "c4_crop" in n
        )

        if is_direct_clean:
            direct_clean_cols.append(c)
        elif is_natural:
            natural_cols.append(c)
        elif is_bad:
            bad_cols.append(c)

    candidates = []

    for c in direct_clean_cols:
        m = truthy_series(df[c])
        nclean = int(m.sum())
        plausible = 0 < nclean < total
        score = 1000
        nn = norm(c)
        if "analysis_natural_grassland_like_no_crop" in nn:
            score += 500
        if "natural_grassland_like_no_crop" in nn:
            score += 400
        if "no_crop_flagged_points" in nn:
            score += 300
        if plausible:
            score += 200
        score += min(nclean, total - nclean if nclean < total else 0)
        candidates.append({
            "source": "direct_clean_column",
            "column": c,
            "n_clean": nclean,
            "n_total": total,
            "plausible": plausible,
            "score": score,
            "mask": m,
        })

    if natural_cols or bad_cols:
        natural_or = pd.Series(False, index=df.index)
        for c in natural_cols:
            natural_or |= truthy_series(df[c])

        bad_or = pd.Series(False, index=df.index)
        for c in bad_cols:
            bad_or |= falsy_or_bad_series(df[c])

        if natural_cols and bad_cols:
            combo = natural_or & (~bad_or)
            label = "OR_natural_AND_NOT_OR_bad"
        elif natural_cols:
            combo = natural_or
            label = "OR_natural_only"
        else:
            combo = ~bad_or
            label = "NOT_OR_bad_only"

        nclean = int(combo.sum())
        plausible = 0 < nclean < total
        score = 700 + (200 if plausible else 0) + min(nclean, total - nclean if nclean < total else 0)
        candidates.append({
            "source": label,
            "column": ";".join(natural_cols + bad_cols),
            "n_clean": nclean,
            "n_total": total,
            "plausible": plausible,
            "score": score,
            "mask": combo,
        })

    audit = []
    for c in direct_clean_cols:
        m = truthy_series(df[c])
        audit.append({"role": "direct_clean_candidate", "column": c, "n_true": int(m.sum()), "n_total": total})
    for c in natural_cols:
        m = truthy_series(df[c])
        audit.append({"role": "natural_candidate", "column": c, "n_true": int(m.sum()), "n_total": total})
    for c in bad_cols:
        m = falsy_or_bad_series(df[c])
        audit.append({"role": "bad_exclusion_candidate", "column": c, "n_true_bad": int(m.sum()), "n_total": total})

    if audit:
        pd.DataFrame(audit).to_csv(TAB / "CLEAN_MASK_COLUMN_CANDIDATE_AUDIT.csv", index=False)

    if not candidates:
        info = {
            "status": "NO_CLEAN_MASK_COLUMNS_FOUND",
            "n_clean": total,
            "n_total": total,
            "used_columns": "",
            "decision": "no clean/crop/land-cover columns found; clean gate not interpretable",
        }
        pd.DataFrame([info]).to_csv(TAB / "CLEAN_LANDCOVER_C4_CROP_MASK_AUDIT.csv", index=False)
        return pd.Series(True, index=df.index), info

    cand_df = pd.DataFrame([{k: v for k, v in r.items() if k != "mask"} for r in candidates])
    cand_df = cand_df.sort_values(["plausible", "score"], ascending=[False, False])
    cand_df.to_csv(TAB / "CLEAN_MASK_DECISION_CANDIDATES.csv", index=False)

    chosen = candidates[int(cand_df.index[0])]
    mask = chosen["mask"].fillna(False).astype(bool)
    nclean = int(mask.sum())

    if nclean == 0 or nclean == total:
        status = "CLEAN_MASK_FOUND_BUT_IMPLAUSIBLE"
    else:
        status = "CLEAN_MASK_INFERRED"

    info = {
        "status": status,
        "n_clean": nclean,
        "n_total": total,
        "used_columns": chosen["column"],
        "decision": chosen["source"],
    }
    pd.DataFrame([info]).to_csv(TAB / "CLEAN_LANDCOVER_C4_CROP_MASK_AUDIT.csv", index=False)
    return mask, info


def bootstrap_loo(df, y_col, mech):
    d, fc, rc, controls, omitted = design(df, y_col, mech)
    if len(d) < MIN_N:
        return {"mechanism_id": mech["id"], "boot_status": "TOO_FEW_CASES", "n": len(d)}
    coefs = []
    n = len(d)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bd = d.iloc[idx]
        r = fit_design(bd, fc, rc, min_n=max(30, int(MIN_N * 0.5)))
        if r.get("status") == "FIT_OK":
            coefs.append(r["coef"])
    if len(coefs) >= 30:
        lo, hi = np.quantile(coefs, [0.025, 0.975])
        boot_ok = bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
    else:
        lo = hi = np.nan
        boot_ok = False
    loo = []
    if n <= 250:
        for i in range(n):
            ld = d.drop(d.index[i])
            r = fit_design(ld, fc, rc, min_n=max(30, MIN_N - 1))
            if r.get("status") == "FIT_OK":
                loo.append(r["coef"])
    if loo:
        medsign = sgn(np.median(loo))
        stability = float(np.mean([sgn(v) == medsign for v in loo])) if medsign != 0 else np.nan
    else:
        stability = np.nan
    return {
        "mechanism_id": mech["id"],
        "boot_status": "FIT_OK" if coefs else "NO_VALID_BOOTSTRAPS",
        "n_boot_valid": len(coefs),
        "boot_ci_low": float(lo),
        "boot_ci_high": float(hi),
        "boot_ci_excludes_zero": boot_ok,
        "loo_sign_stability": stability,
    }


def product_tests(df, exacts, mechs, clean_mask):
    rows = []
    if not exacts:
        return pd.DataFrame(), pd.DataFrame()
    for i, mech in enumerate(mechs, 1):
        if i % 10 == 0:
            print(f"  product gate {i}/{len(mechs)}: {mech['id']}", flush=True)
        for mode, subset in [("all", None), ("clean", clean_mask)]:
            for eo in exacts:
                r = fit_mech(df, eo["outcome_col"], mech, subset=subset, min_n=MIN_N_PRODUCT)
                r.update({k: v for k, v in eo.items() if k != "outcome_col"})
                r["product_outcome_col"] = eo["outcome_col"]
                r["sample_mode"] = mode
                rows.append(r)
    prod = pd.DataFrame(rows)
    if prod.empty:
        return prod, pd.DataFrame()
    if "p" not in prod.columns:
        prod["p"] = np.nan
    prod["product_bh_q"] = np.nan
    for (mid, mode), idx in prod.groupby(["mechanism_id", "sample_mode"]).groups.items():
        prod.loc[idx, "product_bh_q"] = qvals(prod.loc[idx, "p"], "fdr_bh")
    summ = []
    for (mid, mode), g in prod.groupby(["mechanism_id", "sample_mode"]):
        f = g[g["status"] == "FIT_OK"].copy()
        if f.empty:
            summ.append({"mechanism_id": mid, "sample_mode": mode, "product_gate": "FAIL_NO_FITS"})
            continue
        expected = sgn(np.nanmedian(f["coef"]))
        least = f[f["is_gosif_gleam"] == True].sort_values("p").head(1)
        gg_pass = False
        gg_coef = gg_p = gg_q = np.nan
        if len(least):
            lr = least.iloc[0]
            gg_coef = fnum(lr["coef"])
            gg_p = fnum(lr["p"])
            gg_q = fnum(lr["product_bh_q"])
            gg_pass = bool(
                gg_p < 0.05 and gg_q < 0.10 and bool(lr["ci_excludes_zero"])
                and sgn(gg_coef) == expected and fnum(lr["delta_aic_full_minus_reduced"]) < 0
            )
        same = f[np.sign(f["coef"]) == expected]
        opp_sig = f[(np.sign(f["coef"]) == -expected) & (f["p"] < 0.05)]
        sign_consistency = len(same) / len(f) if expected != 0 else np.nan
        try:
            dep_rho = float(stats.spearmanr(f["combo_dependency_rank_sum"], np.abs(f["coef"]), nan_policy="omit").correlation)
        except Exception:
            dep_rho = np.nan
        dep_flag = bool(np.isfinite(dep_rho) and dep_rho > 0.50 and not gg_pass)
        gate = bool(gg_pass and sign_consistency >= 0.75 and len(opp_sig) == 0 and not dep_flag)
        summ.append({
            "mechanism_id": mid,
            "sample_mode": mode,
            "product_gate": "PASS" if gate else "FAIL",
            "n_product_fit_ok": len(f),
            "expected_sign": expected,
            "sign_consistency": sign_consistency,
            "gosif_gleam_pass": gg_pass,
            "gosif_gleam_coef": gg_coef,
            "gosif_gleam_p": gg_p,
            "gosif_gleam_bh_q": gg_q,
            "n_opposite_significant_products": len(opp_sig),
            "dependency_rank_abscoef_spearman": dep_rho,
            "dependency_flag": dep_flag,
        })
    return prod, pd.DataFrame(summ)


def boolish(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).lower() in ["true", "1", "pass", "yes"]


def fail_text(r, gate):
    if gate == "primary":
        reasons = []
        if r.get("status") != "FIT_OK":
            reasons.append(f"status={r.get('status')}")
        if not fnum(r.get("p")) < 0.05:
            reasons.append(f"p={r.get('p')}")
        if not fnum(r.get("bh_q")) < 0.05:
            reasons.append(f"BH q={r.get('bh_q')}")
        if not boolish(r.get("ci_excludes_zero")):
            reasons.append("CI includes zero")
        if not fnum(r.get("delta_r2")) > 0:
            reasons.append(f"ΔR²={r.get('delta_r2')}")
        if not fnum(r.get("delta_aic_full_minus_reduced")) < 0:
            reasons.append(f"ΔAIC={r.get('delta_aic_full_minus_reduced')}")
        if not fnum(r.get("nested_f_p")) < 0.05:
            reasons.append(f"nested p={r.get('nested_f_p')}")
        return "; ".join(reasons) if reasons else "PASS"
    if gate == "bootstrap":
        reasons = []
        if not boolish(r.get("boot_ci_excludes_zero")):
            reasons.append("bootstrap CI includes zero/not available")
        if not fnum(r.get("loo_sign_stability")) >= 0.80:
            reasons.append(f"LOO stability={r.get('loo_sign_stability')}")
        return "; ".join(reasons) if reasons else "PASS"
    if gate == "clean":
        reasons = []
        if str(r.get("clean_status")) != "FIT_OK":
            reasons.append(f"clean status={r.get('clean_status')}")
        if not fnum(r.get("clean_p")) < 0.05:
            reasons.append(f"clean p={r.get('clean_p')}")
        if not boolish(r.get("clean_ci_excludes_zero")):
            reasons.append("clean CI includes zero")
        if not fnum(r.get("clean_delta_aic")) < 0:
            reasons.append(f"clean ΔAIC={r.get('clean_delta_aic')}")
        if np.isfinite(fnum(r.get("coef"))) and np.isfinite(fnum(r.get("clean_coef"))):
            if np.sign(fnum(r.get("coef"))) != np.sign(fnum(r.get("clean_coef"))):
                reasons.append("clean sign differs")
        return "; ".join(reasons) if reasons else "PASS"
    if gate in ["product_all", "product_clean"]:
        prefix = gate
        if str(r.get(f"{prefix}__product_gate")) == "PASS":
            return "PASS"
        reasons = [f"gate={r.get(f'{prefix}__product_gate')}"]
        if not boolish(r.get(f"{prefix}__gosif_gleam_pass")):
            reasons.append(f"GOSIF×GLEAM fail coef={r.get(f'{prefix}__gosif_gleam_coef')} p={r.get(f'{prefix}__gosif_gleam_p')} q={r.get(f'{prefix}__gosif_gleam_bh_q')}")
        if not fnum(r.get(f"{prefix}__sign_consistency")) >= 0.75:
            reasons.append(f"sign consistency={r.get(f'{prefix}__sign_consistency')}")
        if fnum(r.get(f"{prefix}__n_opposite_significant_products")) > 0:
            reasons.append(f"opposite significant products={r.get(f'{prefix}__n_opposite_significant_products')}")
        if boolish(r.get(f"{prefix}__dependency_flag")):
            reasons.append("stronger with more algorithmically entangled products")
        return "; ".join(reasons)
    return "needs independent tower-derived same-feature/uWUE-response test"


def story_category(feature):
    f = str(feature).lower()
    if "c4" in f:
        return "C4/photosynthetic pathway"
    if any(k in f for k in ["lai", "fpar", "evi", "ndvi"]):
        return "canopy structure/productivity"
    if "vpd" in f:
        return "VPD stress"
    if "soil_moisture" in f or "rootzone" in f:
        return "soil moisture"
    if "temp" in f or "temperature" in f:
        return "temperature gradient"
    if "precip" in f or "aridity" in f:
        return "hydroclimate"
    if any(k in f for k in ["sand", "silt", "clay", "soil_texture"]):
        return "soil texture"
    if "root" in f:
        return "rooting"
    if any(k in f for k in ["p50", "psi50", "isohyd"]):
        return "hydraulic trait"
    return "other ecological"


def main():
    print("Stage1B6BK strict + closest mechanism screen", flush=True)
    base_path, df = choose_base_dataset()
    cols_before = len(df.columns)
    df, aux_added = merge_auxiliary(df)
    df, pc1 = compute_soil_pc1(df)
    y_col = choose_outcome(df)
    exacts = exact_product_outcomes(df)
    candidates = candidate_features(df, y_col, exacts)
    mechs = construct_mechanisms(df, candidates)
    cmask, cinfo = clean_landcover_mask(df)

    print(f"Base dataset: {base_path}", flush=True)
    print(f"Rows: {len(df)} Columns before/after aux: {cols_before}/{len(df.columns)}", flush=True)
    print(f"Outcome: {y_col}", flush=True)
    print(f"Strict ecological features: {len(candidates)}", flush=True)
    print(f"Mechanisms: {len(mechs)}", flush=True)
    print(f"Exact product slope outcomes: {len(exacts)}", flush=True)
    print(f"Clean mask: {cinfo['status']} n={cinfo['n_clean']}/{cinfo['n_total']}", flush=True)

    rows = []
    for i, m in enumerate(mechs, 1):
        if i % 100 == 0:
            print(f"  primary model {i}/{len(mechs)}", flush=True)
        rows.append(fit_mech(df, y_col, m))
    primary = pd.DataFrame(rows)
    primary["bh_q"] = qvals(primary["p"], "fdr_bh")
    primary["by_q"] = qvals(primary["p"], "fdr_by")
    primary["holm_p"] = qvals(primary["p"], "holm")
    primary["primary_pass"] = (
        (primary["status"] == "FIT_OK")
        & (primary["p"] < 0.05)
        & (primary["bh_q"] < 0.05)
        & (primary["ci_excludes_zero"] == True)
        & (primary["delta_r2"] > 0)
        & (primary["delta_aic_full_minus_reduced"] < 0)
        & (primary["nested_f_p"] < 0.05)
    )
    primary = primary.sort_values(["primary_pass", "bh_q", "p"], ascending=[False, True, True])
    primary.to_csv(TAB / "PRIMARY_STRICT_ECOLOGICAL_MECHANISM_SCREEN.csv", index=False)

    survivor_ids = primary.loc[
        (primary["status"] == "FIT_OK")
        & (primary["p"] < 0.05)
        & (primary["bh_q"] < 0.10)
        & (primary["ci_excludes_zero"] == True)
        & (primary["delta_aic_full_minus_reduced"] < 0),
        "mechanism_id"
    ].tolist()
    survivor_mechs = [m for m in mechs if m["id"] in set(survivor_ids)]

    boot_rows = []
    for i, m in enumerate(survivor_mechs, 1):
        print(f"  bootstrap/LOO {i}/{len(survivor_mechs)}: {m['id']}", flush=True)
        boot_rows.append(bootstrap_loo(df, y_col, m))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TAB / "BOOTSTRAP_LOO_SURVIVORS.csv", index=False)

    clean_rows = []
    for m in survivor_mechs:
        clean_rows.append(fit_mech(df, y_col, m, subset=cmask, min_n=MIN_N))
    clean = pd.DataFrame(clean_rows)
    if not clean.empty:
        needed_clean_cols = [
            "mechanism_id", "status", "n", "coef", "p", "ci_excludes_zero",
            "delta_aic_full_minus_reduced"
        ]
        for cc in needed_clean_cols:
            if cc not in clean.columns:
                clean[cc] = np.nan
        clean["clean_bh_q"] = qvals(clean["p"], "fdr_bh")
    clean.to_csv(TAB / "CLEAN_LANDCOVER_SENSITIVITY_SURVIVORS.csv", index=False)

    prod, psum = product_tests(df, exacts, survivor_mechs, cmask)
    prod.to_csv(TAB / "EXACT_PRODUCT_SLOPE_TESTS_SURVIVORS.csv", index=False)
    psum.to_csv(TAB / "PRODUCT_DEPENDENCY_GATE_SUMMARY.csv", index=False)

    c4 = primary[primary["feature"].astype(str).map(lambda x: "c4" in norm(x))].copy()
    c4.to_csv(TAB / "project_REQUIRED_C4_FULL_CONTROL_CHECK.csv", index=False)

    gates = primary[primary["mechanism_id"].isin(survivor_ids)].copy()
    if not boot.empty:
        gates = gates.merge(boot, on="mechanism_id", how="left")
    else:
        gates["boot_ci_excludes_zero"] = False
        gates["loo_sign_stability"] = np.nan

    if not clean.empty:
        csmall = clean[["mechanism_id", "status", "n", "coef", "p", "ci_excludes_zero", "delta_aic_full_minus_reduced", "clean_bh_q"]].copy()
        csmall = csmall.rename(columns={
            "status": "clean_status",
            "n": "clean_n",
            "coef": "clean_coef",
            "p": "clean_p",
            "ci_excludes_zero": "clean_ci_excludes_zero",
            "delta_aic_full_minus_reduced": "clean_delta_aic",
        })
        gates = gates.merge(csmall, on="mechanism_id", how="left")
    else:
        gates["clean_status"] = "NOT_RUN"

    if not psum.empty:
        pall = psum[psum["sample_mode"] == "all"].add_prefix("product_all__").rename(columns={"product_all__mechanism_id": "mechanism_id"})
        pcln = psum[psum["sample_mode"] == "clean"].add_prefix("product_clean__").rename(columns={"product_clean__mechanism_id": "mechanism_id"})
        gates = gates.merge(pall, on="mechanism_id", how="left")
        gates = gates.merge(pcln, on="mechanism_id", how="left")
    else:
        gates["product_all__product_gate"] = "NOT_RUN_NO_EXACT_PRODUCT_SLOPE_OUTCOMES"
        gates["product_clean__product_gate"] = "NOT_RUN_NO_EXACT_PRODUCT_SLOPE_OUTCOMES"

    gates["bootstrap_gate"] = (gates["boot_ci_excludes_zero"] == True) & (pd.to_numeric(gates["loo_sign_stability"], errors="coerce").fillna(0) >= 0.80)
    gates["clean_gate"] = (
        (cinfo["status"] == "CLEAN_MASK_INFERRED")
        & (gates["clean_status"] == "FIT_OK")
        & (gates["clean_p"] < 0.05)
        & (gates["clean_ci_excludes_zero"] == True)
        & (np.sign(gates["clean_coef"]) == np.sign(gates["coef"]))
        & (gates["clean_delta_aic"] < 0)
    )
    gates["product_all_gate"] = gates["product_all__product_gate"].astype(str).eq("PASS")
    gates["product_clean_gate"] = gates["product_clean__product_gate"].astype(str).eq("PASS")
    gates["tower_strict_gate"] = False

    gates["FULL_STRICT_STRICT_PASS"] = gates["primary_pass"] & gates["bootstrap_gate"] & gates["clean_gate"] & gates["product_all_gate"] & gates["product_clean_gate"] & gates["tower_strict_gate"]
    gates["SATELLITE_STRICT_PASS_NO_TOWER"] = gates["primary_pass"] & gates["bootstrap_gate"] & gates["clean_gate"] & gates["product_all_gate"] & gates["product_clean_gate"]

    non_tower = ["primary_pass", "bootstrap_gate", "clean_gate", "product_all_gate", "product_clean_gate"]
    full = non_tower + ["tower_strict_gate"]
    gates["non_tower_gate_score_0_to_5"] = gates[non_tower].sum(axis=1)
    gates["full_gate_score_0_to_6"] = gates[full].sum(axis=1)
    gates["story_category"] = gates["feature"].map(story_category)
    gates["primary_failure_reason"] = gates.apply(lambda r: fail_text(r, "primary"), axis=1)
    gates["bootstrap_failure_reason"] = gates.apply(lambda r: fail_text(r, "bootstrap"), axis=1)
    gates["clean_failure_reason"] = gates.apply(lambda r: fail_text(r, "clean"), axis=1)
    gates["product_all_failure_reason"] = gates.apply(lambda r: fail_text(r, "product_all"), axis=1)
    gates["product_clean_failure_reason"] = gates.apply(lambda r: fail_text(r, "product_clean"), axis=1)
    gates["tower_failure_reason"] = gates.apply(lambda r: fail_text(r, "tower"), axis=1)

    def tier(r):
        if bool(r["FULL_STRICT_STRICT_PASS"]):
            return "TIER_0_FULL_STRICT_STRICT_PASS"
        if bool(r["SATELLITE_STRICT_PASS_NO_TOWER"]):
            return "TIER_1_SATELLITE_STRICT_PASS_NO_TOWER"
        if r["non_tower_gate_score_0_to_5"] == 4:
            return "TIER_2_ONE_NON_TOWER_GATE_SHORT"
        if r["non_tower_gate_score_0_to_5"] == 3:
            return "TIER_3_TWO_NON_TOWER_GATES_SHORT"
        if bool(r["primary_pass"]):
            return "TIER_4_PRIMARY_ONLY_OR_WEAK_GATES"
        return "TIER_5_NEAR_DISCOVERY_ONLY"

    gates["closest_tier"] = gates.apply(tier, axis=1)
    gates = gates.sort_values(
        ["FULL_STRICT_STRICT_PASS", "SATELLITE_STRICT_PASS_NO_TOWER", "non_tower_gate_score_0_to_5", "full_gate_score_0_to_6", "bh_q", "p"],
        ascending=[False, False, False, False, True, True],
    )
    gates.to_csv(TAB / "GATED_AND_CLOSEST_STRICT_ECOLOGICAL_SURVIVORS.csv", index=False)

    fullpass = gates[gates["FULL_STRICT_STRICT_PASS"]].copy()
    satpass = gates[gates["SATELLITE_STRICT_PASS_NO_TOWER"]].copy()
    top50 = gates.head(50).copy()
    fullpass.to_csv(TAB / "FULL_STRICT_STRICT_PASSING_MECHANISMS.csv", index=False)
    satpass.to_csv(TAB / "SATELLITE_STRICT_PASSING_MECHANISMS_NO_TOWER.csv", index=False)
    top50.to_csv(TAB / "TOP50_CLOSEST_VALID_MECHANISMS.csv", index=False)

    best_by_category = gates.groupby("story_category", group_keys=False).head(3).copy()
    best_by_category.to_csv(TAB / "BEST_CLOSEST_BY_ECOLOGICAL_CATEGORY.csv", index=False)

    audit = {
        "base_dataset": str(base_path),
        "rows": int(len(df)),
        "columns_before_aux_merge": int(cols_before),
        "columns_after_aux_merge": int(len(df.columns)),
        "auxiliary_columns_added": int(len(aux_added)),
        "primary_outcome": y_col,
        "strict_ecological_features": int(len(candidates)),
        "mechanisms_tested": int(len(mechs)),
        "exact_product_slope_outcomes": int(len(exacts)),
        "clean_mask": cinfo,
        "primary_pass_count": int(primary["primary_pass"].sum()),
        "gate_survivor_count": int(len(survivor_mechs)),
        "FULL_STRICT_strict_pass_count": int(len(fullpass)),
        "satellite_strict_no_tower_pass_count": int(len(satpass)),
        "c4_rows_found": int(len(c4)),
        "seed": SEED,
        "n_boot": N_BOOT,
        "programming_fixes": [
            "Response-derived and product-derived predictor columns are excluded.",
            "Exact product tests use product-specific slope_change outcomes only.",
            "Auxiliary C4, crop, land-cover, trait, soil, and climate columns are merged when point_id or lat/lon keys exist.",
            "Clean land-cover/C4-crop mask is a gate, not a primary filter.",
            "GOSIF×GLEAM is explicitly required for product gate pass.",
            "Closest mechanisms are ranked without converting near-pass into pass.",
        ],
    }
    with open(TAB / "PROGRAMMING_AUDIT.json", "w") as f:
        json.dump(audit, f, indent=2)

    lines = []
    lines.append("Stage1B6BK strict plus closest ecological mechanism screen")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Input")
    lines.append(f"- Base dataset: {base_path}")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Columns before/after auxiliary merge: {cols_before} / {len(df.columns)}")
    lines.append(f"- Auxiliary columns added: {len(aux_added)}")
    lines.append(f"- Primary outcome: {y_col}")
    lines.append("")
    lines.append("Search size")
    lines.append(f"- Strict ecological features: {len(candidates)}")
    lines.append(f"- Mechanisms tested: {len(mechs)}")
    lines.append(f"- Exact product slope outcomes: {len(exacts)}")
    lines.append(f"- Primary pass count: {int(primary['primary_pass'].sum())}")
    lines.append(f"- Survivors sent to expensive gates: {len(survivor_mechs)}")
    lines.append("")
    lines.append("Land-cover / C4-crop clean mask")
    lines.append(f"- Status: {cinfo['status']}")
    lines.append(f"- Clean n: {cinfo['n_clean']} / {cinfo['n_total']}")
    lines.append(f"- Used columns: {cinfo['used_columns'] if cinfo['used_columns'] else 'NONE'}")
    lines.append("")
    lines.append("Final strict result")
    if len(fullpass) == 0:
        lines.append("- FULL_STRICT_STRICT_PASSING_MECHANISMS: NONE")
    else:
        lines.append(f"- FULL_STRICT_STRICT_PASSING_MECHANISMS: {len(fullpass)}")
        cols = ["closest_tier", "mechanism_id", "feature", "story_category", "mechanism_type", "coef", "p", "bh_q", "non_tower_gate_score_0_to_5", "full_gate_score_0_to_6"]
        lines.append(fullpass.head(20)[cols].to_string(index=False))
    if len(satpass) == 0:
        lines.append("- SATELLITE_STRICT_PASSING_MECHANISMS_NO_TOWER: NONE")
    else:
        lines.append(f"- SATELLITE_STRICT_PASSING_MECHANISMS_NO_TOWER: {len(satpass)}")
        cols = ["closest_tier", "mechanism_id", "feature", "story_category", "mechanism_type", "coef", "p", "bh_q", "non_tower_gate_score_0_to_5", "full_gate_score_0_to_6"]
        lines.append(satpass.head(20)[cols].to_string(index=False))
    lines.append("")
    lines.append("Closest valid mechanisms overall")
    if len(gates) == 0:
        lines.append("- None.")
    else:
        cols = ["closest_tier", "mechanism_id", "feature", "story_category", "mechanism_type", "coef", "p", "bh_q", "non_tower_gate_score_0_to_5", "full_gate_score_0_to_6", "primary_pass", "bootstrap_gate", "clean_gate", "product_all_gate", "product_clean_gate", "tower_strict_gate"]
        lines.append(gates.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Best closest mechanisms by ecological category")
    if len(best_by_category) == 0:
        lines.append("- None.")
    else:
        cols = ["closest_tier", "mechanism_id", "feature", "story_category", "mechanism_type", "p", "bh_q", "non_tower_gate_score_0_to_5", "primary_failure_reason", "clean_failure_reason", "product_all_failure_reason", "product_clean_failure_reason"]
        lines.append(best_by_category.head(40)[cols].to_string(index=False))
    lines.append("")
    lines.append("Required C4 full-control check")
    if len(c4) == 0:
        lines.append("- No C4 ecological feature found even after auxiliary merge.")
    else:
        cols = ["mechanism_id", "feature", "status", "n", "coef", "p", "bh_q", "ci_low", "ci_high", "delta_r2", "delta_aic_full_minus_reduced", "controls_used", "controls_omitted"]
        lines.append(c4.head(30)[cols].to_string(index=False))
    lines.append("")
    lines.append("Programming audit")
    lines.append("- Response-derived/product-derived predictors excluded.")
    lines.append("- Exact product tests use product-specific slope_change outcomes only.")
    lines.append("- Clean land-cover/C4-crop filtering is a gate, not a primary prefilter.")
    lines.append("- GOSIF×GLEAM is checked explicitly as the least directly LAI-dependent product pair.")
    lines.append("- Tower strict gate is not auto-passed; it requires independent tower-derived same-feature/uWUE-response evidence.")
    lines.append("")
    lines.append("Important files")
    for p in [
        TAB / "TOP50_CLOSEST_VALID_MECHANISMS.csv",
        TAB / "BEST_CLOSEST_BY_ECOLOGICAL_CATEGORY.csv",
        TAB / "GATED_AND_CLOSEST_STRICT_ECOLOGICAL_SURVIVORS.csv",
        TAB / "FULL_STRICT_STRICT_PASSING_MECHANISMS.csv",
        TAB / "SATELLITE_STRICT_PASSING_MECHANISMS_NO_TOWER.csv",
        TAB / "project_REQUIRED_C4_FULL_CONTROL_CHECK.csv",
        TAB / "STRICT_FEATURE_CANDIDATE_AUDIT.csv",
        TAB / "PROGRAMMING_AUDIT.json",
    ]:
        lines.append(f"- {p}")

    readme = "\n".join(lines)
    (TXT / "READ_ME_strict_plus_closest_mechanism_screen.txt").write_text(readme)
    print("")
    print("DONE.")
    print(f"Outputs written to: {OUT}")
    print("")
    print("Paste this back:")
    print(f"cat {TXT / 'READ_ME_strict_plus_closest_mechanism_screen.txt'}")


if __name__ == "__main__":
    main()
