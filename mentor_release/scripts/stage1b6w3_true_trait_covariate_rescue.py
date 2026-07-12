from pathlib import Path
from datetime import datetime
import math
import json
import numpy as np
import pandas as pd

OUT = Path("results/stage1b6w3_true_trait_covariate_rescue")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
DATA = Path("data/processed/stage1b6w3")
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

BASE = Path("data/processed/stage1b6t/spatial_biome_heterogeneity_input_strict2x2.csv")
FITS = Path("data/processed/stage1b6r/threshold_response_fits_strict_2x2.csv")

SOURCES = [
    Path("results/paper_point_geography_thesis_lock/tables/Table72_high_vpd_point_geography.csv"),
    Path("results/thesis_feasibility_no_tower/trait_model_ready_co2corrected.csv"),
    Path("data/external/soilgrids_texture_by_point.csv"),
    Path("data/external/aridity_by_point.csv"),
]

# Locked final-13 coordinates.
COORDS = {
    "CA-SF3": (54.0916, -106.0053),
    "CN-HaM": (37.607432, 101.332),
    "NL-Hrw": (51.972465, 5.641228),
    "RU-NeC": (62.314844, 129.500075),
    "US-CMW": (31.6637, -110.1777),
    "US-Cop": (38.09, -109.39),
    "US-Dk1": (35.9712, -79.0934),
    "US-Ne1": (41.1651, -96.4766),
    "US-Ne2": (41.1649, -96.4701),
    "US-Ne3": (41.1797, -96.4397),
    "US-SP1": (29.7381, -82.2188),
    "US-Ton": (38.4309, -120.966),
    "US-Var": (38.4133, -120.9508),
}

MAX_NEAREST_KM = 300.0
STRICT_NEAREST_KM = 75.0
MIN_ENV_N = 4
N_PERM = 10000
SEED = 20260629
rng = np.random.default_rng(SEED)

TRUE_COVARIATES = [
    "p50", "psi50", "isohydricity", "rooting_depth",
    "soil_sand", "soil_clay", "soil_silt",
    "soil_sand_mean", "soil_clay_mean", "soil_silt_mean",
    "soil_texture_coarse_index", "soil_texture_fine_index",
    "mean_vpd", "mean_soil_moisture",
    "mean_annual_temperature", "mean_temperature",
    "mean_annual_precipitation", "mean_precipitation",
    "mean_lai", "growing_season_mean_lai",
    "aridity", "aridity_index",
]

def num(s):
    return pd.to_numeric(s, errors="coerce")

def zscore(s):
    s = num(s)
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def covariate_cols(df):
    """
    Return only continuous numeric trait/climate/soil covariates.
    Excludes categorical bins such as aridity_quartile='Q2', region labels,
    status strings, IDs, and lat/lon coordinate columns.
    """
    out = []
    blocked_tokens = [
        "quartile", "quantile", "class", "status", "region", "country",
        "continent", "realm", "biome", "ecoregion", "id", "name",
        "sector", "band", "flag", "high_vpd", "sahel"
    ]

    allowed_tokens = [
        "p50", "psi50", "isohydric", "rooting",
        "soil_sand", "soil_clay", "soil_silt",
        "texture_coarse", "texture_fine",
        "aridity_index", "aridity",
        "mean_vpd", "soil_moisture",
        "temperature", "precipitation",
        "mean_lai", "growing_season_mean_lai"
    ]

    for c in df.columns:
        cl = c.lower()

        if c.startswith("__"):
            continue
        if cl in ["lat", "lon", "latitude", "longitude", "lat_aridity", "lon_aridity"]:
            continue
        if any(tok in cl for tok in blocked_tokens):
            continue
        if not any(tok in cl for tok in allowed_tokens):
            continue

        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 1:
            out.append(c)

    return sorted(set(out))

def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return None
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    rss = float(np.sum((y - pred) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    er = float(beta[1] * (np.max(x) - np.min(x)))
    return {
        "n": int(len(x)),
        "slope": float(beta[1]),
        "intercept": float(beta[0]),
        "r2": float(r2),
        "spearman_r": rho,
        "abs_spearman_r": abs(rho),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "effect_range": er,
        "abs_effect_range": abs(er),
    }

def perm_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    obs = pd.Series(x).corr(pd.Series(y), method="spearman")
    vals = []
    for _ in range(N_PERM):
        vals.append(pd.Series(x).corr(pd.Series(rng.permutation(y)), method="spearman"))
    vals = np.asarray(vals, dtype=float)
    return float((np.sum(np.abs(vals) >= abs(obs)) + 1) / (len(vals) + 1))

def loo_stability(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4:
        return np.nan, np.nan, ""
    full = fit_line(x, y)
    if full is None:
        return np.nan, np.nan, ""
    full_sign = np.sign(full["slope"])
    slopes = []
    signs = []
    for i in range(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[i] = False
        f = fit_line(x[keep], y[keep])
        if f is not None:
            slopes.append(f["slope"])
            signs.append(np.sign(f["slope"]))
    if not slopes:
        return np.nan, np.nan, ""
    return (
        float(np.mean(np.asarray(signs) == full_sign)),
        float(np.median(slopes)),
        ";".join(str(round(v, 5)) for v in slopes),
    )

if not BASE.exists():
    raise FileNotFoundError(f"Missing base table: {BASE}")

df = pd.read_csv(BASE)
df["point_id"] = df["point_id"].astype(str)
df = df.drop_duplicates("point_id").copy()

# Force locked coordinates.
df["lat_locked"] = df["point_id"].map(lambda x: COORDS.get(str(x), (np.nan, np.nan))[0])
df["lon_locked"] = df["point_id"].map(lambda x: COORDS.get(str(x), (np.nan, np.nan))[1])
df["lat"] = df["lat_locked"]
df["lon"] = df["lon_locked"]

# Add fit-derived outcomes.
if FITS.exists():
    fits = pd.read_csv(FITS)
    fits["point_id"] = fits["point_id"].astype(str)
    fits["is_limitation_like"] = fits["response_class"].isin(["breakdown", "saturation", "weakening"])
    wide = (
        fits.groupby(["point_id", "metric"])
        .agg(limitation_fraction=("is_limitation_like", "mean"))
        .reset_index()
        .pivot(index="point_id", columns="metric", values="limitation_fraction")
        .reset_index()
    )
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.rename(columns={
        "log_wue": "limitation_fraction_log_wue_from_fits",
        "log_uwue": "limitation_fraction_log_uwue_from_fits",
    })
    drop_cols = [c for c in wide.columns if c != "point_id" and c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(wide, on="point_id", how="left")

source_audit = []
match_rows = []

for src_path in SOURCES:
    if not src_path.exists():
        source_audit.append({"source": str(src_path), "exists": False, "used": False, "reason": "missing"})
        continue

    raw = pd.read_csv(src_path)
    lat_col = find_col(raw, ["lat", "latitude", "lat_aridity", "target_lat", "point_lat"])
    lon_col = find_col(raw, ["lon", "longitude", "lon_aridity", "target_lon", "point_lon"])
    pid_col = find_col(raw, ["point_id", "site", "site_id", "tower_id", "id"])

    cols = covariate_cols(raw)

    if not cols:
        source_audit.append({
            "source": str(src_path),
            "exists": True,
            "used": False,
            "reason": "no_true_covariate_columns",
            "source_rows": len(raw),
            "lat_col": lat_col,
            "lon_col": lon_col,
            "pid_col": pid_col,
        })
        continue

    src = raw.copy()
    if pid_col:
        src["__pid"] = src[pid_col].astype(str)
    else:
        src["__pid"] = ""

    if lat_col and lon_col:
        src["__lat"] = num(src[lat_col])
        src["__lon"] = num(src[lon_col])
    else:
        src["__lat"] = np.nan
        src["__lon"] = np.nan

    used_any = False

    # exact ID coalesce first
    exact_ids = set(df["point_id"]).intersection(set(src["__pid"]))
    exact_total = 0
    if exact_ids:
        exact = src[src["__pid"].isin(exact_ids)].drop_duplicates("__pid")
        for c in cols:
            mapping = exact.set_index("__pid")[c].to_dict()
            new = pd.to_numeric(df["point_id"].map(mapping), errors="coerce")
            exact_total += int(new.notna().sum())
            if c not in df.columns or df[c].notna().sum() == 0:
                df[c] = new.astype(float)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").combine_first(new).astype(float)
        used_any = exact_total > 0

    # nearest coordinate coalesce
    nearest_total = 0
    nearest_dists = []
    if src["__lat"].notna().any() and src["__lon"].notna().any():
        src_coord = src[src["__lat"].notna() & src["__lon"].notna()].copy()
        for i, row in df.iterrows():
            best_idx = None
            best_d = np.inf
            for j, sr in src_coord.iterrows():
                d = haversine_km(row["lat_locked"], row["lon_locked"], sr["__lat"], sr["__lon"])
                if np.isfinite(d) and d < best_d:
                    best_d = d
                    best_idx = j

            if best_idx is not None and best_d <= MAX_NEAREST_KM:
                nearest_dists.append(best_d)
                match_rows.append({
                    "source": str(src_path),
                    "point_id": row["point_id"],
                    "nearest_km": best_d,
                    "strict_within_75km": best_d <= STRICT_NEAREST_KM,
                    "source_pid": src_coord.loc[best_idx, "__pid"],
                    "source_lat": src_coord.loc[best_idx, "__lat"],
                    "source_lon": src_coord.loc[best_idx, "__lon"],
                })
                for c in cols:
                    val = src_coord.loc[best_idx, c]
                    if pd.notna(val):
                        nearest_total += 1
                        val_num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
                        if pd.notna(val_num):
                            if c not in df.columns:
                                df[c] = np.nan
                            if pd.isna(df.loc[i, c]):
                                df.loc[i, c] = float(val_num)
                used_any = True

    source_audit.append({
        "source": str(src_path),
        "exists": True,
        "used": bool(used_any),
        "source_rows": len(raw),
        "pid_col": pid_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "candidate_cols": ";".join(cols),
        "exact_match_ids": len(exact_ids),
        "exact_nonmissing_total": exact_total,
        "nearest_nonmissing_total": nearest_total,
        "nearest_n_matches": len(nearest_dists),
        "nearest_median_km": float(np.median(nearest_dists)) if nearest_dists else np.nan,
        "nearest_max_km": float(np.max(nearest_dists)) if nearest_dists else np.nan,
        "reason": "" if used_any else "no_match_or_all_missing",
    })

source_audit = pd.DataFrame(source_audit)
matches = pd.DataFrame(match_rows)

source_audit.to_csv(TAB / "Table_PRODUCT02ea_true_covariate_rescue_source_audit.csv", index=False)
matches.to_csv(TAB / "Table_PRODUCT02eb_true_covariate_nearest_match_audit.csv", index=False)

coverage_rows = []
for c in TRUE_COVARIATES:
    if c in df.columns:
        s = num(df[c])
        coverage_rows.append({
            "covariate": c,
            "n_nonmissing": int(s.notna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
            "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            "usable": bool(s.notna().sum() >= 4 and s.nunique(dropna=True) >= 2),
        })

coverage = pd.DataFrame(coverage_rows).sort_values(["usable", "n_nonmissing"], ascending=[False, False])
coverage.to_csv(TAB / "Table_PRODUCT02ec_true_covariate_coverage_after_forced_coords.csv", index=False)

df.to_csv(DATA / "true_trait_environment_input_forced_coords_covariates.csv", index=False)

usable = coverage.loc[coverage["usable"], "covariate"].tolist() if len(coverage) else []

# Environments.
env_cols = []
df["env_Great_Plains_core"] = df["point_id"].isin(["US-Ne1", "US-Ne2", "US-Ne3"])
df["env_US"] = df["point_id"].astype(str).str.startswith("US-")
df["env_Non_US"] = ~df["env_US"]
if "latitude_band_handbuilt" in df.columns:
    for level in df["latitude_band_handbuilt"].dropna().astype(str).unique():
        c = "env_latband_" + level
        df[c] = df["latitude_band_handbuilt"].astype(str).eq(level)
if "longitude_sector_handbuilt" in df.columns:
    for level in df["longitude_sector_handbuilt"].dropna().astype(str).unique():
        c = "env_lonsector_" + level
        df[c] = df["longitude_sector_handbuilt"].astype(str).eq(level)
if "broad_region_handbuilt" in df.columns:
    for level in df["broad_region_handbuilt"].dropna().astype(str).unique():
        c = "env_region_" + level.replace(" ", "_").replace("/", "_")
        df[c] = df["broad_region_handbuilt"].astype(str).eq(level)

envs = []
for c in df.columns:
    if c.startswith("env_"):
        m = df[c].astype(bool)
        if m.sum() >= 2 and (~m).sum() >= 2:
            envs.append(c)

outcomes = [
    "satellite_limitation_mean_fraction",
    "limitation_fraction_log_wue_from_fits",
    "limitation_fraction_log_uwue_from_fits",
]
outcomes = [c for c in outcomes if c in df.columns and num(df[c]).notna().sum() >= 4]

scan_rows = []
inter_rows = []

for outcome in outcomes:
    y_all = num(df[outcome])

    for cov in usable:
        x_all = zscore(df[cov])

        # global
        f = fit_line(x_all, y_all)
        if f:
            loo, loo_slope, loo_slopes = loo_stability(x_all, y_all)
            scan_rows.append({
                "scope": "GLOBAL_ALL_13",
                "environment": "ALL",
                "environment_n": int(df.shape[0]),
                "outcome": outcome,
                "covariate": cov,
                **f,
                "perm_p_spearman": perm_p(x_all, y_all),
                "loo_sign_stability": loo,
                "loo_median_slope": loo_slope,
                "loo_slopes": loo_slopes,
                "claim_strength": "global_true_covariate_screen",
            })

        # within env and interactions
        for env in envs:
            mask = df[env].astype(bool)
            env_n = int(mask.sum())
            xx = x_all[mask]
            yy = y_all[mask]

            f = fit_line(xx, yy)
            if f:
                loo, loo_slope, loo_slopes = loo_stability(xx, yy)
                p = perm_p(xx, yy) if env_n >= 4 else np.nan

                if env_n >= MIN_ENV_N and f["abs_effect_range"] >= 0.20 and f["abs_spearman_r"] >= 0.70 and (pd.isna(p) or p <= 0.15) and (pd.isna(loo) or loo >= 0.80):
                    strength = "candidate_big_true_covariate_environment_effect"
                elif env_n < MIN_ENV_N and f["abs_effect_range"] >= 0.20:
                    strength = "hypothesis_only_small_n"
                else:
                    strength = "weak_or_exploratory"

                scan_rows.append({
                    "scope": "WITHIN_ENVIRONMENT",
                    "environment": env,
                    "environment_n": env_n,
                    "outcome": outcome,
                    "covariate": cov,
                    **f,
                    "perm_p_spearman": p,
                    "loo_sign_stability": loo,
                    "loo_median_slope": loo_slope,
                    "loo_slopes": loo_slopes,
                    "claim_strength": strength,
                })

            # all-site interaction
            tmp = pd.DataFrame({"y": y_all, "x": x_all, "env": mask.astype(float)}).dropna()
            if len(tmp) >= 8 and tmp["x"].nunique() >= 2 and tmp["env"].nunique() == 2:
                X0 = np.column_stack([np.ones(len(tmp)), tmp["x"], tmp["env"]])
                X1 = np.column_stack([np.ones(len(tmp)), tmp["x"], tmp["env"], tmp["x"] * tmp["env"]])
                y = tmp["y"].to_numpy(dtype=float)
                b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
                b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
                rss0 = float(np.sum((y - X0 @ b0) ** 2))
                rss1 = float(np.sum((y - X1 @ b1) ** 2))
                tss = float(np.sum((y - y.mean()) ** 2))
                r20 = 1 - rss0 / tss if tss > 0 else np.nan
                r21 = 1 - rss1 / tss if tss > 0 else np.nan
                delta = r21 - r20 if np.isfinite(r20) and np.isfinite(r21) else np.nan
                if env_n >= MIN_ENV_N and np.isfinite(delta) and delta >= 0.10 and abs(float(b1[3])) >= 0.10:
                    strength_i = "candidate_true_covariate_environment_interaction"
                else:
                    strength_i = "exploratory_or_small_n"

                inter_rows.append({
                    "environment": env,
                    "environment_n": env_n,
                    "outcome": outcome,
                    "covariate": cov,
                    "n_total": int(len(tmp)),
                    "coef_covariate_main": float(b1[1]),
                    "coef_env_main": float(b1[2]),
                    "coef_covariate_x_env": float(b1[3]),
                    "r2_without_interaction": float(r20),
                    "r2_with_interaction": float(r21),
                    "delta_r2_interaction": float(delta),
                    "interaction_abs_effect": abs(float(b1[3])),
                    "claim_strength": strength_i,
                })

scan = pd.DataFrame(scan_rows)
inter = pd.DataFrame(inter_rows)

if len(scan):
    pr = {
        "candidate_big_true_covariate_environment_effect": 0,
        "global_true_covariate_screen": 1,
        "hypothesis_only_small_n": 2,
        "weak_or_exploratory": 3,
    }
    scan["_p"] = scan["claim_strength"].map(pr).fillna(9)
    scan = scan.sort_values(["_p", "abs_effect_range", "abs_spearman_r", "environment_n"], ascending=[True, False, False, False]).drop(columns=["_p"])

if len(inter):
    pr = {
        "candidate_true_covariate_environment_interaction": 0,
        "exploratory_or_small_n": 1,
    }
    inter["_p"] = inter["claim_strength"].map(pr).fillna(9)
    inter = inter.sort_values(["_p", "delta_r2_interaction", "interaction_abs_effect", "environment_n"], ascending=[True, False, False, False]).drop(columns=["_p"])

scan.to_csv(TAB / "Table_PRODUCT02ed_true_covariate_environment_scan.csv", index=False)
inter.to_csv(TAB / "Table_PRODUCT02ee_true_covariate_environment_interactions.csv", index=False)

gp = scan[scan["environment"].eq("env_Great_Plains_core")].copy() if len(scan) else pd.DataFrame()
gp.to_csv(TAB / "Table_PRODUCT02ef_true_covariate_great_plains_diagnostic.csv", index=False)

candidate = scan[scan["claim_strength"].eq("candidate_big_true_covariate_environment_effect")].head(1) if len(scan) else pd.DataFrame()
candidate_i = inter[inter["claim_strength"].eq("candidate_true_covariate_environment_interaction")].head(1) if len(inter) else pd.DataFrame()

if len(candidate):
    b = candidate.iloc[0]
    verdict = "TRUE_COVARIATE_BIG_EFFECT_IN_SPECIFIC_ENVIRONMENT_FOUND"
    safe_claim = (
        f"After forced-coordinate covariate rescue, {b['covariate']} shows the strongest environment-conditioned effect "
        f"within {b['environment']} (n={int(b['environment_n'])}), with effect_range={b['effect_range']:.3f}, "
        f"Spearman r={b['spearman_r']:.3f}, permutation p={b['perm_p_spearman']:.3f}, and LOO sign stability={b['loo_sign_stability']:.3f}. "
        "This is a defensible exploratory trait/climate/soil mechanism hypothesis, not causal proof."
    )
elif len(candidate_i):
    b = candidate_i.iloc[0]
    verdict = "TRUE_COVARIATE_BY_ENVIRONMENT_INTERACTION_FOUND"
    safe_claim = (
        f"After forced-coordinate covariate rescue, {b['covariate']} × {b['environment']} is the strongest interaction "
        f"for {b['outcome']}, with ΔR²={b['delta_r2_interaction']:.3f} and interaction coefficient={b['coef_covariate_x_env']:.3f}. "
        "This supports an environment-dependent mechanism hypothesis, not causal proof."
    )
elif len(usable) == 0:
    verdict = "COVARIATE_RESCUE_STILL_FAILED_NO_USABLE_TRUE_COVARIATES"
    safe_claim = (
        "Even after forcing final-13 coordinates, no true trait/climate/soil covariates could be attached. "
        "This indicates the available covariate products do not cover or match the final-13 sites in the local files. "
        "Do not claim trait proof from this dataset without expanding the covariate source or site set."
    )
else:
    verdict = "NO_BIG_TRUE_COVARIATE_ENVIRONMENT_EFFECT_FOUND"
    safe_claim = (
        "True covariates were attached, but no large environment-conditioned trait/climate/soil effect passed thresholds. "
        "Use spatial/biome heterogeneity as the main result and frame covariates as exploratory."
    )

decision = pd.DataFrame([{
    "generated": datetime.now().isoformat(timespec="seconds"),
    "n_sites": int(df["point_id"].nunique()),
    "n_usable_true_covariates": int(len(usable)),
    "usable_true_covariates": ";".join(usable),
    "n_environment_masks": int(len(envs)),
    "n_scan_rows": int(len(scan)),
    "n_interaction_rows": int(len(inter)),
    "n_candidate_big_true_covariate_environment_effects": int((scan["claim_strength"].eq("candidate_big_true_covariate_environment_effect")).sum()) if len(scan) else 0,
    "n_candidate_true_covariate_environment_interactions": int((inter["claim_strength"].eq("candidate_true_covariate_environment_interaction")).sum()) if len(inter) else 0,
    "verdict": verdict,
    "safe_claim": safe_claim,
    "blocking_next_stage": False,
    "next_stage": "WRITE_MECHANISM_SECTION_IF_FOUND_OR_EXPAND_COVARIATES",
}])
decision.to_csv(TAB / "Table_PRODUCT02eg_true_covariate_rescue_decision.csv", index=False)

# Figures.
figure_status = "NO_FIGURES"
try:
    import matplotlib.pyplot as plt

    if len(coverage):
        cfig = coverage.sort_values("n_nonmissing", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(cfig["covariate"], cfig["n_nonmissing"])
        plt.xlabel("Nonmissing final-13 sites")
        plt.ylabel("Covariate")
        plt.title("Forced-coordinate covariate rescue coverage")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02g_true_covariate_coverage.png", dpi=200)
        plt.close()

    if len(scan):
        sfig = scan.head(15).copy()
        labels = sfig["covariate"].astype(str) + " | " + sfig["environment"].astype(str)
        vals = sfig["effect_range"]
        plt.figure(figsize=(11, 6))
        plt.barh(labels[::-1], vals[::-1])
        plt.xlabel("Effect range on limitation fraction")
        plt.ylabel("Covariate | environment")
        plt.title("Top true covariate environment effects")
        plt.tight_layout()
        plt.savefig(FIG / "Figure_PRODUCT02h_true_covariate_environment_effects.png", dpi=200)
        plt.close()

    figure_status = "FIGURES_WRITTEN"
except Exception as e:
    figure_status = f"FIGURE_WRITE_FAILED: {repr(e)}"

report = []
report.append("# Stage 1B.6W.3 true covariate rescue with forced final-13 coordinates")
report.append("")
report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
report.append("")
report.append("## Decision")
report.append("")
report.append("```text")
report.append(decision.to_string(index=False))
report.append("```")
report.append("")
report.append("## Safe claim")
report.append("")
report.append(safe_claim)
report.append("")
report.append("## Source audit")
report.append("")
report.append("```text")
report.append(source_audit.to_string(index=False))
report.append("```")
report.append("")
report.append("## Nearest-match audit")
report.append("")
report.append("```text")
report.append(matches.head(80).to_string(index=False) if len(matches) else "No nearest matches.")
report.append("```")
report.append("")
report.append("## Coverage after rescue")
report.append("")
report.append("```text")
report.append(coverage.to_string(index=False) if len(coverage) else "No coverage rows.")
report.append("```")
report.append("")
report.append("## Top true covariate environment effects")
report.append("")
report.append("```text")
report.append(scan.head(40).to_string(index=False) if len(scan) else "No scan rows.")
report.append("```")
report.append("")
report.append("## Top true covariate × environment interactions")
report.append("")
report.append("```text")
report.append(inter.head(40).to_string(index=False) if len(inter) else "No interaction rows.")
report.append("```")
report.append("")
report.append("## Great Plains diagnostic")
report.append("")
report.append("```text")
report.append(gp.head(40).to_string(index=False) if len(gp) else "No Great Plains rows.")
report.append("```")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("- This run excludes latitude/longitude as trait variables.")
report.append("- It injects the locked final-13 coordinates, so nearest matching can actually work.")
report.append("- If this still fails, the local trait/covariate files do not cover the strict final-13 sites; expand the covariate source or use all49 for mechanism screening only.")
report.append(f"- Figure status: `{figure_status}`")
report.append("")

(TXT / "STAGE1B6W3_TRUE_COVARIATE_RESCUE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

machine = {
    "stage": "1B.6W.3_true_covariate_rescue",
    "status": str(decision["verdict"].iloc[0]),
    "safe_claim": str(decision["safe_claim"].iloc[0]),
    "outputs": {
        "input": str(DATA / "true_trait_environment_input_forced_coords_covariates.csv"),
        "source_audit": str(TAB / "Table_PRODUCT02ea_true_covariate_rescue_source_audit.csv"),
        "match_audit": str(TAB / "Table_PRODUCT02eb_true_covariate_nearest_match_audit.csv"),
        "coverage": str(TAB / "Table_PRODUCT02ec_true_covariate_coverage_after_forced_coords.csv"),
        "scan": str(TAB / "Table_PRODUCT02ed_true_covariate_environment_scan.csv"),
        "interactions": str(TAB / "Table_PRODUCT02ee_true_covariate_environment_interactions.csv"),
        "great_plains": str(TAB / "Table_PRODUCT02ef_true_covariate_great_plains_diagnostic.csv"),
        "decision": str(TAB / "Table_PRODUCT02eg_true_covariate_rescue_decision.csv"),
        "report": str(TXT / "STAGE1B6W3_TRUE_COVARIATE_RESCUE_REPORT.md"),
    }
}
(TAB / "STAGE1B6W3_TRUE_COVARIATE_RESCUE_SUMMARY.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")

print("\n".join(report))
print("")
print("WROTE", DATA / "true_trait_environment_input_forced_coords_covariates.csv")
print("WROTE", TAB / "Table_PRODUCT02ea_true_covariate_rescue_source_audit.csv")
print("WROTE", TAB / "Table_PRODUCT02eb_true_covariate_nearest_match_audit.csv")
print("WROTE", TAB / "Table_PRODUCT02ec_true_covariate_coverage_after_forced_coords.csv")
print("WROTE", TAB / "Table_PRODUCT02ed_true_covariate_environment_scan.csv")
print("WROTE", TAB / "Table_PRODUCT02ee_true_covariate_environment_interactions.csv")
print("WROTE", TAB / "Table_PRODUCT02ef_true_covariate_great_plains_diagnostic.csv")
print("WROTE", TAB / "Table_PRODUCT02eg_true_covariate_rescue_decision.csv")
print("WROTE", TXT / "STAGE1B6W3_TRUE_COVARIATE_RESCUE_REPORT.md")
print("WROTE figures to", FIG)
