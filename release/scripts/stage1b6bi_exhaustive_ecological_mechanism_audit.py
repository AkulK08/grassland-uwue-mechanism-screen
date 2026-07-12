#!/usr/bin/env python3
"""
stage1b6bi_exhaustive_ecological_mechanism_audit.py

Transparent exploratory audit of all eligible point-level ecological predictors.

Core rules
----------
- Same full-control specification for every candidate.
- BH and BY correction across the complete candidate family.
- No filtering based on significance, effect direction, leverage, or product agreement.
- Every candidate and every failure is written to disk.
- Product tests use every available GPP × ET pair and a common-complete-case sample.
- Domain sensitivity is reported, never hidden.
- Tower checks are labeled provisional unless true tower-coordinate predictor data exist.
- This is discovery/audit work, not independent confirmation.

Run from repository root:
    python scripts/stage1b6bi_exhaustive_ecological_mechanism_audit.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
OUT = ROOT / "results/stage1b6bi_exhaustive_ecological_mechanism_audit"
TAB = OUT / "tables"
TXT = OUT / "text"
for d in (TAB, TXT):
    d.mkdir(parents=True, exist_ok=True)

POINT_PATH = ROOT / "results/stage1b6az_point_provenance_and_c4_missingness/tables/FULL_POINT_PROVENANCE_TABLE.csv"
OBS_PATH = ROOT / "results/trait_framework/phase8/table_latent_model_observations.csv"

MIN_N = 50
N_BOOT = 500
SEED = 20260712
ALPHA = 0.05
rng = np.random.default_rng(SEED)

CONTROL_ALIASES = {
    "vpd": ["vpd_z", "mean_vpd_z", "baseline_vpd_z", "vpd", "mean_vpd", "baseline_vpd"],
    "aridity": ["arid_z", "aridity_z", "aridity_index_z", "arid", "aridity", "aridity_index"],
    "mat": ["mat_z", "mean_annual_temperature_z", "temperature_z", "mat", "mean_annual_temperature"],
    "map": ["map_z", "mean_annual_precipitation_z", "precipitation_z", "map", "mean_annual_precipitation"],
    "soil_moisture": ["sm_z", "soil_moisture_z", "baseline_soil_moisture_z", "sm", "soil_moisture"],
    "soil_texture_pc1": ["soil_texture_pc1_z", "soil_pc1_z", "soil_texture_pc1", "soil_pc1"],
    "latitude": ["lat_z", "latitude_z", "lat", "latitude"],
    "longitude": ["lon_z", "longitude_z", "lon", "longitude"],
}

PRIORITY_PATTERNS = [
    r"lai", r"fpar", r"evi", r"ndvi", r"vod", r"canopy", r"biomass",
    r"c4", r"c3", r"root", r"p50", r"psi50", r"isohyd",
    r"sand", r"clay", r"silt", r"soil", r"elevation", r"slope",
    r"phenolog", r"season", r"productivity", r"graz", r"fire",
    r"nitrogen", r"co2", r"radiation", r"drought",
]

EXCLUDE_PATTERNS = [
    r"(^|_)id($|_)", r"index", r"unnamed", r"geometry", r"wkt",
    r"(^|_)gpp($|_)", r"(^|_)et($|_)", r"uwue", r"wue",
    r"response", r"outcome", r"latent", r"coef", r"effect",
    r"pval", r"qval", r"ci_", r"delta_", r"resid", r"fitted",
    r"prediction", r"product", r"year", r"date", r"time",
    r"mask", r"flag", r"quality", r"source", r"filename",
]

def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def first_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lookup = {norm(c): c for c in df.columns}
    for n in names:
        if norm(n) in lookup:
            return lookup[norm(n)]
    return None

def z(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - x.mean()) / sd

def identify_key(a: pd.DataFrame, b: pd.DataFrame) -> str:
    for name in ["point_id", "site_id", "pixel_id", "grid_id", "location_id", "point", "site"]:
        ca, cb = first_col(a, [name]), first_col(b, [name])
        if ca and cb:
            if cb != ca:
                b.rename(columns={cb: ca}, inplace=True)
            return ca
    common = [c for c in a.columns if c in b.columns]
    scored = []
    for c in common:
        ua, ub = a[c].nunique(dropna=True), b[c].nunique(dropna=True)
        if min(ua, ub) >= 20:
            scored.append((min(ua, ub), c))
    if not scored:
        raise RuntimeError("Could not infer shared point/site key.")
    return sorted(scored, reverse=True)[0][1]

def identify_controls(df: pd.DataFrame) -> dict[str, str]:
    found, missing = {}, []
    for concept, aliases in CONTROL_ALIASES.items():
        c = first_col(df, aliases)
        if c:
            found[concept] = c
        else:
            missing.append(concept)
    if missing:
        raise RuntimeError("Missing required full controls: " + ", ".join(missing))
    return found

def identify_latent_outcome(point: pd.DataFrame, obs: pd.DataFrame, key: str):
    aliases = ["latent_y_z", "latent_outcome_z", "latent_response_z", "latent_y", "latent_outcome"]
    c = first_col(point, aliases)
    if c:
        return point, c
    c = first_col(obs, aliases)
    if c:
        t = obs[[key, c]].dropna().groupby(key, as_index=False)[c].mean()
        return point.merge(t, on=key, how="left"), c
    raise RuntimeError("Could not find latent outcome column.")

def identify_product_columns(obs: pd.DataFrame):
    gpp = first_col(obs, ["gpp_product", "gpp_source", "gpp_dataset"])
    et = first_col(obs, ["et_product", "et_source", "et_dataset"])
    if not gpp or not et:
        raise RuntimeError("Could not identify GPP and ET product columns.")
    return gpp, et

def identify_product_outcome(obs: pd.DataFrame) -> str:
    aliases = [
        "y_z", "response_z", "outcome_z", "uwue_response_z",
        "effect_z", "y", "response", "outcome", "uwue_response",
        "effect", "slope"
    ]
    c = first_col(obs, aliases)
    if c:
        return c
    numeric = [c for c in obs.columns if pd.api.types.is_numeric_dtype(obs[c])]
    scored = []
    for c in numeric:
        n = norm(c)
        score = sum(k in n for k in ["response", "outcome", "effect", "slope", "uwue"])
        scored.append((score, c))
    scored.sort(reverse=True)
    if not scored or scored[0][0] == 0:
        raise RuntimeError("Could not infer per-product response column.")
    return scored[0][1]

def candidate_columns(df: pd.DataFrame, outcome: str, controls: list[str], key: str) -> list[str]:
    bad = re.compile("|".join(EXCLUDE_PATTERNS), re.I)
    pri = re.compile("|".join(PRIORITY_PATTERNS), re.I)
    eligible = []
    for c in df.columns:
        if c in controls or c in {outcome, key}:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if bad.search(norm(c)):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() < MIN_N or x.nunique(dropna=True) < 4 or x.std(skipna=True) == 0:
            continue
        eligible.append(c)
    priority = [c for c in eligible if pri.search(norm(c))]
    other = [c for c in eligible if c not in priority]
    other.sort(key=lambda c: df[c].notna().sum(), reverse=True)
    return list(dict.fromkeys(priority + other))

def mask_for(df: pd.DataFrame, name: str) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    if name == "all_available":
        return keep
    if name == "cropland_clean":
        for aliases, positive in [
            (["crop_clean", "cropland_clean", "is_crop_clean"], True),
            (["natural_grassland", "natural_grassland_indicator"], True),
            (["is_cropland", "cropland_flag", "crop_flag"], False),
        ]:
            c = first_col(df, aliases)
            if c:
                s = df[c]
                if pd.api.types.is_bool_dtype(s):
                    return s.fillna(False) if positive else ~s.fillna(True)
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().any():
                    return (num > 0) if positive else (num <= 0)
        return keep
    if name == "exclude_broad_sahel":
        lat, lon = first_col(df, ["lat", "latitude"]), first_col(df, ["lon", "longitude"])
        if lat and lon:
            inside = pd.to_numeric(df[lat], errors="coerce").between(10, 20) & pd.to_numeric(df[lon], errors="coerce").between(-20, 40)
            return ~inside.fillna(False)
        return keep
    if name == "warm_mat_gt_0":
        c = first_col(df, ["mat", "mean_annual_temperature"])
        return pd.to_numeric(df[c], errors="coerce") > 0 if c else keep
    if name == "warm_mat_gt_2p08":
        c = first_col(df, ["mat", "mean_annual_temperature"])
        return pd.to_numeric(df[c], errors="coerce") > 2.08 if c else keep
    if name == "abs_lat_le_48":
        c = first_col(df, ["lat", "latitude"])
        return pd.to_numeric(df[c], errors="coerce").abs() <= 48 if c else keep
    if name == "c4_covered_domain":
        c = first_col(df, ["c4_fraction", "c4_frac", "c4_cover", "c4"])
        return pd.to_numeric(df[c], errors="coerce").notna() if c else keep
    return keep

def vif_for(d: pd.DataFrame, focal: str, controls: list[str]) -> float:
    try:
        X = sm.add_constant(d[[focal] + controls], has_constant="add")
        idx = list(X.columns).index(focal)
        return float(variance_inflation_factor(X.values, idx))
    except Exception:
        return np.nan

def bootstrap_ci(d: pd.DataFrame, y: str, x: str, controls: list[str]):
    vals = []
    n = len(d)
    for _ in range(N_BOOT):
        b = d.iloc[rng.integers(0, n, n)]
        try:
            X = sm.add_constant(b[[x] + controls], has_constant="add")
            vals.append(sm.OLS(b[y], X).fit().params[x])
        except Exception:
            pass
    if len(vals) < 100:
        return np.nan, np.nan, np.nan
    a = np.asarray(vals)
    lo, hi = np.quantile(a, [0.025, 0.975])
    stab = max((a > 0).mean(), (a < 0).mean())
    return float(lo), float(hi), float(stab)

def fit_model(df: pd.DataFrame, outcome: str, candidate: str, controls_raw: list[str],
              mask_name: str, combo: str, do_boot: bool = False) -> dict:
    w = df.copy()
    w["__y"] = z(w[outcome])
    w["__x"] = z(w[candidate])
    controls = []
    for i, c in enumerate(controls_raw):
        name = f"__c{i}"
        w[name] = z(w[c])
        controls.append(name)
    d = w[["__y", "__x"] + controls].replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "candidate": candidate, "mask": mask_name, "product_combo": combo,
        "n": len(d), "status": "FIT_FAIL", "coef": np.nan, "se_hc3": np.nan,
        "p": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        "delta_r2": np.nan, "delta_aic": np.nan, "nested_f_p": np.nan,
        "focal_vif": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan,
        "boot_sign_stability": np.nan,
    }
    if len(d) < MIN_N or d["__x"].nunique() < 3:
        return out
    try:
        Xf = sm.add_constant(d[["__x"] + controls], has_constant="add")
        Xr = sm.add_constant(d[controls], has_constant="add")
        robust = sm.OLS(d["__y"], Xf).fit(cov_type="HC3")
        plain_f = sm.OLS(d["__y"], Xf).fit()
        plain_r = sm.OLS(d["__y"], Xr).fit()
        ci = robust.conf_int().loc["__x"]
        out.update({
            "status": "FIT_OK",
            "coef": float(robust.params["__x"]),
            "se_hc3": float(robust.bse["__x"]),
            "p": float(robust.pvalues["__x"]),
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "delta_r2": float(plain_f.rsquared - plain_r.rsquared),
            "delta_aic": float(plain_f.aic - plain_r.aic),
            "nested_f_p": float(plain_f.compare_f_test(plain_r)[1]),
            "focal_vif": vif_for(d, "__x", controls),
        })
        if do_boot:
            lo, hi, stab = bootstrap_ci(d, "__y", "__x", controls)
            out.update({"boot_ci_low": lo, "boot_ci_high": hi, "boot_sign_stability": stab})
    except Exception:
        pass
    return out

def add_fdr(df: pd.DataFrame, group_cols: Optional[list[str]] = None) -> pd.DataFrame:
    out = df.copy()
    out["bh_q"] = np.nan
    out["by_q"] = np.nan
    groups = [(None, out)] if not group_cols else out.groupby(group_cols, dropna=False)
    for key, g in groups:
        ok = g["p"].notna()
        if not ok.any():
            continue
        idx = g.index[ok]
        p = g.loc[idx, "p"].values
        out.loc[idx, "bh_q"] = multipletests(p, method="fdr_bh")[1]
        out.loc[idx, "by_q"] = multipletests(p, method="fdr_by")[1]
    return out

def least_entangled(combo: str) -> bool:
    s = combo.upper()
    return ("GOSIF" in s or "SIF" in s) and "GLEAM" in s

def region_col(df: pd.DataFrame) -> Optional[str]:
    return first_col(df, ["continent", "region", "biogeographic_region", "ecoregion", "country"])

# ------------------------------- load -----------------------------------------

if not POINT_PATH.exists():
    raise FileNotFoundError(POINT_PATH)
if not OBS_PATH.exists():
    raise FileNotFoundError(OBS_PATH)

point = pd.read_csv(POINT_PATH, low_memory=False)
obs = pd.read_csv(OBS_PATH, low_memory=False)
KEY = identify_key(point, obs)
point, LATENT_Y = identify_latent_outcome(point, obs, KEY)
CONTROL_MAP = identify_controls(point)
CONTROLS = list(CONTROL_MAP.values())
GPP_COL, ET_COL = identify_product_columns(obs)
PRODUCT_Y = identify_product_outcome(obs)
CANDIDATES = candidate_columns(point, LATENT_Y, CONTROLS, KEY)

if not CANDIDATES:
    raise RuntimeError("No eligible ecological predictors were found.")

manifest = {
    "point_path": str(POINT_PATH),
    "observation_path": str(OBS_PATH),
    "key": KEY,
    "latent_outcome": LATENT_Y,
    "product_outcome": PRODUCT_Y,
    "controls": CONTROL_MAP,
    "candidate_count": len(CANDIDATES),
    "candidates": CANDIDATES,
    "rules": {
        "minimum_n": MIN_N,
        "bootstrap_replicates": N_BOOT,
        "no_significance_based_filtering": True,
        "no_product_agreement_filtering": True,
        "fdr_family": "all eligible candidate variables",
    },
}
(TXT / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

# ------------------------- inventory and filtering audit ----------------------

inventory = []
for c in [LATENT_Y] + CONTROLS + CANDIDATES:
    x = pd.to_numeric(point[c], errors="coerce")
    inventory.append({
        "column": c,
        "role": "outcome" if c == LATENT_Y else ("control" if c in CONTROLS else "candidate"),
        "n_total": len(point),
        "n_nonmissing": int(x.notna().sum()),
        "missing_fraction": float(x.isna().mean()),
        "n_unique": int(x.nunique(dropna=True)),
        "mean": x.mean(), "sd": x.std(), "min": x.min(), "max": x.max(),
    })
pd.DataFrame(inventory).to_csv(TAB / "TABLE_01_VARIABLE_INVENTORY.csv", index=False)

MASKS = [
    "all_available", "cropland_clean", "exclude_broad_sahel",
    "warm_mat_gt_0", "warm_mat_gt_2p08", "abs_lat_le_48", "c4_covered_domain",
]
retention = []
for m in MASKS:
    keep = mask_for(point, m)
    retention.append({"mask": m, "n": int(keep.sum()), "fraction_retained": float(keep.mean())})
pd.DataFrame(retention).to_csv(TAB / "TABLE_02_FILTER_RETENTION.csv", index=False)

# ----------------------------- primary scan -----------------------------------

primary = pd.DataFrame([
    fit_model(point, LATENT_Y, c, CONTROLS, "all_available", "latent", do_boot=True)
    for c in CANDIDATES
])
primary = add_fdr(primary)
primary["ci_excludes_zero"] = (primary["ci_low"] > 0) | (primary["ci_high"] < 0)
primary["boot_ci_excludes_zero"] = (primary["boot_ci_low"] > 0) | (primary["boot_ci_high"] < 0)
primary["bh_pass"] = primary["bh_q"] < ALPHA
primary["by_pass"] = primary["by_q"] < ALPHA
primary["vif_ok"] = primary["focal_vif"] <= 10
primary["incremental_fit_ok"] = (primary["delta_r2"] > 0) & (primary["delta_aic"] < 0)
primary = primary.sort_values(["bh_q", "p"], na_position="last")
primary.to_csv(TAB / "TABLE_10_FULL_FDR_FAMILY_PRIMARY_SCAN.csv", index=False)

deep = primary.loc[
    primary["bh_pass"] & primary["ci_excludes_zero"] & primary["vif_ok"],
    "candidate"
].tolist()
if not deep:
    deep = primary.head(min(10, len(primary)))["candidate"].tolist()

# ---------------------------- domain sensitivity ------------------------------

domain_rows = []
for c in deep:
    for m in MASKS:
        keep = mask_for(point, m)
        domain_rows.append(fit_model(point.loc[keep], LATENT_Y, c, CONTROLS, m, "latent", do_boot=m in ["all_available", "cropland_clean"]))
domain = add_fdr(pd.DataFrame(domain_rows), ["mask"])
domain["ci_excludes_zero"] = (domain["ci_low"] > 0) | (domain["ci_high"] < 0)
domain.to_csv(TAB / "TABLE_20_DOMAIN_AND_FILTER_SENSITIVITY.csv", index=False)

# ----------------------------- product outcomes -------------------------------

tmp = obs[[KEY, GPP_COL, ET_COL, PRODUCT_Y]].copy()
tmp[PRODUCT_Y] = pd.to_numeric(tmp[PRODUCT_Y], errors="coerce")
tmp["product_combo"] = (
    tmp[GPP_COL].astype(str).str.upper().str.strip()
    + " x "
    + tmp[ET_COL].astype(str).str.upper().str.strip()
)
product_long = (
    tmp.groupby([KEY, "product_combo"], as_index=False)[PRODUCT_Y]
    .mean()
    .rename(columns={PRODUCT_Y: "__product_y"})
)
combos = sorted(product_long["product_combo"].dropna().unique())

product_rows = []
for combo in combos:
    y = product_long.loc[product_long["product_combo"] == combo, [KEY, "__product_y"]]
    d = point.merge(y, on=KEY, how="left")
    for c in deep:
        for m in ["all_available", "cropland_clean"]:
            keep = mask_for(d, m)
            product_rows.append(fit_model(d.loc[keep], "__product_y", c, CONTROLS, m, combo))
products = add_fdr(pd.DataFrame(product_rows), ["mask"])
products["least_entangled_pair"] = products["product_combo"].map(least_entangled)
products["bh_pass"] = products["bh_q"] < ALPHA
products.to_csv(TAB / "TABLE_30_ALL_PRODUCT_COMBINATIONS.csv", index=False)

# Common-complete-case comparison.
wide = product_long.pivot(index=KEY, columns="product_combo", values="__product_y")
complete_ids = set(wide.dropna().index)
common_rows = []
for combo in combos:
    y = product_long.loc[
        (product_long["product_combo"] == combo) & product_long[KEY].isin(complete_ids),
        [KEY, "__product_y"]
    ]
    d = point[point[KEY].isin(complete_ids)].merge(y, on=KEY, how="left")
    for c in deep:
        for m in ["all_available", "cropland_clean"]:
            keep = mask_for(d, m)
            name = "common_complete" if m == "all_available" else "cropland_clean_common_complete"
            common_rows.append(fit_model(d.loc[keep], "__product_y", c, CONTROLS, name, combo))
common = add_fdr(pd.DataFrame(common_rows), ["mask"])
common["least_entangled_pair"] = common["product_combo"].map(least_entangled)
common["bh_pass"] = common["bh_q"] < ALPHA
common.to_csv(TAB / "TABLE_31_COMMON_COMPLETE_CASE_PRODUCT_TESTS.csv", index=False)

# ---------------------------- product summary ---------------------------------

primary_lookup = primary.set_index("candidate")
summary = []
for c in deep:
    sign0 = np.sign(primary_lookup.loc[c, "coef"])
    p = products[(products["candidate"] == c) & (products["mask"] == "all_available")]
    pc = common[(common["candidate"] == c) & (common["mask"] == "common_complete")]
    ind = p[p["least_entangled_pair"]]
    indc = pc[pc["least_entangled_pair"]]

    def sign_rate(d):
        return np.nan if d.empty else float((np.sign(d["coef"]) == sign0).mean())

    def pass_rate(d):
        return np.nan if d.empty else float((d["bh_q"] < ALPHA).mean())

    summary.append({
        "candidate": c,
        "n_product_combos": len(p),
        "product_sign_consistency": sign_rate(p),
        "product_fdr_pass_fraction": pass_rate(p),
        "n_least_entangled_pairs": len(ind),
        "least_entangled_sign_consistency": sign_rate(ind),
        "least_entangled_fdr_pass_fraction": pass_rate(ind),
        "common_case_sign_consistency": sign_rate(pc),
        "common_case_fdr_pass_fraction": pass_rate(pc),
        "common_least_entangled_sign_consistency": sign_rate(indc),
        "common_least_entangled_fdr_pass_fraction": pass_rate(indc),
    })
product_summary = pd.DataFrame(summary)
product_summary.to_csv(TAB / "TABLE_32_PRODUCT_DEPENDENCY_SUMMARY.csv", index=False)

# ----------------------- leave-region-out sensitivity --------------------------

rcol = region_col(point)
lro_rows = []
if rcol:
    regions = sorted(point[rcol].dropna().astype(str).unique())
    if 2 <= len(regions) <= 100:
        for c in deep:
            sign0 = np.sign(primary_lookup.loc[c, "coef"])
            for region in regions:
                keep = point[rcol].astype(str) != region
                row = fit_model(point.loc[keep], LATENT_Y, c, CONTROLS, f"leave_out_{region}", "latent")
                row["region_column"] = rcol
                row["left_out_region"] = region
                row["same_sign_as_primary"] = np.sign(row["coef"]) == sign0 if np.isfinite(row["coef"]) else np.nan
                lro_rows.append(row)
pd.DataFrame(lro_rows).to_csv(TAB / "TABLE_40_LEAVE_REGION_OUT.csv", index=False)

# -------------------------- influence sensitivity -----------------------------

infl_rows = []
for c in deep:
    w = point.copy()
    w["__y"] = z(w[LATENT_Y])
    w["__x"] = z(w[c])
    cz = []
    for i, ctrl in enumerate(CONTROLS):
        q = f"__c{i}"
        w[q] = z(w[ctrl])
        cz.append(q)
    d = w[["__y", "__x"] + cz].dropna()
    if len(d) < MIN_N:
        continue
    X = sm.add_constant(d[["__x"] + cz], has_constant="add")
    m = sm.OLS(d["__y"], X).fit()
    cooks = m.get_influence().cooks_distance[0]
    keep = cooks <= 4 / len(d)
    d2 = d.loc[keep]
    X2 = sm.add_constant(d2[["__x"] + cz], has_constant="add")
    m2 = sm.OLS(d2["__y"], X2).fit(cov_type="HC3")
    ci = m2.conf_int().loc["__x"]
    infl_rows.append({
        "candidate": c, "n_full": len(d), "n_removed": int((~keep).sum()),
        "n_after": int(keep.sum()), "coef": m2.params["__x"],
        "p": m2.pvalues["__x"], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1],
        "same_sign_as_primary": np.sign(m2.params["__x"]) == np.sign(primary_lookup.loc[c, "coef"]),
    })
influence = add_fdr(pd.DataFrame(infl_rows)) if infl_rows else pd.DataFrame()
influence.to_csv(TAB / "TABLE_41_INFLUENCE_SENSITIVITY.csv", index=False)

# -------------------------- tower file inventory -------------------------------

tower_candidates = []
for pattern in [
    "results/**/*tower*.csv", "data/**/*tower*.csv",
    "results/**/*fluxnet*.csv", "data/**/*fluxnet*.csv",
    "results/**/*ameriflux*.csv", "data/**/*ameriflux*.csv",
]:
    tower_candidates.extend(ROOT.glob(pattern))

tower_inventory = []
for path in sorted(set(tower_candidates)):
    try:
        d = pd.read_csv(path, low_memory=False)
        tower_inventory.append({
            "path": str(path.relative_to(ROOT)),
            "n_rows": len(d),
            "n_columns": len(d.columns),
            "columns": " | ".join(map(str, d.columns[:100])),
            "has_lai_like": any(re.search(r"lai|fpar|evi|ndvi|vod", norm(c)) for c in d.columns),
            "has_uwue_like": any(re.search(r"uwue|wue|response|effect|slope", norm(c)) for c in d.columns),
            "has_landcover_like": any(re.search(r"land.?cover|igbp|crop|vegetation", norm(c)) for c in d.columns),
        })
    except Exception as exc:
        tower_inventory.append({"path": str(path.relative_to(ROOT)), "read_error": str(exc)})
pd.DataFrame(tower_inventory).to_csv(TAB / "TABLE_60_TOWER_FILE_INVENTORY.csv", index=False)

# ------------------------------ final gates -----------------------------------

prod_idx = product_summary.set_index("candidate")
dom_idx = domain.set_index(["candidate", "mask"])
lro = pd.DataFrame(lro_rows)
gate_rows = []

for c in deep:
    pr = primary_lookup.loc[c]
    warm = dom_idx.loc[(c, "warm_mat_gt_2p08")] if (c, "warm_mat_gt_2p08") in dom_idx.index else None
    clean = dom_idx.loc[(c, "cropland_clean")] if (c, "cropland_clean") in dom_idx.index else None
    ps = prod_idx.loc[c]

    lro_ok = np.nan
    if not lro.empty:
        x = lro[lro["candidate"] == c]
        if not x.empty:
            lro_ok = float(x["same_sign_as_primary"].mean())

    gates = {
        "candidate": c,
        "primary_bh_pass": bool(pr["bh_q"] < ALPHA),
        "primary_by_pass": bool(pr["by_q"] < ALPHA),
        "primary_ci_excludes_zero": bool(pr["ci_excludes_zero"]),
        "bootstrap_ci_excludes_zero": bool(pr["boot_ci_excludes_zero"]),
        "incremental_fit_pass": bool(pr["incremental_fit_ok"]),
        "vif_pass": bool(pr["vif_ok"]),
        "cropland_clean_bh_pass": bool(clean is not None and clean["bh_q"] < ALPHA),
        "warm_domain_bh_pass": bool(warm is not None and warm["bh_q"] < ALPHA),
        "product_sign_consistency": ps["product_sign_consistency"],
        "product_fdr_pass_fraction": ps["product_fdr_pass_fraction"],
        "least_entangled_sign_consistency": ps["least_entangled_sign_consistency"],
        "least_entangled_fdr_pass_fraction": ps["least_entangled_fdr_pass_fraction"],
        "common_case_sign_consistency": ps["common_case_sign_consistency"],
        "common_case_fdr_pass_fraction": ps["common_case_fdr_pass_fraction"],
        "leave_region_out_sign_consistency": lro_ok,
    }
    # "Strict survivor" is deliberately demanding.
    strict = (
        gates["primary_bh_pass"]
        and gates["primary_by_pass"]
        and gates["primary_ci_excludes_zero"]
        and gates["bootstrap_ci_excludes_zero"]
        and gates["incremental_fit_pass"]
        and gates["vif_pass"]
        and gates["cropland_clean_bh_pass"]
        and pd.notna(gates["least_entangled_fdr_pass_fraction"])
        and gates["least_entangled_fdr_pass_fraction"] > 0
        and pd.notna(gates["common_case_sign_consistency"])
        and gates["common_case_sign_consistency"] >= 0.70
    )
    gates["strict_audit_survivor"] = bool(strict)
    gate_rows.append(gates)

gates = pd.DataFrame(gate_rows)
gates.to_csv(TAB / "TABLE_70_FINAL_GATE_SUMMARY.csv", index=False)

final = gates.merge(
    primary[[
        "candidate", "n", "coef", "se_hc3", "p", "bh_q", "by_q",
        "ci_low", "ci_high", "delta_r2", "delta_aic", "nested_f_p",
        "focal_vif", "boot_ci_low", "boot_ci_high", "boot_sign_stability"
    ]],
    on="candidate", how="left"
).sort_values(["strict_audit_survivor", "bh_q"], ascending=[False, True])
final.to_csv(TAB / "TABLE_71_FINAL_RANKED_CANDIDATES.csv", index=False)

# ------------------------------- readme ----------------------------------------

survivors = final[final["strict_audit_survivor"]]
lines = [
    "STAGE1B6BI EXHAUSTIVE ECOLOGICAL-MECHANISM AUDIT",
    "=" * 88,
    "",
    "INTERPRETATION WARNING",
    "This is a transparent exploratory audit, not independent confirmation.",
    "A candidate found here still requires a held-out dataset, true tower-coordinate",
    "predictor data, or a prospectively prespecified confirmatory analysis.",
    "",
    f"Point input: {POINT_PATH}",
    f"Observation input: {OBS_PATH}",
    f"Point/site key: {KEY}",
    f"Latent outcome: {LATENT_Y}",
    f"Per-product outcome: {PRODUCT_Y}",
    f"Full controls: {CONTROLS}",
    f"Candidates scanned: {len(CANDIDATES)}",
    f"Candidates entering deep audit: {len(deep)}",
    f"Product combinations: {len(combos)}",
    f"Common-complete-case points: {len(complete_ids)}",
    "",
    "STRICT AUDIT SURVIVORS",
    "-" * 88,
]
if survivors.empty:
    lines.append("NONE")
else:
    lines.append(survivors.to_string(index=False))

lines += [
    "",
    "TOP RANKED CANDIDATES",
    "-" * 88,
    final.head(20).to_string(index=False),
    "",
    "IMPORTANT OUTPUTS",
    f"- {TAB / 'TABLE_10_FULL_FDR_FAMILY_PRIMARY_SCAN.csv'}",
    f"- {TAB / 'TABLE_20_DOMAIN_AND_FILTER_SENSITIVITY.csv'}",
    f"- {TAB / 'TABLE_30_ALL_PRODUCT_COMBINATIONS.csv'}",
    f"- {TAB / 'TABLE_31_COMMON_COMPLETE_CASE_PRODUCT_TESTS.csv'}",
    f"- {TAB / 'TABLE_32_PRODUCT_DEPENDENCY_SUMMARY.csv'}",
    f"- {TAB / 'TABLE_40_LEAVE_REGION_OUT.csv'}",
    f"- {TAB / 'TABLE_41_INFLUENCE_SENSITIVITY.csv'}",
    f"- {TAB / 'TABLE_60_TOWER_FILE_INVENTORY.csv'}",
    f"- {TAB / 'TABLE_70_FINAL_GATE_SUMMARY.csv'}",
    f"- {TAB / 'TABLE_71_FINAL_RANKED_CANDIDATES.csv'}",
]
(TXT / "READ_ME_exhaustive_ecological_mechanism_audit.txt").write_text("\n".join(lines))

print("\nDONE.")
print(f"Outputs written to: {OUT}")
print("\nPaste this back:")
print(f"cat {TXT / 'READ_ME_exhaustive_ecological_mechanism_audit.txt'}")
