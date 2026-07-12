#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUTDIR = Path("results/trait_framework")
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_RESP = Path("results/reza_final_nature_boot50/fullspec_response_results_raw.csv")
CO2_RESP = Path("results/reza_final_nature_boot50/fullspec_response_results_co2corrected.csv")
RAW_SURF = Path("results/reza_final_nature_boot50/fullspec_vpd_sm_surface_raw.csv")
CO2_SURF = Path("results/reza_final_nature_boot50/fullspec_vpd_sm_surface_co2corrected.csv")

required = [RAW_RESP, CO2_RESP, RAW_SURF, CO2_SURF]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("Missing required input files:\n" + "\n".join(missing))

def clean_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_strings(df):
    df = df.copy()
    for c in ["metric", "gpp_product", "et_product", "stress_definition", "growing_season", "response_class_strict"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    if "metric" in df.columns:
        df["metric"] = df["metric"].str.lower()
    if "gpp_product" in df.columns:
        df["gpp_product"] = df["gpp_product"].str.upper()
    if "et_product" in df.columns:
        df["et_product"] = df["et_product"].str.upper()
    return df

def parse_lon_lat(point_id):
    s = str(point_id).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        return float(s[0]), float(s[1])
    except Exception:
        return np.nan, np.nan

def ensure_slope_cols(df):
    df = df.copy()

    slope_like = {c.lower(): c for c in df.columns}

    if "pre_slope" not in df.columns:
        candidates = [c for c in df.columns if "pre" in c.lower() and "slope" in c.lower()]
        if candidates:
            df["pre_slope"] = pd.to_numeric(df[candidates[0]], errors="coerce")

    if "post_slope" not in df.columns:
        candidates = [c for c in df.columns if "post" in c.lower() and "slope" in c.lower()]
        if candidates:
            df["post_slope"] = pd.to_numeric(df[candidates[0]], errors="coerce")

    if "slope_change" not in df.columns and {"pre_slope", "post_slope"}.issubset(df.columns):
        df["slope_change"] = pd.to_numeric(df["post_slope"], errors="coerce") - pd.to_numeric(df["pre_slope"], errors="coerce")

    for c in ["pre_slope", "post_slope", "slope_change"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

raw = norm_strings(clean_cols(pd.read_csv(RAW_RESP, low_memory=False)))
co2 = norm_strings(clean_cols(pd.read_csv(CO2_RESP, low_memory=False)))
raw_surf = norm_strings(clean_cols(pd.read_csv(RAW_SURF, low_memory=False)))
co2_surf = norm_strings(clean_cols(pd.read_csv(CO2_SURF, low_memory=False)))

raw = ensure_slope_cols(raw)
co2 = ensure_slope_cols(co2)

key_cols = ["point_id", "metric", "gpp_product", "et_product", "stress_definition", "growing_season"]

for df_name, df in [("raw response", raw), ("co2 response", co2)]:
    missing_cols = [c for c in key_cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"{df_name} missing required columns: {missing_cols}")

co2_u = co2[co2["metric"].eq("uwue")].copy()
raw_u = raw[raw["metric"].eq("uwue")].copy()

for c in key_cols:
    co2_u[c] = co2_u[c].astype(str)
    raw_u[c] = raw_u[c].astype(str)

base_cols = key_cols + ["response_class_strict", "pre_slope", "post_slope", "slope_change"]
missing_base = [c for c in base_cols if c not in co2_u.columns]
if missing_base:
    raise SystemExit(f"CO2 uWUE file missing required phenotype columns: {missing_base}")

phen = co2_u[base_cols].copy()
phen["co2_corrected"] = True

lonlat = phen["point_id"].apply(lambda x: pd.Series(parse_lon_lat(x)))
phen["lon"] = lonlat[0]
phen["lat"] = lonlat[1]

raw_match_cols = key_cols + ["pre_slope", "post_slope", "slope_change"]
raw_match = raw_u[raw_match_cols].copy()
raw_match = raw_match.rename(columns={
    "pre_slope": "pre_slope_raw",
    "post_slope": "post_slope_raw",
    "slope_change": "slope_change_raw"
})

phen = phen.merge(raw_match, on=key_cols, how="left")

phen["raw_vs_co2_slope_change_diff"] = phen["slope_change"] - phen["slope_change_raw"]

den = phen["slope_change"].abs() + phen["slope_change_raw"].abs()
phen["raw_vs_co2_stability"] = np.where(
    den > 0,
    1.0 - (phen["raw_vs_co2_slope_change_diff"].abs() / den),
    np.where(phen["slope_change"].notna() & phen["slope_change_raw"].notna(), 1.0, np.nan)
)
phen["raw_vs_co2_stability"] = phen["raw_vs_co2_stability"].clip(lower=0, upper=1)

phen["raw_vs_co2_same_sign"] = np.where(
    phen["slope_change"].notna() & phen["slope_change_raw"].notna(),
    np.sign(phen["slope_change"]) == np.sign(phen["slope_change_raw"]),
    np.nan
)

surface_key_cols = [c for c in key_cols if c in co2_surf.columns]
co2_surf_u = co2_surf[co2_surf["metric"].eq("uwue")].copy() if "metric" in co2_surf.columns else co2_surf.copy()

surface_cols = []
for c in [
    "vpd_partial_effect",
    "sm_partial_effect",
    "vpd_sm_interaction",
    "surface_rss",
    "vpd_partial_effect_se",
    "sm_partial_effect_se",
    "vpd_sm_interaction_se"
]:
    if c in co2_surf_u.columns:
        surface_cols.append(c)

if surface_cols and all(c in co2_surf_u.columns for c in surface_key_cols):
    surf = co2_surf_u[surface_key_cols + surface_cols].copy()
    for c in surface_cols:
        surf[c] = pd.to_numeric(surf[c], errors="coerce")
    phen = phen.merge(surf, on=surface_key_cols, how="left")

if {"vpd_partial_effect", "sm_partial_effect"}.issubset(phen.columns):
    phen["high_stress_surface_sensitivity"] = np.sqrt(
        phen["vpd_partial_effect"].astype(float) ** 2 +
        phen["sm_partial_effect"].astype(float) ** 2
    )
else:
    phen["high_stress_surface_sensitivity"] = np.nan

phen["sat_or_breakdown_secondary"] = phen["response_class_strict"].isin(["saturation", "breakdown"])
phen["breakdown_secondary"] = phen["response_class_strict"].eq("breakdown")
phen["saturation_secondary"] = phen["response_class_strict"].eq("saturation")

first_cols = [
    "point_id",
    "lat",
    "lon",
    "metric",
    "gpp_product",
    "et_product",
    "stress_definition",
    "growing_season",
    "pre_slope",
    "post_slope",
    "slope_change",
    "response_class_strict",
    "co2_corrected",
    "raw_vs_co2_stability"
]

extra_cols = [
    "raw_vs_co2_slope_change_diff",
    "raw_vs_co2_same_sign",
    "high_stress_surface_sensitivity",
    "vpd_partial_effect",
    "sm_partial_effect",
    "vpd_sm_interaction",
    "sat_or_breakdown_secondary",
    "breakdown_secondary",
    "saturation_secondary",
]

ordered = first_cols + [c for c in extra_cols if c in phen.columns]
remaining = [c for c in phen.columns if c not in ordered]
phen = phen[ordered + remaining]

out_detail = OUTDIR / "point_response_phenotypes.csv"
phen.to_csv(out_detail, index=False)

cons = phen.copy()
cons["slope_change_sign"] = np.sign(pd.to_numeric(cons["slope_change"], errors="coerce"))

def sign_agreement(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    med = np.sign(np.nanmedian(s))
    if med == 0:
        return float((s == 0).mean())
    return float((np.sign(s) == med).mean())

consensus = (
    cons.groupby(["point_id", "lat", "lon", "metric"], dropna=False)
    .agg(
        n_phenotype_rows=("point_id", "size"),
        n_product_combos=("gpp_product", lambda s: int(pd.DataFrame({"g": cons.loc[s.index, "gpp_product"], "e": cons.loc[s.index, "et_product"]}).drop_duplicates().shape[0])),
        median_pre_slope=("pre_slope", "median"),
        median_post_slope=("post_slope", "median"),
        median_slope_change=("slope_change", "median"),
        slope_change_iqr=("slope_change", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.75) - pd.to_numeric(s, errors="coerce").quantile(0.25))),
        product_stress_season_sign_agreement=("slope_change", sign_agreement),
        median_raw_vs_co2_stability=("raw_vs_co2_stability", "median"),
        sat_or_breakdown_rate=("sat_or_breakdown_secondary", "mean"),
        breakdown_rate=("breakdown_secondary", "mean"),
        saturation_rate=("saturation_secondary", "mean"),
        median_high_stress_surface_sensitivity=("high_stress_surface_sensitivity", "median"),
        median_vpd_partial_effect=("vpd_partial_effect", "median") if "vpd_partial_effect" in cons.columns else ("high_stress_surface_sensitivity", "median"),
        median_sm_partial_effect=("sm_partial_effect", "median") if "sm_partial_effect" in cons.columns else ("high_stress_surface_sensitivity", "median"),
        median_vpd_sm_interaction=("vpd_sm_interaction", "median") if "vpd_sm_interaction" in cons.columns else ("high_stress_surface_sensitivity", "median"),
    )
    .reset_index()
)

out_consensus = OUTDIR / "point_response_phenotypes_consensus_per_point.csv"
consensus.to_csv(out_consensus, index=False)

summary = {
    "input_raw_response": str(RAW_RESP),
    "input_co2_response": str(CO2_RESP),
    "input_raw_surface": str(RAW_SURF),
    "input_co2_surface": str(CO2_SURF),
    "primary_metric": "co2-corrected uwue",
    "detail_output": str(out_detail),
    "consensus_output": str(out_consensus),
    "detail_shape": list(phen.shape),
    "consensus_shape": list(consensus.shape),
    "unique_points_detail": int(phen["point_id"].nunique()),
    "unique_points_consensus": int(consensus["point_id"].nunique()),
    "product_combos_detail": int(phen[["gpp_product", "et_product"]].drop_duplicates().shape[0]),
    "stress_definitions": sorted(phen["stress_definition"].dropna().unique().tolist()),
    "growing_seasons": sorted(phen["growing_season"].dropna().unique().tolist()),
    "response_class_counts": phen["response_class_strict"].value_counts(dropna=False).to_dict(),
    "median_raw_vs_co2_stability": float(pd.to_numeric(phen["raw_vs_co2_stability"], errors="coerce").median()),
}

with open(OUTDIR / "phase1_response_phenotype_manifest.json", "w") as f:
    json.dump(summary, f, indent=2)

readme = []
readme.append("# Phase 1 response phenotype output")
readme.append("")
readme.append("Primary metric: CO2-corrected uWUE.")
readme.append("")
readme.append("Main output:")
readme.append(f"- `{out_detail}`")
readme.append("")
readme.append("Consensus per-point output:")
readme.append(f"- `{out_consensus}`")
readme.append("")
readme.append("Interpretation:")
readme.append("- Use `post_slope`, `slope_change`, and `high_stress_surface_sensitivity` as primary continuous response phenotypes.")
readme.append("- Use `response_class_strict`, `sat_or_breakdown_secondary`, `breakdown_secondary`, and `saturation_secondary` only as secondary classifications.")
readme.append("- `raw_vs_co2_stability` ranges from 0 to 1, where higher means the raw and CO2-corrected slope-change values agree more closely.")
readme.append("")
readme.append("Manifest:")
readme.append(json.dumps(summary, indent=2))
Path(OUTDIR / "README_phase1_response_phenotypes.md").write_text("\n".join(readme))

print("WROTE", out_detail)
print("WROTE", out_consensus)
print("WROTE", OUTDIR / "phase1_response_phenotype_manifest.json")
print("WROTE", OUTDIR / "README_phase1_response_phenotypes.md")
print(json.dumps(summary, indent=2))
