#!/usr/bin/env python
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/tower_centered_phase19_no_gee")
TAB = OUT / "tables"
TXT = OUT / "text"
FIG = OUT / "figures"
for p in [TAB, TXT, FIG]:
    p.mkdir(parents=True, exist_ok=True)

GOSIF = Path("data/raw/tower_centered_phase19/agents/gosif_tower13_point_timeseries.csv")
GLEAM = Path("data/raw/tower_centered_phase19/agents/gleam_tower13_point_timeseries.csv")
TOWER8 = Path("results/tower_validation_broad_inventory/tables/Table87_tower_8day_wue_stress.csv")
TOWERFIT = Path("results/tower_validation_broad_inventory/tables/Table89_tower_response_phenotypes_primary_by_site.csv")
TARGETS = Path("results/tower_satellite_extraction_targets_FINAL/MAIN_expanded_grassland_savanna_open_coordinates.csv")

def die(msg):
    raise SystemExit("\nERROR: " + msg + "\n")

def read(path):
    if not path.exists():
        die(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)

def zscore_site(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - s.mean(skipna=True)) / sd

def pick_col(cols, candidates, contains=None):
    for c in candidates:
        if c in cols:
            return c
    if contains:
        low = {c.lower(): c for c in cols}
        for key in contains:
            for lc, orig in low.items():
                if key.lower() in lc:
                    return orig
    return None

def lin_slope(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan, np.nan, np.nan
    r = stats.linregress(x[ok], y[ok])
    return float(r.slope), float(r.pvalue), float(r.rvalue)

def fit_piecewise(d, xcol, ycol, min_side=12):
    x = pd.to_numeric(d[xcol], errors="coerce")
    y = pd.to_numeric(d[ycol], errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    dd = d.loc[ok].copy()
    if len(dd) < max(40, min_side * 3):
        return None

    dd = dd.sort_values(xcol)
    x = dd[xcol].astype(float).values
    y = dd[ycol].astype(float).values

    qs = np.linspace(0.20, 0.80, 41)
    candidates = np.unique(np.quantile(x, qs))
    best = None

    for bp in candidates:
        left = x <= bp
        right = x > bp
        if left.sum() < min_side or right.sum() < min_side:
            continue

        sl1, _, _ = lin_slope(x[left], y[left])
        sl2, _, _ = lin_slope(x[right], y[right])
        if pd.isna(sl1) or pd.isna(sl2):
            continue

        a1, b1 = np.polyfit(x[left], y[left], 1)
        a2, b2 = np.polyfit(x[right], y[right], 1)
        pred = np.empty_like(y)
        pred[left] = a1 * x[left] + b1
        pred[right] = a2 * x[right] + b2
        sse = float(np.nansum((y - pred) ** 2))

        if best is None or sse < best["sse"]:
            best = {
                "breakpoint": float(bp),
                "sse": sse,
                "pre_slope": float(sl1),
                "post_slope": float(sl2),
                "slope_change": float(sl2 - sl1),
                "n_fit": int(len(dd)),
                "n_left": int(left.sum()),
                "n_right": int(right.sum()),
            }

    if best is None:
        return None

    # Simple bootstrap probability summaries.
    rng = np.random.default_rng(42)
    post = []
    change = []
    pre = []
    for _ in range(120):
        idx = rng.integers(0, len(dd), len(dd))
        boot = dd.iloc[idx].copy()
        res = fit_piecewise_no_boot(boot, xcol, ycol, min_side=min_side)
        if res is not None:
            pre.append(res["pre_slope"])
            post.append(res["post_slope"])
            change.append(res["slope_change"])

    post = np.array(post, dtype=float)
    change = np.array(change, dtype=float)
    pre = np.array(pre, dtype=float)

    best["pre_slope_boot_median"] = float(np.nanmedian(pre)) if len(pre) else np.nan
    best["post_slope_boot_median"] = float(np.nanmedian(post)) if len(post) else np.nan
    best["slope_change_boot_median"] = float(np.nanmedian(change)) if len(change) else np.nan
    best["p_satellite_saturation_breakdown"] = float(np.nanmean(change < 0)) if len(change) else np.nan
    best["p_satellite_breakdown"] = float(np.nanmean(post < 0)) if len(post) else np.nan
    best["p_satellite_enhancement"] = float(np.nanmean(post > 0)) if len(post) else np.nan

    # Classification mirrors the project logic, but intentionally conservative.
    if best["pre_slope"] > 0 and best["post_slope"] < 0 and best["p_satellite_breakdown"] >= 0.80:
        cls = "breakdown"
    elif best["pre_slope"] > 0 and best["slope_change"] < 0 and best["p_satellite_saturation_breakdown"] >= 0.60:
        cls = "saturation"
    elif best["post_slope"] > 0 and best["p_satellite_enhancement"] >= 0.60:
        cls = "enhancement"
    else:
        cls = "inconclusive"

    best["satellite_response_class"] = cls
    return best

def fit_piecewise_no_boot(d, xcol, ycol, min_side=12):
    x = pd.to_numeric(d[xcol], errors="coerce")
    y = pd.to_numeric(d[ycol], errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    dd = d.loc[ok].copy()
    if len(dd) < max(40, min_side * 3):
        return None
    dd = dd.sort_values(xcol)
    x = dd[xcol].astype(float).values
    y = dd[ycol].astype(float).values
    candidates = np.unique(np.quantile(x, np.linspace(0.20, 0.80, 25)))
    best = None
    for bp in candidates:
        left = x <= bp
        right = x > bp
        if left.sum() < min_side or right.sum() < min_side:
            continue
        sl1, _, _ = lin_slope(x[left], y[left])
        sl2, _, _ = lin_slope(x[right], y[right])
        if pd.isna(sl1) or pd.isna(sl2):
            continue
        a1, b1 = np.polyfit(x[left], y[left], 1)
        a2, b2 = np.polyfit(x[right], y[right], 1)
        pred = np.empty_like(y)
        pred[left] = a1 * x[left] + b1
        pred[right] = a2 * x[right] + b2
        sse = float(np.nansum((y - pred) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "breakpoint": float(bp),
                "sse": sse,
                "pre_slope": float(sl1),
                "post_slope": float(sl2),
                "slope_change": float(sl2 - sl1),
                "n_fit": int(len(dd)),
            }
    return best

print("READING LOCAL TOWER-CENTERED GOSIF/GLEAM")
gosif = read(GOSIF)
gleam = read(GLEAM)
tower8 = read(TOWER8)
towerfit = read(TOWERFIT)
targets = read(TARGETS)

# Normalize satellite point ids.
for df in [gosif, gleam]:
    if "point_id" not in df.columns:
        die("GOSIF/GLEAM missing point_id")
    df["point_id"] = df["point_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])

sat = gosif.merge(
    gleam[["point_id", "date", "et_gleam"]],
    on=["point_id", "date"],
    how="inner"
)

sat = sat.rename(columns={"point_id": "site"})
sat["site"] = sat["site"].astype(str)
sat["date"] = pd.to_datetime(sat["date"])

# Normalize tower 8-day file.
if "site" not in tower8.columns:
    die("Table87 tower 8-day file missing site column.")
tower8["site"] = tower8["site"].astype(str)

date_col = pick_col(
    list(tower8.columns),
    ["date", "start_date", "window_start", "date_8day", "period_start"],
    contains=["date"]
)
if date_col is None:
    die("Could not find date column in Table87. Columns: " + str(list(tower8.columns)))

tower8["date"] = pd.to_datetime(tower8[date_col])

vpd_col = pick_col(
    list(tower8.columns),
    ["vpd_8day_kpa_mean", "vpd_kpa", "vpd", "mean_vpd_kpa"],
    contains=["vpd"]
)
swc_col = pick_col(
    list(tower8.columns),
    ["swc_8day_mean", "soil_moisture", "swc", "sm"],
    contains=["swc", "soil_moisture"]
)
precip_col = pick_col(
    list(tower8.columns),
    ["precip_8day_mm", "precip_mm", "precipitation"],
    contains=["precip"]
)

if vpd_col is None:
    die("Could not find VPD column in Table87. Columns: " + str(list(tower8.columns)))

tower8["vpd_for_stress"] = pd.to_numeric(tower8[vpd_col], errors="coerce")
tower8["vpd_z"] = tower8.groupby("site")["vpd_for_stress"].transform(zscore_site)

if swc_col is not None:
    tower8["swc_for_stress"] = pd.to_numeric(tower8[swc_col], errors="coerce")
    tower8["swc_z"] = tower8.groupby("site")["swc_for_stress"].transform(zscore_site)
else:
    tower8["swc_for_stress"] = np.nan
    tower8["swc_z"] = np.nan

if precip_col is not None:
    tower8["precip_for_stress"] = pd.to_numeric(tower8[precip_col], errors="coerce")
    tower8["precip_z"] = tower8.groupby("site")["precip_for_stress"].transform(zscore_site)
else:
    tower8["precip_for_stress"] = np.nan
    tower8["precip_z"] = np.nan

tower8["stress_compound_vpd_swc"] = tower8["vpd_z"] - tower8["swc_z"]
tower8["stress_vpd_precip_proxy"] = tower8["vpd_z"] - tower8["precip_z"]

# Choose stress by availability per row.
tower8["satellite_validation_stress"] = tower8["stress_compound_vpd_swc"]
tower8.loc[tower8["satellite_validation_stress"].isna(), "satellite_validation_stress"] = tower8.loc[
    tower8["satellite_validation_stress"].isna(), "stress_vpd_precip_proxy"
]

keep = [
    "site", "date",
    "vpd_for_stress", "swc_for_stress", "precip_for_stress",
    "stress_compound_vpd_swc", "stress_vpd_precip_proxy", "satellite_validation_stress"
]
keep = [c for c in keep if c in tower8.columns]

merged = sat.merge(tower8[keep], on=["site", "date"], how="inner")

merged["gpp_gosif"] = pd.to_numeric(merged["gpp_gosif"], errors="coerce")
merged["et_gleam"] = pd.to_numeric(merged["et_gleam"], errors="coerce")
merged["vpd_for_stress"] = pd.to_numeric(merged["vpd_for_stress"], errors="coerce")

merged = merged.replace([np.inf, -np.inf], np.nan)
merged = merged.dropna(subset=["gpp_gosif", "et_gleam", "satellite_validation_stress"])
merged = merged[(merged["gpp_gosif"] > 0) & (merged["et_gleam"] > 0)].copy()

merged["wue_gosif_gleam"] = merged["gpp_gosif"] / merged["et_gleam"]
merged["log_wue_gosif_gleam"] = np.log(merged["wue_gosif_gleam"])

# uWUE-like satellite metric using tower VPD because no GEE/ERA5 extraction.
merged["uwue_gosif_gleam_tower_vpd"] = np.nan
okv = merged["vpd_for_stress"].notna() & (merged["vpd_for_stress"] > 0)
merged.loc[okv, "uwue_gosif_gleam_tower_vpd"] = (
    merged.loc[okv, "gpp_gosif"] * np.sqrt(merged.loc[okv, "vpd_for_stress"]) / merged.loc[okv, "et_gleam"]
)
merged["log_uwue_gosif_gleam_tower_vpd"] = np.log(merged["uwue_gosif_gleam_tower_vpd"])

merged.to_csv(TAB / "Table120_no_gee_tower13_gosif_gleam_merged_timeseries.csv", index=False)
print("WROTE", TAB / "Table120_no_gee_tower13_gosif_gleam_merged_timeseries.csv", merged.shape)

# Fit satellite response by site for WUE and uWUE-like metric.
fit_rows = []
for metric in ["log_wue_gosif_gleam", "log_uwue_gosif_gleam_tower_vpd"]:
    for site, d in merged.groupby("site"):
        res = fit_piecewise(d, "satellite_validation_stress", metric, min_side=12)
        if res is None:
            fit_rows.append({
                "site": site,
                "satellite_metric": metric,
                "satellite_product_combo": "GOSIF_GPP__GLEAM_ET__tower_stress_no_GEE",
                "satellite_response_class": "inconclusive",
                "n_fit": len(d),
            })
            continue
        res["site"] = site
        res["satellite_metric"] = metric
        res["satellite_product_combo"] = "GOSIF_GPP__GLEAM_ET__tower_stress_no_GEE"
        fit_rows.append(res)

fits = pd.DataFrame(fit_rows)
fits.to_csv(TAB / "Table121_no_gee_gosif_gleam_satellite_response_by_site.csv", index=False)
print("WROTE", TAB / "Table121_no_gee_gosif_gleam_satellite_response_by_site.csv", fits.shape)

# Primary satellite metric: log uWUE if available, otherwise log WUE.
primary = fits[fits["satellite_metric"].eq("log_uwue_gosif_gleam_tower_vpd")].copy()
fallback = fits[fits["satellite_metric"].eq("log_wue_gosif_gleam")].copy()
if primary.empty or primary["n_fit"].fillna(0).sum() == 0:
    primary = fallback.copy()

# Merge with tower phenotypes.
towerfit["site"] = towerfit["site"].astype(str)
cols = [
    "site", "response_class", "post_slope", "slope_change",
    "p_tower_saturation_breakdown", "tower_metric", "stress_method",
    "n_fit_8day", "n_years"
]
cols = [c for c in cols if c in towerfit.columns]

comp = primary.merge(towerfit[cols], on="site", how="left", suffixes=("_satellite", "_tower"))
comp = comp.rename(columns={
    "response_class": "tower_response_class",
    "post_slope": "tower_post_slope",
    "slope_change": "tower_slope_change",
})

comp["class_agreement_exact"] = comp["satellite_response_class"].astype(str).eq(comp["tower_response_class"].astype(str))

sat_lim = comp["satellite_response_class"].isin(["saturation", "breakdown"])
tow_lim = comp["tower_response_class"].isin(["saturation", "breakdown"])
comp["class_agreement_limited_vs_enhanced"] = sat_lim.eq(tow_lim)

comp["slope_direction_agreement"] = np.sign(pd.to_numeric(comp["post_slope"], errors="coerce")).eq(
    np.sign(pd.to_numeric(comp["tower_post_slope"], errors="coerce"))
)

comp.to_csv(TAB / "Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv", index=False)
print("WROTE", TAB / "Table122_no_gee_tower_vs_satellite_gosif_gleam_comparison.csv", comp.shape)

summary = {
    "mode": "NO_GEE",
    "satellite_product_combo": "GOSIF_GPP + GLEAM_ET",
    "stress_source": "tower VPD + tower SWC/precip proxy",
    "n_main_tower_sites": int(targets.shape[0]),
    "n_merged_timeseries_rows": int(len(merged)),
    "n_sites_with_satellite_fit": int(primary["site"].nunique()),
    "exact_class_agreement_fraction": float(comp["class_agreement_exact"].mean()) if len(comp) else None,
    "limited_vs_enhanced_agreement_fraction": float(comp["class_agreement_limited_vs_enhanced"].mean()) if len(comp) else None,
    "slope_direction_agreement_fraction": float(comp["slope_direction_agreement"].mean()) if len(comp) else None,
    "satellite_class_counts": primary["satellite_response_class"].value_counts().to_dict() if len(primary) else {},
    "tower_class_counts_in_comparison": comp["tower_response_class"].value_counts().to_dict() if len(comp) and "tower_response_class" in comp else {},
}

(TAB / "Table123_no_gee_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame([summary]).to_csv(TAB / "Table123_no_gee_validation_summary.csv", index=False)
print("WROTE", TAB / "Table123_no_gee_validation_summary.csv")

# Figures.
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    primary["satellite_response_class"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("No-GEE satellite response classes: GOSIF/GLEAM at 13 towers")
    ax.set_ylabel("Sites")
    plt.tight_layout()
    fig.savefig(FIG / "Figure1_no_gee_gosif_gleam_satellite_class_counts.png", dpi=300)
    plt.close(fig)
except Exception as e:
    print("WARNING figure1 failed", e)

try:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, g in comp.groupby("tower_response_class"):
        ax.scatter(g["tower_post_slope"], g["post_slope"], label=cls, s=70)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Tower post-slope")
    ax.set_ylabel("Satellite GOSIF/GLEAM post-slope")
    ax.set_title("No-GEE tower vs satellite post-slope")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(FIG / "Figure2_no_gee_tower_vs_satellite_post_slope.png", dpi=300)
    plt.close(fig)
except Exception as e:
    print("WARNING figure2 failed", e)

report = []
report.append("# Phase 19 no-GEE tower-centered satellite validation")
report.append("")
report.append("## What this run does")
report.append("")
report.append("This is the no-GEE validation path. It uses local GOSIF GPP and local GLEAM ET sampled at the 13 expanded grassland/savanna/open tower coordinates. It uses tower VPD and tower soil moisture or precipitation proxy as the stress axis because Earth Engine/ERA5 extraction is unavailable.")
report.append("")
report.append("## Summary")
report.append("")
for k, v in summary.items():
    report.append(f"- {k}: `{v}`")
report.append("")
report.append("## Interpretation")
report.append("")
report.append("This is a real tower-centered satellite check for the GOSIF-GLEAM product pair, but it is not the full 3x3 satellite product matrix. It is manuscript-useful as a no-GEE validation slice, especially if class agreement or limited-vs-enhanced agreement is strong.")
report.append("")
report.append("## Important limitation")
report.append("")
report.append("Because MODIS/PML/ERA5 products were not extracted without GEE in this run, product arbitration across the full 3x3 matrix is still incomplete. This run tests whether locally sampled GOSIF/GLEAM reproduces the tower-observed phenotype under tower-derived stress.")
report.append("")
report.append("## Main comparison table")
report.append("")
report.append("```text")
show_cols = [c for c in [
    "site", "tower_response_class", "satellite_response_class",
    "tower_post_slope", "post_slope", "tower_slope_change", "slope_change",
    "class_agreement_exact", "class_agreement_limited_vs_enhanced",
    "slope_direction_agreement", "n_fit"
] if c in comp.columns]
report.append(comp[show_cols].to_string(index=False) if len(comp) else "No comparison rows.")
report.append("```")
report.append("")

(TXT / "PHASE19_NO_GEE_GOSIF_GLEAM_VALIDATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
print("")
print("\n".join(report))
print("")
print("DONE NO-GEE PHASE 19")
