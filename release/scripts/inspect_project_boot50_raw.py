#!/usr/bin/env python
from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path("results/project_final_nature_boot50")
INS = OUT / "inspection_raw"
INS.mkdir(parents=True, exist_ok=True)

raw_path = OUT / "fullspec_response_results_raw.csv"
surf_path = OUT / "fullspec_vpd_sm_surface_raw.csv"
arid_path = OUT / "fullspec_aridity_summary_raw.csv"

if not raw_path.exists():
    raise SystemExit(f"Missing {raw_path}. Raw BOOT50 is not written yet.")

df = pd.read_csv(raw_path, low_memory=False)

expected_gpp = ["modis", "gosif", "pml"]
expected_et = ["modis", "gleam", "pml"]
expected_combos = {(g,e) for g in expected_gpp for e in expected_et}
expected_metrics = {"uwue", "iwue", "raw_wue"}
expected_stress = {"zscore", "percentile_joint", "copula_joint", "interaction_surface"}
expected_seasons = {"gpp_threshold", "climate_common", "month_fixed"}

print("\n==============================")
print("RAW BOOT50 RESULT INSPECTION")
print("==============================")
print("file:", raw_path)
print("shape:", df.shape)
print("columns:", list(df.columns))
print("unique points:", df["point_id"].nunique() if "point_id" in df.columns else "NO point_id")

# Basic coverage
combos = (
    df[["gpp_product","et_product"]]
    .drop_duplicates()
    .sort_values(["gpp_product","et_product"])
)
combo_set = set(map(tuple, combos.values.tolist()))
missing_combos = sorted(expected_combos - combo_set)
extra_combos = sorted(combo_set - expected_combos)

print("\n===== PRODUCT COMBOS =====")
print(combos.to_string(index=False))
print("combo count:", len(combo_set))
print("missing combos:", missing_combos)
print("extra combos:", extra_combos)

coverage = (
    df.groupby(["gpp_product","et_product"])
    .agg(rows=("point_id","size"), unique_points=("point_id","nunique"))
    .reset_index()
    .sort_values(["gpp_product","et_product"])
)
coverage.to_csv(INS / "01_combo_coverage.csv", index=False)
print("\n===== COMBO COVERAGE =====")
print(coverage.to_string(index=False))

# Axes coverage
def show_axis(col, expected):
    vals = sorted(df[col].dropna().unique().tolist())
    miss = sorted(set(expected) - set(vals))
    print(f"\n===== {col} =====")
    print(vals)
    print("missing:", miss)
    return vals, miss

metrics, missing_metrics = show_axis("metric", expected_metrics)
stress, missing_stress = show_axis("stress_definition", expected_stress)
seasons, missing_seasons = show_axis("growing_season", expected_seasons)

# Overall classes
class_col = "response_class_strict" if "response_class_strict" in df.columns else None
if class_col is None:
    candidates = [c for c in df.columns if "class" in c.lower()]
    class_col = candidates[0] if candidates else None

if class_col:
    overall_classes = df[class_col].value_counts(dropna=False).rename_axis("response_class").reset_index(name="n")
    overall_classes["fraction"] = overall_classes["n"] / len(df)
    overall_classes.to_csv(INS / "02_overall_response_class_counts.csv", index=False)
    print("\n===== OVERALL RESPONSE CLASSES =====")
    print(overall_classes.to_string(index=False))
else:
    print("\nWARNING: no response class column found")

# Primary uWUE
if class_col and "metric" in df.columns:
    u = df[df["metric"] == "uwue"].copy()
    u_classes = u[class_col].value_counts(dropna=False).rename_axis("response_class").reset_index(name="n")
    u_classes["fraction"] = u_classes["n"] / max(1, len(u))
    u_classes.to_csv(INS / "03_primary_uwue_response_class_counts.csv", index=False)

    print("\n===== PRIMARY uWUE RESPONSE CLASSES =====")
    print(u_classes.to_string(index=False))

    by_combo = (
        u.groupby(["gpp_product","et_product",class_col])
        .size()
        .reset_index(name="n")
        .sort_values(["gpp_product","et_product","n"], ascending=[True, True, False])
    )
    by_combo.to_csv(INS / "04_primary_uwue_class_by_combo_long.csv", index=False)

    by_combo_pivot = by_combo.pivot_table(
        index=["gpp_product","et_product"],
        columns=class_col,
        values="n",
        fill_value=0,
        aggfunc="sum"
    ).reset_index()
    by_combo_pivot.to_csv(INS / "05_primary_uwue_class_by_combo_pivot.csv", index=False)

    print("\n===== PRIMARY uWUE CLASS BY PRODUCT COMBO =====")
    print(by_combo_pivot.to_string(index=False))

    by_stress_season = (
        u.groupby(["stress_definition","growing_season",class_col])
        .size()
        .reset_index(name="n")
        .sort_values(["stress_definition","growing_season","n"], ascending=[True, True, False])
    )
    by_stress_season.to_csv(INS / "06_primary_uwue_class_by_stress_season.csv", index=False)

    # Product dependence score: saturation/breakdown fractions by combo.
    target_classes = ["saturation", "breakdown"]
    tmp = u.copy()
    tmp["sat_or_breakdown"] = tmp[class_col].isin(target_classes)
    tmp["breakdown_only"] = tmp[class_col].eq("breakdown")
    tmp["saturation_only"] = tmp[class_col].eq("saturation")
    product_signal = (
        tmp.groupby(["gpp_product","et_product"])
        .agg(
            n=("point_id","size"),
            points=("point_id","nunique"),
            sat_or_breakdown_frac=("sat_or_breakdown","mean"),
            breakdown_frac=("breakdown_only","mean"),
            saturation_frac=("saturation_only","mean"),
        )
        .reset_index()
        .sort_values("sat_or_breakdown_frac", ascending=False)
    )
    product_signal.to_csv(INS / "07_primary_uwue_product_signal_strength.csv", index=False)
    print("\n===== PRIMARY uWUE SATURATION/BREAKDOWN FRACTION BY COMBO =====")
    print(product_signal.to_string(index=False))

# Accepted transition summaries
accepted_cols = [c for c in df.columns if "accepted" in c.lower()]
print("\n===== ACCEPTED/TRANSITION COLUMNS =====")
print(accepted_cols)

if "accepted_transition" in df.columns:
    acc = (
        df.groupby(["metric","gpp_product","et_product"])
        .agg(
            n=("accepted_transition","size"),
            accepted=("accepted_transition","sum"),
            accepted_frac=("accepted_transition","mean"),
        )
        .reset_index()
        .sort_values(["metric","gpp_product","et_product"])
    )
    acc.to_csv(INS / "08_accepted_transition_by_metric_combo.csv", index=False)
    print("\n===== ACCEPTED TRANSITION BY METRIC × COMBO =====")
    print(acc.to_string(index=False))

# Numeric parameter summaries
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
interesting = [
    c for c in num_cols
    if any(k in c.lower() for k in ["slope", "break", "transition", "bic", "p_", "ci", "lower", "upper", "delta"])
]
print("\n===== INTERESTING NUMERIC COLUMNS =====")
print(interesting)

if interesting:
    param_summary = df[interesting].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).T
    param_summary.to_csv(INS / "09_numeric_parameter_summary.csv")
    print("\n===== NUMERIC PARAMETER SUMMARY =====")
    print(param_summary.to_string())

# Surface inspection
if surf_path.exists():
    surf = pd.read_csv(surf_path, low_memory=False)
    print("\n==============================")
    print("2D VPD × SOIL-MOISTURE SURFACE")
    print("==============================")
    print("file:", surf_path)
    print("shape:", surf.shape)
    print("columns:", list(surf.columns))

    s_num = surf.select_dtypes(include=[np.number]).columns.tolist()
    surface_cols = [
        c for c in s_num
        if any(k in c.lower() for k in ["vpd", "soil", "sm", "theta", "interact", "partial", "coef", "slope"])
    ]

    if {"gpp_product","et_product","metric"}.issubset(surf.columns) and surface_cols:
        surf_summary = (
            surf.groupby(["metric","gpp_product","et_product"])[surface_cols]
            .median()
            .reset_index()
            .sort_values(["metric","gpp_product","et_product"])
        )
        surf_summary.to_csv(INS / "10_surface_partial_effect_medians_by_combo.csv", index=False)
        print("\n===== SURFACE MEDIANS BY METRIC × COMBO =====")
        print(surf_summary.to_string(index=False))
else:
    print("\nWARNING: no surface file yet:", surf_path)

# Aridity inspection
if arid_path.exists():
    ar = pd.read_csv(arid_path, low_memory=False)
    print("\n==============================")
    print("ARIDITY SUMMARY")
    print("==============================")
    print("file:", arid_path)
    print("shape:", ar.shape)
    print("columns:", list(ar.columns))
    ar.head(30).to_csv(INS / "11_aridity_summary_head.csv", index=False)
    print(ar.head(30).to_string(index=False))
else:
    print("\nWARNING: no aridity summary file yet:", arid_path)

# Smoothness / sanity verdict
red_flags = []
warnings = []

if missing_combos:
    red_flags.append(f"Missing product combos: {missing_combos}")
if missing_metrics:
    red_flags.append(f"Missing metrics: {missing_metrics}")
if missing_stress:
    red_flags.append(f"Missing stress definitions: {missing_stress}")
if missing_seasons:
    red_flags.append(f"Missing growing seasons: {missing_seasons}")

if len(combo_set) == 9:
    modis_points = coverage[
        (coverage["gpp_product"].eq("modis")) | (coverage["et_product"].eq("modis"))
    ]["unique_points"].min()
    nonmodis_points = coverage[
        ~(coverage["gpp_product"].eq("modis")) & ~(coverage["et_product"].eq("modis"))
    ]["unique_points"].min()
    if modis_points < nonmodis_points:
        warnings.append(
            f"MODIS-containing combos have fewer QA-usable points: min {modis_points}, non-MODIS min {nonmodis_points}. Report this as a QA limitation."
        )
    if modis_points < 50:
        red_flags.append(f"MODIS-containing combos have very low point count: min {modis_points}")

if "accepted_transition" in df.columns:
    if df["accepted_transition"].sum() == 0:
        red_flags.append("No accepted transitions anywhere.")
    else:
        warnings.append(f"Accepted transitions exist: {int(df['accepted_transition'].sum())} / {len(df)}")

verdict = {
    "raw_result_file": str(raw_path),
    "shape": list(df.shape),
    "unique_points": int(df["point_id"].nunique()) if "point_id" in df.columns else None,
    "combo_count": int(len(combo_set)),
    "missing_combos": missing_combos,
    "missing_metrics": missing_metrics,
    "missing_stress_definitions": missing_stress,
    "missing_growing_seasons": missing_seasons,
    "red_flags": red_flags,
    "warnings": warnings,
}

with open(INS / "12_raw_boot50_inspection_verdict.json", "w") as f:
    json.dump(verdict, f, indent=2)

md = []
md.append("# BOOT50 Raw 3×3 Inspection Verdict\n")
md.append(f"- Result file: `{raw_path}`")
md.append(f"- Shape: `{df.shape}`")
md.append(f"- Unique points: `{verdict['unique_points']}`")
md.append(f"- Product combo count: `{len(combo_set)}`")
md.append(f"- Missing combos: `{missing_combos}`")
md.append(f"- Missing metrics: `{missing_metrics}`")
md.append(f"- Missing stress definitions: `{missing_stress}`")
md.append(f"- Missing growing seasons: `{missing_seasons}`")
md.append("")
md.append("## Red flags")
if red_flags:
    for x in red_flags:
        md.append(f"- {x}")
else:
    md.append("- None from structural inspection.")
md.append("")
md.append("## Warnings")
if warnings:
    for x in warnings:
        md.append(f"- {x}")
else:
    md.append("- None.")
md.append("")
md.append("## Interpretation rule")
md.append("Use this raw BOOT50 run to begin drafting methods and preliminary Gate 1/Gate 2 interpretation. Do not treat the paper's final numerical claims as locked until CO₂-corrected results and BOOT200 validation pass.")
(INS / "README_raw_boot50_inspection.md").write_text("\n".join(md))

print("\n==============================")
print("VERDICT")
print("==============================")
print(json.dumps(verdict, indent=2))
print("\nWROTE inspection files to:", INS)
print("Most important file:", INS / "README_raw_boot50_inspection.md")
