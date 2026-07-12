#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--label", required=True)
args = ap.parse_args()

OUT = Path(args.outdir)
OUT.mkdir(parents=True, exist_ok=True)

expected_metrics = {"uwue", "iwue", "raw_wue"}
expected_stress = {"zscore", "percentile_joint", "copula_joint", "interaction_surface"}
expected_seasons = {"gpp_threshold", "climate_common", "month_fixed"}

required = [
    OUT / "fullspec_response_results_raw.csv",
    OUT / "fullspec_response_results_co2corrected.csv",
    OUT / "fullspec_vpd_sm_surface_raw.csv",
    OUT / "fullspec_vpd_sm_surface_co2corrected.csv",
    OUT / "fullspec_aridity_summary_raw.csv",
    OUT / "fullspec_aridity_summary_co2corrected.csv",
    OUT / "tower_arbiter_status.csv",
    OUT / "hierarchical_trait_proxy_results.csv",
    OUT / "fullspec_implementation_manifest.csv",
    Path("data/external/trait_by_point.csv"),
    Path("docs/algorithm_dependency_table.csv"),
    Path("docs/trait_causal_dag.md"),
]

manifest = []
failures = []

for p in required:
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    manifest.append({"file": str(p), "exists": exists, "size": size})
    if not exists or size <= 0:
        failures.append(f"Missing/empty file: {p}")

pd.DataFrame(manifest).to_csv(OUT / f"final_nature_manifest_{args.label}.csv", index=False)

lines = []
for version in ["raw", "co2corrected"]:
    p = OUT / f"fullspec_response_results_{version}.csv"
    if not p.exists():
        continue

    df = pd.read_csv(p)
    combos = df[["gpp_product", "et_product"]].drop_duplicates()
    combo_n = combos.shape[0]
    point_n = df["point_id"].nunique()
    metrics = set(df["metric"].unique())
    stress = set(df["stress_definition"].unique())
    seasons = set(df["growing_season"].unique())

    lines.append(f"\n==== {args.label} {version.upper()} ====")
    lines.append(f"rows: {len(df)}")
    lines.append(f"unique_points: {point_n}")
    lines.append(f"product_combos: {combo_n}")
    lines.append(f"metrics: {sorted(metrics)}")
    lines.append(f"stress_definitions: {sorted(stress)}")
    lines.append(f"growing_seasons: {sorted(seasons)}")
    lines.append(f"accepted_transitions: {int(df['accepted_transition'].sum())} / {len(df)}")
    lines.append("\nproduct combos:")
    lines.append(combos.sort_values(["gpp_product", "et_product"]).to_string(index=False))
    lines.append("\nresponse classes:")
    lines.append(df["response_class_strict"].value_counts(dropna=False).to_string())

    if combo_n != 9:
        failures.append(f"{version}: expected 9 product combos, got {combo_n}")
    if point_n < 150:
        failures.append(f"{version}: expected >=150 points, got {point_n}")
    if not expected_metrics.issubset(metrics):
        failures.append(f"{version}: missing metrics {sorted(expected_metrics - metrics)}")
    if not expected_stress.issubset(stress):
        failures.append(f"{version}: missing stress definitions {sorted(expected_stress - stress)}")
    if not expected_seasons.issubset(seasons):
        failures.append(f"{version}: missing seasons {sorted(expected_seasons - seasons)}")

    u = df[df["metric"] == "uwue"].copy()
    if len(u):
        thesis = u[u["response_class_strict"].isin(["saturation", "breakdown"])]
        lines.append("\nprimary uWUE class counts:")
        lines.append(u["response_class_strict"].value_counts(dropna=False).to_string())
        lines.append(f"primary_uWUE_saturation_breakdown: {len(thesis)} / {len(u)}")

        cls = (
            u.groupby(["gpp_product", "et_product", "response_class_strict"])
            .size()
            .reset_index(name="n")
            .sort_values(["gpp_product", "et_product", "n"], ascending=[True, True, False])
        )
        cls.to_csv(OUT / f"final_primary_uwue_class_by_combo_{version}_{args.label}.csv", index=False)

        acc = (
            u.groupby(["gpp_product", "et_product"])["accepted_transition"]
            .agg(["sum", "count", "mean"])
            .reset_index()
            .sort_values(["gpp_product", "et_product"])
        )
        lines.append("\nprimary uWUE accepted transition by combo:")
        lines.append(acc.to_string(index=False))

summary = "\n".join(lines)
(OUT / f"final_nature_scientific_summary_{args.label}.txt").write_text(summary)

if failures:
    (OUT / f"FINAL_NATURE_RUN_STATUS_{args.label}.txt").write_text("FAILED\n" + "\n".join(failures) + "\n\n" + summary)
    print("FINAL NATURE VALIDATION FAILED", args.label)
    for f in failures:
        print("FAIL:", f)
    raise SystemExit(2)

(OUT / f"FINAL_NATURE_RUN_STATUS_{args.label}.txt").write_text("PASSED\n\n" + summary)
print("FINAL NATURE VALIDATION PASSED", args.label)
print(summary)
