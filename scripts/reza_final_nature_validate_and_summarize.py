#!/usr/bin/env python
from pathlib import Path
import pandas as pd

OUT = Path("results/reza_final_nature")
OUT.mkdir(parents=True, exist_ok=True)

expected_metrics = {"uwue", "iwue", "raw_wue"}
expected_stress = {"zscore", "percentile_joint", "copula_joint", "interaction_surface"}
expected_seasons = {"gpp_threshold", "climate_common", "month_fixed"}

summary_lines = []
manifest_rows = []

required_files = [
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

for p in required_files:
    manifest_rows.append({"file": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0})

pd.DataFrame(manifest_rows).to_csv(OUT / "final_nature_manifest.csv", index=False)

failures = []
for row in manifest_rows:
    if not row["exists"] or row["size"] <= 0:
        failures.append(f"Missing/empty required file: {row['file']}")

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

    summary_lines.append(f"\n==== {version.upper()} ====")
    summary_lines.append(f"rows: {len(df)}")
    summary_lines.append(f"unique_points: {point_n}")
    summary_lines.append(f"product_combos: {combo_n}")
    summary_lines.append(f"metrics: {sorted(metrics)}")
    summary_lines.append(f"stress_definitions: {sorted(stress)}")
    summary_lines.append(f"growing_seasons: {sorted(seasons)}")
    summary_lines.append(f"accepted_transitions: {int(df['accepted_transition'].sum())} / {len(df)}")
    summary_lines.append("\nproduct combos:")
    summary_lines.append(combos.sort_values(["gpp_product", "et_product"]).to_string(index=False))
    summary_lines.append("\nresponse classes:")
    summary_lines.append(df["response_class_strict"].value_counts(dropna=False).to_string())

    if combo_n != 9:
        failures.append(f"{version}: expected 9 product combos, got {combo_n}")
    if point_n < 150:
        failures.append(f"{version}: expected >=150 points, got {point_n}")
    if not expected_metrics.issubset(metrics):
        failures.append(f"{version}: missing metrics {sorted(expected_metrics - metrics)}")
    if not expected_stress.issubset(stress):
        failures.append(f"{version}: missing stress definitions {sorted(expected_stress - stress)}")
    if not expected_seasons.issubset(seasons):
        failures.append(f"{version}: missing growing seasons {sorted(expected_seasons - seasons)}")

    u = df[df["metric"] == "uwue"].copy()
    if len(u):
        thesis = u[u["response_class_strict"].isin(["saturation", "breakdown"])]
        summary_lines.append("\nprimary uWUE class counts:")
        summary_lines.append(u["response_class_strict"].value_counts(dropna=False).to_string())
        summary_lines.append(f"primary_uWUE_saturation_breakdown: {len(thesis)} / {len(u)}")
        acc = (
            u.groupby(["gpp_product", "et_product"])["accepted_transition"]
            .agg(["sum", "count", "mean"])
            .reset_index()
            .sort_values(["gpp_product", "et_product"])
        )
        summary_lines.append("\nprimary uWUE accepted transition by combo:")
        summary_lines.append(acc.to_string(index=False))

        cls = (
            u.groupby(["gpp_product", "et_product", "response_class_strict"])
            .size()
            .reset_index(name="n")
            .sort_values(["gpp_product", "et_product", "n"], ascending=[True, True, False])
        )
        cls.to_csv(OUT / f"final_primary_uwue_class_by_combo_{version}.csv", index=False)

    combo_summary = (
        df.groupby(["version", "metric", "gpp_product", "et_product", "response_class_strict"])
        .size()
        .reset_index(name="n")
    )
    combo_summary.to_csv(OUT / f"final_response_class_by_combo_{version}.csv", index=False)

summary_text = "\n".join(summary_lines)
(OUT / "final_nature_scientific_summary.txt").write_text(summary_text)

if failures:
    (OUT / "FINAL_NATURE_RUN_STATUS.txt").write_text("FAILED\n" + "\n".join(failures) + "\n\n" + summary_text)
    print("FINAL NATURE VALIDATION FAILED")
    for f in failures:
        print("FAIL:", f)
    raise SystemExit(2)

(OUT / "FINAL_NATURE_RUN_STATUS.txt").write_text("PASSED\n\n" + summary_text)
print("FINAL NATURE VALIDATION PASSED")
print(summary_text)
