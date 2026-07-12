#!/usr/bin/env python

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path("results/trait_framework/phase2")
OUT.mkdir(parents=True, exist_ok=True)

RAW = Path("results/reza_final_nature_boot50/fullspec_response_results_raw.csv")
CO2 = Path("results/reza_final_nature_boot50/fullspec_response_results_co2corrected.csv")

CLASS_ORDER = [
    "enhancement",
    "enhancement_no_accepted_breakpoint",
    "saturation",
    "breakdown",
    "inconclusive",
]

CLASS_LABEL = {
    "enhancement": "Enhancement",
    "enhancement_no_accepted_breakpoint": "Enhancement, no accepted breakpoint",
    "saturation": "Saturation",
    "breakdown": "Breakdown",
    "inconclusive": "Inconclusive",
}


def load(path, version):
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "point_id",
        "metric",
        "gpp_product",
        "et_product",
        "response_class_strict",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing required columns: {missing}")

    df["version"] = version
    df["point_id"] = df["point_id"].astype(str)
    df["metric"] = df["metric"].astype(str).str.lower().str.strip()
    df["gpp_product"] = df["gpp_product"].astype(str).str.upper().str.strip()
    df["et_product"] = df["et_product"].astype(str).str.upper().str.strip()
    df["product_combo"] = df["gpp_product"] + "/" + df["et_product"]

    df["response_class_strict"] = df["response_class_strict"].astype(str).str.strip()
    df.loc[
        ~df["response_class_strict"].isin(CLASS_ORDER),
        "response_class_strict",
    ] = "inconclusive"

    if "stress_definition" in df.columns:
        df["stress_definition"] = df["stress_definition"].astype(str).str.strip()

    if "growing_season" in df.columns:
        df["growing_season"] = df["growing_season"].astype(str).str.strip()

    return df


def class_counts(df, group_cols):
    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))
        total = len(sub)

        for cls in CLASS_ORDER:
            n = int((sub["response_class_strict"] == cls).sum())

            row = dict(base)
            row.update(
                {
                    "response_class": cls,
                    "response_class_label": CLASS_LABEL[cls],
                    "n": n,
                    "total": total,
                    "fraction": n / total if total else np.nan,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def threshold_summary(df, group_cols):
    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))
        total = len(sub)

        n_enh = int((sub["response_class_strict"] == "enhancement").sum())
        n_no_bp = int(
            (sub["response_class_strict"] == "enhancement_no_accepted_breakpoint").sum()
        )
        n_sat = int((sub["response_class_strict"] == "saturation").sum())
        n_break = int((sub["response_class_strict"] == "breakdown").sum())
        n_inc = int((sub["response_class_strict"] == "inconclusive").sum())

        row = dict(base)
        row.update(
            {
                "n_total": total,
                "n_enhancement": n_enh,
                "n_enhancement_no_accepted_breakpoint": n_no_bp,
                "n_saturation": n_sat,
                "n_breakdown": n_break,
                "n_inconclusive": n_inc,
                "enhancement_fraction": n_enh / total if total else np.nan,
                "enhancement_no_accepted_breakpoint_fraction": n_no_bp / total if total else np.nan,
                "saturation_fraction": n_sat / total if total else np.nan,
                "breakdown_fraction": n_break / total if total else np.nan,
                "inconclusive_fraction": n_inc / total if total else np.nan,
                "sat_or_breakdown_fraction": (n_sat + n_break) / total if total else np.nan,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def clustered_bootstrap_threshold(df, group_cols, n_boot=100, seed=42):
    rng = np.random.default_rng(seed)

    point_ids = np.array(sorted(df["point_id"].astype(str).unique()))
    output = []

    if len(point_ids) == 0:
        return pd.DataFrame()

    point_groups = {
        pid: g.copy()
        for pid, g in df.groupby("point_id")
    }

    boot_rows = []

    for b in range(n_boot):
        sampled = rng.choice(point_ids, size=len(point_ids), replace=True)
        boot = pd.concat([point_groups[pid] for pid in sampled], ignore_index=True)

        ts = threshold_summary(boot, group_cols)
        ts["bootstrap"] = b
        boot_rows.append(ts)

    boot_df = pd.concat(boot_rows, ignore_index=True)

    for keys, sub in boot_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        for col in [
            "breakdown_fraction",
            "saturation_fraction",
            "sat_or_breakdown_fraction",
        ]:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[f"{col}_ci_low"] = float(vals.quantile(0.025)) if len(vals) else np.nan
            row[f"{col}_ci_high"] = float(vals.quantile(0.975)) if len(vals) else np.nan

        output.append(row)

    return pd.DataFrame(output)


def merge_ci(summary, ci, group_cols):
    if ci.empty:
        return summary

    return summary.merge(ci, on=group_cols, how="left")


def stacked_bar(table, index_col, out_png, out_pdf, title):
    piv = table.pivot_table(
        index=index_col,
        columns="response_class",
        values="fraction",
        aggfunc="first",
        fill_value=0,
    )

    for cls in CLASS_ORDER:
        if cls not in piv.columns:
            piv[cls] = 0.0

    piv = piv[CLASS_ORDER]
    piv = piv.sort_index()

    fig, ax = plt.subplots(figsize=(14, 7))

    bottom = np.zeros(len(piv))
    x = np.arange(len(piv))

    for cls in CLASS_ORDER:
        vals = piv[cls].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=CLASS_LABEL[cls])
        bottom += vals

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of fitted response classes")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in piv.index], rotation=45, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def safe_first_summary(df, group_cols):
    summary = threshold_summary(df, group_cols)

    if summary.empty:
        out = {c: None for c in group_cols}
        out.update(
            {
                "n_total": 0,
                "n_enhancement": 0,
                "n_enhancement_no_accepted_breakpoint": 0,
                "n_saturation": 0,
                "n_breakdown": 0,
                "n_inconclusive": 0,
                "enhancement_fraction": np.nan,
                "enhancement_no_accepted_breakpoint_fraction": np.nan,
                "saturation_fraction": np.nan,
                "breakdown_fraction": np.nan,
                "inconclusive_fraction": np.nan,
                "sat_or_breakdown_fraction": np.nan,
            }
        )
        return out

    return summary.iloc[0].to_dict()


def main():
    raw = load(RAW, "raw")
    co2 = load(CO2, "co2corrected")

    df = pd.concat([raw, co2], ignore_index=True)
    df.to_csv(OUT / "phase2_input_combined_raw_co2.csv", index=False)

    # Required output table 1: raw vs CO2 for uWUE.
    uwue = df[df["metric"] == "uwue"].copy()

    table_raw_vs_co2 = class_counts(uwue, ["version"])
    table_raw_vs_co2.to_csv(
        OUT / "table_response_class_counts_raw_vs_co2.csv",
        index=False,
    )

    # Required output table 2: by metric.
    table_by_metric = class_counts(df, ["version", "metric"])
    table_by_metric.to_csv(
        OUT / "table_response_class_counts_by_metric.csv",
        index=False,
    )

    # Required output table 3: by product combo.
    table_by_product_combo = class_counts(
        df,
        ["version", "metric", "gpp_product", "et_product", "product_combo"],
    )
    table_by_product_combo.to_csv(
        OUT / "table_response_class_counts_by_product_combo.csv",
        index=False,
    )

    # Extra robustness tables.
    if "stress_definition" in df.columns:
        table_by_stress = class_counts(df, ["version", "metric", "stress_definition"])
        table_by_stress.to_csv(
            OUT / "table_response_class_counts_by_stress_definition.csv",
            index=False,
        )

    if "growing_season" in df.columns:
        table_by_season = class_counts(df, ["version", "metric", "growing_season"])
        table_by_season.to_csv(
            OUT / "table_response_class_counts_by_growing_season.csv",
            index=False,
        )

    # Threshold summaries.
    threshold_raw_vs_co2 = threshold_summary(uwue, ["version"])
    ci_raw_vs_co2 = clustered_bootstrap_threshold(
        uwue,
        ["version"],
        n_boot=100,
        seed=42,
    )
    threshold_raw_vs_co2 = merge_ci(
        threshold_raw_vs_co2,
        ci_raw_vs_co2,
        ["version"],
    )
    threshold_raw_vs_co2.to_csv(
        OUT / "table_threshold_summary_raw_vs_co2_uwue.csv",
        index=False,
    )

    threshold_by_metric = threshold_summary(df, ["version", "metric"])
    ci_by_metric = clustered_bootstrap_threshold(
        df,
        ["version", "metric"],
        n_boot=100,
        seed=42,
    )
    threshold_by_metric = merge_ci(
        threshold_by_metric,
        ci_by_metric,
        ["version", "metric"],
    )
    threshold_by_metric.to_csv(
        OUT / "table_threshold_summary_by_metric.csv",
        index=False,
    )

    threshold_by_product = threshold_summary(
        df,
        ["version", "metric", "gpp_product", "et_product", "product_combo"],
    )
    ci_by_product = clustered_bootstrap_threshold(
        df,
        ["version", "metric", "gpp_product", "et_product", "product_combo"],
        n_boot=100,
        seed=42,
    )
    threshold_by_product = merge_ci(
        threshold_by_product,
        ci_by_product,
        ["version", "metric", "gpp_product", "et_product", "product_combo"],
    )
    threshold_by_product.to_csv(
        OUT / "table_response_threshold_summary.csv",
        index=False,
    )

    # Balanced subset: points with all 9 product combos in CO2 uWUE.
    co2_uwue = df[
        (df["version"] == "co2corrected")
        & (df["metric"] == "uwue")
    ].copy()

    combo_counts = (
        co2_uwue.groupby("point_id")[["gpp_product", "et_product"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_product_combos")
        .reset_index()
    )

    full_points = combo_counts.loc[
        combo_counts["n_product_combos"] == 9,
        "point_id",
    ].astype(str).tolist()

    balanced = df[df["point_id"].isin(full_points)].copy()

    combo_counts.to_csv(
        OUT / "table_product_combo_count_by_point_co2_uwue.csv",
        index=False,
    )

    balanced_manifest = {
        "points_with_all_9_product_combos_in_co2_uwue": len(full_points),
        "total_points": int(df["point_id"].nunique()),
        "balanced_rows": int(len(balanced)),
    }

    if len(balanced) > 0:
        balanced_class = class_counts(balanced, ["version", "metric"])
        balanced_class.to_csv(
            OUT / "table_response_class_counts_balanced_9combo_points.csv",
            index=False,
        )

        balanced_threshold = threshold_summary(balanced, ["version", "metric"])
        balanced_threshold.to_csv(
            OUT / "table_threshold_summary_balanced_9combo_points.csv",
            index=False,
        )

    with open(OUT / "balanced_subset_manifest.json", "w") as f:
        json.dump(balanced_manifest, f, indent=2)

    # Figure 1A: metric/version stacked bars.
    fig_metric = table_by_metric.copy()
    fig_metric["metric_version"] = fig_metric["version"] + " " + fig_metric["metric"]

    stacked_bar(
        fig_metric,
        "metric_version",
        OUT / "Figure1A_response_classes_by_metric_raw_vs_co2.png",
        OUT / "Figure1A_response_classes_by_metric_raw_vs_co2.pdf",
        "Figure 1A. Response classes across WUE metrics and CO2 treatments",
    )

    # Figure 1B: CO2 uWUE by product combo.
    fig_prod = table_by_product_combo[
        (table_by_product_combo["version"] == "co2corrected")
        & (table_by_product_combo["metric"] == "uwue")
    ].copy()

    stacked_bar(
        fig_prod,
        "product_combo",
        OUT / "Figure1B_response_classes_by_product_combo_co2_uwue.png",
        OUT / "Figure1B_response_classes_by_product_combo_co2_uwue.pdf",
        "Figure 1B. Response classes by product combination, CO2-corrected uWUE",
    )

    # Combined Figure 1 with two panels.
    piv_a = fig_metric.pivot_table(
        index="metric_version",
        columns="response_class",
        values="fraction",
        aggfunc="first",
        fill_value=0,
    )

    for cls in CLASS_ORDER:
        if cls not in piv_a.columns:
            piv_a[cls] = 0.0

    piv_a = piv_a[CLASS_ORDER].sort_index()

    piv_b = fig_prod.pivot_table(
        index="product_combo",
        columns="response_class",
        values="fraction",
        aggfunc="first",
        fill_value=0,
    )

    for cls in CLASS_ORDER:
        if cls not in piv_b.columns:
            piv_b[cls] = 0.0

    piv_b = piv_b[CLASS_ORDER].sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 13))

    for ax, piv, title in [
        (
            axes[0],
            piv_a,
            "A. Response classes across metrics and CO2 treatments",
        ),
        (
            axes[1],
            piv_b,
            "B. Response classes across product combinations, CO2-corrected uWUE",
        ),
    ]:
        bottom = np.zeros(len(piv))
        x = np.arange(len(piv))

        for cls in CLASS_ORDER:
            vals = piv[cls].to_numpy()
            ax.bar(x, vals, bottom=bottom, label=CLASS_LABEL[cls])
            bottom += vals

        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of fits")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in piv.index], rotation=45, ha="right")

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUT / "Figure1_response_class_stacked_bars.png", dpi=300)
    fig.savefig(OUT / "Figure1_response_class_stacked_bars.pdf")
    plt.close(fig)

    # Figure 1C: threshold-like fraction by product combo.
    prod_thr = threshold_by_product[
        (threshold_by_product["version"] == "co2corrected")
        & (threshold_by_product["metric"] == "uwue")
    ].copy()

    prod_thr = prod_thr.sort_values(
        "sat_or_breakdown_fraction",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(prod_thr))
    y = prod_thr["sat_or_breakdown_fraction"].to_numpy()

    ax.bar(x, y)

    ci_cols = {
        "sat_or_breakdown_fraction_ci_low",
        "sat_or_breakdown_fraction_ci_high",
    }

    if ci_cols.issubset(prod_thr.columns):
        low = prod_thr["sat_or_breakdown_fraction_ci_low"].to_numpy()
        high = prod_thr["sat_or_breakdown_fraction_ci_high"].to_numpy()

        yerr = np.vstack([
            np.maximum(0, y - low),
            np.maximum(0, high - y),
        ])

        ax.errorbar(x, y, yerr=yerr, fmt="none", capsize=3)

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(prod_thr["product_combo"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Saturation + breakdown fraction")
    ax.set_title("Figure 1C. Threshold-like response fraction by product combination")

    fig.tight_layout()
    fig.savefig(
        OUT / "Figure1C_threshold_like_fraction_by_product_combo.png",
        dpi=300,
    )
    fig.savefig(OUT / "Figure1C_threshold_like_fraction_by_product_combo.pdf")
    plt.close(fig)

    co2_uwue_threshold = safe_first_summary(co2_uwue, ["version"])
    raw_uwue_threshold = safe_first_summary(
        df[
            (df["version"] == "raw")
            & (df["metric"] == "uwue")
        ],
        ["version"],
    )

    phase2_claim_supported = (
        co2_uwue_threshold["breakdown_fraction"] < 0.5
        and co2_uwue_threshold["sat_or_breakdown_fraction"] < 0.5
        and raw_uwue_threshold["breakdown_fraction"] < 0.5
        and raw_uwue_threshold["sat_or_breakdown_fraction"] < 0.5
    )

    manifest = {
        "phase": "Phase 2: Test no universal threshold claim",
        "input_raw": str(RAW),
        "input_co2corrected": str(CO2),
        "raw_shape": list(raw.shape),
        "co2_shape": list(co2.shape),
        "combined_shape": list(df.shape),
        "unique_points": int(df["point_id"].nunique()),
        "metrics": sorted(df["metric"].dropna().unique().tolist()),
        "product_combos": sorted(df["product_combo"].dropna().unique().tolist()),
        "n_product_combos": int(
            df[["gpp_product", "et_product"]].drop_duplicates().shape[0]
        ),
        "co2_uwue_threshold_summary": co2_uwue_threshold,
        "raw_uwue_threshold_summary": raw_uwue_threshold,
        "balanced_subset": balanced_manifest,
        "phase2_claim_supported": bool(phase2_claim_supported),
        "claim_text": (
            "Grassland WUE does not exhibit a universal breakdown threshold; "
            "breakdown and saturation are minority response classes."
        ),
    }

    with open(OUT / "phase2_no_universal_threshold_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    readme = []
    readme.append("# Phase 2: No universal threshold")
    readme.append("")
    readme.append("## Claim tested")
    readme.append("")
    readme.append("Grassland WUE does not exhibit a universal breakdown threshold.")
    readme.append("")
    readme.append("## Primary decision rule")
    readme.append("")
    readme.append(
        "The claim is supported if breakdown and saturation+breakdown remain minority "
        "classes in both raw and CO2-corrected uWUE."
    )
    readme.append("")
    readme.append("## CO2-corrected uWUE threshold summary")
    readme.append("")
    readme.append(pd.DataFrame([co2_uwue_threshold]).to_string(index=False))
    readme.append("")
    readme.append("## Raw uWUE threshold summary")
    readme.append("")
    readme.append(pd.DataFrame([raw_uwue_threshold]).to_string(index=False))
    readme.append("")
    readme.append("## Verdict")
    readme.append("")

    if phase2_claim_supported:
        readme.append(
            "PHASE 2 CLAIM SUPPORTED: breakdown and saturation+breakdown are "
            "minority response classes."
        )
    else:
        readme.append(
            "PHASE 2 CLAIM NOT SUPPORTED under the current decision rule."
        )

    readme.append("")
    readme.append("## Interpretation")
    readme.append("")
    readme.append(
        "The correct scientific interpretation is response heterogeneity, not universal "
        "collapse. Binary breakdown should remain a secondary classification. The trait "
        "phase should use continuous response phenotypes such as post_slope, slope_change, "
        "and high-stress surface sensitivity."
    )
    readme.append("")
    readme.append("## Outputs")
    readme.append("")

    for p in sorted(OUT.glob("*")):
        readme.append(f"- `{p}`")

    Path(OUT / "README_phase2_no_universal_threshold.md").write_text(
        "\n".join(readme)
    )

    print("DONE Phase 2.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
