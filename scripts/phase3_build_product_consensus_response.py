#!/usr/bin/env python
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE1_DETAIL = Path("results/trait_framework/point_response_phenotypes.csv")
PHASE1_CONSENSUS = Path("results/trait_framework/point_response_phenotypes_consensus_per_point.csv")
PHASE2_THRESHOLD = Path("results/trait_framework/phase2/table_response_threshold_summary.csv")

OUTDIR = Path("results/trait_framework/phase3")
OUTDIR.mkdir(parents=True, exist_ok=True)

MAIN_OUT = Path("results/trait_framework/point_product_consensus_response.csv")

GPP_ORDER = ["MODIS", "GOSIF", "PML"]
ET_ORDER = ["MODIS", "GLEAM", "PML"]

ALL_COMBOS = [
    "MODIS/MODIS",
    "MODIS/GLEAM",
    "MODIS/PML",
    "GOSIF/MODIS",
    "GOSIF/GLEAM",
    "GOSIF/PML",
    "PML/MODIS",
    "PML/GLEAM",
    "PML/PML",
]

SUBSETS = {
    "all": ALL_COMBOS,
    "independent": [
        "GOSIF/GLEAM",
        "GOSIF/MODIS",
        "MODIS/GLEAM",
    ],
    "pml_containing": [
        "MODIS/PML",
        "GOSIF/PML",
        "PML/MODIS",
        "PML/GLEAM",
        "PML/PML",
    ],
    "gosif_gpp": [
        "GOSIF/MODIS",
        "GOSIF/GLEAM",
        "GOSIF/PML",
    ],
    "gleam_et": [
        "MODIS/GLEAM",
        "GOSIF/GLEAM",
        "PML/GLEAM",
    ],
}

CLASS_SATBREAK = {"saturation", "breakdown"}

def die(msg):
    raise SystemExit("\nERROR: " + msg + "\n")

def parse_point_id(pid):
    s = str(pid).replace(",", "_").split("_")
    if len(s) < 2:
        return np.nan, np.nan
    try:
        lon = float(s[0])
        lat = float(s[1])
        return lon, lat
    except Exception:
        return np.nan, np.nan

def slug_combo(combo):
    return str(combo).lower().replace("/", "_").replace("-", "_").replace(" ", "_")

def safe_median(x):
    x = pd.to_numeric(x, errors="coerce")
    if x.notna().sum() == 0:
        return np.nan
    return float(x.median())

def safe_mean_bool(x):
    if len(x) == 0:
        return np.nan
    return float(pd.Series(x).astype(float).mean())

def sign_nonzero(x):
    if pd.isna(x):
        return np.nan
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def direction_agreement(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan

    signs = vals.apply(sign_nonzero)
    signs = signs[signs != 0]

    if len(signs) == 0:
        return 1.0

    frac_pos = float((signs > 0).mean())
    frac_neg = float((signs < 0).mean())
    return max(frac_pos, frac_neg)

def negative_fraction(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float((vals < 0).mean())

def positive_fraction(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float((vals > 0).mean())

def iqr(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.quantile(0.75) - vals.quantile(0.25))

def mad_to_consensus(vals):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    med = float(vals.median())
    return float((vals - med).abs().median())

def classify_product_family(combo):
    gpp, et = combo.split("/")
    return {
        "gpp_product": gpp,
        "et_product": et,
        "product_combo": combo,
        "is_pml_containing": ("PML" in [gpp, et]),
        "is_independent_subset": combo in SUBSETS["independent"],
        "is_gosif_gpp": gpp == "GOSIF",
        "is_gleam_et": et == "GLEAM",
    }

def main():
    if not PHASE1_DETAIL.exists():
        die(f"Missing Phase 1 detail table: {PHASE1_DETAIL}\nRun Phase 1 first.")

    df = pd.read_csv(PHASE1_DETAIL, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "point_id",
        "lat",
        "lon",
        "metric",
        "gpp_product",
        "et_product",
        "pre_slope",
        "post_slope",
        "slope_change",
        "response_class_strict",
        "raw_vs_co2_stability",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        die(f"Phase 1 detail table missing required columns: {missing}")

    df["point_id"] = df["point_id"].astype(str)
    df["metric"] = df["metric"].astype(str).str.lower().str.strip()
    df["gpp_product"] = df["gpp_product"].astype(str).str.upper().str.strip()
    df["et_product"] = df["et_product"].astype(str).str.upper().str.strip()
    df["product_combo"] = df["gpp_product"] + "/" + df["et_product"]
    df["response_class_strict"] = df["response_class_strict"].astype(str).str.strip()

    # Phase 3 is explicitly for CO2-corrected uWUE from Phase 1.
    df = df[df["metric"].eq("uwue")].copy()

    for c in [
        "lat",
        "lon",
        "pre_slope",
        "post_slope",
        "slope_change",
        "raw_vs_co2_stability",
        "high_stress_surface_sensitivity",
        "vpd_partial_effect",
        "sm_partial_effect",
        "vpd_sm_interaction",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["sat_or_breakdown"] = df["response_class_strict"].isin(CLASS_SATBREAK)
    df["breakdown"] = df["response_class_strict"].eq("breakdown")
    df["saturation"] = df["response_class_strict"].eq("saturation")
    df["negative_slope_change"] = pd.to_numeric(df["slope_change"], errors="coerce") < 0
    df["positive_slope_change"] = pd.to_numeric(df["slope_change"], errors="coerce") > 0

    # ---------------------------------------------------------------------
    # Step 1: collapse within each point × product_combo.
    # This prevents product combos with more rows from dominating.
    # ---------------------------------------------------------------------
    agg = {
        "lat": ("lat", "first"),
        "lon": ("lon", "first"),
        "n_rows_product_combo": ("point_id", "size"),
        "median_pre_slope": ("pre_slope", safe_median),
        "median_post_slope": ("post_slope", safe_median),
        "median_slope_change": ("slope_change", safe_median),
        "slope_change_iqr": ("slope_change", iqr),
        "negative_slope_fraction_within_combo": ("negative_slope_change", "mean"),
        "positive_slope_fraction_within_combo": ("positive_slope_change", "mean"),
        "satbreak_fraction_within_combo": ("sat_or_breakdown", "mean"),
        "breakdown_fraction_within_combo": ("breakdown", "mean"),
        "saturation_fraction_within_combo": ("saturation", "mean"),
        "median_raw_vs_co2_stability": ("raw_vs_co2_stability", safe_median),
    }

    optional_cols = [
        "high_stress_surface_sensitivity",
        "vpd_partial_effect",
        "sm_partial_effect",
        "vpd_sm_interaction",
    ]
    for c in optional_cols:
        if c in df.columns:
            agg[f"median_{c}"] = (c, safe_median)

    combo_level = (
        df.groupby(
            ["point_id", "metric", "gpp_product", "et_product", "product_combo"],
            dropna=False,
        )
        .agg(**agg)
        .reset_index()
    )

    combo_level["product_slope_sign"] = combo_level["median_slope_change"].apply(sign_nonzero)
    combo_level["product_has_negative_slope_change"] = combo_level["median_slope_change"] < 0
    combo_level["product_has_positive_slope_change"] = combo_level["median_slope_change"] > 0
    combo_level["product_satbreak_any_signal"] = combo_level["satbreak_fraction_within_combo"] > 0

    fam_rows = []
    for combo in sorted(combo_level["product_combo"].dropna().unique()):
        fam_rows.append(classify_product_family(combo))
    product_family_table = pd.DataFrame(fam_rows)

    combo_level = combo_level.merge(
        product_family_table,
        on=["gpp_product", "et_product", "product_combo"],
        how="left",
    )

    combo_level.to_csv(OUTDIR / "point_product_combo_level_response.csv", index=False)
    product_family_table.to_csv(OUTDIR / "product_family_table.csv", index=False)

    # ---------------------------------------------------------------------
    # Step 2: subset-level consensus per point.
    # ---------------------------------------------------------------------
    point_base = (
        combo_level.groupby("point_id", dropna=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"), metric=("metric", "first"))
        .reset_index()
    )

    consensus = point_base.copy()

    subset_diagnostics = []

    for subset_name, subset_combos in SUBSETS.items():
        sub = combo_level[combo_level["product_combo"].isin(subset_combos)].copy()

        if sub.empty:
            # Still create empty columns.
            for col in [
                f"consensus_slope_change_{subset_name}",
                f"consensus_post_slope_{subset_name}",
                f"consensus_pre_slope_{subset_name}",
                f"product_agreement_{subset_name}",
                f"negative_slope_fraction_{subset_name}",
                f"positive_slope_fraction_{subset_name}",
                f"satbreak_fraction_{subset_name}",
                f"breakdown_fraction_{subset_name}",
                f"saturation_fraction_{subset_name}",
                f"raw_vs_co2_stability_{subset_name}",
                f"n_product_combos_{subset_name}",
                f"n_rows_{subset_name}",
                f"slope_change_iqr_{subset_name}",
                f"slope_change_mad_to_consensus_{subset_name}",
            ]:
                consensus[col] = np.nan
            continue

        out = (
            sub.groupby("point_id", dropna=False)
            .agg(
                **{
                    f"consensus_slope_change_{subset_name}": ("median_slope_change", safe_median),
                    f"consensus_post_slope_{subset_name}": ("median_post_slope", safe_median),
                    f"consensus_pre_slope_{subset_name}": ("median_pre_slope", safe_median),
                    f"product_agreement_{subset_name}": ("median_slope_change", direction_agreement),
                    f"negative_slope_fraction_{subset_name}": ("median_slope_change", negative_fraction),
                    f"positive_slope_fraction_{subset_name}": ("median_slope_change", positive_fraction),
                    f"satbreak_fraction_{subset_name}": ("satbreak_fraction_within_combo", safe_median),
                    f"satbreak_fraction_mean_{subset_name}": ("satbreak_fraction_within_combo", "mean"),
                    f"breakdown_fraction_{subset_name}": ("breakdown_fraction_within_combo", "mean"),
                    f"saturation_fraction_{subset_name}": ("saturation_fraction_within_combo", "mean"),
                    f"raw_vs_co2_stability_{subset_name}": ("median_raw_vs_co2_stability", safe_median),
                    f"n_product_combos_{subset_name}": ("product_combo", "nunique"),
                    f"n_rows_{subset_name}": ("n_rows_product_combo", "sum"),
                    f"slope_change_iqr_{subset_name}": ("median_slope_change", iqr),
                    f"slope_change_mad_to_consensus_{subset_name}": ("median_slope_change", mad_to_consensus),
                }
            )
            .reset_index()
        )

        # Optional surface fields, if present.
        optional_combo_cols = [
            c for c in combo_level.columns
            if c.startswith("median_high_stress_surface_sensitivity")
            or c.startswith("median_vpd_partial_effect")
            or c.startswith("median_sm_partial_effect")
            or c.startswith("median_vpd_sm_interaction")
        ]

        if optional_combo_cols:
            surf = sub.groupby("point_id", dropna=False)[optional_combo_cols].median(numeric_only=True).reset_index()
            rename = {}
            for c in optional_combo_cols:
                rename[c] = f"{c}_{subset_name}"
            surf = surf.rename(columns=rename)
            out = out.merge(surf, on="point_id", how="left")

        consensus = consensus.merge(out, on="point_id", how="left")

        subset_diagnostics.append({
            "subset": subset_name,
            "requested_combos": ",".join(subset_combos),
            "available_rows": int(len(sub)),
            "points_with_at_least_one_combo": int(sub["point_id"].nunique()),
            "median_n_product_combos_per_point": float(
                sub.groupby("point_id")["product_combo"].nunique().median()
            ),
            "max_n_product_combos_per_point": int(
                sub.groupby("point_id")["product_combo"].nunique().max()
            ),
            "expected_n_combos": len(subset_combos),
        })

    subset_diagnostics = pd.DataFrame(subset_diagnostics)
    subset_diagnostics.to_csv(OUTDIR / "subset_diagnostics.csv", index=False)

    # ---------------------------------------------------------------------
    # Step 3: product-specific sensitivity variables.
    # ---------------------------------------------------------------------
    product_wide = combo_level[[
        "point_id",
        "product_combo",
        "median_slope_change",
        "median_post_slope",
        "median_pre_slope",
        "satbreak_fraction_within_combo",
        "breakdown_fraction_within_combo",
        "saturation_fraction_within_combo",
        "median_raw_vs_co2_stability",
        "negative_slope_fraction_within_combo",
        "positive_slope_fraction_within_combo",
    ]].copy()

    wide_parts = []
    for combo in ALL_COMBOS:
        sub = product_wide[product_wide["product_combo"].eq(combo)].copy()
        slug = slug_combo(combo)
        rename = {
            "median_slope_change": f"slope_change_{slug}",
            "median_post_slope": f"post_slope_{slug}",
            "median_pre_slope": f"pre_slope_{slug}",
            "satbreak_fraction_within_combo": f"satbreak_fraction_{slug}",
            "breakdown_fraction_within_combo": f"breakdown_fraction_{slug}",
            "saturation_fraction_within_combo": f"saturation_fraction_{slug}",
            "median_raw_vs_co2_stability": f"raw_vs_co2_stability_{slug}",
            "negative_slope_fraction_within_combo": f"negative_slope_fraction_{slug}",
            "positive_slope_fraction_within_combo": f"positive_slope_fraction_{slug}",
        }
        keep = ["point_id"] + list(rename.keys())
        sub = sub[keep].rename(columns=rename)
        wide_parts.append(sub)

    product_specific = point_base[["point_id"]].copy()
    for part in wide_parts:
        product_specific = product_specific.merge(part, on="point_id", how="left")

    product_specific.to_csv(OUTDIR / "product_specific_sensitivity_variables.csv", index=False)

    consensus = consensus.merge(product_specific, on="point_id", how="left")

    # ---------------------------------------------------------------------
    # Step 4: extra agreement/disagreement variables.
    # ---------------------------------------------------------------------
    # Deviation of each product combo from all-product consensus.
    for combo in ALL_COMBOS:
        slug = slug_combo(combo)
        c = f"slope_change_{slug}"
        if c in consensus.columns:
            consensus[f"slope_change_deviation_from_all_{slug}"] = (
                consensus[c] - consensus["consensus_slope_change_all"]
            )
            consensus[f"abs_slope_change_deviation_from_all_{slug}"] = (
                consensus[f"slope_change_deviation_from_all_{slug}"].abs()
            )

    # Agreement of independent vs all and PML vs independent.
    if "consensus_slope_change_independent" in consensus.columns:
        consensus["independent_minus_all_slope_change"] = (
            consensus["consensus_slope_change_independent"] - consensus["consensus_slope_change_all"]
        )
        consensus["independent_all_same_sign"] = (
            np.sign(consensus["consensus_slope_change_independent"]) ==
            np.sign(consensus["consensus_slope_change_all"])
        )

    if "consensus_slope_change_pml_containing" in consensus.columns and "consensus_slope_change_independent" in consensus.columns:
        consensus["pml_minus_independent_slope_change"] = (
            consensus["consensus_slope_change_pml_containing"] -
            consensus["consensus_slope_change_independent"]
        )
        consensus["pml_independent_same_sign"] = (
            np.sign(consensus["consensus_slope_change_pml_containing"]) ==
            np.sign(consensus["consensus_slope_change_independent"])
        )

    # Label complete 9-product support.
    consensus["has_all_9_product_combos"] = consensus["n_product_combos_all"].eq(9)
    consensus["has_independent_subset_complete"] = consensus["n_product_combos_independent"].eq(len(SUBSETS["independent"]))
    consensus["has_pml_containing_subset_complete"] = consensus["n_product_combos_pml_containing"].eq(len(SUBSETS["pml_containing"]))
    consensus["has_gosif_gpp_subset_complete"] = consensus["n_product_combos_gosif_gpp"].eq(len(SUBSETS["gosif_gpp"]))
    consensus["has_gleam_et_subset_complete"] = consensus["n_product_combos_gleam_et"].eq(len(SUBSETS["gleam_et"]))

    # Reorder core columns.
    core_cols = [
        "point_id",
        "lat",
        "lon",
        "metric",

        "consensus_slope_change_all",
        "consensus_post_slope_all",
        "consensus_pre_slope_all",
        "product_agreement_all",
        "negative_slope_fraction_all",
        "positive_slope_fraction_all",
        "satbreak_fraction_all",
        "satbreak_fraction_mean_all",
        "breakdown_fraction_all",
        "saturation_fraction_all",
        "raw_vs_co2_stability_all",
        "n_product_combos_all",
        "has_all_9_product_combos",

        "consensus_slope_change_independent",
        "consensus_post_slope_independent",
        "consensus_pre_slope_independent",
        "product_agreement_independent",
        "negative_slope_fraction_independent",
        "positive_slope_fraction_independent",
        "satbreak_fraction_independent",
        "satbreak_fraction_mean_independent",
        "breakdown_fraction_independent",
        "saturation_fraction_independent",
        "raw_vs_co2_stability_independent",
        "n_product_combos_independent",
        "has_independent_subset_complete",

        "consensus_slope_change_pml_containing",
        "consensus_post_slope_pml_containing",
        "product_agreement_pml_containing",
        "negative_slope_fraction_pml_containing",
        "satbreak_fraction_pml_containing",
        "raw_vs_co2_stability_pml_containing",
        "n_product_combos_pml_containing",
        "has_pml_containing_subset_complete",

        "consensus_slope_change_gosif_gpp",
        "consensus_post_slope_gosif_gpp",
        "product_agreement_gosif_gpp",
        "negative_slope_fraction_gosif_gpp",
        "satbreak_fraction_gosif_gpp",
        "raw_vs_co2_stability_gosif_gpp",
        "n_product_combos_gosif_gpp",
        "has_gosif_gpp_subset_complete",

        "consensus_slope_change_gleam_et",
        "consensus_post_slope_gleam_et",
        "product_agreement_gleam_et",
        "negative_slope_fraction_gleam_et",
        "satbreak_fraction_gleam_et",
        "raw_vs_co2_stability_gleam_et",
        "n_product_combos_gleam_et",
        "has_gleam_et_subset_complete",

        "independent_minus_all_slope_change",
        "independent_all_same_sign",
        "pml_minus_independent_slope_change",
        "pml_independent_same_sign",
    ]

    core_cols = [c for c in core_cols if c in consensus.columns]
    other_cols = [c for c in consensus.columns if c not in core_cols]
    consensus = consensus[core_cols + other_cols]

    MAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    consensus.to_csv(MAIN_OUT, index=False)

    # ---------------------------------------------------------------------
    # Step 5: Figure 2 heatmaps.
    # ---------------------------------------------------------------------
    # Product-level summary across point × product_combo.
    product_summary = (
        combo_level.groupby(["gpp_product", "et_product", "product_combo"], dropna=False)
        .agg(
            n_points=("point_id", "nunique"),
            n_point_product_rows=("point_id", "size"),
            satbreak_fraction=("satbreak_fraction_within_combo", "mean"),
            breakdown_fraction=("breakdown_fraction_within_combo", "mean"),
            saturation_fraction=("saturation_fraction_within_combo", "mean"),
            median_slope_change=("median_slope_change", "median"),
            median_post_slope=("median_post_slope", "median"),
            raw_vs_co2_stability=("median_raw_vs_co2_stability", "median"),
            product_negative_slope_fraction=("product_has_negative_slope_change", "mean"),
        )
        .reset_index()
    )

    # Product disagreement = median absolute deviation from all-product point consensus.
    tmp = combo_level.merge(
        consensus[["point_id", "consensus_slope_change_all"]],
        on="point_id",
        how="left",
    )
    tmp["abs_deviation_from_all_consensus"] = (
        tmp["median_slope_change"] - tmp["consensus_slope_change_all"]
    ).abs()

    disagreement = (
        tmp.groupby(["gpp_product", "et_product", "product_combo"], dropna=False)
        .agg(
            median_abs_deviation_from_all_consensus=("abs_deviation_from_all_consensus", "median"),
            mean_abs_deviation_from_all_consensus=("abs_deviation_from_all_consensus", "mean"),
        )
        .reset_index()
    )

    product_summary = product_summary.merge(
        disagreement,
        on=["gpp_product", "et_product", "product_combo"],
        how="left",
    )

    product_summary.to_csv(OUTDIR / "product_matrix_summary_for_figure2.csv", index=False)

    def matrix_from_product_summary(value_col):
        mat = pd.DataFrame(index=GPP_ORDER, columns=ET_ORDER, dtype=float)
        for _, r in product_summary.iterrows():
            g = r["gpp_product"]
            e = r["et_product"]
            if g in GPP_ORDER and e in ET_ORDER:
                mat.loc[g, e] = r[value_col]
        return mat

    mat_sat = matrix_from_product_summary("satbreak_fraction")
    mat_stab = matrix_from_product_summary("raw_vs_co2_stability")
    mat_dis = matrix_from_product_summary("median_abs_deviation_from_all_consensus")

    def annotate_heatmap(ax, mat, title, fmt="{:.2f}"):
        im = ax.imshow(mat.values.astype(float), aspect="auto")
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("ET product")
        ax.set_ylabel("GPP product")
        ax.set_title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, fmt.format(val), ha="center", va="center")
                else:
                    ax.text(j, i, "NA", ha="center", va="center")
        return im

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

    im0 = annotate_heatmap(
        axes[0],
        mat_sat,
        "A. Saturation/breakdown fraction",
        fmt="{:.2f}",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = annotate_heatmap(
        axes[1],
        mat_stab,
        "B. Raw-vs-CO2 stability",
        fmt="{:.2f}",
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = annotate_heatmap(
        axes[2],
        mat_dis,
        "C. Product disagreement\n(abs. deviation from consensus)",
        fmt="{:.2f}",
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Figure 2. Product-matrix consensus diagnostics for CO2-corrected uWUE",
        fontsize=14,
    )
    fig.savefig(OUTDIR / "Figure2_product_matrix_consensus_diagnostics.png", dpi=300)
    fig.savefig(OUTDIR / "Figure2_product_matrix_consensus_diagnostics.pdf")
    plt.close(fig)

    # Additional Figure 2 individual panels.
    for name, mat, title in [
        ("Figure2A_satbreak_fraction_heatmap", mat_sat, "Saturation/breakdown fraction"),
        ("Figure2B_raw_vs_co2_stability_heatmap", mat_stab, "Raw-vs-CO2 stability"),
        ("Figure2C_product_disagreement_heatmap", mat_dis, "Product disagreement"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        im = annotate_heatmap(ax, mat, title, fmt="{:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(OUTDIR / f"{name}.png", dpi=300)
        fig.savefig(OUTDIR / f"{name}.pdf")
        plt.close(fig)

    # ---------------------------------------------------------------------
    # Step 6: summary / manifest.
    # ---------------------------------------------------------------------
    def finite_count(col):
        if col not in consensus.columns:
            return None
        return int(pd.to_numeric(consensus[col], errors="coerce").notna().sum())

    manifest = {
        "phase": "Phase 3: Build product-consensus response phenotypes",
        "input_phase1_detail": str(PHASE1_DETAIL),
        "output_main": str(MAIN_OUT),
        "output_combo_level": str(OUTDIR / "point_product_combo_level_response.csv"),
        "output_product_specific": str(OUTDIR / "product_specific_sensitivity_variables.csv"),
        "output_figure2": str(OUTDIR / "Figure2_product_matrix_consensus_diagnostics.png"),
        "n_points": int(consensus["point_id"].nunique()),
        "main_shape": list(consensus.shape),
        "combo_level_shape": list(combo_level.shape),
        "product_combos_present": sorted(combo_level["product_combo"].dropna().unique().tolist()),
        "n_product_combos_present": int(combo_level["product_combo"].nunique()),
        "subset_diagnostics": subset_diagnostics.to_dict(orient="records"),
        "finite_core_counts": {
            "consensus_slope_change_all": finite_count("consensus_slope_change_all"),
            "consensus_post_slope_all": finite_count("consensus_post_slope_all"),
            "product_agreement_all": finite_count("product_agreement_all"),
            "satbreak_fraction_all": finite_count("satbreak_fraction_all"),
            "consensus_slope_change_independent": finite_count("consensus_slope_change_independent"),
            "consensus_post_slope_independent": finite_count("consensus_post_slope_independent"),
            "product_agreement_independent": finite_count("product_agreement_independent"),
            "satbreak_fraction_independent": finite_count("satbreak_fraction_independent"),
        },
        "complete_subset_counts": {
            "has_all_9_product_combos": int(consensus["has_all_9_product_combos"].sum()),
            "has_independent_subset_complete": int(consensus["has_independent_subset_complete"].sum()),
            "has_pml_containing_subset_complete": int(consensus["has_pml_containing_subset_complete"].sum()),
            "has_gosif_gpp_subset_complete": int(consensus["has_gosif_gpp_subset_complete"].sum()),
            "has_gleam_et_subset_complete": int(consensus["has_gleam_et_subset_complete"].sum()),
        },
        "product_matrix_summary": product_summary.to_dict(orient="records"),
    }

    with open(OUTDIR / "phase3_product_consensus_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    readme = []
    readme.append("# Phase 3: Product-consensus response phenotypes")
    readme.append("")
    readme.append("## Goal")
    readme.append("")
    readme.append("Avoid relying on one unvalidated product combination by creating product-consensus response phenotypes.")
    readme.append("")
    readme.append("## Main output")
    readme.append("")
    readme.append(f"- `{MAIN_OUT}`")
    readme.append("")
    readme.append("## Other outputs")
    readme.append("")
    for p in [
        OUTDIR / "point_product_combo_level_response.csv",
        OUTDIR / "product_specific_sensitivity_variables.csv",
        OUTDIR / "product_matrix_summary_for_figure2.csv",
        OUTDIR / "subset_diagnostics.csv",
        OUTDIR / "Figure2_product_matrix_consensus_diagnostics.png",
        OUTDIR / "Figure2_product_matrix_consensus_diagnostics.pdf",
        OUTDIR / "phase3_product_consensus_manifest.json",
    ]:
        readme.append(f"- `{p}`")
    readme.append("")
    readme.append("## Product subsets")
    readme.append("")
    for name, combos in SUBSETS.items():
        readme.append(f"- `{name}`: {', '.join(combos)}")
    readme.append("")
    readme.append("## Interpretation")
    readme.append("")
    readme.append("The primary trait-model response should be one of the consensus response variables, not a single product combination.")
    readme.append("")
    readme.append("Recommended primary outcomes for the next phase:")
    readme.append("")
    readme.append("- `consensus_slope_change_independent`")
    readme.append("- `consensus_post_slope_independent`")
    readme.append("- `consensus_slope_change_all`")
    readme.append("- `satbreak_fraction_all`")
    readme.append("- `product_agreement_all` as an uncertainty/robustness variable")
    readme.append("")
    readme.append("## Manifest")
    readme.append("")
    readme.append(json.dumps(manifest, indent=2))

    Path(OUTDIR / "README_phase3_product_consensus.md").write_text("\n".join(readme))

    print("DONE Phase 3.")
    print("")
    print(f"WROTE {MAIN_OUT}")
    print(f"WROTE {OUTDIR / 'point_product_combo_level_response.csv'}")
    print(f"WROTE {OUTDIR / 'product_specific_sensitivity_variables.csv'}")
    print(f"WROTE {OUTDIR / 'product_matrix_summary_for_figure2.csv'}")
    print(f"WROTE {OUTDIR / 'Figure2_product_matrix_consensus_diagnostics.png'}")
    print(f"WROTE {OUTDIR / 'phase3_product_consensus_manifest.json'}")
    print("")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
