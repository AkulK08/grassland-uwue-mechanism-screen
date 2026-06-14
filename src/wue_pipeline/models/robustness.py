"""Gate 2 robustness scoring."""

from __future__ import annotations
import pandas as pd
import numpy as np


def response_fraction_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    classes = ["enhancement", "saturation", "reversal", "inconclusive", "insufficient_data"]
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        n = len(g)
        row["n"] = n
        for c in classes:
            row[f"frac_{c}"] = float((g["response_class"] == c).mean()) if n else np.nan
        row["dominant_class"] = g["response_class"].value_counts().idxmax() if n else "none"
        row["median_pre_slope"] = float(g["pre_slope"].median())
        row["median_post_slope"] = float(g["post_slope"].median())
        row["median_slope_change"] = float(g["slope_change"].median())
        row["median_breakpoint"] = float(g["breakpoint"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_gate2_success(summary: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the success criterion: same shape across >=2 GPP, >=2 ET, >=2 stress, >=2 GS."""
    rows = []
    for response_class, g in summary.groupby("dominant_class"):
        if response_class in ["inconclusive", "insufficient_data", "none"]:
            continue
        rows.append({
            "response_class": response_class,
            "n_gpp_products": g["gpp_product"].nunique(),
            "n_et_products": g["et_product"].nunique(),
            "n_stress_definitions": g["stress_definition"].nunique(),
            "n_growing_seasons": g["growing_season"].nunique(),
            "passes_gate2": bool(
                g["gpp_product"].nunique() >= 2 and
                g["et_product"].nunique() >= 2 and
                g["stress_definition"].nunique() >= 2 and
                g["growing_season"].nunique() >= 2
            ),
        })
    if not rows:
        return pd.DataFrame(columns=["response_class", "n_gpp_products", "n_et_products", "n_stress_definitions", "n_growing_seasons", "passes_gate2"])
    return pd.DataFrame(rows)


def product_sensitivity_iqr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["pre_slope", "post_slope", "slope_change", "breakpoint"]:
        rows.append({
            "metric": metric,
            "iqr_across_product_matrix": float(df.groupby(["gpp_product", "et_product"])[metric].median().quantile(0.75) - df.groupby(["gpp_product", "et_product"])[metric].median().quantile(0.25)),
            "global_median": float(df[metric].median()),
        })
    return pd.DataFrame(rows)
