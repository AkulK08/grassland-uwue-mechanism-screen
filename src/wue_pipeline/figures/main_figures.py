"""Manuscript figure generation."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..config import ProjectConfig


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def figure1_conceptual(cfg: ProjectConfig):
    x = np.linspace(-2, 4, 300)
    y_enh = 0.12 * x
    y_sat = 0.16 * x - 0.12 * np.maximum(x - 1.3, 0)
    y_rev = 0.18 * x - 0.32 * np.maximum(x - 1.3, 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y_enh, label="Enhancement")
    ax.plot(x, y_sat, label="Saturation")
    ax.plot(x, y_rev, label="Reversal / breakdown")
    ax.axvline(1.3, linestyle="--", linewidth=1)
    ax.set_xlabel("Compound atmospheric-soil moisture stress")
    ax.set_ylabel("log(WUE)")
    ax.set_title("Operational response-shape classes")
    ax.legend(frameon=False)
    _save(fig, cfg.file("figures", "figure1_conceptual_response.png"))


def figure2_gate1(cfg: ProjectConfig):
    path = cfg.file("tables", "gate1_pixel_results.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["enhancement", "saturation", "reversal", "inconclusive", "insufficient_data"]
    counts = df["response_class"].value_counts().reindex(order).fillna(0)
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Pilot pixels")
    ax.set_title("Gate 1 response classifications")
    ax.tick_params(axis="x", rotation=35)
    _save(fig, cfg.file("figures", "figure2_gate1_response.png"))


def figure3_product_matrix(cfg: ProjectConfig):
    path = cfg.file("tables", "gate2_robustness_matrix.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    # Focus on zscore/gpp_threshold pooled over aridity for clean matrix.
    d = df[(df["stress_definition"] == "zscore") & (df["growing_season"] == "gpp_threshold")]
    pivot = d.groupby(["gpp_product", "et_product"])["frac_reversal"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("ET product")
    ax.set_ylabel("GPP product")
    ax.set_title("Gate 2 product matrix: reversal fraction")
    fig.colorbar(im, ax=ax, label="Fraction reversal")
    _save(fig, cfg.file("figures", "figure3_product_matrix.png"))


def figure4_tower_validation(cfg: ProjectConfig):
    path = cfg.file("tables", "gate3_tower_validation.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    if len(df) == 0:
        return
    conc = df.groupby(["gpp_product", "et_product"])["concordant"].mean().reset_index()
    conc["label"] = conc["gpp_product"] + " / " + conc["et_product"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(conc["label"], conc["concordant"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tower-satellite concordance")
    ax.set_title("Gate 3 validation by product family")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, cfg.file("figures", "figure4_tower_validation.png"))


def figure5_traits(cfg: ProjectConfig):
    path = cfg.file("tables", "phase4_shap_importance.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "mean_abs_shap" not in df:
        return
    df = df.sort_values("mean_abs_shap", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["variable"], df["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_title("Phase 4 trait/climate predictor importance")
    _save(fig, cfg.file("figures", "figure5_trait_importance.png"))


def generate_all_figures(cfg: ProjectConfig):
    figure1_conceptual(cfg)
    figure2_gate1(cfg)
    figure3_product_matrix(cfg)
    figure4_tower_validation(cfg)
    figure5_traits(cfg)
