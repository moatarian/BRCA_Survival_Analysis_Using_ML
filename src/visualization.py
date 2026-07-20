"""
visualization.py
=================
Publication-quality plotting helpers shared across all notebooks. A single,
validated color system (adapted from the project's data-viz style guide) is
applied consistently: a fixed-order 8-hue categorical palette, a one-hue blue
sequential ramp for magnitude, and a blue<->red diverging pair for
signed/SHAP-style contributions. Every figure is saved as both PNG (300 dpi)
and PDF (vector) into ``figures/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import PATHS

# --------------------------------------------------------------------------- #
# Color system (validated categorical / sequential / diverging palette)
# --------------------------------------------------------------------------- #
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_BLUE_RED = "RdBu_r"  # red=positive/high, blue=negative/low -- matches SHAP convention when reversed
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

SURGERY_COLORS = {"Mastectomy": CATEGORICAL[0], "BCS": CATEGORICAL[4]}
CLASS_COLORS = {"Living": CATEGORICAL[1], "Deceased": CATEGORICAL[7]}


def set_publication_style() -> None:
    """Apply a consistent, publication-quality matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_PRIMARY,
        "axes.titlecolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def save_figure(fig: plt.Figure, name: str, subdir: Optional[str] = None) -> None:
    """Save a matplotlib figure as both PNG and PDF into figures/[subdir]/."""
    out_dir = PATHS.figures / subdir if subdir else PATHS.figures
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def annotate_source(ax: plt.Axes, text: str = "Source: METABRIC (cBioPortal), reproduction analysis") -> None:
    ax.text(1.0, -0.12, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK_MUTED, style="italic")


# --------------------------------------------------------------------------- #
# Generic reusable chart builders
# --------------------------------------------------------------------------- #
def plot_missingness_heatmap(df: pd.DataFrame, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False, cmap=["#ffffff", CATEGORICAL[7]], ax=ax, yticklabels=False)
    ax.set_title("Missing Data Pattern")
    ax.set_xlabel("Feature")
    return fig, ax


def plot_class_distribution(counts_before: pd.Series, counts_after: pd.Series, figsize=(11, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, counts, title in zip(axes, [counts_before, counts_after], ["Before SMOTE", "After SMOTE"]):
        colors = [CLASS_COLORS.get(c, CATEGORICAL[0]) for c in counts.index]
        bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.55)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=10, color=INK_PRIMARY)
        ax.set_title(title)
        ax.set_ylabel("Count")
    fig.suptitle("Class Distribution Before and After SMOTE", fontweight="bold")
    return fig, axes


def plot_correlation_heatmap(corr: pd.DataFrame, figsize=(12, 10), title="Feature Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=DIVERGING_BLUE_RED, center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, linecolor=SURFACE, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_roc_curves(curves: dict, figsize=(7, 6), title="ROC Curves"):
    """``curves``: {model_name: (fpr, tpr, auc)}"""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return fig, ax


def plot_pr_curves(curves: dict, figsize=(7, 6), title="Precision-Recall Curves"):
    """``curves``: {model_name: (recall, precision, ap)}"""
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, (recall, precision, ap)) in enumerate(curves.items()):
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    return fig, ax


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], figsize=(5.5, 5), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap=SEQUENTIAL_BLUE, cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5, linecolor=SURFACE)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig, ax


def plot_calibration_curve(curves: dict, figsize=(7, 6), title="Calibration Curves"):
    """``curves``: {model_name: (prob_true, prob_pred)}"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, label="Perfectly calibrated")
    for i, (name, (prob_true, prob_pred)) in enumerate(curves.items()):
        ax.plot(prob_pred, prob_true, marker="o", label=name, color=CATEGORICAL[i % len(CATEGORICAL)], linewidth=1.6)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig, ax


def plot_learning_curve(train_sizes, train_scores, val_scores, figsize=(7, 5.5), title="Learning Curve"):
    fig, ax = plt.subplots(figsize=figsize)
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)
    ax.plot(train_sizes, train_mean, "o-", color=CATEGORICAL[0], label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=CATEGORICAL[0])
    ax.plot(train_sizes, val_mean, "o-", color=CATEGORICAL[5], label="Cross-validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color=CATEGORICAL[5])
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return fig, ax


def plot_feature_importance(importance: pd.Series, top_n=15, figsize=(8, 7), title="Feature Importance"):
    top = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index.astype(str), top.values, color=CATEGORICAL[0])
    ax.set_title(title)
    ax.set_xlabel(importance.name or "Importance")
    return fig, ax


def plot_model_comparison_bars(df: pd.DataFrame, metric: str, group_col="Classifier", split_col="Training/Testing",
                                figsize=(10, 6), title=None):
    """Grouped bar chart comparing a metric across classifiers, split by train/test."""
    pivot = df.pivot(index=group_col, columns=split_col, values=metric)
    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax, color=[CATEGORICAL[0], CATEGORICAL[5]], width=0.7)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by Classifier")
    ax.legend(title=None, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return fig, ax
