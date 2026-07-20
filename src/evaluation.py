"""
evaluation.py
==============
Model evaluation utilities reproducing the paper's Tables 2 & 3 (Accuracy,
Precision, Recall, F1, ROC-AUC on train and test splits for every classifier)
plus the additional evaluation artefacts requested by the project brief
(PR-AUC, confusion matrices, calibration curves, learning curves).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, learning_curve

from .utils import RANDOM_SEED


@dataclass
class SplitMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def compute_metrics(y_true, y_pred, y_proba) -> SplitMetrics:
    """Compute the five metrics reported in the paper's Tables 2 & 3."""
    return SplitMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba),
    )


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, SplitMetrics]:
    """Fit-free evaluation: expects an already-fitted model."""
    results = {}
    for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred
        results[split_name] = compute_metrics(y, y_pred, y_proba)
    return results


def metrics_to_row(model_name: str, split: str, metrics: SplitMetrics) -> Dict:
    return {
        "Classifier": model_name,
        "Training/Testing": split,
        "Accuracy": round(metrics.accuracy, 3),
        "Precision": round(metrics.precision, 3),
        "Recall": round(metrics.recall, 3),
        "F1-score": round(metrics.f1, 3),
        "ROC-AUC": round(metrics.roc_auc, 3),
    }


def build_comparison_table(all_results: Dict[str, Dict[str, SplitMetrics]]) -> pd.DataFrame:
    """Assemble a long-format table matching the paper's Table 2 / Table 3 layout."""
    rows = []
    for model_name, splits in all_results.items():
        for split_name, metrics in splits.items():
            rows.append(metrics_to_row(model_name, split_name, metrics))
    df = pd.DataFrame(rows)
    order = list(all_results.keys())
    df["Classifier"] = pd.Categorical(df["Classifier"], categories=order, ordered=True)
    return df.sort_values(["Classifier", "Training/Testing"]).reset_index(drop=True)


def pr_auc(y_true, y_proba) -> float:
    return float(average_precision_score(y_true, y_proba))


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_calibration_curve(y_true, y_proba, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return prob_true, prob_pred


def get_learning_curve(model, X, y, random_state: int = RANDOM_SEED):
    """Learning curve (train/val score vs. training-set size) via 5-fold CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=random_state,
    )
    return train_sizes, train_scores, val_scores
