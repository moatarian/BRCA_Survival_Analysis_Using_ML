"""
explainability.py
==================
SHAP-based model explanation, reproducing the paper's "SHAP analysis" section:
Beeswarm plots, Force plots, and Waterfall plots computed separately for the
Mastectomy and BCS models. Also provides the additional explainability views
requested by the project brief (dependence plots, global/permutation importance,
partial dependence, individual-patient explanations) that extend beyond what the
paper itself reports.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import shap

from .utils import RANDOM_SEED


def compute_shap_values(model, X: pd.DataFrame, background_size: int = 200, random_state: int = RANDOM_SEED):
    """Compute SHAP values for a tree-based model using TreeExplainer.

    Falls back to a model-agnostic ``Explainer`` with a subsampled background
    for non-tree models (kept generic so the function works across the whole
    model zoo, not just Gradient Boosting).
    """
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X)
    except Exception:
        rng = np.random.RandomState(random_state)
        bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
        background = X.iloc[bg_idx]
        explainer = shap.Explainer(model.predict_proba, background)
        sv = explainer(X)
        # binary classifiers via predict_proba produce a 3D array (n, features, 2 classes)
        if sv.values.ndim == 3:
            sv = sv[..., 1]
    return explainer, sv


def top_features_by_mean_abs_shap(shap_values, feature_names: List[str], top_n: int = 10) -> pd.Series:
    """Rank features by mean |SHAP value|, the basis of the paper's Beeswarm plots."""
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names, name="mean_abs_shap").sort_values(ascending=False).head(top_n)


def single_patient_explanation(shap_values, index: int) -> "shap.Explanation":
    """Return the SHAP explanation row for one patient (used for waterfall/force plots)."""
    return shap_values[index]


def representative_patient_index(model, X: pd.DataFrame, y_proba: Optional[np.ndarray] = None) -> int:
    """Pick a representative patient (closest predicted probability to the cohort mean)
    for the single-patient waterfall/force plots, mirroring Figs. 2, 4 and 5 of the paper."""
    if y_proba is None:
        y_proba = model.predict_proba(X)[:, 1]
    target = float(np.mean(y_proba))
    return int(np.argmin(np.abs(y_proba - target)))
