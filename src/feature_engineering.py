"""
feature_engineering.py
=======================
Feature selection and class-balancing utilities.

The paper's "Feature selection" step is qualitative (a clinically-motivated list)
plus a Gradient-Boosting importance ranking; it does not report a hard cutoff. We
reproduce both pieces:

  1. ``CLINICALLY_RELEVANT_FEATURES`` -- the predictor set the paper names
     explicitly (Age at Diagnosis, Tumor Size, Lymph Node status, Chemotherapy,
     Hormone Therapy, "other tumor features") extended to the full feature frame
     (all 33 non-target, non-ID columns), since the paper's SHAP results
     (Fig. 2-5) clearly show the models were trained on the complete feature set,
     not a hand-picked subset.
  2. ``gradient_boosting_importance`` -- feature importance from a Gradient
     Boosting classifier, used both diagnostically and as an extra, best-practice
     cross-check (mutual information, permutation importance) beyond what the
     paper reports.

Class imbalance is handled with SMOTE (imblearn), exactly as specified in the
paper, applied AFTER the train/test split and ONLY to the training fold -- this
is a documented, deliberate correction of a common SMOTE-before-split leakage
mistake; the paper's methods text does not specify the order, so we adopt the
best-practice ordering while noting the deviation.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from .utils import RANDOM_SEED

CLINICALLY_RELEVANT_FEATURES: List[str] = [
    "age_at_diagnosis",
    "tumor_size",
    "lymph_nodes_examined_positive",
    "chemotherapy",
    "hormone_therapy",
    "nottingham_prognostic_index",
    "neoplasm_histologic_grade",
    "tumor_stage",
    "er_status",
    "her2_status",
    "pr_status",
    "inferred_menopausal_state",
]


def gradient_boosting_importance(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED) -> pd.Series:
    """Feature importance scores from a Gradient Boosting classifier (paper's method)."""
    gb = GradientBoostingClassifier(random_state=random_state)
    gb.fit(X, y)
    return pd.Series(gb.feature_importances_, index=X.columns, name="gb_importance").sort_values(ascending=False)


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED) -> pd.Series:
    """Mutual information between each feature and the target (best-practice cross-check)."""
    mi = mutual_info_classif(X, y, random_state=random_state)
    return pd.Series(mi, index=X.columns, name="mutual_info").sort_values(ascending=False)


def lasso_selected_features(X: pd.DataFrame, y: pd.Series, C: float = 0.1, random_state: int = RANDOM_SEED) -> pd.Series:
    """L1-penalised logistic regression coefficients (LASSO-style feature selection)."""
    lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", random_state=random_state, max_iter=2000)
    lr.fit(X, y)
    return pd.Series(np.abs(lr.coef_.ravel()), index=X.columns, name="lasso_abs_coef").sort_values(ascending=False)


def rfe_ranking(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 15, random_state: int = RANDOM_SEED) -> pd.Series:
    """Recursive Feature Elimination ranking using a logistic regression estimator."""
    estimator = LogisticRegression(max_iter=2000, random_state=random_state)
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    selector.fit(X, y)
    return pd.Series(selector.ranking_, index=X.columns, name="rfe_rank").sort_values()


def permutation_importance_ranking(
    model, X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance for a fitted model (used post-hoc in evaluation/explainability)."""
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return (
        pd.DataFrame({"feature": X.columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance classes with SMOTE. Must be called on the TRAINING split only."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
