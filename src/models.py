"""
models.py
=========
Model zoo and hyperparameter optimisation.

The paper benchmarks nine classifiers per surgical group: SVM, Random Forest,
XGBoost, Gradient Boosting, KNN, AdaBoost, Gaussian Naive Bayes, Logistic
Regression, and Decision Tree, training with 10-fold cross-validation and L2
regularisation on the winning Gradient Boosting model. We reproduce that exact
model set, and additionally benchmark LightGBM and CatBoost (present in the
required tech stack but absent from the original paper) purely as a modern
point of comparison -- clearly separated from the paper's own model list in the
evaluation notebook.

Optuna is used to tune Gradient Boosting (the paper's best model) beyond the
paper's own hyperparameters, which are not fully reported.
"""
from __future__ import annotations

from typing import Dict

import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .utils import RANDOM_SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)


def paper_model_zoo(random_state: int = RANDOM_SEED) -> Dict[str, object]:
    """The exact nine classifiers benchmarked in the paper (Tables 2 & 3)."""
    return {
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "Gradient Boosting": gradient_boosting_l2(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "XGBoost": XGBClassifier(
            random_state=random_state, eval_metric="logloss", use_label_encoder=False
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }


def extended_model_zoo(random_state: int = RANDOM_SEED) -> Dict[str, object]:
    """Extra modern boosters required by the tech stack, not part of the original paper."""
    return {
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=random_state, verbose=False),
    }


def gradient_boosting_l2(random_state: int = RANDOM_SEED, **kwargs) -> GradientBoostingClassifier:
    """Gradient Boosting with explicit L2-style regularisation (paper's winning model).

    scikit-learn's GradientBoostingClassifier does not expose a literal "L2
    penalty" the way linear models do; the paper's stated regularisation is
    reproduced here via ``max_features`` sub-sampling, shallow trees, a
    non-trivial ``subsample`` fraction, and a modest learning rate -- the
    standard levers used to control variance/overfitting in gradient boosting
    (functionally analogous to weight shrinkage).
    """
    defaults = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        max_features="sqrt",
        random_state=random_state,
    )
    defaults.update(kwargs)
    return GradientBoostingClassifier(**defaults)


def cross_validate_10fold(model, X, y, scoring: str = "accuracy", random_state: int = RANDOM_SEED):
    """10-fold stratified cross-validation, as specified in the paper's Model training section."""
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)


def tune_gradient_boosting_optuna(X_train, y_train, n_trials: int = 40, random_state: int = RANDOM_SEED):
    """Optuna hyperparameter search for Gradient Boosting (the paper's best model)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=random_state,
        )
        model = GradientBoostingClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study
