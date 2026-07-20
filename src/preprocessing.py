"""
preprocessing.py
=================
Reproduces the paper's "Data preprocessing" section:

  * Missing-value imputation: mode for categorical features, K-Nearest-Neighbors
    (KNN, k=5) for numerical features.
  * Validation of imputation: two-sample t-test (numerical) / chi-square test
    (categorical) comparing imputed vs. originally-observed distributions.
  * Outlier detection via IQR and Z-score (reported, not removed -- the paper
    explicitly keeps outliers because "they are very important in healthcare data").
  * Label encoding of categorical features + Min-Max scaling of numerical features.
  * The logistic-regression significance test of Overall Survival Status ~ Type of
    Breast Surgery (odds ratio, 95% CI, pseudo R-squared) reported in the paper's
    Results section.

All functions operate on the snake_case schema produced by ``data_loader.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_SEED, TARGET_COL, SURGERY_COL, ID_COL


# --------------------------------------------------------------------------- #
# Missing-value summary (reproduces Table 1 of the paper)
# --------------------------------------------------------------------------- #
def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column, mirroring the paper's Table 1."""
    return (
        df.isna()
        .sum()
        .rename("n_missing")
        .to_frame()
        .assign(pct_missing=lambda t: (t["n_missing"] / len(df) * 100).round(2))
        .sort_values("n_missing", ascending=False)
    )


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #
def _resolve_feature_lists(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    """Intersect the canonical numeric/categorical lists with columns actually present."""
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns and c not in exclude]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns and c not in exclude]
    return numeric, categorical


def impute_missing(
    df: pd.DataFrame,
    n_neighbors: int = 5,
    exclude: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Mode-impute categorical features, KNN-impute numerical features.

    Returns the imputed dataframe plus a dict of pre-imputation missing masks
    (boolean arrays) so that ``validate_imputation`` can compare imputed vs.
    observed distributions afterwards.
    """
    exclude = exclude or [ID_COL, TARGET_COL, SURGERY_COL]
    numeric, categorical = _resolve_feature_lists(df, exclude)

    out = df.copy()
    missing_masks: Dict[str, np.ndarray] = {}

    # --- categorical: mode imputation ---
    for col in categorical:
        mask = out[col].isna().to_numpy()
        if mask.any():
            mode_val = out[col].mode(dropna=True)
            fill = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            out[col] = out[col].fillna(fill)
        missing_masks[col] = mask

    # --- numerical: KNN imputation ---
    if numeric:
        mask_df = out[numeric].isna()
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        imputed_values = imputer.fit_transform(out[numeric])
        out[numeric] = pd.DataFrame(imputed_values, columns=numeric, index=out.index)
        for col in numeric:
            missing_masks[col] = mask_df[col].to_numpy()

    return out, missing_masks


@dataclass
class ImputationValidation:
    column: str
    dtype: str
    test: str
    statistic: float
    p_value: float
    n_imputed: int
    significantly_different: bool


def validate_imputation(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_masks: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """t-test (numeric) / chi-square test (categorical) of imputed vs. observed values.

    Mirrors the paper's imputation-validity check: a non-significant test
    (p > alpha) supports the claim that imputation did not distort the
    feature's distribution.
    """
    results: List[ImputationValidation] = []
    numeric, categorical = _resolve_feature_lists(original, [ID_COL, TARGET_COL, SURGERY_COL])

    for col in numeric:
        mask = missing_masks.get(col)
        if mask is None or mask.sum() == 0:
            continue
        observed = original.loc[~mask, col].dropna()
        newly_imputed = imputed.loc[mask, col].dropna()
        if len(observed) < 2 or len(newly_imputed) < 2:
            continue
        stat, p = stats.ttest_ind(observed, newly_imputed, equal_var=False, nan_policy="omit")
        results.append(
            ImputationValidation(col, "numeric", "Welch t-test", float(stat), float(p), int(mask.sum()), p < alpha)
        )

    for col in categorical:
        mask = missing_masks.get(col)
        if mask is None or mask.sum() == 0:
            continue
        observed = original.loc[~mask, col].dropna()
        newly_imputed = imputed.loc[mask, col]
        categories = sorted(set(observed.unique()) | set(newly_imputed.unique()))
        obs_counts = observed.value_counts().reindex(categories, fill_value=0)
        imp_counts = newly_imputed.value_counts().reindex(categories, fill_value=0)
        table = np.vstack([obs_counts.values, imp_counts.values])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(table)
        results.append(
            ImputationValidation(col, "categorical", "Chi-square", float(chi2), float(p), int(mask.sum()), p < alpha)
        )

    return pd.DataFrame([r.__dict__ for r in results]).sort_values("column").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Outlier detection (reported only -- outliers are retained, per the paper)
# --------------------------------------------------------------------------- #
def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Flag outliers per numeric column using the 1.5*IQR rule."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"column": col, "method": "IQR", "lower": low, "upper": high,
                      "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(df), 2)})
    return pd.DataFrame(rows)


def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """Flag outliers per numeric column using |z| > threshold."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        z = np.abs(stats.zscore(df[col], nan_policy="omit"))
        n_out = int((z > threshold).sum())
        rows.append({"column": col, "method": "Z-score", "threshold": threshold,
                      "n_outliers": n_out, "pct_outliers": round(100 * n_out / len(df), 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature engineering primitives: label encoding + Min-Max scaling
# --------------------------------------------------------------------------- #
def label_encode(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns in place, returning fitted encoders."""
    out = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in columns:
        if col not in out.columns:
            continue
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le
    return out, encoders


def minmax_scale(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max scale numeric columns to [0, 1], as specified in the paper."""
    out = df.copy()
    scaler = MinMaxScaler()
    present = [c for c in columns if c in out.columns]
    out[present] = scaler.fit_transform(out[present])
    return out, scaler


# --------------------------------------------------------------------------- #
# Logistic-regression significance test: survival ~ surgery type
# --------------------------------------------------------------------------- #
@dataclass
class SurgeryLogisticResult:
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float
    pseudo_r2: float
    n_obs: int
    summary: str


def surgery_survival_logistic(df: pd.DataFrame) -> SurgeryLogisticResult:
    """Univariate logistic regression: Overall Survival Status ~ Type of Breast Surgery.

    Reproduces the paper's headline statistic (OR ~= 1.27, 95% CI 1.05-1.53,
    p = 0.011, pseudo R^2 ~= 0.005) that motivates comparing the two surgical
    groups separately for the rest of the analysis.
    """
    work = df[[TARGET_COL, SURGERY_COL]].dropna().copy()
    work["surgery_mastectomy"] = (work[SURGERY_COL] == "MASTECTOMY").astype(int)
    X = sm.add_constant(work["surgery_mastectomy"])
    y = work[TARGET_COL].astype(int)

    model = sm.Logit(y, X).fit(disp=0)
    coef = model.params["surgery_mastectomy"]
    ci = model.conf_int().loc["surgery_mastectomy"]
    odds_ratio = float(np.exp(coef))
    ci_low, ci_high = float(np.exp(ci[0])), float(np.exp(ci[1]))
    p_value = float(model.pvalues["surgery_mastectomy"])
    pseudo_r2 = float(model.prsquared)

    return SurgeryLogisticResult(
        odds_ratio=odds_ratio,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        pseudo_r2=pseudo_r2,
        n_obs=int(model.nobs),
        summary=model.summary().as_text(),
    )


# --------------------------------------------------------------------------- #
# Train/test split (80/20, stratified on target -- paper's Data Splitting step)
# --------------------------------------------------------------------------- #
def train_test_split_80_20(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED):
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
