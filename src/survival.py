"""
survival.py
============
Time-to-event survival analysis.

The paper itself never performs a formal survival analysis (Kaplan-Meier, log-rank,
Cox PH) despite being framed around "Overall Survival Status" -- it treats survival
as a static binary classification target instead. Because METABRIC provides genuine
time-to-event fields (Overall Survival (Months) + Overall Survival Status, and
Relapse Free Status (Months) + Relapse Free Status), this module adds the survival
analysis the paper's own title implies but does not execute, as a methodologically
stronger complement to the classification pipeline. This is documented explicitly
as an addition, not a reproduction, in notebook 08.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

from .utils import SURGERY_COL


@dataclass
class KMGroupResult:
    group: str
    n: int
    n_events: int
    median_survival: float
    kmf: KaplanMeierFitter


def fit_km_by_group(
    df: pd.DataFrame, duration_col: str, event_col: str, group_col: str = SURGERY_COL
) -> Dict[str, KMGroupResult]:
    """Fit a Kaplan-Meier estimator per level of ``group_col`` (Mastectomy vs BCS)."""
    results: Dict[str, KMGroupResult] = {}
    for group_name, sub in df.groupby(group_col):
        sub = sub.dropna(subset=[duration_col, event_col])
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], event_observed=sub[event_col], label=str(group_name))
        results[str(group_name)] = KMGroupResult(
            group=str(group_name),
            n=len(sub),
            n_events=int(sub[event_col].sum()),
            median_survival=float(kmf.median_survival_time_),
            kmf=kmf,
        )
    return results


def logrank_between_groups(
    df: pd.DataFrame, duration_col: str, event_col: str, group_col: str = SURGERY_COL
) -> Tuple[float, float]:
    """Two-group log-rank test (Mastectomy vs BCS). Returns (test_statistic, p_value)."""
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) != 2:
        raise ValueError(f"log-rank test requires exactly 2 groups, got {groups}")
    a = df[df[group_col] == groups[0]].dropna(subset=[duration_col, event_col])
    b = df[df[group_col] == groups[1]].dropna(subset=[duration_col, event_col])
    result = logrank_test(a[duration_col], b[duration_col], event_observed_A=a[event_col], event_observed_B=b[event_col])
    return float(result.test_statistic), float(result.p_value)


def fit_cox_ph(
    df: pd.DataFrame, duration_col: str, event_col: str, covariates: list[str], penalizer: float = 0.1
) -> CoxPHFitter:
    """Fit a (ridge-penalised) Cox proportional-hazards model on the given covariates."""
    cols = [duration_col, event_col, *covariates]
    work = df[cols].dropna().copy()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(work, duration_col=duration_col, event_col=event_col)
    return cph


def hazard_ratio_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Tidy hazard-ratio table (HR, 95% CI, p-value) from a fitted Cox model."""
    summary = cph.summary.copy()
    out = pd.DataFrame({
        "covariate": summary.index,
        "hazard_ratio": summary["exp(coef)"],
        "ci_lower": summary["exp(coef) lower 95%"],
        "ci_upper": summary["exp(coef) upper 95%"],
        "p_value": summary["p"],
    }).reset_index(drop=True)
    return out.sort_values("p_value")


def risk_stratify(cph: CoxPHFitter, df: pd.DataFrame, covariates: list[str], n_groups: int = 3) -> pd.Series:
    """Stratify patients into risk tertiles/quartiles by Cox partial hazard (risk score)."""
    work = df[covariates].dropna()
    risk_score = cph.predict_partial_hazard(work)
    labels = ["Low", "Medium", "High"] if n_groups == 3 else [f"Q{i+1}" for i in range(n_groups)]
    bins = pd.qcut(risk_score, q=n_groups, labels=labels)
    return pd.Series(bins, index=work.index, name="risk_group")
