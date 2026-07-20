"""
utils.py
========
Project-wide configuration, reproducibility helpers, and path management for the
reproduction of:

    Asfaw, B. B. & Tegaw, E. M. (2025). "Explainable machine learning to compare
    the overall survival status between patients receiving mastectomy and breast
    conserving surgeries." Scientific Reports 15:10700.
    https://doi.org/10.1038/s41598-025-91064-2

Everything that must stay constant across the whole analysis (random seed, output
directories, plotting style, the canonical feature list) lives here so that every
notebook and module shares a single source of truth.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed every source of randomness we rely on.

    The paper does not report a seed, so we fix one to make *our* pipeline
    deterministic. This is a documented deviation that improves on the original.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optuna / lightgbm / xgboost respect the global numpy seed, but be explicit
        import lightgbm as lgb  # noqa: F401
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _find_project_root(start: Path | None = None) -> Path:
    """Walk upwards until we find the directory that contains ``Data`` and ``Paper``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "Data").is_dir() and (parent / "Paper").is_dir():
            return parent
    # Fallback: parent of the src/ directory
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Canonical project directories. Created on access if missing."""

    root: Path = field(default_factory=_find_project_root)

    @property
    def data(self) -> Path:
        return self.root / "Data"

    @property
    def metabric(self) -> Path:
        return self.data / "brca_metabric"

    @property
    def paper(self) -> Path:
        return self.root / "Paper"

    @property
    def figures(self) -> Path:
        return self._ensure(self.root / "figures")

    @property
    def models(self) -> Path:
        return self._ensure(self.root / "models")

    @property
    def outputs(self) -> Path:
        return self._ensure(self.root / "outputs")

    @property
    def reports(self) -> Path:
        return self._ensure(self.root / "reports")

    @staticmethod
    def _ensure(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p


PATHS = Paths()


# --------------------------------------------------------------------------- #
# Canonical column semantics (paper's 34-feature framing)
# --------------------------------------------------------------------------- #
# Target and the two "grouping"/identifier columns.
TARGET_COL: str = "overall_survival_status"
SURGERY_COL: str = "type_of_breast_surgery"
ID_COL: str = "patient_id"

# The two surgical arms compared in the paper.
SURGERY_GROUPS: Dict[str, str] = {
    "Mastectomy": "MASTECTOMY",
    "BCS": "BREAST CONSERVING",
}

# Columns that leak the target. The paper INCLUDES these (reproduced faithfully),
# but notebook 09 also runs a leakage-free sensitivity analysis using this list.
LEAKAGE_COLS: List[str] = [
    "patients_vital_status",       # essentially the target relabelled
    "overall_survival_months",     # survival time -> deterministic of status at cutoff
    "relapse_free_status",         # "relapse or death" -> contains death signal
    "relapse_free_status_months",  # time-to-event, same leakage family
]

# Empirically, "Patient's Vital Status" (Living / Died of Disease / Died of Other
# Causes) is near-synonymous with the target and drives test accuracy to ~1.00 for
# every classifier -- a trivial leak the paper's own reported numbers (0.55-0.95
# range) are NOT consistent with reproducing. Excluding only this one column
# reproduces the paper's Table 2/3 accuracy range closely (e.g. Gradient Boosting,
# Mastectomy: 0.846/0.917 here vs. 0.864/0.840 reported), while Relapse Free Status
# and Overall Survival (Months) -- which the paper explicitly keeps and discusses
# via SHAP -- are retained. This is therefore the column set used for the
# "paper-faithful" reproduction track (notebooks 05-08); the full LEAKAGE_COLS
# exclusion above is used only for the leakage-free sensitivity track in notebook 09.
PAPER_FAITHFUL_EXCLUDE_COLS: List[str] = ["patients_vital_status"]

# Numerical vs categorical feature roles AFTER the patient/sample merge and
# the snake_case renaming performed in data_loader.py. Membership is validated
# at runtime against the actual dataframe, so an unexpected column never breaks
# the pipeline silently.
NUMERIC_FEATURES: List[str] = [
    "age_at_diagnosis",
    "lymph_nodes_examined_positive",
    "mutation_count",
    "nottingham_prognostic_index",
    "overall_survival_months",
    "relapse_free_status_months",
    "tumor_size",
    "tumor_stage",
    "neoplasm_histologic_grade",
    "cohort",
]

CATEGORICAL_FEATURES: List[str] = [
    "cancer_type",
    "cancer_type_detailed",
    "cellularity",
    "chemotherapy",
    "pam50_claudin_low_subtype",
    "er_status_measured_by_ihc",
    "er_status",
    "her2_status_measured_by_snp6",
    "her2_status",
    "tumor_other_histologic_subtype",
    "hormone_therapy",
    "inferred_menopausal_state",
    "integrative_cluster",
    "primary_tumor_laterality",
    "oncotree_code",
    "pr_status",
    "radio_therapy",
    "sex",
    "three_gene_classifier_subtype",
    "patients_vital_status",
    "relapse_free_status",
]


def banner(msg: str, char: str = "=", width: int = 78) -> str:
    """Return a formatted section banner for console/notebook logging."""
    pad = max(width - len(msg) - 2, 0)
    left = pad // 2
    right = pad - left
    return f"{char * left} {msg} {char * right}"
