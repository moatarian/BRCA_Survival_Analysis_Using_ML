"""
data_loader.py
==============
Load the raw METABRIC (cBioPortal export) clinical files and reconstruct the flat,
34-feature table that the paper used (the Kaggle ``breast-cancer-metabric`` CSV is
simply ``data_clinical_patient`` merged with ``data_clinical_sample`` on the patient
id). All column names are normalised to snake_case so the rest of the pipeline can
rely on a single, stable schema.

Substitution documented here (paper feature -> METABRIC column):
    * "Mutation Count"  ->  TMB_NONSYNONYMOUS   (non-synonymous mutation burden;
      the cBioPortal export exposes the mutation count under this header).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .utils import PATHS, TARGET_COL, SURGERY_COL

# cBioPortal header code  ->  snake_case name used throughout the project
_RENAME: Dict[str, str] = {
    # ---- patient file ----
    "PATIENT_ID": "patient_id",
    "LYMPH_NODES_EXAMINED_POSITIVE": "lymph_nodes_examined_positive",
    "NPI": "nottingham_prognostic_index",
    "CELLULARITY": "cellularity",
    "CHEMOTHERAPY": "chemotherapy",
    "COHORT": "cohort",
    "ER_IHC": "er_status_measured_by_ihc",
    "HER2_SNP6": "her2_status_measured_by_snp6",
    "HORMONE_THERAPY": "hormone_therapy",
    "INFERRED_MENOPAUSAL_STATE": "inferred_menopausal_state",
    "SEX": "sex",
    "INTCLUST": "integrative_cluster",
    "AGE_AT_DIAGNOSIS": "age_at_diagnosis",
    "OS_MONTHS": "overall_survival_months",
    "OS_STATUS": "overall_survival_status",
    "CLAUDIN_SUBTYPE": "pam50_claudin_low_subtype",
    "THREEGENE": "three_gene_classifier_subtype",
    "VITAL_STATUS": "patients_vital_status",
    "LATERALITY": "primary_tumor_laterality",
    "RADIO_THERAPY": "radio_therapy",
    "HISTOLOGICAL_SUBTYPE": "tumor_other_histologic_subtype",
    "BREAST_SURGERY": "type_of_breast_surgery",
    "RFS_MONTHS": "relapse_free_status_months",
    "RFS_STATUS": "relapse_free_status",
    # ---- sample file ----
    "CANCER_TYPE": "cancer_type",
    "CANCER_TYPE_DETAILED": "cancer_type_detailed",
    "ER_STATUS": "er_status",
    "HER2_STATUS": "her2_status",
    "GRADE": "neoplasm_histologic_grade",
    "ONCOTREE_CODE": "oncotree_code",
    "PR_STATUS": "pr_status",
    "SAMPLE_TYPE": "sample_type",
    "TUMOR_SIZE": "tumor_size",
    "TUMOR_STAGE": "tumor_stage",
    "TMB_NONSYNONYMOUS": "mutation_count",  # documented substitution
}

# Columns that exist in the export but are not part of the paper's 34-feature set.
# SAMPLE_ID is not covered by the rename map, so it keeps its raw upper-case name.
_DROP_AFTER_MERGE = ["SAMPLE_ID", "sample_id", "sample_type"]


def _read_cbioportal_clinical(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical file.

    The first four lines are ``#``-prefixed metadata; the fifth line is the real,
    machine-readable header (e.g. ``PATIENT_ID``). We skip the four metadata rows.
    """
    return pd.read_csv(path, sep="\t", skiprows=4, dtype=str, keep_default_na=True)


def load_raw_clinical() -> pd.DataFrame:
    """Merge the patient- and sample-level clinical tables into one wide frame."""
    patient = _read_cbioportal_clinical(PATHS.metabric / "data_clinical_patient.txt")
    sample = _read_cbioportal_clinical(PATHS.metabric / "data_clinical_sample.txt")

    df = patient.merge(sample, on="PATIENT_ID", how="inner", validate="one_to_one")
    df = df.rename(columns=_RENAME)
    df = df.drop(columns=[c for c in _DROP_AFTER_MERGE if c in df.columns])
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric-looking columns loaded as strings to real numbers."""
    numeric_like = [
        "lymph_nodes_examined_positive",
        "nottingham_prognostic_index",
        "cohort",
        "age_at_diagnosis",
        "overall_survival_months",
        "relapse_free_status_months",
        "neoplasm_histologic_grade",
        "tumor_size",
        "tumor_stage",
        "mutation_count",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode Overall Survival Status as a binary label.

    Raw values are ``"0:LIVING"`` / ``"1:DECEASED"``. We keep DECEASED as the
    positive class (label 1), matching the paper's framing where the majority
    (deceased) class is the focus of the survival comparison.
    """
    raw = df[TARGET_COL].astype(str).str.upper()
    mapped = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mapped[raw.str.contains("DECEAS")] = 1
    mapped[raw.str.contains("LIVING")] = 0
    df[TARGET_COL] = mapped
    return df


def _normalise_surgery(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the surgery label to {'MASTECTOMY', 'BREAST CONSERVING'}."""
    s = df[SURGERY_COL].astype(str).str.upper().str.strip()
    s = s.where(~s.str.contains("MASTECT"), "MASTECTOMY")
    s = s.where(~s.str.contains("CONSERV"), "BREAST CONSERVING")
    s = s.replace({"NAN": pd.NA})
    df[SURGERY_COL] = s
    return df


def load_clinical(drop_missing_target: bool = True) -> pd.DataFrame:
    """Return the analysis-ready (still un-imputed) clinical frame.

    Parameters
    ----------
    drop_missing_target
        If True, rows with a missing Overall Survival Status are removed. The paper
        models 2,007 patients (1,330 deceased + 677 living); dropping the missing
        target reproduces that modelling population.
    """
    df = load_raw_clinical()
    df = _coerce_numeric(df)
    df = _encode_target(df)
    df = _normalise_surgery(df)
    if drop_missing_target:
        df = df[df[TARGET_COL].notna()].reset_index(drop=True)
        df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def split_by_surgery(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the cohort into the two surgical arms compared in the paper."""
    out: Dict[str, pd.DataFrame] = {}
    s = df[SURGERY_COL].astype(str)
    out["Mastectomy"] = df[s == "MASTECTOMY"].reset_index(drop=True)
    out["BCS"] = df[s == "BREAST CONSERVING"].reset_index(drop=True)
    return out


def dataset_overview() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience helper returning (raw_all_patients, target_filtered)."""
    raw = load_raw_clinical()
    modelling = load_clinical(drop_missing_target=True)
    return raw, modelling


if __name__ == "__main__":  # quick smoke test
    raw = load_raw_clinical()
    print("raw merged shape:", raw.shape)
    model_df = load_clinical()
    print("modelling shape:", model_df.shape)
    print(model_df[TARGET_COL].value_counts(dropna=False))
    for name, part in split_by_surgery(model_df).items():
        print(name, part.shape, dict(part[TARGET_COL].value_counts()))
