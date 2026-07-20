# Explainable ML Reproduction — Mastectomy vs. Breast Conserving Surgery Survival (METABRIC)

> Asfaw, B. B. & Tegaw, E. M. (2025). *Explainable machine learning to compare the overall survival status between patients receiving mastectomy and breast conserving surgeries.* **Scientific Reports**, 15:10700. https://doi.org/10.1038/s41598-025-91064-2

The paper compares Overall Survival Status (Living / Deceased) between breast cancer patients who underwent **Mastectomy** vs. **Breast Conserving Surgery (BCS)**, using nine ML classifiers on the METABRIC cohort and SHAP for explainability. This project reproduces that methodology as faithfully as possible on the raw METABRIC data available locally, documents every necessary substitution, and extends the analysis with a data-leakage audit, a formal survival analysis, and a systematic paper-vs-reproduction comparison.

## Key findings

- **The paper's methodology is substantially reproducible.** Gradient Boosting is confirmed as the strongest classifier in both surgery groups; BCS shows better survival outcomes than Mastectomy in both the paper's classification framing and our added Kaplan-Meier/Cox analysis; Relapse Free Status and Age at Diagnosis are confirmed as top SHAP predictors.
- **A data-leakage issue was identified and quantified.** Including `Patient's Vital Status` as a predictor drives every classifier to ~100% accuracy — inconsistent with the paper's own reported 55–95% range. Excluding only this column reproduces the paper's Table 2/3 numbers closely (see `reports/three_way_leakage_comparison.csv`).
- **The paper never performs a formal survival analysis** despite its "Overall Survival" framing. Notebook 08 adds Kaplan-Meier curves, log-rank tests, and a multivariable Cox model — confirming Mastectomy remains an independent mortality risk factor (HR≈1.24, p<0.001) even after adjusting for tumour severity.
- **A discrepancy in the paper's own numbers** (Fig. 1 reports 2,007 patients; Table 1 implies 1,981) is documented and resolved in favour of the internally-consistent figure.

Full discussion: `notebooks/09_Reproducing_Paper_Results.ipynb`.

## Dataset

- **Paper's source**: Kaggle `gunesevitan/breast-cancer-metabric` (flat, 34-column CSV export of METABRIC).
- **This project's source**: `Data/brca_metabric/` — the raw cBioPortal study download. `src/data_loader.py` merges `data_clinical_patient.txt` + `data_clinical_sample.txt` on Patient ID to reconstruct the paper's exact 34-feature schema (verified: 2,509 patients × 34 features, exact match).
- **One documented substitution**: paper's *Mutation Count* → cBioPortal's `TMB_NONSYNONYMOUS` (non-synonymous tumour mutational burden), the closest available per-patient mutation-burden field in the raw export.

## Project structure

```
Data/                      Raw METABRIC (cBioPortal) files
Paper/                     Source PDF
notebooks/                 01–09, fully executed, in dependency order
src/                       Modular, reusable pipeline code (see below)
figures/                   ~50 figures x {PNG, PDF}, organised by notebook
models/                    24 trained models (joblib): 9 paper classifiers + Optuna-tuned
                            Gradient Boosting + LightGBM/CatBoost, x2 surgery groups
outputs/                   Persisted intermediate datasets (imputed, SMOTE-balanced train/test)
reports/                   ~26 CSV/TXT tables: missingness, imputation validation, feature
                            selection, model metrics, SHAP rankings, Cox hazard ratios,
                            paper-vs-reproduction comparisons
```

### `src/` modules

| Module | Responsibility |
|---|---|
| `utils.py` | Paths, random seed, canonical feature lists, leakage-column policy |
| `data_loader.py` | Load & merge raw cBioPortal clinical files into the paper's schema |
| `preprocessing.py` | Imputation (mode/KNN), validation (t-test/chi²), outliers, encoding, scaling, the surgery→survival logistic regression |
| `feature_engineering.py` | Feature selection (GB importance, mutual info, LASSO, RFE), SMOTE |
| `models.py` | The paper's 9-classifier zoo, LightGBM/CatBoost extension, Optuna tuning |
| `evaluation.py` | Accuracy/Precision/Recall/F1/ROC-AUC, PR-AUC, confusion matrix, calibration, learning curves |
| `explainability.py` | SHAP value computation and ranking helpers |
| `survival.py` | Kaplan-Meier, log-rank, Cox PH, risk stratification (lifelines) |
| `visualization.py` | Shared publication-quality plotting style and chart builders |

## Setup

Requires **Python 3.10** (matching the paper's environment).

```bash
pip install -r requirements.txt
```

## Running the analysis

Notebooks are numbered in dependency order — each persists intermediate artefacts (`outputs/`, `models/`, `reports/`) that later notebooks load, so they should be run **in order** the first time:

1. `01_Paper_Review.ipynb` — paper summary, workflow diagram (no data dependency)
2. `02_Data_Exploration.ipynb` — load & explore the reconstructed cohort
3. `03_Data_Preprocessing.ipynb` — imputation, validation, outliers, logistic regression; persists imputed per-group datasets
4. `04_Feature_Engineering.ipynb` — encoding, scaling, feature selection, SMOTE; persists model-ready train/test arrays
5. `05_Model_Training.ipynb` — trains all classifiers + Optuna tuning; persists models and metric tables
6. `06_Model_Evaluation.ipynb` — ROC/PR/calibration/learning curves, cross-model comparison
7. `07_Explainability.ipynb` — SHAP waterfall/beeswarm/force/dependence, permutation importance, PDP
8. `08_Survival_Analysis.ipynb` — Kaplan-Meier, log-rank, Cox PH, risk stratification (methodological addition)
9. `09_Reproducing_Paper_Results.ipynb` — side-by-side comparison with the paper's published Tables 2–3

To re-run headlessly end-to-end:

```bash
for nb in notebooks/0*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

## Reproducibility

- Global random seed fixed to `42` (`src/utils.set_seeds`) — not reported in the original paper.
- All figures saved as both PNG (300 dpi) and PDF (vector) to `figures/`.
- Every model is persisted with `joblib` to `models/`; every intermediate dataset and metrics table is persisted to `outputs/` / `reports/`, so any notebook can be re-run independently once its upstream dependencies have executed once.
- Library versions are pinned in `requirements.txt`. The original paper used an older stack (Python 3.10, pandas 2.0.3, scikit-learn 1.3.0, imbalanced-learn 0.10.1, SHAP 0.41.0, Google Colab, 2024–2025); minor numerical differences versus the paper's tables are expected from this drift and are discussed in Notebook 09.

## Substitutions & deviations from the paper (full list in Notebook 09, §6)

| Paper | This reproduction | Reason |
|---|---|---|
| Kaggle flat CSV | cBioPortal `data_clinical_patient` + `data_clinical_sample` merge | Only source available locally; verified exact schema/shape match |
| Mutation Count | `TMB_NONSYNONYMOUS` | Closest available field in the raw export |
| `Patient's Vital Status` inclusion unspecified | Excluded from the paper-faithful modelling track | Included, it drives every classifier to ~100% accuracy — inconsistent with the paper's own reported numbers |
| SMOTE split-order unspecified | Applied strictly to the training fold, after the train/test split | Avoids test-set leakage from synthetic-neighbour information |
| No survival analysis performed | Kaplan-Meier + log-rank + Cox PH added (Notebook 08) | The paper's own framing implies it; METABRIC supports it directly |
