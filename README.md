# PMSM thermal prediction study

This repository contains the data workflow, models, evaluations, and manuscript for predicting permanent-magnet (PM) temperature in a permanent-magnet synchronous motor. The current target is the dataset column `pm`; it is not stator-winding temperature.

## Current evaluation

The main comparison is a nested five-fold profile-grouped evaluation of 69 operating profiles. Hyperparameters were selected in grouped inner folds and neural models were repeated across three seeds. Neural fits used all available outer-training windows, with a five-epoch cap and batch size 512. Ridge sampled at most 200,000 windows per fit. History length was 128 and forecast horizon was 20 steps. Neural training ran on an NVIDIA RTX 3050 Laptop GPU.

| Model | Pooled 20-step MAE (Celsius) | 95% profile-cluster bootstrap CI (Celsius) | RMSE (Celsius) |
|---|---:|---:|---:|
| Direct GRU | **0.09212** | 0.07572-0.11124 | 0.19519 |
| Ridge direct | 0.09433 | 0.07988-0.11050 | **0.18107** |
| Operating-conditioned thermal model | 0.09673 | 0.08066-0.11483 | 0.19873 |
| Persistence | 0.12269 | 0.10600-0.14119 | 0.22615 |

Direct GRU has the lowest pooled MAE, 4.77% below the operating-conditioned model. This is a numerical lead, not a confirmed superiority result: the Holm-adjusted Nadeau-Bengio outer-fold test for GRU versus the operating-conditioned model was *p* = 0.182. Profile-level Wilcoxon/bootstrap analyses favored GRU, but they do not replace independent fold-level evidence. The 20-step forecasts used recorded future ambient temperature, so they represent offline conditional forecasts.

The neural models used all available training windows, but training was capped at five epochs. This was not an exhaustive search over training duration. Exact fold-specific hyperparameters and outputs are in [`research/results/final/nested_cv_gpu/`](research/results/final/nested_cv_gpu/README_RESULTS.md). Earlier reduced-budget screens remain separately documented and should not be pooled with these results.

## Repository layout

```text
dataset/                 Source data (kept out of version control)
notebooks/               Exploration, training, and explainability notebooks
research/experiments/    Reusable ablation, nested-CV, and statistics code
research/results/        Evaluation tables, tuning logs, and diagnostics
research/reports/        Study report and PDF manuscript copy
PIML_PMSM_Research_Report.tex  Editable manuscript source
Explainability/          SHAP and thermal-limit analysis
PMSM_EDSS/               Streamlit inference application
figures/                 Project figures
models/                  Saved model artifacts (large files ignored)
```

## Data and reproduction

Place `measures_v2.csv` at `dataset/measures_v2.csv`. The primary notebook is `notebooks/PIML_training.ipynb`. Activate the `piml_motor` Conda environment before running Python commands.

Reproduce the nested evaluation from the repository root with:

```bash
python research/experiments/nested_cv.py --data dataset/measures_v2.csv --out research/results/final/nested_cv_gpu --outer-folds 5 --inner-folds 2 --seeds 0 1 2 --epochs 5 --batch 512 --train-frac 1.0 --device cuda
python research/experiments/stat_tests.py --results research/results/final/nested_cv_gpu --mode cv --ref oc_full --boot 10000 --thin 20 --seed 2026
```

Use a fresh output directory when launching a new run. The nested output contains the manifest, selected hyperparameters, inner-fold scores, per-profile metrics, window errors, and statistical tables. See [`research/README_ablation.md`](research/README_ablation.md) for the experiment inventory and [`research/reports/PROJECT_STUDY_REPORT.md`](research/reports/PROJECT_STUDY_REPORT.md) for the consolidated findings and limitations.

## Dashboard and analysis

Launch the local application from the repository root:

```bash
streamlit run PMSM_EDSS/app.py
```

The application accepts a 128-row sensor history. SHAP and thermal-limit analysis are described in [`Explainability/README.md`](Explainability/README.md). The thermal-limit output is a forecast proxy, not empirical remaining useful life; the dataset has no run-to-failure labels.

## Dataset and artifact handling

The source data are the [Electric Motor Temperature dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature). Raw datasets, checkpoints, and per-window error arrays are excluded from Git because of their size. Aggregate result tables, run manifests, and concise reports are retained for reproducibility.