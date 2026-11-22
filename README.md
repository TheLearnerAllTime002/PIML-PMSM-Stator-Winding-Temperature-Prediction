# PMSM temperature prediction research project

This repository studies profile-disjoint prediction of PMSM temperatures using recurrent, physics-guided models. The current training target in `PIML_training.ipynb` is `pm` (permanent-magnet temperature); do not describe it as stator-winding temperature unless the target is changed.

## Project layout

```text
dataset/                 Source dataset
notebooks/               Exploration, training, and explainability notebooks
research/experiments/    Ablation runner and statistical analysis
research/results/        Study outputs, tuning logs, and diagnostics
research/reports/        Study report and PDF manuscript copy
PIML_PMSM_Research_Report.tex  Editable manuscript source (root)
Explainability/          SHAP analysis and saved analysis outputs
PMSM_EDSS/               Streamlit inference application
figures/                 Project figures
models/                  Saved model artifacts
```

See [the study report](research/reports/PROJECT_STUDY_REPORT.md) for the run record, current results, limitations, and next steps. The experiment protocol and commands are in [research/README_ablation.md](research/README_ablation.md).

## Data and notebooks

Place the source `measures_v2.csv` at `dataset/measures_v2.csv`. Open the primary notebook with:

```bash
jupyter notebook notebooks/PIML_training.ipynb
```

The other notebook variants are retained in `notebooks/` for reference.

## Reproduce a study run

From the repository root with the `piml_motor` environment activated:

```bash
python research/experiments/ablation_suite.py --data dataset/measures_v2.csv --split paper --groups all --seeds 0 1 2 --out research/results/paper
python research/experiments/stat_tests.py --results research/results/paper --mode paper
```

For the focused five-fold evaluation, change `--split paper` to `--split cv --folds 5`. See the study report before interpreting results: the saved runs use a reduced training budget and are exploratory.

## Local dashboard

Run the Streamlit application from the repository root:

```bash
streamlit run PMSM_EDSS/app.py
```

It loads model artifacts from `models/` and forecasts PM temperature from a 128-row sensor history. Consult [Explainability/README.md](Explainability/README.md) for SHAP analysis details.

## Source dataset

The project uses the [Electric Motor Temperature dataset](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature). Keep downloaded data and trained weights out of version control; `.gitignore` preserves aggregate research tables and run logs while excluding large source and window-error files.
