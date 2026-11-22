# Ablation and statistical testing workflow

The reusable runner and analysis code are in `research/experiments/`. Study outputs live under `research/results/`. Run commands from the repository root with the `piml_motor` Conda environment active.

## Experiments

The runner includes persistence, linear extrapolation and Ridge baselines; thermal, LSTM and GRU architectures; cooling/heat/parameter ablations; engineered-feature ablations; loss ablations; history lengths; and model-capacity variants. Every comparison uses common forecast origins and scaling fitted on training profiles only.

## Commands

```bash
# List registered experiments
python research/experiments/ablation_suite.py --list

# Synthetic pipeline smoke check
python research/experiments/ablation_suite.py --synthetic --groups baselines architecture --epochs 1 --train_frac 0.05 --seeds 0 --out research/results/smoke
python research/experiments/stat_tests.py --results research/results/smoke --mode paper --boot 500

# Paper split
python research/experiments/ablation_suite.py --data dataset/measures_v2.csv --split paper --groups all --seeds 0 1 2 --out research/results/paper
python research/experiments/stat_tests.py --results research/results/paper --mode paper

# Profile-grouped five-fold CV
python research/experiments/ablation_suite.py --data dataset/measures_v2.csv --split cv --folds 5 --groups all --seeds 0 1 2 --train_frac 0.25 --long_horizon 500 --out research/results/cv
python research/experiments/stat_tests.py --results research/results/cv --mode cv
```

The real-data runs already in `research/results/final/` used a reduced budget (1% of training windows, 3 epochs) and should be treated as exploratory. The full command above is a recommended next study configuration, but hyperparameter selection should be nested inside the outer CV before making confirmatory claims.

`--train_frac` randomly samples training windows each epoch and must be reported. `--lr_override MODEL=LR` can be repeated to assign model-specific learning rates. `--validation_only` trains and logs validation performance without evaluating test profiles. The runner resumes finished `(experiment, split, seed)` combinations in an output directory; use a new output directory when changing configuration.

## Statistical protocol

- The profile is the unit of inference. Overlapping windows are not independent observations.
- Report pooled MAE/RMSE/R² with profile-cluster bootstrap intervals and paired profile-level deltas.
- Compare each variant to `oc_full` using paired Wilcoxon tests with Holm correction; BH-FDR is exploratory. Report effect sizes and confidence intervals.
- Use Friedman/Nemenyi for multi-model ranking and Nadeau–Bengio correction for fold comparisons.
- DM tests are secondary. The implementation thins by default every 20 origins, uses Bartlett HAC with `floor(sqrt(n))` bandwidth capped at `n−1`, applies the Harvey–Leybourne–Newbold correction, and reports batch-means checks. Treat the test as supporting evidence and report bandwidth sensitivity.
- Equivalence requires a justified practical margin set before seeing results. The default 0.005 °C is only a software default, not a justified scientific margin.

## Current study summary

See [the consolidated study report](reports/PROJECT_STUDY_REPORT.md), [the all-variant screen](results/final/ablation_screen_gpu/README_RESULTS.md), and [the focused CV report](results/final/core_cv_gpu/README_RESULTS.md). The dataset target used is `pm`, permanent-magnet temperature. The 20-step protocol uses recorded future ambient temperature, so document that assumption.
