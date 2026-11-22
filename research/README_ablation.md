# Experiment and statistical analysis guide

The reusable runners are in `research/experiments/`; each completed run has its own directory under `research/results/`. Use the `piml_motor` Conda environment and run commands from the repository root. The project currently predicts `pm` (permanent-magnet temperature). The reported 20-step protocol supplies recorded future ambient temperature.

## Completed evaluation

The nested comparison has five profile-grouped outer folds and two grouped inner folds. All 69 profiles appear in an outer test fold once. Inner validation selects neural learning rate and stopping epoch and Ridge alpha; final neural models are refit on the outer training profiles with three seeds. Neural training uses all available training windows, capped at five epochs and batch size 512. Ridge uses at most 200,000 sampled windows per fit. Forecast history is 128 samples and horizon is 20 steps.

| Model | 20-step MAE ( Celsius) | 95% profile-cluster bootstrap CI ( Celsius) | RMSE ( Celsius) |
|---|---:|---:|---:|
| Direct GRU | **0.09212** | 0.07572-0.11124 | 0.19519 |
| Ridge direct | 0.09433 | 0.07988-0.11050 | **0.18107** |
| `oc_full` | 0.09673 | 0.08066-0.11483 | 0.19873 |
| Persistence | 0.12269 | 0.10600-0.14119 | 0.22615 |

The overall Friedman test was significant (four models, 69 profiles; chi^2 = 112.88, *p* = 2.64  x  10-^24). In paired profile comparisons against `oc_full`, GRU reduced MAE by 0.00461  Celsius (profile-bootstrap CI -0.00645 to -0.00292  Celsius; Holm-adjusted Wilcoxon *p* = 2.22  x  10-8). Ridge reduced pooled MAE by 0.00241  Celsius, but its bootstrap interval crossed zero (-0.00576 to 0.00107  Celsius). The conservative Nadeau-Bengio outer-fold test did not confirm a GRU advantage (mean fold difference -0.00452  Celsius; Holm-adjusted *p* = 0.182); Ridge was also not significant (*p* = 0.353). Report GRU as the current numerical leader, not as a statistically established best model.

The neural learning-rate grid was {1e-4, 3e-4}, weight decay 1e-5, and maximum epochs five. Chosen GRU rates were 1e-4 or 3e-4, with four or five epochs. `oc_full` used the same selected rate/epoch values. Ridge alpha was selected from {0.01, 0.1, 1, 10, 100}; fold-wise choices were 0.01, 0.1, or 100. Exact choices and inner scores are in `results/final/nested_cv_gpu/tuning/selected_hyperparameters.csv` and `inner_fold_scores.csv`.

Diebold-Mariano tests are secondary profile-level diagnostics. Forecast origins were thinned by 20; the largest HAC bandwidth was 46 and no bandwidth exceeded `n_thinned - 1`. Results varied by profile. The nested run used all available windows but a five-epoch cap; do not describe it as an exhaustive epoch or hyperparameter search. The future-ambient input is recorded truth and should be identified as an offline conditional forecast.

## Experiment inventory

- `results/final/nested_cv_gpu/`: current nested evaluation and statistics.
- `results/final/ablation_screen_gpu/`: earlier 32-variant paper-split screen, one seed, 1% windows and three epochs; exploratory only.
- `results/final/core_cv_gpu/`: earlier five-model grouped CV, 1% windows and three epochs; non-nested screening run.
- `results/diagnostics/ablation_smoke_real/`: one-epoch CPU pipeline diagnostic; not comparative evidence.

Do not combine pooled metrics from these runs as though they came from a common test protocol. The earlier reduced-budget values are preserved in their run-specific result files.

## Commands

List registered models:

```bash
python research/experiments/ablation_suite.py --list
```

Run the nested evaluation and its statistical analysis:

```bash
python research/experiments/nested_cv.py --data dataset/measures_v2.csv --out research/results/final/nested_cv_gpu --outer-folds 5 --inner-folds 2 --seeds 0 1 2 --epochs 5 --batch 512 --train-frac 1.0 --device cuda
python research/experiments/stat_tests.py --results research/results/final/nested_cv_gpu --mode cv --ref oc_full --boot 10000 --thin 20 --seed 2026
```

Run the 32-variant screen or the older focused CV only when intentionally reproducing those historical runs; use new output paths if changing any configuration because the runners resume completed experiment/split/seed combinations.

## Statistical conventions

- Profiles, rather than overlapping forecast windows, are the main unit for paired inference and bootstrap resampling.
- The analysis reports pooled MAE and RMSE, profile-cluster confidence intervals, paired profile deltas, Wilcoxon tests with Holm adjustment, and BH-adjusted values as exploratory FDR results.
- Friedman/Nemenyi summaries rank models across profiles. Nadeau-Bengio corrected tests use the outer-fold comparisons and provide a conservative check on the profile-level conclusions.
- Diebold-Mariano output uses a thinning stride at least as large as the forecast horizon, Bartlett HAC variance, and a bandwidth capped by the thinned sample count. It remains a secondary diagnostic because many profile-level comparisons are possible and their errors are serially dependent.
- The software default equivalence margin of 0.005  Celsius has not been justified against sensor accuracy or an application tolerance; do not interpret it as a scientific equivalence result.

The data target, future-input scenario, history length, horizon, selected parameters, hardware, and software versions are recorded in each run's manifest and log. The current nested manifest is `results/final/nested_cv_gpu/manifest.json`.
