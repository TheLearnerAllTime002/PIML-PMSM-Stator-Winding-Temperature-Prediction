# Nested profile-grouped evaluation

This is the current primary comparison of four 20-step PM-temperature predictors. Five outer folds held out disjoint source profiles, and all 69 profiles were tested once. Within each outer training set, two grouped inner folds selected the neural learning rate and epoch count and the Ridge regularization parameter. Neural models were refit on all outer-training windows with three seeds; Ridge and persistence are deterministic baselines.

## Evaluation setup

- Target: `pm` (permanent-magnet temperature).
- Forecast context: 128 historical samples, 20-step horizon, with recorded future ambient temperature supplied.
- Neural training: all available outer-training windows (`train_frac=1.0`), batch 512, maximum five epochs, weight decay 1e-5. Ridge sampled at most 200,000 training windows per fit.
- Tuning: neural learning rates {1e-4, 3e-4}; Ridge alpha {0.01, 0.1, 1, 10, 100}; grouped inner profile MAE selection.
- Neural hardware/software: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), PyTorch 2.5.1+cu121, `piml_motor` environment.

The run used the full available window fraction, but training stopped after at most five epochs. It is not an exhaustive epoch-budget or broad hyperparameter search.

## Pooled results

| Model | MAE ( Celsius) | 95% profile-cluster bootstrap CI ( Celsius) | RMSE ( Celsius) | Seed SD of MAE ( Celsius) |
|---|---:|---:|---:|---:|
| Direct GRU | **0.09212** | 0.07572-0.11124 | 0.19519 | 0.00085 |
| Ridge direct | 0.09433 | 0.07988-0.11050 | **0.18107** | - |
| `oc_full` | 0.09673 | 0.08066-0.11483 | 0.19873 | 0.00070 |
| Persistence | 0.12269 | 0.10600-0.14119 | 0.22615 | - |

Direct GRU is the numerical leader in pooled MAE, 4.77% lower than `oc_full`. Paired profile Wilcoxon-Holm testing favored GRU; the profile-bootstrap interval for the GRU-minus-reference MAE difference was -0.00645 to -0.00292  Celsius. Ridge's corresponding interval crossed zero (-0.00576 to 0.00107  Celsius).

The overall Friedman test was significant (chi^2 = 112.88, *p* = 2.64  x  10-^24). The conservative Nadeau-Bengio corrected comparison across five outer folds did not establish a GRU advantage over `oc_full`: mean fold difference -0.00452  Celsius, *t* = -2.22, Holm-adjusted *p* = 0.182. Ridge was also not significant (*p* = 0.353); persistence performed worse than `oc_full` (*p* = 0.00875). The supported conclusion is that GRU leads numerically, while superiority remains unconfirmed.

## Horizon behavior and DM diagnostics

Window-weighted horizon-wise MAE increases as the prediction extends farther into the future. At step 1, GRU is 0.03665  Celsius, Ridge 0.03571  Celsius, `oc_full` 0.03556  Celsius, and persistence 0.03679  Celsius. At step 20, the values are 0.11141, 0.11860, 0.12360, and 0.17849  Celsius, respectively. `stats_cv/horizon_all_models.csv` contains the curves and profile-cluster intervals plotted in the manuscript.

Diebold-Mariano analyses were run per held-out profile after thinning origins by 20. The largest HAC bandwidth was 46; there were zero instances where it exceeded `n_thinned - 1`. Profile-level DM results varied, so they remain secondary diagnostics rather than the basis for a model ranking.

## Tuning and files

Selected neural settings varied by outer fold. GRU used learning rate 1e-4 or 3e-4 and four or five epochs. `oc_full` used the same choices. Ridge selected alpha 0.01, 0.1, or 100. The complete fold-wise selections and validation values are in `tuning/selected_hyperparameters.csv` and `tuning/inner_fold_scores.csv`.

`manifest.json` records the protocol, dataset target, device, libraries, and search grids. `run_log.jsonl` records inner and outer stages; `profile_metrics.csv` and `errors/` contain evaluation details. Statistical CSV/LaTeX summaries and figures are under `stats_cv/`. The manuscript uses `figures/nested_cv_mae.png` and `figures/nested_cv_horizon.png`.
