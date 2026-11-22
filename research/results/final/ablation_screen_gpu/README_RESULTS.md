# 32-variant ablation screen

This earlier screen evaluated the 32 registered variants on a profile-grouped paper split: 48 training, 10 validation, and 11 test profiles. The run used the `piml_motor` environment and an RTX 3050 Laptop GPU (6 GB). It trained for three epochs using 1% of available windows per epoch, batch size 32, AdamW weight decay 1e-5, one seed, history 128, and a 20-step forecast. Neural checkpoint selection used validation profiles. The 500-step diagnostic fed predictions recursively while supplying recorded future operating inputs.

## Main 20-step test results

| Model | Pooled test MAE ( degC) |
|---|---:|
| Ridge direct | 0.11191 |
| Direct GRU | 0.12336 |
| Direct LSTM | 0.12357 |
| `oc_full` | 0.12602 |
| Persistence | 0.13501 |
| Linear extrapolation | 0.21586 |

Cooling-linear recorded 0.10918  degC in this screen. It was a post-hoc screen among many variants, however, and no paired profile-level comparison with `oc_full` survived global Holm correction across the 32 variants. It should not be interpreted as a discovered best method.

The 500-step chained diagnostic reported MAE of 1.503  degC for `oc_full`, 1.076  degC for direct GRU, and 2.424  degC for persistence. This was a single-seed, 11-profile diagnostic and is not directly comparable with the notebook's constant-forcing result or the 20-step nested evaluation.

## Interpretation

The screen is historical exploratory evidence, not the primary model comparison. Its reduced training budget, single seed, small holdout, and post-hoc selection make it unsuitable for a superiority claim. The later nested evaluation is reported in [`../nested_cv_gpu/README_RESULTS.md`](../nested_cv_gpu/README_RESULTS.md) and in the project study report.

The screen used `oc_full` learning rate 1e-3, direct GRU 1e-4, and 3e-4 for other neural variants; Ridge alpha was fixed at 1.0. The compact validation screen later slightly favored 3e-4 for `oc_full`. These choices were not jointly nested in this all-variant run.

## Reproducibility files

- `profile_metrics.csv` contains profile-level summaries.
- `stats_paper/` contains pooled tables, bootstrap intervals, paired tests, Friedman/Nemenyi results, DM diagnostics, and `forest_plot.png`.
- `run_log.jsonl` records model settings, validation histories, parameter diagnostics, and elapsed time.
- `long_horizon.csv` and `stats_paper/long_horizon_summary.csv` contain chained-rollout diagnostics.
- `errors/` contains per-window errors and is excluded from version control because of its size.
