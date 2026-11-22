# Real-data ablation diagnostic (not publication results)

Run date: 2026-11-22 (Asia/Calcutta)

## Scope and configuration

This is a pipeline check on the paper split, not a confirmatory study or a tuned model comparison.

- Data: repository-root `measures_v2.csv`; 1,330,816 rows and 69 profiles.
- Split: the README paper split, profile grouped, random state 42; 48 train / 10 validation / 11 test profiles.
- Models: persistence, linear extrapolation, ridge direct forecast, `oc_full`, `global_adaptive`.
- Neural training: seed 0, one epoch, 5% random training-window subsample, CPU, batch 512, AdamW learning rate 3e-4, weight decay 1e-5, gradient clip 1.0, best validation MAE checkpoint.
- Forecast: 20 steps, 128-sample maximum history; all models use the same 184,699 forecast windows.
- Statistics: 1,000 profile-cluster bootstrap replicates; profile-level Wilcoxon with Holm and BH adjustments; Friedman/Nemenyi; DM on absolute error thinned every 20 origins, default HAC bandwidth `floor(sqrt(n_thinned))`, and batch-means support check.

## Diagnostic metrics

Pooled test MAE (°C): ridge 0.11188; global-adaptive 0.12707; oc_full 0.12748; persistence 0.13501; linear extrapolation 0.21586. These values are exploratory: one seed, one epoch, and 5% of training windows are not enough to compare trained models fairly. The ridge result indicates that a serious linear/direct baseline is essential in the full study. The oc_full selected epoch was epoch 1 (validation MAE 0.11904 °C). The separate `direct_lstm`/`direct_gru` registry variants were not part of this subset; `ridge_direct` is a one-parameter Ridge regression per horizon over the flattened 128-step feature history, current temperature, and future ambient values.

For `oc_full` versus persistence, this run found pooled delta MAE `+0.00754 °C` (variant minus reference; persistence is the variant), with 95% profile-bootstrap CI `[-0.00022, 0.01512]`; the 11-profile Wilcoxon p was 0.0674 and Holm-adjusted p 0.1611. The small holdout is inconclusive.

The combined DM and batch-means p-values agreed on this diagnostic split for persistence (0.00889 / 0.02479), linear extrapolation (2.08e-69 / 3.15e-27), and ridge (2.13e-11 / 8.09e-7), but they are secondary, heavily window-dependent comparisons and do not replace the profile-level analysis. Bandwidth was within its valid sample-count limit for all profiles: default `floor(sqrt(n))` on 102–1,817 thinned origins (approximately 10–42 lags), below `n`.

## Artifacts

- `summary_table.csv`, `pairwise_vs_ref.csv`, `friedman_ranks.csv`, `nemenyi_pairwise.csv`, `dm_tests.csv`, `horizon_table.csv`, LaTeX tables, and `forest_plot.png` are in `stats_paper/`.
- `run_log.jsonl` preserves the actual settings, neural validation history, parameter diagnostics, and elapsed time.
- Per-window errors are in `errors/`.

## Limitations / next run

Do not cite these diagnostic numbers as final paper evidence. The current runner implements the paper split and CV, but the diagnostic was deliberately shortened; no hyperparameter search was performed. First fix/confirm the evaluation estimand (20-step forecast origin and future ambient availability), verify physical units, and compare direct baselines fairly. Then select hyperparameters using grouped validation folds only, lock them, and run the all-profile five-fold CV with three or more seeds and full training windows. DM results are supporting evidence only; preserve a predeclared lag cap and sensitivity report. The README's default 0.005 °C equivalence margin needs a sensor/application-based justification before use.
