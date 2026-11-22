# PMSM temperature study: run record and next steps

**Status: completed exploratory GPU study; publication study still requires a nested, full-budget rerun.**

## What was run

The active `piml_motor` Conda environment used PyTorch 2.5.1+cu121 on an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB). The data file contained 1,330,816 rows and 69 profile IDs. For window construction, the runner internally indexed contiguous runs by first-seen file order; this matches the notebook's `profile_id.unique()` split order. The profile IDs were disjoint across the five outer folds; each profile was a test profile once. Scaling was fitted on the training profiles in each split.

All training used 1% of available training windows per epoch, 3 epochs, batch size 32, AdamW, weight decay 1e-5, and gradient clipping at 1.0. This is a deliberately reduced screen to fit the available GPU budget, not a final training budget. Neural models used three seeds in CV; the deterministic baselines used one fit. Checkpoints were selected using validation profiles. Learning-rate screening compared 1e-4, 3e-4, and 1e-3 on the paper split with two seeds and validation-only scoring. Selected rates for the focused CV were 3e-4 for `oc_full` and 1e-4 for `direct_gru`; `direct_lstm` used 3e-4. Ridge used alpha=1.0 and was not tuned.

The paper-split screen ran all 32 registered experiments, each with one seed, and wrote profile metrics, window errors, training histories, and 500-step chained-rollout diagnostics. The focused CV compared persistence, Ridge, `oc_full`, direct LSTM, and direct GRU across five grouped folds. Forecasts were 20 steps from a 128-sample history. The protocol supplies true future ambient temperature during the 20-step rollout; that is an oracle-input assumption and must be stated in any comparison.

## Preliminary results

Five-fold profile-grouped CV (69 held-out profiles total), pooled 20-step MAE:

| Model | MAE °C | 95% profile-cluster bootstrap CI | Seeds |
|---|---:|---:|---:|
| Ridge direct | 0.09440 | 0.08034–0.11013 | 1 |
| Direct GRU | 0.10510 | 0.08976–0.12213 | 3 |
| Direct LSTM | 0.10561 | 0.09031–0.12314 | 3 |
| Operating-conditioned thermal (`oc_full`) | 0.10947 | 0.09440–0.12721 | 3 |
| Persistence | 0.12269 | 0.10552–0.14137 | 1 |

The Friedman test found an overall model difference (5 models, 69 profiles; chi-square 118.35, p=1.20e-24). Paired profile-level Wilcoxon tests with Holm correction versus `oc_full` favored Ridge (delta MAE −0.01507 °C, adjusted p=3.00e-9), direct GRU (−0.00437 °C, p=0.0255), and direct LSTM (−0.00386 °C, p=0.0326). The Nadeau–Bengio corrected fold tests did **not** survive Holm correction (adjusted p=0.110 for Ridge, 0.524 for GRU, 0.524 for LSTM). Treat the profile-level result as exploratory, especially given the reduced training budget and non-nested hyperparameter screen.

In the separate single-seed paper-split ablation screen, pooled MAE was 0.11191 °C for Ridge, 0.12336 °C for direct GRU, and 0.12602 °C for `oc_full`. Linear cooling scored 0.10918 °C, but that variant was not selected using validation results and its family-adjusted paired comparison was not significant (Holm p=0.0606). Do not claim it as a discovery without a pre-specified validation selection and a fresh evaluation.

The 500-step chained evaluation gave MAE 1.503 °C for `oc_full`, 1.076 °C for direct GRU, and 2.424 °C for persistence on the 11-profile paper holdout. It fed back predicted temperature while using recorded future operating signals and ambient data. This shortened, single-seed result is a diagnostic, not evidence of validated long-term performance.

The DM analysis thinned origins by 20 steps and used Bartlett HAC bandwidth `floor(sqrt(n))`, capped at `n−1`. Across outputs, thinned sample counts were 102–2,192 and bandwidths 10–46; no bandwidth exceeded its available sample count. DM and batch-means evidence varied across folds, so neither should be the headline result.

## Interpretation and study risks

The dataset source identifies `pm` as **permanent-magnet temperature**, not stator-winding temperature, and reports a 2 Hz sample rate (0.5 seconds per row). A 20-step forecast therefore covers 10 seconds; 500 steps cover 250 seconds. The manuscript title and target description must match the actual `pm` target. The other temperature column `stator_winding` is present in the CSV but is not the model target used in this runner. [Dataset description and field definitions](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature).

The current engineering features are predictive transformations. `P_cu` uses an assumed fixed resistance of 0.05 ohm; it is a proxy unless that resistance and the current convention are supported for this motor. Check the dq power convention and units before calling all engineered quantities measured physical power or loss. The near-zero first observations are startup behavior, not proof that the complete channels are normalized.

The exceptionally high R² values mostly reflect the slowly changing temperature level and should not be the headline. Report MAE/RMSE by forecast horizon, profile-level uncertainty, and errors during transitions. Windows overlap heavily, so inference must remain at profile level.

The 1% window subsampling and three-epoch training were used to keep the run practical on a 6 GB GPU. The validation sweep used the same paper split whose profiles are included in outer CV, so the CV is **not nested** and is not a fully independent estimate after model/hyperparameter screening. Ridge has one deterministic fit and fixed alpha. The 11-profile paper split has low statistical power. No sensor-based equivalence margin was justified, so do not interpret the default 0.005 °C equivalence flag.

## Modular project layout

The repository now separates notebooks, reusable experiment code, results, and reports. The exact result directories are grouped under `research/results/final/`, tuning logs under `research/results/tuning/`, and the earlier smoke/pilot artifacts under `research/results/diagnostics/`. Dataset files remain under `dataset/`; the identical root-level dataset copy was left in place for existing workflows. The modified LaTeX draft remains at the repository root because moving it was denied by the filesystem.

## Files

- Main 32-variant screen and paper statistics: [screen README](../results/final/ablation_screen_gpu/README_RESULTS.md), [summary table](../results/final/ablation_screen_gpu/stats_paper/summary_table.csv), [paired comparisons](../results/final/ablation_screen_gpu/stats_paper/pairwise_vs_ref.csv), [DM results](../results/final/ablation_screen_gpu/stats_paper/dm_tests.csv), [long-horizon summary](../results/final/ablation_screen_gpu/stats_paper/long_horizon_summary.csv), [run log](../results/final/ablation_screen_gpu/run_log.jsonl).
- Focused CV statistics: [CV README](../results/final/core_cv_gpu/README_RESULTS.md), [summary table](../results/final/core_cv_gpu/stats_cv/summary_table.csv), [paired comparisons](../results/final/core_cv_gpu/stats_cv/pairwise_vs_ref.csv), [Friedman ranks](../results/final/core_cv_gpu/stats_cv/friedman_ranks.csv), [Nadeau–Bengio results](../results/final/core_cv_gpu/stats_cv/nadeau_bengio.csv), [DM results](../results/final/core_cv_gpu/stats_cv/dm_tests.csv), [run log](../results/final/core_cv_gpu/run_log.jsonl).
- Validation-only architecture and learning-rate trials: [architecture screen](../results/tuning/gpu_tuning_validation/validation_log.jsonl), [1e-4 trials](../results/tuning/tune_lr_0p0001/validation_log.jsonl), [3e-4 trials](../results/tuning/tune_lr_0p0003/validation_log.jsonl), [1e-3 trials](../results/tuning/tune_lr_0p001/validation_log.jsonl).
- Runner and statistical code: [ablation runner](../experiments/ablation_suite.py), [statistical analysis](../experiments/stat_tests.py), [workflow guide](../README_ablation.md).

## Results wording for a draft

> In an exploratory five-fold profile-grouped evaluation of 69 PMSM operating profiles, Ridge direct forecasting achieved a pooled 20-step MAE of 0.0944 °C (95% profile-cluster bootstrap CI 0.0803–0.1101 °C), compared with 0.1051 °C for a direct GRU, 0.1056 °C for a direct LSTM, 0.1095 °C for the operating-conditioned thermal model, and 0.1227 °C for persistence. Profile-level paired tests favored Ridge and the direct recurrent models over the thermal reference after Holm correction; corrected fold-level tests did not remain significant after multiplicity correction. Because training used a 1% window sample for three epochs and model/hyperparameter screening was not nested within the outer folds, these results are screening evidence and require confirmation with a pre-specified nested evaluation and full training budget.

This paragraph is intentionally qualified; update it only after the confirmatory rerun.

## What to tell your mentor and peer

> I modularized the project into notebooks, reusable experiment code, results, and reports. I implemented a reproducible GPU ablation pipeline comparing 32 variants and added profile-cluster bootstrap intervals, Wilcoxon-Holm/BH, Friedman–Nemenyi, corrected fold tests, and a capped-HAC Diebold–Mariano analysis. I also ran a validation-only learning-rate screen and a five-fold profile-grouped comparison. In the reduced-budget run, Ridge had the lowest pooled 20-step MAE (0.0944 °C), followed by direct GRU (0.1051 °C), direct LSTM (0.1056 °C), the adaptive thermal model (0.1095 °C), and persistence (0.1227 °C). Paired profile tests favored these alternatives over the thermal reference, but corrected fold tests did not remain significant after Holm adjustment. Since training used 1% of windows for 3 epochs and tuning was not nested, these are preliminary screening results; I’m proposing a nested, full-budget evaluation before making a superiority claim.

The new research contribution is the **evaluation framework and honest baseline comparison**, not a claim that the current model is already best. The key next decision with the team is to confirm whether the paper targets permanent-magnet or stator-winding temperature and agree on the future-input scenario before the confirmatory run.

## Next work for journal submission

1. Confirm whether the paper target is PM temperature or stator-winding temperature; change the target, title, and all claims together if needed.
2. Confirm the future-input scenario. Report forecasts with known future ambient separately from a deployable scenario that forecasts or holds ambient and operating inputs.
3. Build a nested grouped-CV protocol: outer folds for final performance, inner profile folds for learning rate, weight decay, capacity, history length, loss weights, and Ridge alpha. Keep each outer test fold untouched during all choices.
4. Increase the training budget in stages on the GPU (for example, 5%, 25%, then 100% windows) and monitor validation convergence and seed variation. Use the same budget for comparable models; save code revision, environment package versions, split IDs, and all trial settings.
5. Run the pre-specified core comparison with at least three seeds. Expand the all-variant ablation to the locked nested protocol only for components central to the paper’s contribution.
6. Recompute the 20-step and 500-step evaluations under a matched, explicit input protocol. Add profile-level horizon curves, calibration/uncertainty if available, transition/error-case plots, and comparison against the Ridge/direct models.
7. Freeze a primary outcome, multiplicity family, and equivalence margin before confirmatory testing. Use profile-cluster bootstrap and paired profile-level inference as primary; retain DM only as secondary with bandwidth sensitivity and batch-means agreement.
8. Rewrite the paper around what survives confirmation. Avoid digital-twin and RUL claims unless validated on suitable operating/degradation data.

- Editable manuscript source: [PIML_PMSM_Research_Report.tex](../../PIML_PMSM_Research_Report.tex).
- Existing PDF copy: [PIML_PISM_Research_Report.pdf](PIML_PISM_Research_Report.pdf). The PDF has not been regenerated from the updated source because no LaTeX compiler is available in this environment.

