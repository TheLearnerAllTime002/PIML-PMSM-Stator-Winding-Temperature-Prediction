# PMSM temperature prediction study

## Current status

The current primary result is a nested five-fold, profile-grouped evaluation of permanent-magnet temperature (`pm`). It used five outer folds and two inner folds per outer training set. Neural models used all available training windows and three seeds, with a five-epoch cap; Ridge sampled at most 200,000 windows per fit. The forecast uses 128 historical samples and a 20-step horizon with recorded future ambient temperature supplied.

| Model | Pooled 20-step MAE ( Celsius) | 95% profile-cluster bootstrap CI ( Celsius) | RMSE ( Celsius) |
|---|---:|---:|---:|
| Direct GRU | **0.09212** | 0.07572-0.11124 | 0.19519 |
| Ridge direct | 0.09433 | 0.07988-0.11050 | **0.18107** |
| Operating-conditioned (`oc_full`) | 0.09673 | 0.08066-0.11483 | 0.19873 |
| Persistence | 0.12269 | 0.10600-0.14119 | 0.22615 |

Direct GRU has the lowest pooled MAE, 4.77% lower than `oc_full`. Its profile-level paired comparison favored GRU, with a profile-bootstrap confidence interval for the MAE difference of -0.00645 to -0.00292  Celsius. However, the corrected outer-fold Nadeau-Bengio test was not significant (mean fold difference -0.00452  Celsius; Holm-adjusted *p* = 0.182). Ridge was also not significant versus `oc_full` (*p* = 0.353). The result supports describing GRU as the numerical leader, but not as a proven superior model.

This run used the full available fraction of training windows, but only up to five epochs. It is not an exhaustive search over epoch counts or model settings. Recorded future ambient values are supplied in the forecast, so the estimates describe offline conditional forecasts rather than a fully input-free deployment scenario.

## Nested protocol and tuned settings

The outer splits group profiles so each of the 69 profiles is held out once. Within each outer training portion, two grouped inner folds select the neural learning rate and epoch count using macro profile MAE; Ridge alpha is selected on the same inner partitions. Neural models use all available outer-training windows, with a five-epoch maximum and batch size 512. Ridge alpha is selected from {0.01, 0.1, 1, 10, 100}, and each Ridge fit samples at most 200,000 training windows. Neural fitting used an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB), PyTorch 2.5.1+cu121, and the `piml_motor` Conda environment.

GRU selected learning rate 1e-4 or 3e-4 and four or five epochs, depending on the outer fold. `oc_full` selected the same learning-rate and epoch choices. Ridge selected alpha 0.01, 0.1, or 100. Fold-level settings and inner validation scores are recorded under `results/final/nested_cv_gpu/tuning/`.

## Statistical results

The overall Friedman test found differences among the four methods (chi^2 = 112.88, *p* = 2.64  x  10-^24). The mean ranks were 1.81 for GRU, 1.86 for Ridge, 2.48 for `oc_full`, and 3.86 for persistence. Profile-level Wilcoxon-Holm testing favored GRU versus `oc_full` (adjusted *p* = 2.22  x  10-8), while the paired Ridge bootstrap interval crossed zero. Since profiles recur across training sets and the profile tests use out-of-fold predictions, the outer-fold corrected analysis is the conservative basis for the model claim.

DM tests were run per held-out profile after thinning forecast origins by 20. HAC bandwidths respected the `n_thinned - 1` limit; the maximum was 46 and there were no violations. DM results varied by profile and are secondary diagnostics.

## Forecast horizon

GRU horizon-specific MAE was 0.04133  Celsius at step 1 and 0.12403  Celsius at step 20. Ridge was 0.04026  Celsius at step 1 and 0.12956  Celsius at step 20; `oc_full` was 0.04031 and 0.13505  Celsius; persistence was 0.04151 and 0.19318  Celsius. The error increase with forecast distance should accompany the pooled metric in any report.

## Earlier experiments

The 32-variant ablation screen and focused five-model CV remain useful records of model development, but both used reduced training budgets and non-nested screening. Their results should not be merged with the nested result as if they used one protocol. The 32-variant screen used one seed, 1% of training windows, and three epochs. The focused CV used three epochs and 1% of windows; learning-rate screening was performed on profiles that also occur in its outer folds. A one-epoch CPU diagnostic was solely a pipeline check.

The 32-variant screen's pooled paper-split MAEs included Ridge 0.11191  Celsius, direct GRU 0.12336  Celsius, direct LSTM 0.12357  Celsius, `oc_full` 0.12602  Celsius, and persistence 0.13501  Celsius. Focused reduced-budget CV placed Ridge first at 0.09440  Celsius, then GRU 0.10510  Celsius, LSTM 0.10561  Celsius, `oc_full` 0.10947  Celsius, and persistence 0.12269  Celsius. Different splits and budgets explain why these numbers differ from the nested result.

## Dataset and interpretation

The dataset has 1,330,816 rows across 69 operating profiles and is sampled at 2 Hz. The model target is the `pm` channel, meaning permanent-magnet temperature, not stator-winding temperature. A 20-step forecast corresponds to 10 seconds at this sampling rate. The copper-loss feature is computed using an assumed fixed resistance of 0.05 ohm and should be described as a proxy unless supported by motor-specific electrical data.

The high coefficients of determination are partly driven by slowly varying temperature levels. Report MAE and RMSE by horizon and profile, along with transition cases. The dataset does not contain failure or degradation labels; the thermal-limit output is a horizon-to-threshold proxy, not validated component remaining useful life.

## Project organization

```text
dataset/                 Source data
notebooks/               Training and analysis notebooks
research/experiments/    Reusable runners and statistical code
research/results/        Run artifacts, metrics, and tuning logs
research/reports/        Study report and PDF copy
Explainability/          SHAP and thermal-limit analysis
PMSM_EDSS/               Streamlit inference application
figures/                 Evaluation and notebook figures
models/                  Saved checkpoints and scalers
```

Raw data, checkpoints, and per-window error arrays are excluded from Git because of their size. Aggregate tables, manifests, and concise reports are retained.

## Result files

- Nested comparison and complete run record: `results/final/nested_cv_gpu/README_RESULTS.md`.
- Earlier 32-variant screen: `results/final/ablation_screen_gpu/README_RESULTS.md`.
- Earlier reduced-budget grouped CV: `results/final/core_cv_gpu/README_RESULTS.md`.
- Diagnostic pipeline run: `results/diagnostics/ablation_smoke_real/README_RESULTS.md`.
- Experiment commands and statistical conventions: `README_ablation.md`.
- Editable manuscript: [`PIML_PMSM_Research_Report.tex`](../../PIML_PMSM_Research_Report.tex).

## Suggested wording for a progress update

> I completed a nested five-fold profile-grouped comparison of four PMSM permanent-magnet temperature predictors using the GPU. Direct GRU had the lowest pooled 20-step MAE at 0.0921  Celsius, followed by Ridge at 0.0943  Celsius, the operating-conditioned model at 0.0967  Celsius, and persistence at 0.1227  Celsius. Profile-level tests favored GRU, but the Holm-adjusted outer-fold test did not establish a significant advantage over the operating-conditioned model (*p* = 0.182). I logged the fold-specific hyperparameters and statistical outputs and updated the manuscript. The run used all available training windows with a five-epoch cap, so it was not an exhaustive training-duration search.

The study's current contribution is a reproducible, profile-grouped evaluation with nested tuning and an explicit comparison to direct and persistence baselines. The numerical GRU lead is promising, but an independent or expanded profile evaluation and a deployment-realistic future-input scenario are still needed before a journal-level superiority claim.
