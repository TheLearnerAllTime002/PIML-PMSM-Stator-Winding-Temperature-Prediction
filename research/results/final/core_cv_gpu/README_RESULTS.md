# Focused profile-grouped CV results

Run: GPU, `piml_motor`, NVIDIA RTX 3050 Laptop (6 GB); five folds; all 69 profiles tested once; train/validation/test sizes per fold were 48/7/13–14 profiles; 3 seeds for neural models; 3 epochs; 1% random training windows/epoch; batch 32; AdamW weight decay 1e-5. Learning rate was 3e-4 for `oc_full` and direct LSTM and 1e-4 for direct GRU. Persistence and Ridge are deterministic and used one fit; Ridge alpha was fixed at 1.0. History 128, horizon 20, recorded future ambient supplied.

Pooled MAE (°C), with 95% profile-cluster bootstrap CI:

| Model | MAE | 95% CI |
|---|---:|---:|
| Ridge direct | 0.09440 | 0.08034–0.11013 |
| Direct GRU | 0.10510 | 0.08976–0.12213 |
| Direct LSTM | 0.10561 | 0.09031–0.12314 |
| `oc_full` | 0.10947 | 0.09440–0.12721 |
| Persistence | 0.12269 | 0.10552–0.14137 |

Friedman: chi-square 118.35, p=1.20e-24. Profile-level Wilcoxon-Holm versus `oc_full` favored Ridge, direct GRU and direct LSTM. Nadeau–Bengio corrected fold tests did not survive Holm adjustment (p-adjusted 0.110, 0.524 and 0.524, respectively). This is exploratory CV, not a nested confirmatory estimate: learning-rate screening occurred on the paper split, and those profiles also appear in the outer CV evaluation. In addition, 1% windows/3 epochs is a reduced training budget. The result does not support a final superiority claim.

Outputs: `stats_cv/` contains summary, paired tests, Friedman/Nemenyi, Nadeau–Bengio and DM tables. `run_log.jsonl` records all seeds, hyperparameters, validation histories and parameter diagnostics. DM used thinning=20 and HAC bandwidth 10–46 for per-profile samples 102–2,192; the implementation caps bandwidth at n−1 and the output contains `hac_bandwidth`.
