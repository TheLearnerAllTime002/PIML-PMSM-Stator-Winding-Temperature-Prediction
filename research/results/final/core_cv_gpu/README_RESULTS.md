# Earlier five-fold profile-grouped comparison

This run evaluated persistence, Ridge, `oc_full`, direct LSTM, and direct GRU over five profile-grouped folds. Each of the 69 profiles was held out once. The GPU run used three seeds for neural models and one deterministic fit for Ridge and persistence. Training was limited to three epochs, 1% of the available windows per epoch, and batch size 32. Ridge alpha was fixed at 1.0. The forecast used 128 historical samples and supplied recorded future ambient temperature for its 20-step horizon.

## Pooled 20-step results

| Model | MAE ( degC) | 95% profile-cluster bootstrap CI ( degC) | Seeds |
|---|---:|---:|---:|
| Ridge direct | **0.09440** | 0.08034-0.11013 | 1 |
| Direct GRU | 0.10510 | 0.08976-0.12213 | 3 |
| Direct LSTM | 0.10561 | 0.09031-0.12314 | 3 |
| `oc_full` | 0.10947 | 0.09440-0.12721 | 3 |
| Persistence | 0.12269 | 0.10552-0.14137 | 1 |

Friedman's test found an overall difference across the five models (chi^2 = 118.35, *p* = 1.20  x  10-^24). Profile-paired Wilcoxon-Holm comparisons favored Ridge, GRU, and LSTM over `oc_full`; however, Nadeau-Bengio corrected tests across the outer folds did not remain significant after Holm adjustment (adjusted *p* = 0.110 for Ridge and 0.524 for both GRU and LSTM).

This was a non-nested screening run: learning-rate screening had used a paper split whose profiles also appear in outer CV. Together with the 1%-window/three-epoch budget and fixed Ridge alpha, this means the metrics are not confirmatory. The later nested comparison in [`../nested_cv_gpu/README_RESULTS.md`](../nested_cv_gpu/README_RESULTS.md) is the current primary evaluation; do not combine the two result tables as a single experiment.

## Outputs

`stats_cv/` contains summary, horizon, paired-profile, Friedman/Nemenyi, Nadeau-Bengio, and DM tables. `run_log.jsonl` records seeds, settings, validation histories, and parameter diagnostics. The analysis thinned DM forecast origins by 20; per-profile HAC bandwidths were 10-46 for thinned samples of 102-2,192 and remained within the sample-count cap. The DM comparisons are secondary because of serial dependence and variation across profiles.
