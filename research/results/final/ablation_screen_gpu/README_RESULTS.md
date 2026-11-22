# Ablation screen results

Run: GPU, `piml_motor`, NVIDIA RTX 3050 Laptop (6 GB); 32 registry variants; paper split of 48 train / 10 validation / 11 test profiles; 3 epochs; 1% random training windows/epoch; batch 32; one seed. Best validation profile MAE checkpoint selected. AdamW weight decay 1e-5; model learning rates are in `run_log.jsonl` (oc_full 1e-3, direct_gru 1e-4, other neural variants 3e-4). Forecast history 128, horizon 20. The run also evaluated 500-step chained forecasts for trainable models and persistence, using observed future operating inputs.

Pooled test MAE (°C): Ridge 0.11191; direct GRU 0.12336; direct LSTM 0.12357; `oc_full` 0.12602; persistence 0.13501; linear extrapolation 0.21586. Cooling-linear scored 0.10918 but is a post-hoc screening result. Across 32 variants, no paired profile-level Wilcoxon comparison with `oc_full` survived global Holm correction. This split has 11 test profiles and only one seed; treat all variant deltas as exploratory.

The 500-step chained diagnostic produced MAE 1.503 °C for `oc_full`, 1.076 °C for direct GRU, and 2.424 °C for persistence. It is not comparable to the notebook's constant-forcing diagnostic and is not final long-horizon evidence.

Full tables, error vectors, plots, profile bootstrap, Friedman/Nemenyi, Wilcoxon-Holm/BH, DM, and HAC bandwidth fields are in `stats_paper/`; the exact training, validation histories and model parameters are in `run_log.jsonl`.

Limitations: the training window fraction is 1%, training is 3 epochs, only one seed was used, Ridge alpha was fixed at 1.0, and the all-variant screen used a 1e-3 learning rate for `oc_full` even though the compact validation screen slightly favored 3e-4. Do not present these as optimized or confirmatory journal results.
