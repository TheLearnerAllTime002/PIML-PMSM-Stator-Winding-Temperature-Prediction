# SHAP and thermal-limit RUL analysis

Run from the repository root:

```powershell
python Explainability/shap_rul_analysis.py
```

The default run loads `models/rq4_v2_best.pth`, applies the saved
`models/feature_scaler.pkl`, and writes these files to `Explainability/results/`:

- `shap_global_importance.csv` — mean absolute SHAP values aggregated over the 128-step input history.
- `shap_values.npy` — sample-level SHAP values for reproducibility.
- `thermal_limit_rul.csv` — forecast steps until PM temperature reaches the configured limit.
- `analysis_metadata.json` — model, horizon, seed, and RUL definition.

The RUL output is a thermal-limit RUL proxy, not empirical component RUL. The
dataset contains no failure/degradation labels. It assumes the latest copper
loss remains constant and forecasts 20 steps. Change the limit, sample count,
or output directory, for example:

```powershell
python Explainability/shap_rul_analysis.py --temperature-limit 100 --max-samples 1024
```

For publication, report the right-censoring rate (`thermal_limit_rul_steps ==
forecast_horizon_steps`) and do not call this a validated lifetime estimate
until run-to-failure data or a validated ageing/failure criterion is available.
