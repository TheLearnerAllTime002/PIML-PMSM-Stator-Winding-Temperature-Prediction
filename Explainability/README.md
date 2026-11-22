# SHAP feature analysis and thermal-limit forecast

The explainability script summarizes feature contributions for the saved recurrent model and optionally estimates how many forecast steps remain before PM temperature reaches a configured thermal limit.

Run from the repository root in the environment containing the project's Python dependencies:

```powershell
python Explainability/shap_rul_analysis.py
```

By default, the script loads `models/rq4_v2_best.pth`, applies `models/feature_scaler.pkl`, and writes its outputs under `Explainability/results/`:

- `shap_global_importance.csv` contains mean absolute SHAP values aggregated across the 128-sample input history.
- `shap_values.npy` stores sample-level contributions for the evaluated inputs.
- `thermal_limit_rul.csv` records forecast steps until the selected PM-temperature limit is reached or the forecast ends.
- `analysis_metadata.json` records the model, horizon, seed, and thermal-limit definition.

For example, set a 100  degC threshold and evaluate up to 1,024 samples with:

```powershell
python Explainability/shap_rul_analysis.py --temperature-limit 100 --max-samples 1024
```

The temperature-limit output is a forecast horizon proxy, not validated component remaining useful life. The dataset has no failure or degradation labels. This analysis holds the latest copper-loss input constant during its 20-step forecast. Report the fraction of cases where the threshold is not reached within the forecast horizon (`thermal_limit_rul_steps == forecast_horizon_steps`) and state the chosen limit and input assumption. The nested model comparison elsewhere in the repository evaluates prediction error; it does not validate this lifetime proxy.
