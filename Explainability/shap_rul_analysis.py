"""Reproducible SHAP and thermal-limit RUL analysis for the saved PMSM model.

This repository contains temperature trajectories, but no observed failure or
degradation variable.  Consequently, the RUL output is explicitly a
thermal-limit RUL proxy: forecast steps until PM temperature reaches
``--temperature-limit`` under a persistence operating-condition assumption.
It must not be reported as empirical component-life RUL without degradation
labels or a validated failure criterion.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


FEATURES = ["u_q", "coolant", "u_d", "motor_speed", "i_d", "i_q",
            "torque", "ambient", "I_mag", "V_mag", "P_elec", "P_mech", "P_cu"]
ENERGY_FEATURES = ["P_cu", "P_elec", "P_mech", "I_mag", "motor_speed", "torque"]
BASE_COLUMNS = ["u_q", "coolant", "u_d", "motor_speed", "i_d", "i_q",
                "torque", "ambient"]


class RQ4ThermalStateModel(nn.Module):
    def __init__(self, input_dim=13, energy_dim=6, hidden_dim=64, horizon=20):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2,
                               batch_first=True, dropout=0.1)
        self.state_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 3))
        self.heat_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.Tanh(),
                                       nn.Linear(32, 1))
        self.dynamic_head = nn.Sequential(nn.Linear(hidden_dim + 3, 32), nn.Tanh(),
                                          nn.Linear(32, 3))
        self.residual_head = nn.Sequential(nn.Linear(hidden_dim + 3, 32), nn.Tanh(),
                                           nn.Linear(32, 1), nn.Tanh())

    def forward(self, x, t_current, t_ambient, pcu_future=None):
        encoded, _ = self.encoder(x)
        context = encoded[:, -1, :]
        thermal_state = self.state_head(context)
        base_heat = self.heat_head(context).squeeze(-1)
        dynamic_input = torch.cat([context, thermal_state], dim=1)
        params = self.dynamic_head(dynamic_input)
        heat_gain, cooling_gain, inertia = [torch.sigmoid(params[:, i]) for i in range(3)]
        temperature, predictions = t_current, []
        for k in range(self.horizon):
            pcu = pcu_future[:, k] if pcu_future is not None else torch.zeros_like(temperature)
            power_term = torch.tanh(pcu)
            heating = heat_gain * power_term
            cooling = cooling_gain * torch.tanh((temperature - t_ambient[:, k]) / 25.0)
            temperature = temperature + 0.15 * inertia * (heating - cooling)
            residual = 0.05 * self.residual_head(dynamic_input).squeeze(-1)
            predictions.append(temperature + residual)
        return torch.stack(predictions, dim=1)


def rollout(t_current, ambient, outputs):
    heat, alpha, beta, gamma, residual = outputs
    predictions, temperature = [], t_current
    for k in range(ambient.shape[1]):
        delta = alpha * (ambient[:, k] - temperature)
        delta = delta + beta * heat + gamma * residual[:, k]
        delta = 0.15 * torch.tanh(delta / 0.15)
        temperature = temperature + delta
        predictions.append(temperature)
    return torch.stack(predictions, dim=1)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["I_mag"] = np.sqrt(df["i_d"] ** 2 + df["i_q"] ** 2)
    df["V_mag"] = np.sqrt(df["u_d"] ** 2 + df["u_q"] ** 2)
    df["P_elec"] = df["u_d"] * df["i_d"] + df["u_q"] * df["i_q"]
    df["P_mech"] = df["torque"] * (2 * np.pi * df["motor_speed"] / 60.0)
    df["P_cu"] = df["I_mag"] ** 2 * 0.05
    return df


def load_scaler(path: Path) -> StandardScaler:
    try:
        import joblib
        scaler = joblib.load(path)
    except Exception:
        with path.open("rb") as fh:
            scaler = pickle.load(fh)
    if not hasattr(scaler, "transform") or getattr(scaler, "n_features_in_", None) != len(FEATURES):
        raise ValueError(f"{path} is not the expected 13-feature scaler")
    return scaler


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    features = checkpoint.get("features", FEATURES)
    if list(features) != FEATURES:
        raise ValueError(f"Checkpoint feature order differs from this analysis: {features}")
    model = RQ4ThermalStateModel(input_dim=checkpoint.get("input_dim", 13),
                                 hidden_dim=checkpoint.get("hidden_dim", 64),
                                 horizon=checkpoint.get("horizon", 20)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_samples(df, scaler, seq_len, horizon, profiles=None, max_samples=512):
    xs, es, tc, amb, actual, meta = [], [], [], [], [], []
    selected = set(profiles) if profiles else None
    for pid, group in df.groupby("profile_id", sort=True):
        if selected is not None and int(pid) not in selected:
            continue
        group = group.reset_index(drop=True)
        if len(group) < seq_len + horizon:
            continue
        scaled = scaler.transform(group[FEATURES]).astype(np.float32)
        starts = np.linspace(0, len(group) - seq_len - horizon,
                             min(8, len(group) - seq_len - horizon + 1), dtype=int)
        for start in np.unique(starts):
            end = start + seq_len
            xs.append(scaled[start:end])
            es.append(scaled[end - 1][[FEATURES.index(f) for f in ENERGY_FEATURES]])
            tc.append(group["pm"].iloc[end - 1])
            amb.append(group["ambient"].iloc[end:end + horizon].to_numpy(np.float32))
            actual.append(group["pm"].iloc[end:end + horizon].to_numpy(np.float32))
            meta.append((int(pid), int(end - 1)))
    if not xs:
        raise ValueError("No valid profile sequences found")
    order = np.linspace(0, len(xs) - 1, min(max_samples, len(xs)), dtype=int)
    return (np.asarray(xs)[order], np.asarray(es)[order], np.asarray(tc)[order],
            np.asarray(amb)[order], np.asarray(actual)[order], [meta[i] for i in order])


def explain_shap(model, x, feature_names, output_dir, device, nsamples):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required. Install it with: python -m pip install shap") from exc

    class FlatWrapper(nn.Module):
        def forward(self, flat):
            seq = flat[:, :x.shape[1] * x.shape[2]].reshape(-1, x.shape[1], x.shape[2])
            pcu = seq[:, -1, 12].unsqueeze(1).expand(-1, model.horizon)
            # Explain mean forecast; the full horizon remains in RUL outputs.
            return model(seq, torch.zeros(len(flat), device=flat.device),
                         torch.zeros(len(flat), model.horizon, device=flat.device), pcu).mean(dim=1, keepdim=True)

    flat = x.reshape(len(x), -1)
    background = torch.from_numpy(flat[:min(64, len(flat))]).to(device)
    explain = torch.from_numpy(flat[:min(nsamples, len(flat))]).to(device)
    # The wrapper's zero temperature/ambient baseline makes this a model-output
    # explanation and avoids introducing future measured values into SHAP.
    # cuDNN does not expose recurrent backward passes in eval mode.  SHAP
    # needs gradients, so disable dropout and use the deterministic training
    # kernel only for this explanation pass.
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    explainer = shap.GradientExplainer(FlatWrapper(), background)
    values = explainer.shap_values(explain)
    model.eval()
    values = values[0] if isinstance(values, list) else values
    values = np.asarray(values).squeeze()
    seq_values = values[:, :x.shape[1] * x.shape[2]].reshape(-1, x.shape[1], x.shape[2])
    feature_values = np.abs(seq_values).mean(axis=1)
    combined = feature_values.mean(axis=0)
    names = [f"{f} (sequence)" for f in feature_names]
    result = pd.DataFrame({"feature": names, "mean_abs_shap": combined})
    result = result.sort_values("mean_abs_shap", ascending=False)
    result.to_csv(output_dir / "shap_global_importance.csv", index=False)
    with (output_dir / "shap_values.npy").open("wb") as fh:
        np.save(fh, values)
    return result


@torch.no_grad()
def compute_rul(model, x, energy, temperature, ambient, actual, meta, limit, device, max_rul):
    x_t = torch.from_numpy(x).to(device)
    energy_t = torch.from_numpy(energy).to(device)
    pcu = x_t[:, -1, 12].unsqueeze(1).expand(-1, model.horizon)
    pred = model(x_t, torch.from_numpy(temperature).to(device),
                 torch.from_numpy(ambient).to(device), pcu)
    pred = pred.cpu().numpy()
    rows = []
    for i, (pid, index) in enumerate(meta):
        hit = np.flatnonzero(pred[i] >= limit)
        rul = int(hit[0] + 1) if len(hit) else max_rul
        actual_hit = np.flatnonzero(actual[i] >= limit)
        rows.append({"profile_id": pid, "origin_index": index, "predicted_temperature_C": float(pred[i, 0]),
                     "temperature_limit_C": limit, "thermal_limit_rul_steps": rul,
                     "observed_horizon_hit_step": int(actual_hit[0] + 1) if len(actual_hit) else np.nan,
                     "forecast_max_C": float(pred[i].max()), "actual_max_C": float(actual[i].max())})
    return pd.DataFrame(rows), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("measures_v2.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/rq4_v2_best.pth"))
    parser.add_argument("--scaler", type=Path, default=Path("models/feature_scaler.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("Explainability/results"))
    parser.add_argument("--temperature-limit", type=float, default=120.0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--shap-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.model, device)
    scaler = load_scaler(args.scaler)
    df = engineer(pd.read_csv(args.data))
    horizon = int(checkpoint.get("horizon", 20))
    x, energy, tc, amb, actual, meta = make_samples(df, scaler, args.seq_len, horizon, max_samples=args.max_samples)
    importance = explain_shap(model, x, FEATURES, args.output_dir, device, args.shap_samples)
    rul, predictions = compute_rul(model, x, energy, tc, amb, actual, meta,
                                   args.temperature_limit, device, horizon)
    rul.to_csv(args.output_dir / "thermal_limit_rul.csv", index=False)
    metrics = {"model": str(args.model), "checkpoint_metrics": {k: checkpoint[k] for k in checkpoint if k in ("mae", "rmse", "r2", "val_mae", "val_rmse", "val_r2")},
               "temperature_limit_C": args.temperature_limit, "sequence_length": args.seq_len,
               "forecast_horizon_steps": horizon, "samples": len(rul), "device": str(device),
               "rul_definition": "steps until forecast PM temperature reaches the configured limit; persistence operating condition",
               "warning": "Proxy RUL only: dataset contains no observed failure/degradation labels."}
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(importance.head(10).to_string(index=False))
    print(f"Saved SHAP and thermal-limit RUL outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
