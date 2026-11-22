#!/usr/bin/env python3
"""
ablation_suite.py
=================
Ablation / baseline / robustness experiments for the PMSM adaptive thermal
state-space model (OperatingConditionAdaptiveModel in PIML_training.ipynb).

Design rules (so results are journal-defensible)
------------------------------------------------
* Profile-disjoint splits only.  `--split paper` reproduces the notebook split
  exactly (train_test_split on profile_id.unique(), random_state=42, 70/15/15).
  `--split cv` runs grouped K-fold over ALL profiles (every profile is a test
  profile exactly once) which gives ~69 independent test units instead of 11.
* StandardScaler is fitted on training profiles of each split only.
* Every experiment is evaluated on IDENTICAL test windows, whatever its history
  length (window "end" indices are defined with the maximum history, 128).
* Training recipe = notebook recipe: AdamW(lr 3e-4, wd 1e-5), grad-clip 1.0,
  batch 512, 5 epochs, loss = MSE + 0.01*smoothness + 1e-5*variation penalty.
  Difference: the checkpoint with the best VALIDATION MAE is kept
  (use --no_val_select to use the last epoch, as the notebook did).
* Persistence baseline uses T at the forecast origin (standard definition).
* Results are stored per PROFILE (the unit for statistics) plus window-level
  error vectors for Diebold-Mariano tests.  See stat_tests.py.

Usage examples
--------------
  python ablation_suite.py --list
  python ablation_suite.py --synthetic --groups baselines architecture \
         --epochs 1 --train_frac 0.05 --seeds 0 --out smoke        # smoke test
  python research/experiments/ablation_suite.py --data dataset/measures_v2.csv --split paper \
         --groups all --seeds 0 1 2 --out research/results/paper
  python research/experiments/ablation_suite.py --data dataset/measures_v2.csv --split cv --folds 5 \
         --groups architecture features --seeds 0 1 2 --train_frac 0.25 \
         --out research/results/cv
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Constants copied from the notebook
# --------------------------------------------------------------------------
FEATURES = ["u_q", "coolant", "u_d", "motor_speed", "i_d", "i_q", "torque",
            "ambient", "I_mag", "V_mag", "P_elec", "P_mech", "P_cu"]
OPERATING = ["motor_speed", "torque", "I_mag", "V_mag", "P_elec", "P_mech",
             "P_cu", "ambient", "coolant"]
ENGINEERED = ["I_mag", "V_mag", "P_elec", "P_mech", "P_cu"]
TARGET, PID = "pm", "profile_id"
R_S = 0.05            # fixed resistance proxy used in the notebook
HORIZON = 20
MAX_HIST = 128        # history used to define the common set of evaluation windows
SOURCE_GROUP_BY_RUN = np.array([], dtype=np.int32)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identical to notebook BLOCK 5."""
    df = df.copy()
    df["I_mag"] = np.sqrt(df["i_d"] ** 2 + df["i_q"] ** 2)
    df["V_mag"] = np.sqrt(df["u_d"] ** 2 + df["u_q"] ** 2)
    df["P_elec"] = df["u_d"] * df["i_d"] + df["u_q"] * df["i_q"]
    df["omega"] = 2 * np.pi * df["motor_speed"] / 60
    df["P_mech"] = df["torque"] * df["omega"]
    df["P_cu"] = df["I_mag"] ** 2 * R_S
    return df


def make_synthetic(n_profiles=14, length=1500, seed=0) -> pd.DataFrame:
    """Tiny first-order-thermal synthetic data set. ONLY for smoke-testing the code."""
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_profiles):
        n = length + rng.randint(0, 400)
        seg = rng.randint(80, 250)

        def pw():
            return np.repeat(rng.randn(n // seg + 2), seg)[:n]

        speed, torque = pw(), pw()
        iq = 0.8 * torque + 0.05 * rng.randn(n)
        i_d = -0.5 * np.abs(torque) + 0.05 * rng.randn(n)
        ud = -0.3 * speed * iq + 0.05 * rng.randn(n)
        uq = 0.5 * speed + 0.2 * i_d + 0.05 * rng.randn(n)
        amb = 22 + 3 * rng.randn() + np.cumsum(rng.randn(n)) * 0.002
        cool = amb - 2 + 0.5 * rng.randn(n) * 0.1
        T = np.zeros(n)
        T[0] = amb[0] + 5 * rng.rand()
        for t in range(1, n):
            P = i_d[t] ** 2 + iq[t] ** 2
            T[t] = T[t - 1] + 0.02 * (8 * P - 0.15 * (T[t - 1] - amb[t]))
        rows.append(pd.DataFrame(dict(u_q=uq, coolant=cool, u_d=ud, motor_speed=speed,
                                      i_d=i_d, i_q=iq, torque=torque, ambient=amb,
                                      pm=T, profile_id=100 + p)))
    return pd.concat(rows, ignore_index=True)


def load_dataset(path: str, synthetic: bool):
    global SOURCE_GROUP_BY_RUN
    df = make_synthetic() if synthetic else pd.read_csv(path)
    # Internally number contiguous source runs in first-seen file order. This
    # preserves the notebook's unique().split order, prevents windows across
    # run boundaries, and lets CV keep any repeated source label together.
    raw_pid = df[PID].to_numpy()
    run_start = np.r_[True, raw_pid[1:] != raw_pid[:-1]]
    run_id = np.cumsum(run_start) - 1
    df["source_profile_id"] = raw_pid
    df[PID] = run_id.astype(np.int32)
    profile_ids = np.arange(int(run_id[-1]) + 1)
    SOURCE_GROUP_BY_RUN = (df.groupby(PID, sort=True)["source_profile_id"].first()
                           .to_numpy())
    df = add_physics_features(df)
    df = df.reset_index(drop=True)
    return df, profile_ids


def paper_split(profile_ids):
    tr, tmp = train_test_split(profile_ids, test_size=0.30, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42)
    return [("paper", tr, va, te)]


def cv_splits(profile_ids, n_folds, seed=0, val_frac=0.12):
    """Group contiguous runs sharing an original profile label into outer folds."""
    rng = np.random.RandomState(seed)
    ids = np.asarray(sorted(profile_ids))
    # Multiple non-adjacent runs can share the same source profile label.
    # Their group label is stored by load_dataset and passed via module global.
    group_by_run = SOURCE_GROUP_BY_RUN
    blocks = [ids[group_by_run == g] for g in np.unique(group_by_run)]
    blocks = [b for b in blocks if len(b)]
    rng.shuffle(blocks)
    folds = [[] for _ in range(n_folds)]
    sizes = np.zeros(n_folds, dtype=int)
    for block in sorted(blocks, key=len, reverse=True):
        k = int(np.argmin(sizes))
        folds[k].extend(block.tolist())
        sizes[k] += len(block)
    folds = [np.asarray(sorted(f)) for f in folds]
    out = []
    for k in range(n_folds):
        te = folds[k]
        rest = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        rest = np.random.RandomState(seed + 100 + k).permutation(rest)
        n_val = max(1, int(round(val_frac * len(rest))))
        out.append((f"cv{k}", rest[n_val:], rest[:n_val], te))
    return out


class WindowSet:
    """Row arrays of one split + the set of valid forecast origins ('ends')."""

    def __init__(self, df_split: pd.DataFrame, scaler: StandardScaler):
        self.Xs = scaler.transform(df_split[FEATURES].values).astype(np.float32)
        self.T = df_split[TARGET].values.astype(np.float32)
        self.amb = df_split["ambient"].values.astype(np.float32)
        self.pid = df_split[PID].values
        change = np.flatnonzero(self.pid[1:] != self.pid[:-1]) + 1
        starts, stops = np.r_[0, change], np.r_[change, len(self.pid)]
        ends, seg_stop = [], []
        for a, b in zip(starts, stops):
            lo, hi = a + MAX_HIST - 1, b - HORIZON - 1
            if hi >= lo:
                e = np.arange(lo, hi + 1)
                ends.append(e)
                seg_stop.append(np.full(len(e), b))
        self.ends = np.concatenate(ends)
        self.seg_stop = np.concatenate(seg_stop)
        self._cache: Dict[tuple, np.ndarray] = {}

    def __len__(self):
        return len(self.ends)

    def _cols(self, cols: Tuple[int, ...]):
        if cols not in self._cache:
            self._cache[cols] = np.ascontiguousarray(self.Xs[:, list(cols)])
        return self._cache[cols]

    def batch_ends(self, e: np.ndarray, cfg: "Cfg", device):
        L = cfg.seq_len
        feat_idx = tuple(FEATURES.index(f) for f in cfg.features)
        rows = e[:, None] + np.arange(-L + 1, 1)[None, :]
        x = self._cols(feat_idx)[rows]
        op_idx = [FEATURES.index(f) for f in cfg.operating]
        op = self.Xs[e][:, op_idx] if op_idx else np.zeros((len(e), 0), np.float32)
        fut = e[:, None] + np.arange(1, HORIZON + 1)[None, :]
        Ta = self.amb[fut]
        if cfg.ambient_future == "last":              # no knowledge of future ambient
            Ta = np.repeat(self.amb[e][:, None], HORIZON, axis=1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return t(x), t(op), t(self.T[e]), t(Ta), t(self.T[fut])

    def batch(self, idx, cfg, device):
        return self.batch_ends(self.ends[idx], cfg, device)


# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    name: str = "oc_full"
    group: str = "architecture"
    kind: str = "oc"          # oc | global | direct | persistence | linear_extrap | ridge
    features: tuple = tuple(FEATURES)
    operating: tuple = tuple(OPERATING)
    seq_len: int = 128
    hidden: int = 64
    layers: int = 2
    rnn: str = "lstm"
    use_temporal: bool = True
    cooling: str = "tanh"     # tanh | linear | none
    heat_bound: bool = True
    param_bound: str = "sigmoid"   # sigmoid | softplus
    ambient_future: str = "true"   # true | last
    smooth_w: float = 0.01
    var_w: float = 1e-5
    param_w: float = 1e-3
    delta_w: float = 0.0

    @property
    def trainable(self):
        return self.kind in ("oc", "global", "direct")


def build_registry(max_hist: int) -> Dict[str, Cfg]:
    base = Cfg()
    R: Dict[str, Cfg] = {}

    def add(group, name, **kw):
        R[name] = replace(base, name=name, group=group, **kw)

    def drop(cols):
        return dict(features=tuple(f for f in FEATURES if f not in cols),
                    operating=tuple(o for o in OPERATING if o not in cols))

    # -- classical / trivial baselines (no neural training) ------------------
    add("baselines", "persistence", kind="persistence")
    add("baselines", "linear_extrap", kind="linear_extrap")
    add("baselines", "ridge_direct", kind="ridge")
    # -- architecture ---------------------------------------------------------
    add("architecture", "oc_full")
    add("architecture", "global_adaptive", kind="global", operating=())
    add("architecture", "no_operating_branch", operating=())
    add("architecture", "no_temporal_memory", use_temporal=False)
    add("architecture", "direct_lstm", kind="direct")
    add("architecture", "direct_gru", kind="direct", rnn="gru")
    # -- physics structure of the state-space update ---------------------------
    add("physics_structure", "cooling_linear", cooling="linear")
    add("physics_structure", "cooling_none", cooling="none")
    add("physics_structure", "heat_unbounded", heat_bound=False)
    add("physics_structure", "params_softplus", param_bound="softplus")
    add("physics_structure", "ambient_last_value", ambient_future="last")
    # -- physics-feature ablation ---------------------------------------------
    add("features", "no_engineered", **drop(ENGINEERED))
    add("features", "no_Pcu", **drop(["P_cu"]))
    add("features", "no_power_features", **drop(["P_elec", "P_mech", "P_cu"]))
    add("features", "no_magnitudes", **drop(["I_mag", "V_mag"]))
    add("features", "no_coolant", **drop(["coolant"]))
    add("features", "no_ambient_input", **drop(["ambient"]))
    add("features", "only_engineered_plus_env",
        **dict(features=tuple(ENGINEERED + ["coolant", "ambient"]),
               operating=tuple(o for o in OPERATING if o in ENGINEERED + ["coolant", "ambient"])))
    # -- physics-loss ablation --------------------------------------------------
    add("loss", "mse_only", smooth_w=0.0, var_w=0.0)
    add("loss", "no_smoothness", smooth_w=0.0)
    add("loss", "no_variation_penalty", var_w=0.0)
    add("loss", "delta_consistency", delta_w=0.1)
    # -- history length ------------------------------------------------------------
    for L in (8, 16, 32, 64):
        add("history", f"hist_{L}", seq_len=L)
    if max_hist >= 256:
        add("history", "hist_256", seq_len=256)
    # -- capacity ----------------------------------------------------------------------
    add("capacity", "hidden_32", hidden=32)
    add("capacity", "hidden_128", hidden=128)
    add("capacity", "layers_1", layers=1)
    return R


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, rnn):
        super().__init__()
        cls = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = cls(in_dim, hidden, num_layers=layers, batch_first=True,
                       dropout=0.1 if layers > 1 else 0.0)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out[:, -1, :]


def mlp(i, h, o, last=None):
    layers = [nn.Linear(i, h), nn.Tanh(), nn.Linear(h, o)]
    if last is not None:
        layers.append(last)
    return nn.Sequential(*layers)


def rollout(Tc, Ta, heat, alpha, beta, gamma, H, cooling):
    """Notebook's adaptive_thermal_rollout, with switchable cooling term."""
    T, out = Tc, []
    for k in range(H):
        d = T - Ta[:, k]
        if cooling == "tanh":
            cool = beta * torch.tanh(d / 10.0)
        elif cooling == "linear":
            cool = beta * d / 10.0
        else:
            cool = torch.zeros_like(T)
        T = T + alpha * (gamma * heat - cool)
        out.append(T)
    return torch.stack(out, dim=1)


class ThermalNet(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        n_f, n_o = len(cfg.features), len(cfg.operating)
        self.enc = Encoder(n_f, cfg.hidden, cfg.layers, cfg.rnn) if cfg.use_temporal else None
        self.op_enc = (nn.Sequential(nn.Linear(n_o, 32), nn.LayerNorm(32), nn.Tanh(),
                                     nn.Linear(32, 32), nn.Tanh()) if n_o > 0 else None)
        d_in = (cfg.hidden if self.enc is not None else 0) + (32 if self.op_enc is not None else 0)
        if d_in == 0:
            raise ValueError("model needs a temporal encoder or operating features")
        heat_last = nn.Tanh() if cfg.heat_bound else None
        if cfg.kind == "oc":
            self.fusion = nn.Sequential(nn.Linear(d_in, 32), nn.Tanh())
            self.heat = mlp(32, 16, 1, heat_last)
            self.a, self.b, self.g = mlp(32, 16, 1), mlp(32, 16, 1), mlp(32, 16, 1)
        elif cfg.kind == "global":      # notebook AdaptiveThermalStateSpace
            self.heat = mlp(d_in, 32, 1, heat_last)
            self.raw = nn.Parameter(torch.tensor([-3.0, -3.0, -1.0]))
        elif cfg.kind == "direct":      # information-matched, physics-free baseline
            self.head = nn.Sequential(nn.Linear(d_in + HORIZON, 128), nn.Tanh(),
                                      nn.Linear(128, HORIZON))
        else:
            raise ValueError(cfg.kind)

    def _context(self, x, op):
        parts = []
        if self.enc is not None:
            parts.append(self.enc(x))
        if self.op_enc is not None:
            parts.append(self.op_enc(op))
        return torch.cat(parts, 1) if len(parts) > 1 else parts[0]

    def _pos(self, raw):
        if self.cfg.param_bound == "sigmoid":
            return 0.005 + 0.20 * torch.sigmoid(raw)
        return 0.005 + 0.15 * F.softplus(raw)

    def forward(self, x, op, Tc, Ta):
        cfg, c = self.cfg, self._context(x, op)
        if cfg.kind == "direct":
            inc = self.head(torch.cat([c, (Tc[:, None] - Ta) / 10.0], 1))
            return Tc[:, None] + torch.cumsum(inc, 1), {}
        if cfg.kind == "oc":
            h = self.fusion(c)
            heat = self.heat(h).squeeze(-1)
            alpha = self._pos(self.a(h).squeeze(-1))
            beta = self._pos(self.b(h).squeeze(-1))
            g = self.g(h).squeeze(-1)
            gamma = torch.sigmoid(g) if cfg.param_bound == "sigmoid" else 0.72 * F.softplus(g)
        else:  # global
            heat = self.heat(c).squeeze(-1)
            s = torch.sigmoid(self.raw)
            alpha, beta, gamma = (s[i].expand_as(heat) for i in range(3))
        pred = rollout(Tc, Ta, heat, alpha, beta, gamma, HORIZON, cfg.cooling)
        return pred, dict(alpha=alpha, beta=beta, gamma=gamma, heat=heat)


def compute_loss(cfg: Cfg, pred, Tf, Tc, aux):
    mse = F.mse_loss(pred, Tf)
    loss = mse
    if cfg.smooth_w > 0:
        loss = loss + cfg.smooth_w * ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
    if cfg.kind == "oc" and cfg.var_w > 0:
        pen = sum(torch.exp(-20.0 * aux[k].std()) for k in ("alpha", "beta", "gamma"))
        loss = loss + cfg.var_w * pen
    if cfg.kind == "global" and cfg.param_w > 0:
        loss = loss + cfg.param_w * (aux["alpha"].mean() + aux["beta"].mean())
    if cfg.delta_w > 0:
        dp = torch.diff(torch.cat([Tc[:, None], pred], 1), dim=1)
        dt = torch.diff(torch.cat([Tc[:, None], Tf], 1), dim=1)
        loss = loss + cfg.delta_w * F.mse_loss(dp, dt)
    return loss, mse


# --------------------------------------------------------------------------
# Training / prediction
# --------------------------------------------------------------------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_nn(model, cfg, ws, idx, device, bs=2048):
    model.eval()
    P = []
    for i in range(0, len(idx), bs):
        x, op, Tc, Ta, _ = ws.batch(idx[i:i + bs], cfg, device)
        pred, _ = model(x, op, Tc, Ta)
        P.append(pred.float().cpu().numpy())
    return np.concatenate(P)


def targets(ws, idx):
    e = ws.ends[idx]
    return ws.T[e[:, None] + np.arange(1, HORIZON + 1)[None, :]]


def train_nn(cfg, train_ws, val_ws, args, device, seed):
    set_seed(seed)
    model = ThermalNet(cfg).to(device)
    lr = args.lr_overrides.get(cfg.name, args.lr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    # Sample validation origins within each profile, so long profiles do not
    # monopolize checkpoint selection and every held-out profile contributes.
    val_groups = []
    val_pids = val_ws.pid[val_ws.ends]
    per_profile = max(1, args.val_windows // max(1, len(np.unique(val_pids))))
    for p in np.unique(val_pids):
        cand = np.flatnonzero(val_pids == p)
        val_groups.append(cand[np.linspace(0, len(cand) - 1,
                                           min(per_profile, len(cand))).astype(int)])
    val_idx = np.concatenate(val_groups)
    best, best_state, best_ep, hist = np.inf, None, -1, []
    n = len(train_ws)
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)[: int(n * args.train_frac)]
        run = 0.0
        for i in range(0, len(perm), args.batch):
            x, op, Tc, Ta, Tf = train_ws.batch(perm[i:i + args.batch], cfg, device)
            pred, aux = model(x, op, Tc, Ta)
            loss, _ = compute_loss(cfg, pred, Tf, Tc, aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
        vp = predict_nn(model, cfg, val_ws, val_idx, device)
        vy = targets(val_ws, val_idx)
        ae = np.abs(vp - vy).mean(axis=1)
        vpids = val_ws.pid[val_ws.ends[val_idx]]
        val_macro = float(np.mean([ae[vpids == p].mean() for p in np.unique(vpids)]))
        val_pooled = float(np.abs(vp - vy).mean())
        hist.append(dict(epoch=ep + 1, train_loss=run / max(1, len(perm) // args.batch),
                         val_macro_profile_mae=val_macro, val_pooled_mae=val_pooled))
        vmae = val_macro
        if vmae < best:
            best, best_ep = vmae, ep + 1
            best_state = copy.deepcopy(model.state_dict())
    if args.val_select and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = args.epochs
    return model, dict(best_val_mae=best, best_epoch=best_ep, history=hist,
                       n_params=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def param_diagnostics(model, cfg, ws, device, n=20000):
    """Distribution of the learned alpha/beta/gamma/heat (detects parameter collapse)."""
    if cfg.kind not in ("oc", "global"):
        return {}
    idx = np.linspace(0, len(ws) - 1, min(n, len(ws))).astype(int)
    model.eval()
    acc = {k: [] for k in ("alpha", "beta", "gamma", "heat")}
    for i in range(0, len(idx), 2048):
        x, op, Tc, Ta, _ = ws.batch(idx[i:i + 2048], cfg, device)
        _, aux = model(x, op, Tc, Ta)
        for k in acc:
            acc[k].append(aux[k].float().cpu().numpy())
    out = {}
    for k, v in acc.items():
        v = np.concatenate(v)
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


# ---- non-neural baselines ---------------------------------------------------
def predict_persistence(ws, idx):
    return np.repeat(ws.T[ws.ends[idx]][:, None], HORIZON, axis=1)


def predict_linear(ws, idx, lag=8):
    e = ws.ends[idx]
    slope = (ws.T[e] - ws.T[e - lag]) / lag
    return ws.T[e][:, None] + slope[:, None] * np.arange(1, HORIZON + 1)[None, :]


def ridge_features(ws, idx):
    e = ws.ends[idx]
    Tc = ws.T[e]
    Ta = ws.amb[e[:, None] + np.arange(1, HORIZON + 1)[None, :]]
    slopes = [(Tc - ws.T[e - l]) / l for l in (8, 32, 127)]
    lagmean = np.mean([ws.Xs[e - j] for j in range(0, 32, 4)], axis=0)
    return np.column_stack([ws.Xs[e], lagmean, (Tc - 60) / 20, (Tc - Ta.mean(1)) / 20] + slopes)


def fit_ridge(train_ws, max_n=200000, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_ws), min(max_n, len(train_ws)), replace=False)
    y = targets(train_ws, idx) - train_ws.T[train_ws.ends[idx]][:, None]
    return Ridge(alpha=1.0).fit(ridge_features(train_ws, idx), y)


def predict_ridge(model, ws, idx):
    return ws.T[ws.ends[idx]][:, None] + model.predict(ridge_features(ws, idx))


# --------------------------------------------------------------------------
# Evaluation -> per-profile table + window-level errors
# --------------------------------------------------------------------------
def profile_table(pred, y, pids, meta):
    rows, err = [], pred.astype(np.float64) - y.astype(np.float64)
    for p in np.unique(pids):
        m = pids == p
        e, yy = err[m], y[m].astype(np.float64)
        ae = np.abs(e)
        row = dict(meta, profile=int(p), n_windows=int(m.sum()), n_elem=int(e.size),
                   sum_ae=float(ae.sum()), sum_se=float((e ** 2).sum()),
                   sum_y=float(yy.sum()), sum_y2=float((yy ** 2).sum()),
                   mae=float(ae.mean()), rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
        mh = ae.mean(0)
        for k in range(HORIZON):
            row[f"mae_h{k + 1:02d}"] = float(mh[k])
        rows.append(row)
    return rows


def save_window_errors(path, pred, y, pids, ends):
    err = pred.astype(np.float64) - y
    ae, se = np.abs(err), err ** 2
    np.savez_compressed(path, pid=pids.astype(np.int32), end=ends.astype(np.int64),
                        ae_mean=ae.mean(1).astype(np.float32), se_mean=se.mean(1).astype(np.float32),
                        ae_h01=ae[:, 0].astype(np.float32), ae_h10=ae[:, 9].astype(np.float32),
                        ae_h20=ae[:, 19].astype(np.float32))


@torch.no_grad()
def long_horizon_eval(model, cfg, ws, device, n_steps, n_starts, meta):
    """Chained 20-step forecasts. Temperature is fed back (closed loop); exogenous
    inputs (currents, voltages, speed, torque, ambient) are the TRUE recorded ones.
    This is *my* protocol - align it with your RQ4 500-step protocol before quoting."""
    blocks = n_steps // HORIZON
    if model is not None:
        model.eval()
    rows = []
    for p in np.unique(ws.pid[ws.ends]):
        cand = np.flatnonzero((ws.pid[ws.ends] == p) & (ws.ends + n_steps <= ws.seg_stop - 1))
        if len(cand) == 0:
            continue
        sel = cand[np.linspace(0, len(cand) - 1, min(n_starts, len(cand))).astype(int)]
        e0 = ws.ends[sel]
        Tc = torch.from_numpy(ws.T[e0]).to(device)
        preds = []
        for b in range(blocks):
            x, op, _, Ta, _ = ws.batch_ends(e0 + b * HORIZON, cfg, device)
            if cfg.kind == "persistence":
                pred = Tc[:, None].repeat(1, HORIZON)
            else:
                pred, _ = model(x, op, Tc, Ta)
            preds.append(pred)
            Tc = pred[:, -1]
        pred = torch.cat(preds, 1).cpu().numpy()
        true = ws.T[e0[:, None] + 1 + np.arange(blocks * HORIZON)[None, :]]
        err = pred - true
        row = dict(meta, profile=int(p), n_starts=len(sel), mae=float(np.abs(err).mean()),
                   rmse=float(np.sqrt((err ** 2).mean())), final_err=float(err[:, -1].mean()),
                   final_abs_err=float(np.abs(err[:, -1]).mean()),
                   drift_per_step=float(np.mean(np.diff(err, axis=1))))
        for s in (20, 100, 250, n_steps):
            if s <= blocks * HORIZON:
                row[f"mae_step{s}"] = float(np.abs(err[:, s - 1]).mean())
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/measures_v2.csv")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data, smoke test only")
    ap.add_argument("--out", default="research/results/runs")
    ap.add_argument("--split", choices=["paper", "cv"], default="paper")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--groups", nargs="+", default=["all"],
                    help="all | baselines architecture physics_structure features loss history capacity")
    ap.add_argument("--only", nargs="*", default=None, help="explicit experiment names")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_override", action="append", default=[], metavar="MODEL=LR",
                    help="model-specific learning rate; can be repeated")
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="fraction of training windows sampled each epoch (speed knob; report it!)")
    ap.add_argument("--val_windows", type=int, default=40000)
    ap.add_argument("--val_profiles", type=int, default=0,
                    help="subsample whole validation profiles for model selection; 0 uses all")
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--no_val_select", dest="val_select", action="store_false")
    ap.add_argument("--max_hist", type=int, default=128)
    ap.add_argument("--long_horizon", type=int, default=0, help="e.g. 500; 0 disables")
    ap.add_argument("--lh_starts", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--validation_only", action="store_true",
                    help="train/log validation results only; never predict or save test results")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    args.lr_overrides = {}
    for item in args.lr_override:
        name, value = item.split("=", 1)
        args.lr_overrides[name] = float(value)

    global MAX_HIST
    MAX_HIST = max(128, args.max_hist)
    registry = build_registry(MAX_HIST)
    if args.list:
        for c in registry.values():
            print(f"{c.group:18s} {c.name:26s} kind={c.kind:13s} L={c.seq_len:3d} "
                  f"nfeat={len(c.features)} nop={len(c.operating)}")
        return
    names = list(registry) if (args.only is None and "all" in args.groups) else \
        [n for n, c in registry.items() if (args.only and n in args.only) or
         (not args.only and c.group in args.groups)]
    if "oc_full" not in names and any(registry[n].trainable for n in names):
        names.insert(0, "oc_full")      # always keep the reference model

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(os.path.join(args.out, "errors"), exist_ok=True)
    print(f"device={device}  experiments={len(names)}  seeds={args.seeds}  split={args.split}")

    df, profile_ids = load_dataset(args.data, args.synthetic)
    splits = paper_split(profile_ids) if args.split == "paper" else cv_splits(profile_ids, args.folds)

    pm_path = os.path.join(args.out, "profile_metrics.csv")
    lh_path = os.path.join(args.out, "long_horizon.csv")
    done = set()
    if os.path.exists(pm_path):
        d = pd.read_csv(pm_path, usecols=["exp", "split", "seed"]).drop_duplicates()
        done = set(map(tuple, d.values.tolist()))

    for split_id, tr, va, te in splits:
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "leakage"
        tr_df, va_df, te_df = (df[df[PID].isin(s)] for s in (tr, va, te))
        scaler = StandardScaler().fit(tr_df[FEATURES].values)         # TRAIN profiles only
        tr_ws, va_ws, te_ws = (WindowSet(d_, scaler) for d_ in (tr_df, va_df, te_df))
        if args.val_profiles and len(va) > args.val_profiles:
            rng = np.random.RandomState(9000 + int(split_id[-1]) if split_id.startswith("cv") else 9042)
            val_keep = set(rng.choice(np.asarray(va), args.val_profiles, replace=False))
            va_df_model = va_df[va_df[PID].isin(val_keep)]
            va_ws = WindowSet(va_df_model, scaler)
        print(f"[{split_id}] profiles tr/va/te = {len(tr)}/{len(va)}/{len(te)}  "
              f"windows = {len(tr_ws)}/{len(va_ws)}/{len(te_ws)}")
        te_idx = np.arange(len(te_ws))
        y_te = targets(te_ws, te_idx)
        pids = te_ws.pid[te_ws.ends]

        for name in names:
            cfg = registry[name]
            seeds = args.seeds if cfg.trainable else [0]
            for seed in seeds:
                if (name, split_id, seed) in done:
                    print(f"  skip {name} s{seed} (done)")
                    continue
                t0 = time.time()
                info, model = {}, None
                if cfg.kind == "persistence":
                    pred = predict_persistence(te_ws, te_idx)
                elif cfg.kind == "linear_extrap":
                    pred = predict_linear(te_ws, te_idx)
                elif cfg.kind == "ridge":
                    rmodel = fit_ridge(tr_ws)
                    pred = predict_ridge(rmodel, te_ws, te_idx)
                else:
                    model, info = train_nn(cfg, tr_ws, va_ws, args, device, seed)
                    if args.validation_only:
                        meta = dict(exp=name, group=cfg.group, split=split_id, seed=seed)
                        with open(os.path.join(args.out, "validation_log.jsonl"), "a") as f:
                            f.write(json.dumps(dict(meta, seconds=time.time() - t0,
                                                    cfg={k: (list(v) if isinstance(v, tuple) else v)
                                                         for k, v in cfg.__dict__.items()},
                                                    train_frac=args.train_frac, epochs=args.epochs,
                                                    lr=args.lr_overrides.get(name, args.lr), wd=args.wd, batch=args.batch,
                                                    val_profiles=sorted(map(int, va)), **info)) + "\n")
                        print(f"  {name:26s} s{seed} val_macro={info['best_val_mae']:.5f}  "
                              f"({time.time() - t0:.0f}s)")
                        del model
                        continue
                    pred = predict_nn(model, cfg, te_ws, te_idx, device, bs=args.eval_batch)
                    info.update(param_diagnostics(model, cfg, te_ws, device))
                meta = dict(exp=name, group=cfg.group, split=split_id, seed=seed)
                rows = profile_table(pred, y_te, pids, meta)
                save_window_errors(os.path.join(args.out, "errors", f"{split_id}__{name}__s{seed}.npz"),
                                   pred, y_te, pids, te_ws.ends)
                if args.long_horizon and (cfg.trainable or cfg.kind == "persistence"):
                    append_csv(lh_path, long_horizon_eval(model, cfg, te_ws, device,
                                                          args.long_horizon, args.lh_starts, meta))
                append_csv(pm_path, rows)
                pooled = sum(r["sum_ae"] for r in rows) / sum(r["n_elem"] for r in rows)
                with open(os.path.join(args.out, "run_log.jsonl"), "a") as f:
                    f.write(json.dumps(dict(meta, pooled_mae=pooled, seconds=time.time() - t0,
                                            cfg={k: (list(v) if isinstance(v, tuple) else v)
                                                 for k, v in cfg.__dict__.items()},
                                            train_frac=args.train_frac, epochs=args.epochs,
                                            lr=args.lr_overrides.get(name, args.lr), wd=args.wd,
                                            batch=args.batch, device=str(device), **info)) + "\n")
                print(f"  {name:26s} s{seed}  pooled MAE={pooled:.5f}  ({time.time() - t0:.0f}s)")
                del model


if __name__ == "__main__":
    main()
