#!/usr/bin/env python3
"""Nested profile-grouped CV for the primary PMSM forecasting comparison.

The outer profile fold is held out until the final prediction. In each outer
training set, a three-fold grouped inner CV selects neural learning rate and
epoch (by macro validation-profile MAE), and Ridge alpha. Final neural models
are then refit on every outer-training profile for the selected epoch count.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import ablation_suite as suite

NEURAL = ("oc_full", "direct_gru")
MODELS = ("persistence", "ridge_direct", *NEURAL)
LR_GRID = (1e-4, 3e-4)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def args_for(name, lr, epochs, batch, train_frac, val_select):
    return SimpleNamespace(lr_overrides={name: lr}, lr=lr, wd=1e-5,
                           epochs=epochs, batch=batch, train_frac=train_frac,
                           val_windows=40000, val_select=val_select)


def macro_profile_mae(pred, ws):
    y = suite.targets(ws, np.arange(len(ws)))
    ae = np.abs(pred - y).mean(axis=1)
    pids = ws.pid[ws.ends]
    return float(np.mean([ae[pids == p].mean() for p in np.unique(pids)]))


def fit_ridge_alpha(train_ws, alpha, seed=0, max_n=200000):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(train_ws), min(max_n, len(train_ws)), replace=False)
    y = suite.targets(train_ws, idx) - train_ws.T[train_ws.ends[idx]][:, None]
    return Ridge(alpha=alpha).fit(suite.ridge_features(train_ws, idx), y)


def write_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="dataset/measures_v2.csv")
    ap.add_argument("--out", default="research/results/final/nested_cv_gpu")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=2)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    suite.MAX_HIST = 128
    registry = suite.build_registry(suite.MAX_HIST)
    df, profile_ids = suite.load_dataset(args.data, False)
    out = args.out
    os.makedirs(os.path.join(out, "errors"), exist_ok=True)
    os.makedirs(os.path.join(out, "tuning"), exist_ok=True)

    outer = suite.cv_splits(profile_ids, args.outer_folds, seed=2026)
    manifest = dict(
        protocol="nested profile-grouped cross-validation",
        target=suite.TARGET, metric="inner macro profile MAE; outer pooled window MAE",
        outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        outer_seed=2026, inner_seed_base=7100, outer_training_seeds=args.seeds,
        neural_learning_rate_grid=list(LR_GRID), ridge_alpha_grid=list(RIDGE_GRID),
        weight_decay=1e-5, epochs_max=args.epochs, batch=args.batch,
        train_frac=args.train_frac, horizon=suite.HORIZON, history=suite.MAX_HIST,
        future_ambient="recorded true ambient values", device=str(device),
        torch_version=torch.__version__, cuda_name=torch.cuda.get_device_name(0)
        if device.type == "cuda" else None,
        profile_grouping="all contiguous runs with the same source profile label stay together",
        models=list(MODELS),
    )
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    metrics_path = os.path.join(out, "profile_metrics.csv")
    tuning_path = os.path.join(out, "tuning", "inner_fold_scores.csv")
    tuning_summary_path = os.path.join(out, "tuning", "selected_hyperparameters.csv")
    runlog_path = os.path.join(out, "run_log.jsonl")
    done = set()
    if os.path.exists(metrics_path):
        old = pd.read_csv(metrics_path, usecols=["exp", "split", "seed"])
        done = set(map(tuple, old.drop_duplicates().values.tolist()))

    all_ids = np.asarray(profile_ids)
    for outer_i, (_, _, _, test_ids) in enumerate(outer):
        split_id = f"cv{outer_i}"
        test_ids = np.asarray(test_ids)
        train_ids = np.setdiff1d(all_ids, test_ids)
        assert not (set(train_ids) & set(test_ids))
        inner_base = suite.cv_splits(train_ids, args.inner_folds, seed=7100 + outer_i)
        inner_parts = []
        for inner_i, (_, _, _, valid_ids) in enumerate(inner_base):
            valid_ids = np.asarray(valid_ids)
            fit_ids = np.setdiff1d(train_ids, valid_ids)
            scale_df = df[df[suite.PID].isin(fit_ids)]
            valid_df = df[df[suite.PID].isin(valid_ids)]
            scaler = StandardScaler().fit(scale_df[suite.FEATURES].values)
            inner_parts.append((
                f"inner{inner_i}", fit_ids, valid_ids,
                suite.WindowSet(scale_df, scaler), suite.WindowSet(valid_df, scaler),
            ))

        selected = {}
        for name in NEURAL:
            candidates = []
            cfg = registry[name]
            for lr in LR_GRID:
                scores, best_epochs = [], []
                for inner_i, (inner_id, fit_ids, valid_ids, tr_ws, va_ws) in enumerate(inner_parts):
                    inner_seed = 7100 + outer_i * 100 + inner_i
                    train_args = args_for(name, lr, args.epochs, args.batch,
                                          args.train_frac, True)
                    t0 = time.time()
                    model, info = suite.train_nn(cfg, tr_ws, va_ws, train_args,
                                                 device, inner_seed)
                    scores.append(float(info["best_val_mae"]))
                    best_epochs.append(int(info["best_epoch"]))
                    write_jsonl(runlog_path, dict(
                        stage="inner_tuning", outer_split=split_id, inner_split=inner_id,
                        exp=name, lr=lr, weight_decay=1e-5, seed=inner_seed,
                        fit_profiles=fit_ids.tolist(), validation_profiles=valid_ids.tolist(),
                        train_windows=len(tr_ws), validation_windows=len(va_ws),
                        train_frac=args.train_frac, epochs=args.epochs, batch=args.batch,
                        best_val_macro_mae=scores[-1], best_epoch=best_epochs[-1],
                        seconds=time.time() - t0, device=str(device)))
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    append = dict(outer_split=split_id, inner_split=inner_id,
                                  exp=name, lr=lr, wd=1e-5, seed=inner_seed,
                                  best_val_macro_mae=scores[-1], best_epoch=best_epochs[-1])
                    pd.DataFrame([append]).to_csv(tuning_path, mode="a",
                        header=not os.path.exists(tuning_path), index=False)
                candidates.append(dict(lr=lr, mean_val=float(np.mean(scores)),
                                       epochs=int(np.clip(round(statistics.median(best_epochs)),
                                                          1, args.epochs)),
                                       fold_scores=scores, fold_epochs=best_epochs))
            chosen = min(candidates, key=lambda x: (x["mean_val"], x["lr"]))
            selected[name] = chosen

        # Select Ridge alpha on the same inner validation profiles.
        ridge_candidates = []
        for alpha in RIDGE_GRID:
            scores = []
            for inner_i, (inner_id, fit_ids, valid_ids, tr_ws, va_ws) in enumerate(inner_parts):
                model = fit_ridge_alpha(tr_ws, alpha, seed=outer_i * 100 + inner_i)
                idx = np.arange(len(va_ws))
                pred = suite.predict_ridge(model, va_ws, idx)
                score = macro_profile_mae(pred, va_ws)
                scores.append(score)
                write_jsonl(runlog_path, dict(stage="inner_tuning", outer_split=split_id,
                    inner_split=inner_id, exp="ridge_direct", alpha=alpha,
                    fit_profiles=fit_ids.tolist(), validation_profiles=valid_ids.tolist(),
                    train_windows=len(tr_ws), validation_windows=len(va_ws),
                    best_val_macro_mae=score, seconds=None, device="cpu"))
            ridge_candidates.append(dict(alpha=alpha, mean_val=float(np.mean(scores)),
                                         fold_scores=scores))
        selected["ridge_direct"] = min(ridge_candidates,
                                       key=lambda x: (x["mean_val"], x["alpha"]))
        selected["persistence"] = dict(alpha=None, lr=None, epochs=None)

        summary = []
        for name in MODELS:
            hp = selected[name]
            summary.append(dict(outer_split=split_id, exp=name, lr=hp.get("lr"),
                                ridge_alpha=hp.get("alpha"),
                                selected_epochs=hp.get("epochs"),
                                inner_mean_macro_mae=hp.get("mean_val")))
        pd.DataFrame(summary).to_csv(tuning_summary_path, mode="a",
            header=not os.path.exists(tuning_summary_path), index=False)
        print(f"[{split_id}] selected: " + ", ".join(
            f"{n}={selected[n].get('lr', selected[n].get('alpha'))}" for n in MODELS),
            flush=True)

        # The cached inner windows are no longer needed for final fitting.
        del inner_parts

        outer_train_df = df[df[suite.PID].isin(train_ids)]
        test_df = df[df[suite.PID].isin(test_ids)]
        scaler = StandardScaler().fit(outer_train_df[suite.FEATURES].values)
        tr_ws = suite.WindowSet(outer_train_df, scaler)
        te_ws = suite.WindowSet(test_df, scaler)
        test_idx = np.arange(len(te_ws))
        y_test = suite.targets(te_ws, test_idx)
        pids = te_ws.pid[te_ws.ends]
        for name in MODELS:
            seeds = args.seeds if name in NEURAL else [0]
            for seed in seeds:
                if (name, split_id, seed) in done:
                    print(f"  skip completed {name} seed={seed}", flush=True)
                    continue
                t0 = time.time()
                hp = selected[name]
                info, model = {}, None
                if name == "persistence":
                    pred = suite.predict_persistence(te_ws, test_idx)
                elif name == "ridge_direct":
                    model = fit_ridge_alpha(tr_ws, hp["alpha"], seed=seed)
                    pred = suite.predict_ridge(model, te_ws, test_idx)
                else:
                    model_args = args_for(name, hp["lr"], hp["epochs"],
                                          args.batch, args.train_frac, False)
                    model, info = suite.train_nn(registry[name], tr_ws, tr_ws,
                                                 model_args, device, seed)
                    pred = suite.predict_nn(model, registry[name], te_ws, test_idx,
                                            device, bs=256)
                meta = dict(exp=name, group=registry[name].group,
                            split=split_id, seed=seed)
                rows = suite.profile_table(pred, y_test, pids, meta)
                pd.DataFrame(rows).to_csv(metrics_path, mode="a",
                    header=not os.path.exists(metrics_path), index=False)
                suite.save_window_errors(os.path.join(out, "errors",
                    f"{split_id}__{name}__s{seed}.npz"), pred, y_test, pids, te_ws.ends)
                pooled_mae = sum(r["sum_ae"] for r in rows) / sum(r["n_elem"] for r in rows)
                write_jsonl(runlog_path, dict(stage="outer_test", **meta,
                    pooled_mae=pooled_mae, selected_hyperparameters=hp,
                    outer_train_profiles=train_ids.tolist(), test_profiles=test_ids.tolist(),
                    train_windows=len(tr_ws), test_windows=len(te_ws),
                    train_frac=args.train_frac,
                    epochs=hp.get("epochs", 0) or 0, batch=args.batch,
                    weight_decay=1e-5 if name in NEURAL else None,
                    seconds=time.time() - t0, device=str(device), **info))
                done.add((name, split_id, seed))
                print(f"  {name:14s} seed={seed} outer pooled MAE={pooled_mae:.5f} C "
                      f"({time.time() - t0:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
