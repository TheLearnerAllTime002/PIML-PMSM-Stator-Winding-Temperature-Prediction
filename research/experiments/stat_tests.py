#!/usr/bin/env python3
"""
stat_tests.py
=============
Statistical analysis of the outputs of ablation_suite.py.

Unit of analysis = the operating PROFILE (not the window, not the row).
Windows inside a profile are strongly autocorrelated (stride-1, 128+20 samples),
so row/window-level p-values are wildly over-confident.  Everything below
therefore (a) aggregates to profiles first, and (b) uses window-level data only
inside Diebold-Mariano with HAC variance and non-overlapping thinning.

Outputs (in --out):
  summary_table.csv / .tex       pooled MAE/RMSE/R2 + cluster-bootstrap 95% CI, seed SD
  pairwise_vs_ref.csv / .tex     delta-MAE vs reference, bootstrap CI, Wilcoxon, sign test,
                                 paired t, effect sizes, Holm/BH-adjusted p, equivalence
  friedman_nemenyi.csv           Friedman test, average ranks, Nemenyi p-values, CD
  dm_tests.csv                   per-profile Diebold-Mariano (HAC + HLN), Stouffer combination
  nadeau_bengio.csv              corrected resampled t-test (only for --mode cv)
  horizon_table.csv              pooled MAE per forecast step with CI (reference + baselines)
  long_horizon_summary.csv       (if long_horizon.csv exists)
  forest_plot.png                delta-MAE forest plot

Sign convention: delta = metric(variant) - metric(reference).
  delta > 0  -> variant is WORSE than the reference (the removed/changed component helped).
  delta < 0  -> variant is better.
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

H = 20
HCOLS = [f"mae_h{k:02d}" for k in range(1, H + 1)]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        adj[i] = prev
    return adj


def load_results(res_dir, mode):
    df = pd.read_csv(os.path.join(res_dir, "profile_metrics.csv"))
    df = df[df["split"] == "paper"] if mode == "paper" else df[df["split"].str.startswith("cv")]
    if df.empty:
        raise SystemExit(f"no rows for mode={mode} in {res_dir}")
    return df


def seed_average(df):
    """Average over seeds -> one row per (exp, split, profile). Sums are identical windows, so mean is exact."""
    keys = ["exp", "group", "split", "profile"]
    num = ["n_windows", "n_elem", "sum_ae", "sum_se", "sum_y", "sum_y2", "mae", "rmse", "bias"] + HCOLS
    return df.groupby(keys, as_index=False)[num].mean()


def pooled(prof, col="sum_ae"):
    return prof[col].sum() / prof["n_elem"].sum()


def pooled_r2(prof):
    n = prof["n_elem"].sum()
    sst = prof["sum_y2"].sum() - prof["sum_y"].sum() ** 2 / n
    return 1.0 - prof["sum_se"].sum() / sst


def aligned(prof, exps):
    """Dict exp -> DataFrame indexed by (split, profile), restricted to the common profile set."""
    d = {e: prof[prof.exp == e].set_index(["split", "profile"]).sort_index() for e in exps}
    common = None
    for v in d.values():
        common = v.index if common is None else common.intersection(v.index)
    return {e: v.loc[common] for e, v in d.items()}


def boot_indices(n, B, rng):
    return rng.randint(0, n, size=(B, n))


# --------------------------------------------------------------------------
# 1. summary table
# --------------------------------------------------------------------------
def summary_table(raw, prof, B, rng):
    rows = []
    for e, g in prof.groupby("exp"):
        g = g.sort_values(["split", "profile"])
        sa, ne = g["sum_ae"].values, g["n_elem"].values
        bi = boot_indices(len(g), B, rng)
        bm = sa[bi].sum(1) / ne[bi].sum(1)
        br = np.sqrt(g["sum_se"].values[bi].sum(1) / ne[bi].sum(1))
        # seed variability (training randomness), pooled MAE per seed
        r = np.array([pooled(x) for _, x in raw[raw.exp == e].groupby("seed")])
        rows.append(dict(exp=e, group=g["group"].iloc[0], n_profiles=len(g), n_seeds=len(r),
                         MAE=pooled(g), MAE_lo=np.percentile(bm, 2.5), MAE_hi=np.percentile(bm, 97.5),
                         RMSE=np.sqrt(pooled(g, "sum_se")), RMSE_lo=np.percentile(br, 2.5),
                         RMSE_hi=np.percentile(br, 97.5), R2=pooled_r2(g),
                         macroMAE=g["mae"].mean(), medianMAE=g["mae"].median(),
                         seed_sd_MAE=float(r.std(ddof=1)) if len(r) > 1 else np.nan,
                         MAE_h01=g["mae_h01"].mean(), MAE_h10=g["mae_h10"].mean(), MAE_h20=g["mae_h20"].mean()))
    out = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# 2. pairwise vs reference
# --------------------------------------------------------------------------
def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def pairwise_vs_ref(prof, ref, B, rng, margin):
    exps = [e for e in prof.exp.unique()]
    al = aligned(prof, exps)
    R = al[ref]
    rows = []
    for e in exps:
        if e == ref:
            continue
        V = al[e]
        d = (V["mae"] - R["mae"]).values            # per-profile delta MAE (macro)
        n = len(d)
        sa_v, sa_r, ne = V["sum_ae"].values, R["sum_ae"].values, R["n_elem"].values
        pooled_v, pooled_r = sa_v.sum() / ne.sum(), sa_r.sum() / ne.sum()
        bi = boot_indices(n, B, rng)
        bd = (sa_v[bi].sum(1) - sa_r[bi].sum(1)) / ne[bi].sum(1)               # pooled delta
        brel = 100 * bd / (sa_r[bi].sum(1) / ne[bi].sum(1))                      # relative %
        try:
            w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
            p_w = float(w.pvalue)
        except ValueError:
            p_w = 1.0
        nz = d[d != 0]
        p_sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
        t = stats.ttest_1samp(d, 0.0) if n > 1 and d.std() > 0 else None
        lo90, hi90 = np.percentile(bd, [5, 95])
        rows.append(dict(
            exp=e, group=V["group"].iloc[0], n_profiles=n,
            MAE_ref=pooled_r, MAE_var=pooled_v, delta_MAE=pooled_v - pooled_r,
            delta_lo=np.percentile(bd, 2.5), delta_hi=np.percentile(bd, 97.5),
            rel_delta_pct=100 * (pooled_v - pooled_r) / pooled_r,
            rel_lo=np.percentile(brel, 2.5), rel_hi=np.percentile(brel, 97.5),
            mean_profile_delta=d.mean(), median_profile_delta=np.median(d),
            frac_profiles_variant_worse=float((d > 0).mean()),
            p_wilcoxon=p_w, p_sign=float(p_sign),
            p_paired_t=float(t.pvalue) if t is not None else np.nan,
            rank_biserial=rank_biserial(d), cohen_dz=float(d.mean() / d.std(ddof=1)) if d.std() > 0 else np.nan,
            equivalent_within_margin=bool(lo90 > -margin and hi90 < margin),   # 90% CI inside +-margin (TOST-style)
        ))
    out = pd.DataFrame(rows)
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    out["p_wilcoxon_bh"] = benjamini_hochberg(out["p_wilcoxon"])
    out["p_sign_holm"] = holm(out["p_sign"])
    # Holm family = all comparisons in this table; also within-group adjustment
    out["p_wilcoxon_holm_within_group"] = np.nan
    for g, idx in out.groupby("group").groups.items():
        out.loc[idx, "p_wilcoxon_holm_within_group"] = holm(out.loc[idx, "p_wilcoxon"].values)
    out["verdict"] = np.where(out["p_wilcoxon_holm"] < 0.05,
                              np.where(out["delta_MAE"] > 0, "variant significantly worse", "variant significantly better"),
                              "no significant difference")
    return out.sort_values(["group", "delta_MAE"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Friedman + Nemenyi
# --------------------------------------------------------------------------
def friedman_nemenyi(prof, exps, label):
    al = aligned(prof, exps)
    M = np.column_stack([al[e]["mae"].values for e in exps])      # profiles x models
    N, K = M.shape
    if K < 3 or N < 5:
        return pd.DataFrame(), pd.DataFrame()
    chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(K)])
    ranks = np.apply_along_axis(stats.rankdata, 1, M)
    avg = ranks.mean(0)
    se = np.sqrt(K * (K + 1) / (12.0 * N))
    q_crit = stats.studentized_range.ppf(0.95, K, np.inf)
    cd = q_crit * se
    top = pd.DataFrame(dict(family=label, exp=exps, avg_rank=avg, n_profiles=N, K=K,
                            friedman_chi2=chi, friedman_p=p, nemenyi_CD_0_05=cd)).sort_values("avg_rank")
    pw = []
    for i in range(K):
        for j in range(i + 1, K):
            diff = abs(avg[i] - avg[j])
            pw.append(dict(family=label, a=exps[i], b=exps[j], rank_diff=diff,
                           p_nemenyi=float(stats.studentized_range.sf(diff / se, K, np.inf)),
                           significant_0_05=bool(diff > cd)))
    return top, pd.DataFrame(pw)


# --------------------------------------------------------------------------
# 4. Diebold-Mariano on window-level losses (per profile, HAC, HLN-corrected)
# --------------------------------------------------------------------------
def dm_stat(d, h, bw=None):
    """d_t = L_B - L_A ; positive mean => A better.  Newey-West/Bartlett long-run variance."""
    n = len(d)
    d = d - 0.0
    mu = d.mean()
    x = d - mu
    if bw is None:
        # Thermal loss differentials are highly persistent; the Newey-West 'automatic' rule
        # (~4*(n/100)^(2/9)) was found anti-conservative in simulation, so use sqrt(n).
        bw = max(1, int(np.floor(np.sqrt(n))))
    # A HAC lag cannot exceed the available number of lagged pairs.
    bw = min(int(bw), max(0, n - 1))
    lrv = (x @ x) / n
    for k in range(1, bw + 1):
        w = 1 - k / (bw + 1)
        lrv += 2 * w * (x[k:] @ x[:-k]) / n
    lrv = max(lrv, 1e-18)
    dm = mu / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)    # Harvey-Leybourne-Newbold
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p, bw


def dm_batch_means(d, n_batches=20):
    """Batch-means t-test on the loss differential (robust to strong persistence; fewer df)."""
    n = len(d)
    if n < 2 * n_batches:
        n_batches = max(2, n // 2)
    m = np.array([b.mean() for b in np.array_split(d, n_batches)])
    sd = m.std(ddof=1)
    if sd == 0:
        return 0.0, 1.0
    t = m.mean() / (sd / np.sqrt(len(m)))
    return float(t), float(2 * stats.t.sf(abs(t), df=len(m) - 1))


def load_errors(res_dir, split, exp, key):
    files = sorted(glob.glob(os.path.join(res_dir, "errors", f"{split}__{exp}__s*.npz")))
    if not files:
        return None
    arrs = [np.load(f) for f in files]
    pid, end = arrs[0]["pid"], arrs[0]["end"]
    for a in arrs:
        assert np.array_equal(a["pid"], pid) and np.array_equal(a["end"], end), "window mismatch"
    return pid, end, np.mean([a[key] for a in arrs], axis=0)      # loss averaged over seeds


def dm_tests(res_dir, ref, exps, split, loss="ae_mean", thin=20, bw=None):
    L = {e: load_errors(res_dir, split, e, loss) for e in set(exps) | {ref}}
    if L.get(ref) is None:
        return pd.DataFrame()
    pid, end, lref = L[ref]
    rows = []
    for e in exps:
        if e == ref or L[e] is None:
            continue
        assert np.array_equal(L[e][0], pid) and np.array_equal(L[e][1], end)
        lvar = L[e][2]
        zs, ws, ps, zb = [], [], [], []
        for p in np.unique(pid):
            m = np.flatnonzero(pid == p)[::thin]                # thinning -> (almost) non-overlapping targets
            if len(m) < 30:
                continue
            d = lvar[m].astype(np.float64) - lref[m].astype(np.float64)   # >0 => reference better
            stat, pv, bw_used = dm_stat(d, h=1, bw=bw)
            tb, pb = dm_batch_means(d)
            zs.append(stat); ps.append(pv); ws.append(np.sqrt(len(m)))
            zb.append(np.sign(tb) * stats.norm.isf(max(pb, 1e-300) / 2))
            rows.append(dict(split=split, exp=e, profile=int(p), n_thinned=len(m),
                             hac_bandwidth=bw_used, loss=loss,
                             mean_loss_diff_var_minus_ref=float(d.mean()), dm_stat=float(stat), p=float(pv),
                             p_batchmeans=pb))
        if zs:
            zs, ws = np.array(zs), np.array(ws)
            zc = (ws * zs).sum() / np.sqrt((ws ** 2).sum())    # Liptak/Stouffer weighted combination
            zbc = (ws * np.array(zb)).sum() / np.sqrt((ws ** 2).sum())
            rows.append(dict(split=split, exp=e, profile=-1, n_thinned=int(sum(ws ** 2)), loss=loss,
                             mean_loss_diff_var_minus_ref=np.nan, dm_stat=float(zc),
                             p=float(2 * stats.norm.sf(abs(zc))),
                             p_batchmeans=float(2 * stats.norm.sf(abs(zbc)))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    per = out.profile >= 0
    out.loc[per, "p_holm_within_exp"] = out[per].groupby("exp")["p"].transform(lambda s: holm(s.values))
    return out


# --------------------------------------------------------------------------
# 5. Nadeau-Bengio corrected resampled t-test (CV only)
# --------------------------------------------------------------------------
def nadeau_bengio(prof, ref):
    rows = []
    folds = sorted(prof.split.unique())
    K = len(folds)
    if K < 3:
        return pd.DataFrame()

    def fold_mae(e):
        g = prof[prof.exp == e]
        return np.array([pooled(g[g.split == f]) for f in folds])

    r = fold_mae(ref)
    for e in prof.exp.unique():
        if e == ref:
            continue
        d = fold_mae(e) - r
        var = d.var(ddof=1)
        if var <= 0:
            continue
        t = d.mean() / np.sqrt((1.0 / K + 1.0 / (K - 1)) * var)      # n_test/n_train = 1/(K-1)
        rows.append(dict(exp=e, folds=K, mean_fold_delta=d.mean(), t=float(t),
                         p=float(2 * stats.t.sf(abs(t), K - 1))))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"])
    return out


# --------------------------------------------------------------------------
# 6. horizon table, long-horizon, plotting, latex
# --------------------------------------------------------------------------
def horizon_table(prof, exps, B, rng):
    rows = []
    for e in exps:
        g = prof[prof.exp == e].sort_values(["split", "profile"])
        w = g["n_windows"].values
        for k, c in enumerate(HCOLS, 1):
            v = g[c].values
            bi = boot_indices(len(g), B, rng)
            bm = (v[bi] * w[bi]).sum(1) / w[bi].sum(1)
            rows.append(dict(exp=e, step=k, MAE=float((v * w).sum() / w.sum()),
                             lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5)))
    return pd.DataFrame(rows)


def long_horizon_summary(res_dir, ref, mode):
    path = os.path.join(res_dir, "long_horizon.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[d["split"] == "paper"] if mode == "paper" else d[d["split"].str.startswith("cv")]
    g = d.groupby(["exp", "split", "profile"], as_index=False).mean(numeric_only=True)
    ref_g = g[g.exp == ref].set_index(["split", "profile"])
    rows = []
    for e, x in g.groupby("exp"):
        x = x.set_index(["split", "profile"])
        row = dict(exp=e, n_profiles=len(x), MAE=x["mae"].mean(), MAE_median=x["mae"].median(),
                   final_abs_err=x["final_abs_err"].mean(), drift_per_step=x["drift_per_step"].mean())
        if e != ref and len(ref_g):
            idx = x.index.intersection(ref_g.index)
            dd = (x.loc[idx, "mae"] - ref_g.loc[idx, "mae"]).values
            try:
                row["p_wilcoxon_vs_ref"] = float(stats.wilcoxon(dd).pvalue)
            except ValueError:
                row["p_wilcoxon_vs_ref"] = np.nan
            row["frac_profiles_worse_than_ref"] = float((dd > 0).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "p_wilcoxon_vs_ref" in out:
        m = out["p_wilcoxon_vs_ref"].notna()
        out.loc[m, "p_holm"] = holm(out.loc[m, "p_wilcoxon_vs_ref"].values)
    return out.sort_values("MAE")


def forest_plot(pw, path, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pw.sort_values("delta_MAE").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.5))
    y = np.arange(len(d))[::-1]
    col = np.where(d["p_wilcoxon_holm"] < 0.05, "#c0392b", "#7f8c8d")
    ax.hlines(y, d["delta_lo"], d["delta_hi"], color=col, lw=2)
    ax.scatter(d["delta_MAE"], y, color=col, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["exp"])
    ax.set_xlabel(f"$\\Delta$MAE vs {ref} (°C), 95% profile-cluster bootstrap CI\n"
                  "red = Holm-adjusted Wilcoxon p < 0.05")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def to_latex(df, path, cols, fmt):
    sub = df[cols].copy()
    with open(path, "w") as f:
        f.write(sub.to_latex(index=False, float_format=fmt, escape=True))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="research/results/runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["paper", "cv"], default="paper")
    ap.add_argument("--ref", default="oc_full")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--margin", type=float, default=0.005, help="equivalence margin in deg C (state & justify it!)")
    ap.add_argument("--thin", type=int, default=20, help="DM thinning stride (>= forecast horizon)")
    ap.add_argument("--dm_bw", type=int, default=None, help="HAC bandwidth (default: floor(sqrt(n)))")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    out = args.out or os.path.join(args.results, f"stats_{args.mode}")
    os.makedirs(out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    raw = load_results(args.results, args.mode)
    prof = seed_average(raw)
    exps = list(prof.exp.unique())
    if args.ref not in exps:
        raise SystemExit(f"reference '{args.ref}' not in results: {exps}")

    # 1 summary
    S = summary_table(raw, prof, args.boot, rng)
    S.to_csv(os.path.join(out, "summary_table.csv"), index=False)
    to_latex(S, os.path.join(out, "summary_table.tex"),
             ["exp", "MAE", "MAE_lo", "MAE_hi", "RMSE", "R2", "seed_sd_MAE"], "%.4f")
    print("\n=== SUMMARY (pooled, profile-cluster bootstrap CI) ===")
    print(S[["exp", "n_profiles", "n_seeds", "MAE", "MAE_lo", "MAE_hi", "RMSE", "R2", "seed_sd_MAE"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # 2 pairwise
    P = pairwise_vs_ref(prof, args.ref, args.boot, rng, args.margin)
    P.to_csv(os.path.join(out, "pairwise_vs_ref.csv"), index=False)
    to_latex(P, os.path.join(out, "pairwise_vs_ref.tex"),
             ["exp", "delta_MAE", "delta_lo", "delta_hi", "rel_delta_pct", "p_wilcoxon", "p_wilcoxon_holm"], "%.4g")
    print(f"\n=== PAIRWISE vs {args.ref} (delta>0: variant worse) ===")
    print(P[["exp", "delta_MAE", "delta_lo", "delta_hi", "rel_delta_pct", "p_wilcoxon", "p_wilcoxon_holm", "verdict"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    forest_plot(P, os.path.join(out, "forest_plot.png"), args.ref)

    # 3 Friedman / Nemenyi, overall and per group (+ reference)
    tops, pws = [], []
    families = {"all": exps}
    for g, sub in prof.groupby("group"):
        fam = sorted(set(sub.exp) | {args.ref})
        if len(fam) >= 3:
            families[g] = fam
    for label, fam in families.items():
        t, p = friedman_nemenyi(prof, fam, label)
        tops.append(t)
        pws.append(p)
    pd.concat(tops).to_csv(os.path.join(out, "friedman_ranks.csv"), index=False)
    pd.concat(pws).to_csv(os.path.join(out, "nemenyi_pairwise.csv"), index=False)

    # 4 Diebold-Mariano
    dm_all = []
    for split in sorted(raw.split.unique()):
        dm_all.append(dm_tests(args.results, args.ref, [e for e in exps if e != args.ref], split,
                               "ae_mean", args.thin, args.dm_bw))
    dm = pd.concat([d for d in dm_all if len(d)], ignore_index=True) if any(len(d) for d in dm_all) else pd.DataFrame()
    if len(dm):
        dm.to_csv(os.path.join(out, "dm_tests.csv"), index=False)
        comb = dm[dm.profile == -1][["split", "exp", "dm_stat", "p", "p_batchmeans"]]
        print("\n=== Diebold-Mariano, Stouffer-combined over profiles (positive => reference better) ===")
        print(comb.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 5 Nadeau-Bengio (CV)
    if args.mode == "cv":
        nb = nadeau_bengio(prof, args.ref)
        nb.to_csv(os.path.join(out, "nadeau_bengio.csv"), index=False)
        print("\n=== Nadeau-Bengio corrected t-test over CV folds ===")
        print(nb.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # 6 horizon + long horizon
    keep = [e for e in ["persistence", "linear_extrap", "ridge_direct", args.ref, "global_adaptive", "direct_lstm"]
            if e in exps]
    horizon_table(prof, keep, min(args.boot, 2000), rng).to_csv(os.path.join(out, "horizon_table.csv"), index=False)
    L = long_horizon_summary(args.results, args.ref, args.mode)
    if len(L):
        L.to_csv(os.path.join(out, "long_horizon_summary.csv"), index=False)
        print("\n=== LONG-HORIZON (chained 20-step) ===")
        print(L.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nWrote tables/plots to {out}")


if __name__ == "__main__":
    main()
