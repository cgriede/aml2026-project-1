"""Inverse-density sample weights against an unweighted XGBoost.

Two ways to build the weight, both fit on the outer training fold only:

- bin: 5-year age counts, w ∝ 1 / p(y)^α
- lds: 1-year counts smoothed by a Gaussian (Yang et al. 2021), then the same inverse

α is 0.3, 0.5, or 0.7. LDS σ is 2 or 3 years. Weights are clipped at 8× the
median unnormalized weight, then scaled to mean 1. The outer test fold is
scored with plain R². sample_weight is not passed to that score, and the
early-stopping slice is unweighted.

The unweighted model stays unless a weighted arm beats it by >= 0.01 mean
paired R² and that gap exceeds the standard error of the five fold deltas.

lib/ is imported, not edited.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import (  # noqa: E402
    CV_FOLDS,
    RANDOM_STATE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)
from select_features import corr as corr_fs  # noqa: E402

# (name, spec). spec is None, ("bin", alpha), or ("lds", alpha, sigma_years).
ARMS = (
    ("none", None),
    ("bin_a03", ("bin", 0.3)),
    ("bin_a05", ("bin", 0.5)),
    ("bin_a07", ("bin", 0.7)),
    ("lds_s2_a03", ("lds", 0.3, 2.0)),
    ("lds_s2_a05", ("lds", 0.5, 2.0)),
    ("lds_s2_a07", ("lds", 0.7, 2.0)),
    ("lds_s3_a03", ("lds", 0.3, 3.0)),
    ("lds_s3_a05", ("lds", 0.5, 3.0)),
    ("lds_s3_a07", ("lds", 0.7, 3.0)),
)
SERIES = tuple(name for name, _ in ARMS)
INCUMBENT = "none"
FLOOR = 0.01
K = 150
CLIP = 8.0
BIN_WIDTH = 5.0
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "010" / "results" / "density_weights.json"
PLOT = ROOT / "experiments" / "010" / "results" / "density_weights.png"
INK = "#1c1c1c"
BIN_COLOR = "#1f6b4a"
LDS_COLOR = "#c4552a"


def _k150():
    path = ROOT / "experiments" / "001" / "run_k150_matrix.py"
    spec = importlib.util.spec_from_file_location("k150_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bin_index(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, y, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def _histogram(y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(_bin_index(y, edges), minlength=len(edges) - 1).astype(float)
    return counts, _bin_index(y, edges)


def _edges(y: np.ndarray, width: float) -> np.ndarray:
    lo = width * np.floor(float(y.min()) / width)
    hi = width * np.ceil(float(y.max()) / width)
    if hi <= lo:
        hi = lo + width
    n_bins = int(round((hi - lo) / width))
    return lo + width * np.arange(n_bins + 1)


def _from_density(p: np.ndarray, alpha: float) -> tuple[np.ndarray, dict]:
    raw = np.power(np.maximum(p, 1e-12), -float(alpha))
    cap = CLIP * float(np.median(raw))
    clipped = np.minimum(raw, cap)
    w = clipped / clipped.mean()
    info = {
        "max": float(w.max()),
        "p95": float(np.quantile(w, 0.95)),
        "clipped_frac": float(np.mean(raw > cap + 1e-9)),
    }
    return w, info


def _weights(y: np.ndarray, spec: tuple) -> tuple[np.ndarray, dict]:
    y = np.asarray(y, dtype=float).ravel()
    kind = spec[0]
    alpha = float(spec[1])
    if kind == "bin":
        counts, idx = _histogram(y, _edges(y, BIN_WIDTH))
        p = counts[idx] / counts.sum()
        return _from_density(p, alpha)
    if kind == "lds":
        sigma = float(spec[2])
        counts, idx = _histogram(y, _edges(y, 1.0))
        half = int(np.ceil(3 * sigma))
        xs = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-0.5 * (xs / sigma) ** 2)
        kernel /= kernel.sum()
        smooth = np.convolve(counts, kernel, mode="same")
        smooth = np.maximum(smooth, 1e-6)
        return _from_density(smooth[idx], alpha)
    raise ValueError(kind)


def _check() -> None:
    crowded = np.concatenate([np.full(100, 70.0), np.array([90.0])])
    w, info = _weights(crowded, ("bin", 1.0))
    assert abs(w.mean() - 1.0) < 1e-9
    assert info["clipped_frac"] > 0
    assert w[-1] > w[0] * 5

    y = np.concatenate([np.full(100, 70.0), np.array([71.0, 90.0])])
    w_lds, _ = _weights(y, ("lds", 1.0, 2.0))
    assert abs(w_lds.mean() - 1.0) < 1e-9
    # 71 borrows mass from 70. 90 does not, so it keeps the larger weight.
    assert w_lds[-1] > w_lds[-2] > w_lds[0]

    flat = np.concatenate([np.full(30, age) for age in (62.0, 67.0, 72.0, 77.0)])
    w_flat, flat_info = _weights(flat, ("bin", 0.7))
    assert flat_info["clipped_frac"] == 0.0
    assert w_flat.max() - w_flat.min() < 1e-9

    mean, se = _paired(np.array([0.02, 0.02, 0.02, 0.02, 0.02]))
    assert mean >= FLOOR and mean > se
    mean, se = _paired(np.array([0.005, 0.005, 0.005, 0.005, 0.005]))
    assert not (mean >= FLOOR and mean > se)
    print("weight check ok", flush=True)


def _metrics(y, pred) -> dict:
    y = np.ravel(y)
    pred = np.ravel(pred)
    err = pred - y
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


def _booster(n_estimators: int, early_stopping: bool, smoke: bool) -> XGBRegressor:
    if smoke:
        return XGBRegressor(n_estimators=20, max_depth=2, random_state=RANDOM_STATE, n_jobs=2)
    kwargs = dict(
        n_estimators=n_estimators,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )
    if early_stopping:
        kwargs["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
    return XGBRegressor(**kwargs)


def _fit_kwargs(w):
    return {} if w is None else {"sample_weight": w}


def _fit(X, y, spec, smoke: bool):
    """Fit on the training fold. Early-stopping rows are not weighted and not scored."""
    y = np.asarray(y, dtype=float).ravel()
    if smoke:
        w = None if spec is None else _weights(y, spec)[0]
        model = _booster(20, False, True)
        model.fit(X, y, **_fit_kwargs(w))
        return model, 20, w
    idx = np.arange(len(y))
    fit_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
    w_probe = None if spec is None else _weights(y[fit_idx], spec)[0]
    probe = _booster(XGB_N_ESTIMATORS, True, False)
    probe.fit(
        X.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        verbose=False,
        **_fit_kwargs(w_probe),
    )
    trees = int(probe.best_iteration) + 1
    w = None if spec is None else _weights(y, spec)[0]
    model = _booster(trees, False, False)
    model.fit(X, y, **_fit_kwargs(w))
    return model, trees, w


def _decide(fold_rows: list[dict]) -> dict:
    table = {}
    for name in SERIES:
        r2 = np.array([row[name]["r2"] for row in fold_rows])
        rmse = np.array([row[name]["rmse"] for row in fold_rows])
        mae = np.array([row[name]["mae"] for row in fold_rows])
        table[name] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)),
            "r2_folds": [float(v) for v in r2],
            "rmse_mean": float(rmse.mean()),
            "mae_mean": float(mae.mean()),
        }
    challengers = {}
    for name in SERIES:
        if name == INCUMBENT:
            continue
        delta = np.array([row[name]["r2"] - row[INCUMBENT]["r2"] for row in fold_rows])
        mean, se = _paired(delta)
        challengers[name] = {
            "mean": mean,
            "se": se,
            "clears": bool(mean >= FLOOR and mean > se),
            "folds": [float(v) for v in delta],
        }
    clearing = [name for name, row in challengers.items() if row["clears"]]
    keep = max(clearing, key=lambda name: table[name]["r2_mean"]) if clearing else INCUMBENT
    return {"keep": keep, "incumbent": INCUMBENT, "challengers": challengers, "table": table}


def _tails(y: np.ndarray, pred: dict[str, np.ndarray]) -> dict:
    lo, hi = np.quantile(y, [0.1, 0.9])
    bands = {
        "young": y <= lo,
        "mid": (y > lo) & (y < hi),
        "old": y >= hi,
    }
    out = {"young_max": float(lo), "old_min": float(hi)}
    for name in SERIES:
        err = np.abs(pred[name] - y)
        out[name] = {band: float(err[mask].mean()) for band, mask in bands.items()}
        out[name]["n_young"] = int(bands["young"].sum())
        out[name]["n_old"] = int(bands["old"].sum())
    return out


def _curve(age: np.ndarray, values: np.ndarray) -> dict:
    edges = _edges(age, BIN_WIDTH)
    idx = _bin_index(age, edges)
    centers, means = [], []
    for i in range(len(edges) - 1):
        mask = idx == i
        if not mask.any():
            continue
        centers.append(float(edges[i] + BIN_WIDTH / 2))
        means.append(float(values[mask].mean()))
    return {"age": centers, "value": means}


def _plot(y: np.ndarray, pred: dict[str, np.ndarray], curves: dict, verdict: dict, smoke: bool) -> None:
    table = verdict["table"]
    best_bin = max((n for n in SERIES if n.startswith("bin_")), key=lambda n: table[n]["r2_mean"])
    best_lds = max((n for n in SERIES if n.startswith("lds_")), key=lambda n: table[n]["r2_mean"])
    show = (INCUMBENT, best_bin, best_lds)
    colors = {INCUMBENT: INK, best_bin: BIN_COLOR, best_lds: LDS_COLOR}
    edges = _edges(y, BIN_WIDTH)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for name in show:
        err = np.abs(pred[name] - y)
        mae = []
        centers = []
        for i in range(len(edges) - 1):
            mask = _bin_index(y, edges) == i
            if not mask.any():
                continue
            centers.append(edges[i] + BIN_WIDTH / 2)
            mae.append(err[mask].mean())
        axes[0].plot(centers, mae, color=colors[name], linewidth=1.6, label=name)
    axes[0].set_xlabel("True age")
    axes[0].set_ylabel("MAE")
    axes[0].legend(frameon=False)
    axes[0].set_title("Out-of-fold error")
    for name in show:
        if name not in curves:
            continue
        axes[1].plot(curves[name]["age"], curves[name]["value"], color=colors[name], linewidth=1.6, label=name)
    axes[1].axhline(1.0, color="#bdbdbd", linewidth=1.0)
    axes[1].set_xlabel("True age")
    axes[1].set_ylabel("Mean training weight")
    axes[1].legend(frameon=False)
    axes[1].set_title("Fold 0 weights")
    for ax in axes:
        ax.grid(color="#f0f0f0")
        ax.set_axisbelow(True)
    fig.tight_layout()
    path = PLOT.with_name("density_weights.smoke.png") if smoke else PLOT
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _print(verdict: dict, tails: dict) -> None:
    print(f"keep={verdict['keep']}", flush=True)
    print(f"{'arm':<12} {'r2':>8} {'rmse':>8} {'mae':>8} {'vs none':>8}", flush=True)
    base = verdict["table"][INCUMBENT]["r2_mean"]
    for name in SERIES:
        row = verdict["table"][name]
        gap = "" if name == INCUMBENT else f"{row['r2_mean'] - base:+.3f}"
        print(
            f"{name:<12} {row['r2_mean']:8.3f} {row['rmse_mean']:8.3f} {row['mae_mean']:8.3f} {gap:>8}",
            flush=True,
        )
    print(
        f"tails  young<={tails['young_max']:.1f}  old>={tails['old_min']:.1f}",
        flush=True,
    )
    print(f"{'arm':<12} {'young':>8} {'mid':>8} {'old':>8}", flush=True)
    for name in SERIES:
        row = tails[name]
        print(f"{name:<12} {row['young']:8.3f} {row['mid']:8.3f} {row['old']:8.3f}", flush=True)


def _acquire() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(json.loads(LOCK.read_text()).get("pid", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
            except OSError:
                pid = -1
        if pid > 0:
            raise SystemExit(f"pipeline run held by pid {pid}: {LOCK}")
        LOCK.unlink()
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(
        fd,
        json.dumps(
            {"slug": "inverse_density", "pid": os.getpid(), "script": str(Path(__file__))}
        ).encode(),
    )
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool = False) -> dict:
    _check()
    k150 = _k150()
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    y = pd.Series(np.ravel(y), index=X.index)
    if smoke:
        X, y = X.iloc[:180].copy(), y.iloc[:180].copy()
    n = len(X)
    k = 12 if smoke else K
    folds_n = 2 if smoke else CV_FOLDS
    splitter = KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)

    oof = {name: np.full(n, np.nan) for name in SERIES}
    y_oof = np.full(n, np.nan)
    fold_rows = []
    fold_meta = []
    curves = {}

    for fid, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fid}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = k150._fold_frames(X, y, tr, te)
        cols = corr_fs.select(X_tr[keep], y_tr, k=min(k, len(keep)))
        X_tr, X_te = X_tr[cols], X_te[cols]
        y_tr_v = np.ravel(y_tr)
        y_te_v = np.ravel(y_te)
        y_oof[te] = y_te_v
        scores = {}
        for name, spec in ARMS:
            model, trees, w = _fit(X_tr, y_tr_v, spec, smoke)
            pred = np.ravel(model.predict(X_te))
            oof[name][te] = pred
            scores[name] = _metrics(y_te_v, pred)
            info = {"max": 1.0, "p95": 1.0, "clipped_frac": 0.0} if w is None else _weights(y_tr_v, spec)[1]
            print(
                f"  {name:<12} r2 {scores[name]['r2']:.3f}  trees {trees}  "
                f"w_max {info['max']:.2f}  clipped {info['clipped_frac']:.2f}",
                flush=True,
            )
            if fid == 0 and w is not None:
                curves[name] = _curve(y_tr_v, w)
            fold_meta.append(
                {
                    "fold": fid,
                    "arm": name,
                    "trees": int(trees),
                    "weight": info,
                    "r2": scores[name]["r2"],
                }
            )
        fold_rows.append(scores)

    verdict = _decide(fold_rows)
    pred = {name: oof[name] for name in SERIES}
    tails = _tails(y_oof, pred)
    _print(verdict, tails)
    _plot(y_oof, pred, curves, verdict, smoke)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": k,
        "folds_n": folds_n,
        "clip": CLIP,
        "bin_years": BIN_WIDTH,
        "incumbent": INCUMBENT,
        "verdict": verdict,
        "tails": tails,
        "folds": fold_meta,
        "weight_curves_fold0": curves,
        "y": [float(v) for v in y_oof],
        "pred": {name: [float(v) for v in oof[name]] for name in SERIES},
    }
    out = OUT.with_name("density_weights.smoke.json") if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out}", flush=True)
    return payload


def main() -> None:
    smoke = "--smoke" in sys.argv
    if not smoke:
        _acquire()
    try:
        run(smoke)
    finally:
        if not smoke:
            _release()


if __name__ == "__main__":
    main()
