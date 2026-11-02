"""Configs for NnEnsembler. Records R² and MAE every epoch.

The width and learning rate in run_weight_net.py were one untested guess.
This run compares that guess with a few neighbors, and with the top 10 or
25 |corr| columns added as inputs. Columns are chosen on the outer training
fold only. The blend is still a weighted sum of the four model predictions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "004"))

from run_weight_net import (  # noqa: E402
    BASES,
    SRC,
    NnEnsembler,
    _metrics,
    predict_weights,
)

OUT = ROOT / "experiments/004/results/explore_ensembler.json"
PLOT = ROOT / "experiments/004/results/explore_ensembler.png"
INK = "#1c1c1c"
CONFIGS = (
    {"name": "h8", "hidden": 8, "lr": 1e-2, "k": 0},
    {"name": "h16", "hidden": 16, "lr": 1e-2, "k": 0},
    {"name": "h32", "hidden": 32, "lr": 1e-2, "k": 0},
    {"name": "h16, lr 1e-3", "hidden": 16, "lr": 1e-3, "k": 0},
    {"name": "h16, lr 1e-1", "hidden": 16, "lr": 1e-1, "k": 0},
    {"name": "h16 + 10 features", "hidden": 16, "lr": 1e-2, "k": 10},
    {"name": "h16 + 25 features", "hidden": 16, "lr": 1e-2, "k": 25},
    {"name": "h32, lr 1e-3 + 10 features", "hidden": 32, "lr": 1e-3, "k": 10},
)


def _k150():
    path = ROOT / "experiments/001/run_k150_matrix.py"
    spec = importlib.util.spec_from_file_location("k150_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rank(X, y) -> list[str]:
    scores = X.corrwith(pd.Series(np.ravel(y), index=X.index)).abs()
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index]


def _folds(y: np.ndarray):
    k150 = _k150()
    X, y_frame, _ = k150.load()
    X = X.drop(columns=k150._dead_columns(X) + k150._insane_columns(X))
    if len(X) != len(y):
        raise SystemExit(f"row mismatch {len(X)} vs {len(y)}")
    splitter = k150.KFold(n_splits=5, shuffle=True, random_state=42)
    prepared = []
    for tr, te in splitter.split(X):
        X_tr, y_tr, X_te, _, keep = k150._fold_frames(X, y_frame, tr, te)
        ranked = _rank(X_tr[keep], y_tr)
        prepared.append((tr, te, X_tr[ranked].to_numpy(), X_te[ranked].to_numpy()))
    return prepared


def _mean_curve(histories: list[list[float]]) -> np.ndarray:
    width = min(len(row) for row in histories)
    return np.mean([row[:width] for row in histories], axis=0)


def run() -> None:
    payload = json.loads(Path(SRC).read_text())
    y = np.array(payload["y"], dtype=float)
    columns = {name: np.array(payload["pred"][name], dtype=float) for name in BASES}
    bag = _metrics(y, np.array(payload["pred"]["xgboost_bag"]))
    prepared = _folds(y)
    summaries = []
    curves = []
    print(f"{'config':<28} {'test R2':>8} {'test MAE':>9} {'best epoch':>11}", flush=True)
    for cfg in CONFIGS:
        blend = np.full(len(y), np.nan)
        histories = []
        epochs = []
        for fold_id, (tr, te, X_tr, X_te) in enumerate(prepared):
            train_pred = {name: values[tr] for name, values in columns.items()}
            test_pred = {name: values[te] for name, values in columns.items()}
            extra_tr = extra_te = None
            if cfg["k"]:
                extra_tr, extra_te = X_tr[:, : cfg["k"]], X_te[:, : cfg["k"]]
            model = NnEnsembler(hidden=cfg["hidden"], lr=cfg["lr"], seed=42 + fold_id)
            model.fit(train_pred, y[tr], extra_tr)
            _, hat = predict_weights(test_pred, model, extra_te)
            blend[te] = hat
            histories.append(model.history["val_r2"])
            epochs.append(model.best_epoch)
        score = _metrics(y, blend)
        row = {
            "name": cfg["name"],
            "hidden": cfg["hidden"],
            "lr": cfg["lr"],
            "k": cfg["k"],
            "test_r2": score["r2"],
            "test_mae": score["mae"],
            "best_epoch_mean": float(np.mean(epochs)),
            "val_r2": _mean_curve(histories).tolist(),
        }
        summaries.append(row)
        curves.append(row["val_r2"])
        print(
            f"{cfg['name']:<28} {score['r2']:8.3f} {score['mae']:9.3f} {np.mean(epochs):11.0f}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"bag_r2": bag["r2"], "bag_mae": bag["mae"], "configs": summaries}))
    _plot(summaries, bag["r2"])
    print(f"wrote {OUT}", flush=True)


def _plot(summaries: list[dict], bag_r2: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for row in summaries:
        ax.plot(row["val_r2"], linewidth=1.5, label=row["name"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation R²")
    ax.set_title("Validation R² while training", loc="left", color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    order = sorted(summaries, key=lambda row: row["test_r2"])
    ax.barh([row["name"] for row in order], [row["test_r2"] for row in order], color="#1f6b4a")
    ax.axvline(bag_r2, color="#c4552a", linewidth=0.9, linestyle="--", label="XGB bag")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0.50, 0.55)
    ax.set_xlabel("Outer-fold R²")
    ax.set_title("Held-out blend", loc="left", color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#f0f0f0")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)
    print(PLOT, flush=True)


if __name__ == "__main__":
    run()
