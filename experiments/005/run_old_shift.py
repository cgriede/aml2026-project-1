"""Old-age detector plus one constant, on the exp 3 XGBoost bag.

The bag's out-of-fold prediction is the age. A logistic detector, fit on the
training fold only, says who is in the top fifth of training age. One shift,
the mean inner-fold residual of those old training rows, is added to the
people the detector flags. There is no second age model.

The bag stays unless hard or soft clears +0.01 mean paired R² and that gap
exceeds the standard error of the five fold deltas.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, RANDOM_STATE, XGB_SEEDS  # noqa: E402

ARMS = ("bag", "hard", "soft")
INCUMBENT = "bag"
FLOOR = 0.01
K = 150
OLD_RATE = 0.2
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "005" / "results" / "old_shift.json"
PLOT = ROOT / "experiments" / "005" / "results" / "old_shift.png"
BAG_SRC = ROOT / "experiments" / "003" / "results" / "models.json"
INK = "#1c1c1c"
DOT = "#1f6b4a"
LINE = "#c4552a"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(y, pred) -> dict:
    y = np.ravel(y)
    pred = np.ravel(pred)
    err = pred - y
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
    }


def _rank(X: pd.DataFrame, y) -> list[str]:
    scores = X.corrwith(pd.Series(np.ravel(y), index=X.index)).abs()
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index]


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


def _shift(models, X_tr, y_tr, edge: float, smoke: bool) -> float:
    """Mean (age - bag) on old rows, from inner folds that never fit those rows."""
    y = np.ravel(y_tr)
    n_splits = 2 if smoke else 3
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    seeds = (RANDOM_STATE,) if smoke else XGB_SEEDS
    parts = []
    for inner_tr, inner_te in splitter.split(X_tr):
        preds = [
            np.ravel(models._xgb_refit(X_tr.iloc[inner_tr], y[inner_tr], seed, smoke).predict(X_tr.iloc[inner_te]))
            for seed in seeds
        ]
        hat = np.mean(preds, axis=0)
        old = y[inner_te] >= edge
        if old.any():
            parts.append(y[inner_te][old] - hat[old])
    if not parts:
        return 0.0
    return float(np.mean(np.concatenate(parts)))


def _detect(X_tr, y_tr, X_te, edge: float):
    y = np.ravel(y_tr)
    label = (y >= edge).astype(int)
    gate = LogisticRegression(max_iter=1000)
    gate.fit(X_tr, label)
    p_tr = gate.predict_proba(X_tr)[:, list(gate.classes_).index(1)]
    p_te = gate.predict_proba(X_te)[:, list(gate.classes_).index(1)]
    thr = float(np.quantile(p_tr, 1.0 - float(label.mean())))
    return p_te, thr


def _decide(fold_rows: list[dict]) -> dict:
    table = {}
    for name in ARMS:
        r2 = np.array([row[name]["r2"] for row in fold_rows])
        rmse = np.array([row[name]["rmse"] for row in fold_rows])
        table[name] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)),
            "r2_folds": [float(v) for v in r2],
            "rmse_mean": float(rmse.mean()),
            "mae_mean": float(np.mean([row[name]["mae"] for row in fold_rows])),
        }
    challengers = {}
    for name in ARMS:
        if name == INCUMBENT:
            continue
        delta = np.array([row[name]["r2"] - row[INCUMBENT]["r2"] for row in fold_rows])
        mean, se = _paired(delta)
        challengers[name] = {"mean": mean, "se": se, "clears": bool(mean >= FLOOR and mean > se)}
    clearing = [name for name, row in challengers.items() if row["clears"]]
    keep = max(clearing, key=lambda name: table[name]["r2_mean"]) if clearing else INCUMBENT
    return {"keep": keep, "incumbent": INCUMBENT, "challengers": challengers, "table": table}


def _bands(y: np.ndarray, pred: dict[str, np.ndarray]) -> list[dict]:
    edges = np.quantile(y, [0, 0.2, 0.4, 0.6, 0.8, 1])
    bands = np.digitize(y, edges[1:-1], right=True)
    rows = []
    for band in range(5):
        mask = bands == band
        rec = {
            "band": int(band + 1),
            "age_min": float(y[mask].min()),
            "age_max": float(y[mask].max()),
            "n": int(mask.sum()),
        }
        for name in ARMS:
            err = pred[name][mask] - y[mask]
            rec[name] = {"mae": float(np.mean(np.abs(err))), "bias": float(err.mean())}
        rows.append(rec)
    return rows


def _plot(y: np.ndarray, pred: dict[str, np.ndarray], bands: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    lo, hi = float(y.min()) - 2, float(y.max()) + 2
    for ax, name, title in (
        (axes[0], "bag", "Bag"),
        (axes[1], "hard", "Bag plus old shift"),
    ):
        ax.scatter(y, pred[name], s=8, alpha=0.35, color=DOT, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=LINE, linewidth=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True age")
        ax.set_ylabel("Predicted age")
        ax.set_title(title, loc="left", color=INK)
        ax.set_aspect("equal", adjustable="box")
    labels = [f"{row['age_min']:.0f}-{row['age_max']:.0f}" for row in bands]
    xs = np.arange(len(bands))
    width = 0.36
    axes[2].bar(xs - width / 2, [row["bag"]["mae"] for row in bands], width, color="#bdbdbd", label="Bag")
    axes[2].bar(xs + width / 2, [row["hard"]["mae"] for row in bands], width, color=DOT, label="Hard shift")
    axes[2].set_xticks(xs, labels, rotation=30, ha="right")
    axes[2].set_ylabel("MAE (years)")
    axes[2].set_xlabel("True age")
    axes[2].set_title("MAE by quintile", loc="left", color=INK)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#f0f0f0")
        ax.tick_params(colors=INK, labelsize=8)
    fig.tight_layout()
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=140)


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
    os.write(fd, json.dumps({"slug": "old_shift", "pid": os.getpid()}).encode())
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool) -> dict:
    models = _load("run_models", ROOT / "experiments" / "003" / "run_models.py")
    k150 = models._k150()
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    X = X.drop(columns=drop)
    stored_bag = None
    if not smoke:
        stored = json.loads(BAG_SRC.read_text())
        y_stored = np.asarray(stored["y"], dtype=float)
        if not np.allclose(y_stored, np.ravel(y)):
            raise SystemExit("exp 3 predictions are not on these rows")
        stored_bag = np.asarray(stored["pred"]["xgboost_bag"], dtype=float)
    if smoke:
        X, y = X.iloc[:180].copy(), y.iloc[:180].copy()
    n = len(X)
    folds_n = 2 if smoke else CV_FOLDS
    splitter = k150.KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.full(n, np.nan) for name in ARMS}
    y_oof = np.full(n, np.nan)
    fold_rows = []
    diagnostics = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = k150._fold_frames(X, y, tr, te)
        cols = _rank(X_tr[keep], y_tr)[: 12 if smoke else min(K, len(keep))]
        X_tr, X_te = X_tr[cols], X_te[cols]
        y_tr_v = np.ravel(y_tr)
        y_te_v = np.ravel(y_te)
        edge = float(np.quantile(y_tr_v, 1.0 - OLD_RATE))
        shift = _shift(models, X_tr, y_tr, edge, smoke)
        p_te, thr = _detect(X_tr, y_tr, X_te, edge)
        if stored_bag is None:
            bag = np.mean(
                [np.ravel(models._xgb_refit(X_tr, y_tr, seed, True).predict(X_te)) for seed in (RANDOM_STATE,)],
                axis=0,
            )
        else:
            bag = stored_bag[te]
        flagged = p_te >= thr
        pred = {
            "bag": bag,
            "hard": bag + flagged.astype(float) * shift,
            "soft": bag + p_te * shift,
        }
        y_oof[te] = y_te_v
        scores = {}
        for name in ARMS:
            oof[name][te] = pred[name]
            scores[name] = _metrics(y_te_v, pred[name])
        truly_old = y_te_v >= edge
        auc = float(roc_auc_score(truly_old.astype(int), p_te)) if truly_old.any() and (~truly_old).any() else float("nan")
        diagnostics.append(
            {
                "fold": fold_id,
                "edge": edge,
                "shift": shift,
                "threshold": thr,
                "flagged": int(flagged.sum()),
                "n_test": int(len(te)),
                "detector_auc": auc,
                "old_mae_bag": float(np.mean(np.abs(bag[truly_old] - y_te_v[truly_old]))) if truly_old.any() else None,
                "old_mae_hard": float(np.mean(np.abs(pred["hard"][truly_old] - y_te_v[truly_old]))) if truly_old.any() else None,
            }
        )
        print(
            f"  edge {edge:.1f}  shift {shift:+.2f}  flagged {int(flagged.sum())}/{len(te)}  "
            f"auc {auc:.3f}  hard R2 {scores['hard']['r2']:.3f}  bag R2 {scores['bag']['r2']:.3f}",
            flush=True,
        )
        fold_rows.append(scores)
    verdict = _decide(fold_rows)
    print(f"keep={verdict['keep']}", flush=True)
    for name in ARMS:
        row = verdict["table"][name]
        gap = "" if name == INCUMBENT else f"{verdict['challengers'][name]['mean']:+.3f}"
        print(f"{name:<8} {row['r2_mean']:8.3f} {row['rmse_mean']:8.3f} {gap:>8}", flush=True)
    bands = [] if smoke else _bands(y_oof, oof)
    if not smoke:
        _plot(y_oof, oof, bands)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": 12 if smoke else K,
        "folds_n": folds_n,
        "old_rate": OLD_RATE,
        "verdict": verdict,
        "folds": diagnostics,
        "age_bands": bands,
        "y": [float(v) for v in y_oof],
        "pred": {name: [float(v) for v in oof[name]] for name in ARMS},
    }
    out = OUT.with_name("old_shift.smoke.json") if smoke else OUT
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
