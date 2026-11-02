"""Model comparison on one shared |corr| top-150 set.

Every model sees the same outer folds and the same columns. The test fold
is score-only. xgboost_bag stays unless a challenger beats it by >= 0.01
mean paired R² and that gap exceeds the fold standard error.

age_mixture and age_residual are the experimental arms: specialists for
low, mid, and high training-fold age, mixed by a logistic gate on X.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from train.ridge import fit as fit_ridge  # noqa: E402
from train.svr import fit as fit_svr  # noqa: E402
from train.random_forest import fit as fit_forest  # noqa: E402
from train.xgboost import fit as fit_xgb  # noqa: E402
from GLOBAL import CV_FOLDS, RANDOM_STATE, XGB_SEEDS  # noqa: E402

MODELS = (
    "ridge",
    "elastic_net",
    "knn",
    "svr",
    "random_forest",
    "extra_trees",
    "hist_gbr",
    "mlp",
    "xgboost",
    "xgboost_bag",
    "age_mixture",
    "age_residual",
)
INCUMBENT = "xgboost_bag"
FLOOR = 0.01
K = 150
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "003" / "results" / "models.json"


def _k150():
    path = ROOT / "experiments" / "001" / "run_k150_matrix.py"
    spec = importlib.util.spec_from_file_location("k150_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(y, pred) -> dict:
    y = np.ravel(y)
    pred = np.ravel(pred)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def _rank(X: pd.DataFrame, y) -> list[str]:
    scores = X.corrwith(pd.Series(np.ravel(y), index=X.index)).abs()
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index]


def _xgb_refit(X, y, seed: int, smoke: bool):
    y = np.ravel(y)
    if smoke:
        return fit_xgb(X, y, n_estimators=20, random_state=seed)
    X_fit, X_va, y_fit, y_va = _k150().train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    probe = fit_xgb(X_fit, y_fit, X_va, y_va, random_state=seed)
    trees = int(probe.best_iteration) + 1
    return fit_xgb(X, y, n_estimators=trees, random_state=seed)


def _align_proba(gate, X, n_bins: int) -> np.ndarray:
    raw = gate.predict_proba(X)
    out = np.zeros((len(X), n_bins))
    for i, label in enumerate(gate.classes_):
        out[:, int(label)] = raw[:, i]
    return out


def _tertiles(y) -> np.ndarray:
    y = np.ravel(y)
    edges = np.quantile(y, [1 / 3, 2 / 3])
    return np.digitize(y, edges)


def _age_mixture(X_tr, y_tr, X_te, smoke: bool) -> np.ndarray:
    y = np.ravel(y_tr)
    bins = _tertiles(y)
    gate = LogisticRegression(max_iter=1000)
    gate.fit(X_tr, bins)
    preds = []
    for band in range(3):
        mask = bins == band
        take = mask if int(mask.sum()) >= 30 else np.ones(len(y), dtype=bool)
        model = _xgb_refit(X_tr.iloc[np.flatnonzero(take)], y[take], RANDOM_STATE + band, smoke)
        preds.append(np.ravel(model.predict(X_te)))
    proba = _align_proba(gate, X_te, 3)
    return (proba * np.column_stack(preds)).sum(axis=1)


def _age_residual(global_model, X_tr, y_tr, X_te) -> np.ndarray:
    y = np.ravel(y_tr)
    resid = y - np.ravel(global_model.predict(X_tr))
    bins = _tertiles(y)
    gate = LogisticRegression(max_iter=1000)
    gate.fit(X_tr, bins)
    corr = []
    for band in range(3):
        mask = np.flatnonzero(bins == band)
        model = Ridge(alpha=1.0, solver="svd")
        if len(mask) >= 5:
            model.fit(X_tr.iloc[mask], resid[mask])
            corr.append(np.ravel(model.predict(X_te)))
        else:
            corr.append(np.zeros(len(X_te)))
    proba = _align_proba(gate, X_te, 3)
    return np.ravel(global_model.predict(X_te)) + (proba * np.column_stack(corr)).sum(axis=1)


def _predict(name: str, X_tr, y_tr, X_te, smoke: bool, xgb_model=None) -> np.ndarray:
    y = np.ravel(y_tr)
    if name == "ridge":
        return np.ravel(fit_ridge(X_tr, y).predict(X_te))
    if name == "elastic_net":
        model = ElasticNetCV(
            l1_ratio=0.5,
            cv=2 if smoke else 3,
            alphas=5 if smoke else 20,
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        model.fit(X_tr, y)
        return np.ravel(model.predict(X_te))
    if name == "knn":
        model = KNeighborsRegressor(n_neighbors=5 if smoke else 25, weights="distance")
        model.fit(X_tr, y)
        return np.ravel(model.predict(X_te))
    if name == "svr":
        return np.ravel(fit_svr(X_tr, y).predict(X_te))
    if name == "random_forest":
        return np.ravel(fit_forest(X_tr, y).predict(X_te))
    if name == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=20 if smoke else 200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        model.fit(X_tr, y)
        return np.ravel(model.predict(X_te))
    if name == "hist_gbr":
        model = HistGradientBoostingRegressor(
            max_iter=20 if smoke else 300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y)
        return np.ravel(model.predict(X_te))
    if name == "mlp":
        model = MLPRegressor(
            hidden_layer_sizes=(32,) if smoke else (64, 32),
            max_iter=30 if smoke else 400,
            early_stopping=True,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y)
        return np.ravel(model.predict(X_te))
    if name == "xgboost":
        model = xgb_model if xgb_model is not None else _xgb_refit(X_tr, y, RANDOM_STATE, smoke)
        return np.ravel(model.predict(X_te))
    if name == "xgboost_bag":
        preds = [_xgb_refit(X_tr, y, seed, smoke).predict(X_te) for seed in XGB_SEEDS]
        return np.mean(preds, axis=0)
    if name == "age_mixture":
        return _age_mixture(X_tr, y, X_te, smoke)
    if name == "age_residual":
        model = xgb_model if xgb_model is not None else _xgb_refit(X_tr, y, RANDOM_STATE, smoke)
        return _age_residual(model, X_tr, y, X_te)
    raise ValueError(name)


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


def _decide(fold_rows: list[dict]) -> dict:
    table = {}
    for name in MODELS:
        r2 = np.array([row[name]["r2"] for row in fold_rows])
        rmse = np.array([row[name]["rmse"] for row in fold_rows])
        table[name] = {
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)),
            "r2_folds": [float(v) for v in r2],
            "rmse_mean": float(rmse.mean()),
        }
    challengers = {}
    for name in MODELS:
        if name == INCUMBENT:
            continue
        delta = np.array([row[name]["r2"] - row[INCUMBENT]["r2"] for row in fold_rows])
        mean, se = _paired(delta)
        challengers[name] = {
            "mean": mean,
            "se": se,
            "clears": bool(mean >= FLOOR and mean > se),
        }
    clearing = [name for name, row in challengers.items() if row["clears"]]
    keep = max(clearing, key=lambda name: table[name]["r2_mean"]) if clearing else INCUMBENT
    return {"keep": keep, "incumbent": INCUMBENT, "challengers": challengers, "table": table}


def _age_table(y: np.ndarray, pred: dict[str, np.ndarray]) -> list[dict]:
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
        for name in MODELS:
            err = pred[name][mask] - y[mask]
            rec[name] = {"mae": float(np.mean(np.abs(err))), "bias": float(err.mean())}
        rows.append(rec)
    return rows


def _print(verdict: dict) -> None:
    print(f"keep={verdict['keep']}", flush=True)
    order = sorted(MODELS, key=lambda name: verdict["table"][name]["r2_mean"], reverse=True)
    print(f"{'model':<16} {'r2':>8} {'rmse':>8} {'vs bag':>8}", flush=True)
    for name in order:
        row = verdict["table"][name]
        gap = "" if name == INCUMBENT else f"{verdict['challengers'][name]['mean']:+.3f}"
        print(f"{name:<16} {row['r2_mean']:8.3f} {row['rmse_mean']:8.3f} {gap:>8}", flush=True)


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
    os.write(fd, json.dumps({"slug": "model_selection", "pid": os.getpid()}).encode())
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool) -> dict:
    k150 = _k150()
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    if smoke:
        X, y = X.iloc[:180].copy(), y.iloc[:180].copy()
    n = len(X)
    folds_n = 2 if smoke else CV_FOLDS
    splitter = k150.KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    oof = {name: np.full(n, np.nan) for name in MODELS}
    y_oof = np.full(n, np.nan)
    fold_rows = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = k150._fold_frames(X, y, tr, te)
        cols = _rank(X_tr[keep], y_tr)[: min(K, len(keep)) if not smoke else 12]
        X_tr, X_te = X_tr[cols], X_te[cols]
        y_oof[te] = np.ravel(y_te)
        xgb_model = _xgb_refit(X_tr, y_tr, RANDOM_STATE, smoke)
        scores = {}
        for name in MODELS:
            pred = _predict(name, X_tr, y_tr, X_te, smoke, xgb_model=xgb_model)
            oof[name][te] = pred
            scores[name] = _metrics(y_te, pred)
            print(f"  {name:<16} {scores[name]['r2']:.3f}", flush=True)
        fold_rows.append(scores)
    verdict = _decide(fold_rows)
    _print(verdict)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": 12 if smoke else K,
        "folds_n": folds_n,
        "verdict": verdict,
        "age_bands": [] if smoke else _age_table(y_oof, oof),
        "y": [float(v) for v in y_oof],
        "pred": {name: [float(v) for v in oof[name]] for name in MODELS},
    }
    out = OUT.with_name("models.smoke.json") if smoke else OUT
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
