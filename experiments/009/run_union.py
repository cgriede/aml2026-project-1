"""Two specialists on different feature sets, then a leakage-free blend.

corr_xgb stays unless mean or nnls beats it by >= 0.01 mean paired R²
and that gap exceeds the standard error of the five fold deltas.

lib/ is imported, not edited. Selection, scaling, and the correlation cull
fit on the outer training fold only. Defect columns are dropped once.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, RANDOM_STATE  # noqa: E402
from select_features import corr as corr_fs  # noqa: E402
from select_features import rrelieff as rrelieff_fs  # noqa: E402
from train.svr import fit as fit_svr  # noqa: E402
from train.xgboost import fit as fit_xgb  # noqa: E402

SERIES = ("corr_xgb", "rrelieff_svr", "mean", "nnls")
INCUMBENT = "corr_xgb"
CHALLENGERS = ("mean", "nnls")
FLOOR = 0.01
K = 150
INNER_FOLDS = 3
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "009" / "results" / "union.json"


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
        "mae": float(np.mean(np.abs(y - pred))),
    }


def _jaccard(a, b) -> float:
    left, right = set(a), set(b)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


def _xgb_refit(X, y, smoke: bool, seed: int = RANDOM_STATE):
    y = np.ravel(y)
    if smoke:
        # 20 trees at the full-run learning rate (0.03) and min_child_weight 20
        # stay on the training mean. Same smoke booster as exp 001.
        model = XGBRegressor(n_estimators=20, max_depth=2, random_state=seed, n_jobs=2)
        model.fit(X, y)
        return model
    X_fit, X_va, y_fit, y_va = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    probe = fit_xgb(X_fit, y_fit, X_va, y_va, random_state=seed)
    trees = int(probe.best_iteration) + 1
    return fit_xgb(X, y, n_estimators=trees, random_state=seed)


def _predict_corr_xgb(X_tr, y_tr, X_te, cols, smoke: bool) -> np.ndarray:
    model = _xgb_refit(X_tr[cols], y_tr, smoke)
    return np.ravel(model.predict(X_te[cols]))


def _predict_rrelieff_svr(X_tr, y_tr, X_te, cols) -> np.ndarray:
    model = fit_svr(X_tr[cols], y_tr)
    return np.ravel(model.predict(X_te[cols]))


def _inner_oof(X_tr, y_tr, corr_cols, rrel_cols, smoke: bool) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold specialist preds on the outer training rows. Columns fixed."""
    n = len(X_tr)
    oof_corr = np.full(n, np.nan)
    oof_svr = np.full(n, np.nan)
    folds_n = 2 if smoke else INNER_FOLDS
    splitter = KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    for inner_tr, inner_te in splitter.split(X_tr):
        X_i, X_j = X_tr.iloc[inner_tr], X_tr.iloc[inner_te]
        y_i = y_tr.iloc[inner_tr] if hasattr(y_tr, "iloc") else y_tr[inner_tr]
        oof_corr[inner_te] = _predict_corr_xgb(X_i, y_i, X_j, corr_cols, smoke)
        oof_svr[inner_te] = _predict_rrelieff_svr(X_i, y_i, X_j, rrel_cols)
    return oof_corr, oof_svr


def _fit_nnls(oof_corr, oof_svr, y) -> LinearRegression:
    stack = np.column_stack([oof_corr, oof_svr])
    model = LinearRegression(positive=True, fit_intercept=True)
    model.fit(stack, np.ravel(y))
    return model


def _apply_nnls(model: LinearRegression, pred_corr, pred_svr) -> np.ndarray:
    stack = np.column_stack([np.ravel(pred_corr), np.ravel(pred_svr)])
    return np.ravel(model.predict(stack))


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
    for name in CHALLENGERS:
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
    return {
        "keep": keep,
        "incumbent": INCUMBENT,
        "challengers": challengers,
        "table": table,
    }


def _print(verdict: dict, jaccards: list[float]) -> None:
    print(f"keep={verdict['keep']}  mean_jaccard={np.mean(jaccards):.3f}", flush=True)
    print(f"{'arm':<16} {'r2':>8} {'rmse':>8} {'mae':>8} {'vs corr':>8}", flush=True)
    for name in SERIES:
        row = verdict["table"][name]
        if name == INCUMBENT:
            gap = ""
        elif name in verdict["challengers"]:
            gap = f"{verdict['challengers'][name]['mean']:+.3f}"
        else:
            delta = row["r2_mean"] - verdict["table"][INCUMBENT]["r2_mean"]
            gap = f"{delta:+.3f}"
        print(
            f"{name:<16} {row['r2_mean']:8.3f} {row['rmse_mean']:8.3f} "
            f"{row['mae_mean']:8.3f} {gap:>8}",
            flush=True,
        )


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
        json.dumps({"slug": "feature_union", "pid": os.getpid(), "script": str(Path(__file__))}).encode(),
    )
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool = False) -> dict:
    k150 = _k150()
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    nan_count = X.isna().sum(axis=1).astype(float).to_numpy()
    y = pd.Series(np.ravel(y), index=X.index)
    if smoke:
        X, y = X.iloc[:180].copy(), y.iloc[:180].copy()
        nan_count = nan_count[:180]
    n = len(X)
    k = 12 if smoke else K
    folds_n = 2 if smoke else CV_FOLDS
    splitter = KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)

    oof = {name: np.full(n, np.nan) for name in SERIES}
    y_oof = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    fold_rows = []
    jaccards = []
    fold_meta = []
    nnls_weights = []

    for fid, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fid}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = k150._fold_frames(X, y, tr, te)
        if not X_tr.index.equals(y_tr.index) or not X_te.index.equals(y_te.index):
            raise RuntimeError(f"fold {fid} X/y index mismatch")
        X_tr, X_te = X_tr[keep], X_te[keep]
        corr_cols = corr_fs.select(X_tr, y_tr, k=min(k, len(keep)))
        rrel_cols = rrelieff_fs.select(X_tr, y_tr, k=min(k, len(keep)))
        jac = _jaccard(corr_cols, rrel_cols)
        jaccards.append(jac)
        print(f"  jaccard={jac:.3f}  corr={len(corr_cols)}  rrelieff={len(rrel_cols)}", flush=True)
        print(f"  corr X {X_tr[corr_cols].shape} {list(corr_cols)[:5]}", flush=True)
        print(f"  rrelieff X {X_tr[rrel_cols].shape} {list(rrel_cols)[:5]}", flush=True)

        pred_corr = _predict_corr_xgb(X_tr, y_tr, X_te, corr_cols, smoke)
        pred_svr = _predict_rrelieff_svr(X_tr, y_tr, X_te, rrel_cols)
        pred_mean = 0.5 * (pred_corr + pred_svr)

        oof_corr, oof_svr = _inner_oof(X_tr, y_tr, corr_cols, rrel_cols, smoke)
        blender = _fit_nnls(oof_corr, oof_svr, y_tr)
        pred_nnls = _apply_nnls(blender, pred_corr, pred_svr)
        nnls_weights.append(
            {
                "fold": fid,
                "coef_corr_xgb": float(blender.coef_[0]),
                "coef_rrelieff_svr": float(blender.coef_[1]),
                "intercept": float(blender.intercept_),
            }
        )
        print(
            f"  nnls w=[{blender.coef_[0]:.3f}, {blender.coef_[1]:.3f}] "
            f"b={blender.intercept_:.3f}",
            flush=True,
        )

        preds = {
            "corr_xgb": pred_corr,
            "rrelieff_svr": pred_svr,
            "mean": pred_mean,
            "nnls": pred_nnls,
        }
        y_te_v = np.ravel(y_te)
        y_oof[te] = y_te_v
        fold_id[te] = fid
        scores = {}
        for name in SERIES:
            oof[name][te] = preds[name]
            scores[name] = _metrics(y_te_v, preds[name])
            print(
                f"  {name:<16} r2 {scores[name]['r2']:.3f}  pred_std {np.std(preds[name]):.2f}",
                flush=True,
            )
        fold_rows.append(scores)
        fold_meta.append(
            {
                "fold": fid,
                "jaccard": jac,
                "n_corr": len(corr_cols),
                "n_rrelieff": len(rrel_cols),
                "scores": scores,
            }
        )

    verdict = _decide(fold_rows)
    _print(verdict, jaccards)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": k,
        "folds_n": folds_n,
        "incumbent": INCUMBENT,
        "verdict": verdict,
        "jaccard_mean": float(np.mean(jaccards)),
        "jaccard_folds": [float(v) for v in jaccards],
        "nnls_weights": nnls_weights,
        "folds": fold_meta,
        "y": [float(v) for v in y_oof],
        "fold_id": [int(v) for v in fold_id],
        "nan_count": [float(v) for v in nan_count],
        "pred": {name: [float(v) for v in oof[name]] for name in SERIES},
    }
    out = OUT.with_name("union.smoke.json") if smoke else OUT
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
