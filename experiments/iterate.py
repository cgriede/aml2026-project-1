"""Shared CV harness for isolated pipeline iterations."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, FS_CORR_MAX, RANDOM_STATE  # noqa: E402
from clean import clean  # noqa: E402
from load import load  # noqa: E402
from remove_outliers import remove_outliers  # noqa: E402
from scale import scale  # noqa: E402
from select_features import _corr_select, _drop_high_corr, _mi_select  # noqa: E402
from train import Trainer  # noqa: E402

OUT = ROOT / "runs" / "iterations.jsonl"


def prepare(drop_outliers: bool = True, scale_first: bool = False):
    X, y, _ = load()
    X, _ = clean(X)
    if scale_first:
        X, y, _ = scale(X, y)
    if drop_outliers:
        X, y = remove_outliers(X, y)
    if not scale_first:
        X, y, _ = scale(X, y)
    keep = _drop_high_corr(X, y, FS_CORR_MAX)
    return X[keep], y


def select(X, y, method: str, k: int | None = None):
    if method == "mi":
        cols = _mi_select(X, y)
    elif method == "corr":
        cols = _corr_select(X, y, k=k)
    elif method == "all":
        cols = list(X.columns)
        print(f"all post-cull: {len(cols)}")
    else:
        raise ValueError(method)
    return X[cols], y


def cv(X, y) -> dict:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    yv = np.ravel(y)
    ridge, xgb, trees = [], [], []
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        trainer = Trainer(X_tr, y_tr)
        r = trainer.train_baseline()
        x = trainer.train_xgboost(X_te, y_te)
        ridge.append(float(r2_score(y_te, r.predict(X_te))))
        xgb.append(float(r2_score(y_te, x.predict(X_te))))
        trees.append(int(x.best_iteration))
    ridge, xgb = np.asarray(ridge), np.asarray(xgb)
    return {
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "ridge_cv_mean": float(ridge.mean()),
        "ridge_cv_std": float(ridge.std()),
        "ridge_folds": [float(s) for s in ridge],
        "xgb_cv_mean": float(xgb.mean()),
        "xgb_cv_std": float(xgb.std()),
        "xgb_folds": [float(s) for s in xgb],
        "xgb_best_iteration": [int(t) for t in trees],
    }


SEEDS = (42, 43, 44)


def cv_seed_mean(X, y, seeds=SEEDS) -> dict:
    from xgboost import XGBRegressor

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    yv = np.ravel(y)
    ridge, xgb = [], []
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        trainer = Trainer(X_tr, y_tr)
        r = trainer.train_baseline()
        ridge.append(float(r2_score(y_te, r.predict(X_te))))
        preds = []
        for seed in seeds:
            kwargs = dict(
                n_estimators=2000,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=7,
                early_stopping_rounds=50,
                random_state=seed,
            )
            m = XGBRegressor(**kwargs)
            m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            preds.append(m.predict(X_te))
        xgb.append(float(r2_score(y_te, np.mean(preds, axis=0))))
    ridge, xgb = np.asarray(ridge), np.asarray(xgb)
    return {
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "ridge_cv_mean": float(ridge.mean()),
        "ridge_cv_std": float(ridge.std()),
        "ridge_folds": [float(s) for s in ridge],
        "xgb_cv_mean": float(xgb.mean()),
        "xgb_cv_std": float(xgb.std()),
        "xgb_folds": [float(s) for s in xgb],
        "xgb_best_iteration": [],
    }


def cv_xgb_ridge(X, y) -> dict:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    yv = np.ravel(y)
    ridge, mix = [], []
    for tr, te in kf.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        trainer = Trainer(X_tr, y_tr)
        r = trainer.train_baseline()
        x = trainer.train_xgboost(X_te, y_te)
        rp, xp = r.predict(X_te), x.predict(X_te)
        ridge.append(float(r2_score(y_te, rp)))
        mix.append(float(r2_score(y_te, 0.5 * rp + 0.5 * xp)))
    ridge, mix = np.asarray(ridge), np.asarray(mix)
    return {
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "ridge_cv_mean": float(ridge.mean()),
        "ridge_cv_std": float(ridge.std()),
        "ridge_folds": [float(s) for s in ridge],
        "xgb_cv_mean": float(mix.mean()),
        "xgb_cv_std": float(mix.std()),
        "xgb_folds": [float(s) for s in mix],
        "xgb_best_iteration": [],
    }


def log(iteration: int, hyp: str, verdict: str, row: dict) -> dict:
    rec = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iteration": iteration,
        "hyp": hyp,
        "verdict": verdict,
        **row,
    }
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(
        f"I{iteration} {hyp}: n={rec['n_train']}×{rec['n_features']}  "
        f"Ridge {rec['ridge_cv_mean']:.4f}±{rec['ridge_cv_std']:.4f}  "
        f"XGB {rec['xgb_cv_mean']:.4f}±{rec['xgb_cv_std']:.4f}  "
        f"{np.round(rec['xgb_folds'], 4)}  {verdict}",
        flush=True,
    )
    return rec
