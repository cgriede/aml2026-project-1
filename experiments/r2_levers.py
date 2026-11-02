"""One-at-a-time R2 levers vs current lib/main.py.

Baseline = dead-col drop + IF∩SVM + corr-cull 0.95 + MI median + stock XGB
(lr=0.1, 100 trees, early_stop=20). Each arm changes exactly one of those.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, FS_CORR_MAX, RANDOM_STATE  # noqa: E402
from clean import clean  # noqa: E402
from load import load  # noqa: E402
from remove_outliers import remove_outliers  # noqa: E402
from select_features import _drop_high_corr, _mi_select  # noqa: E402

OUT = ROOT / "runs" / "r2_levers.jsonl"

STOCK_XGB = dict(n_estimators=100, learning_rate=0.1, early_stopping_rounds=20)
SEARCH_XGB = dict(n_estimators=2000, learning_rate=0.03, early_stopping_rounds=50)
SEARCH_GRID = {
    "max_depth": (3, 4, 5),
    "subsample": (0.6, 0.8),
    "colsample_bytree": (0.6, 0.8),
    "min_child_weight": (5, 10),
}
# Midpoint of the recommended box, scored before looking at the grid winner.
SEARCH_PRIOR = dict(max_depth=4, subsample=0.7, colsample_bytree=0.7, min_child_weight=7)
SEEDS = (42, 43, 44)
RIDGE_K = 50


def _ravel(y) -> np.ndarray:
    return np.ravel(y)


def _kfold():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _fit_xgb(X_tr, y_tr, X_te, y_te, seed=RANDOM_STATE, **kwargs):
    model = XGBRegressor(random_state=seed, **kwargs)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return model


def _cv_xgb(X, y, **xgb_kwargs) -> dict:
    scores, trees = [], []
    yv = _ravel(y)
    for tr, te in _kfold().split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        model = _fit_xgb(X_tr, y_tr, X_te, y_te, **xgb_kwargs)
        scores.append(float(r2_score(y_te, model.predict(X_te))))
        trees.append(int(model.best_iteration))
    return _pack(scores, trees=trees)


def _cv_seed_mean(X, y) -> dict:
    scores = []
    yv = _ravel(y)
    for tr, te in _kfold().split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        preds = [
            _fit_xgb(X_tr, y_tr, X_te, y_te, seed=s, **STOCK_XGB).predict(X_te)
            for s in SEEDS
        ]
        scores.append(float(r2_score(y_te, np.mean(preds, axis=0))))
    return _pack(scores)


def _cv_xgb_ridge(X, y) -> dict:
    scores = []
    yv = _ravel(y)
    for tr, te in _kfold().split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = yv[tr], yv[te]
        xgb = _fit_xgb(X_tr, y_tr, X_te, y_te, **STOCK_XGB)
        corr = X_tr.corrwith(pd.Series(y_tr, index=X_tr.index)).abs()
        cols = corr.nlargest(min(RIDGE_K, X_tr.shape[1])).index
        ridge = Ridge(alpha=1.0, solver="svd")
        ridge.fit(X_tr[cols], y_tr)
        pred = 0.5 * xgb.predict(X_te) + 0.5 * ridge.predict(X_te[cols])
        scores.append(float(r2_score(y_te, pred)))
    return _pack(scores)


def _pack(scores, **extra) -> dict:
    a = np.asarray(scores, dtype=float)
    return {
        "cv_mean": float(a.mean()),
        "cv_std": float(a.std()),
        "folds": [float(s) for s in a],
        **extra,
    }


def _prepare(drop_outliers: bool, mi_cut: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y, _ = load()
    X, _ = clean(X)
    if drop_outliers:
        X, y = remove_outliers(X, y)
    keep = _drop_high_corr(X, y, FS_CORR_MAX)
    X = X[keep]
    if mi_cut:
        X = X[_mi_select(X, y)]
    print(f"prepared drop_outliers={drop_outliers} mi_cut={mi_cut}  {X.shape}")
    return X, y


def _log(row: dict) -> None:
    OUT.parent.mkdir(exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **row}
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"{row['arm']:22s}  n={row.get('n_train')}×{row.get('n_features')}  "
        f"R2 {row['cv_mean']:.4f} ± {row['cv_std']:.4f}  {np.round(row['folds'], 4)}"
        + (f"  {row['note']}" if row.get("note") else ""),
        flush=True,
    )


def _record(arm: str, X, y, result: dict, note: str = "") -> dict:
    row = {
        "arm": arm,
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "note": note,
        **result,
    }
    _log(row)
    return row


def _search(X, y) -> list[dict]:
    rows = []
    rows.append(
        _record(
            "B_search_prior",
            X,
            y,
            _cv_xgb(X, y, **SEARCH_XGB, **SEARCH_PRIOR),
            note=str(SEARCH_PRIOR),
        )
    )
    keys = list(SEARCH_GRID)
    for values in product(*(SEARCH_GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        rows.append(
            _record(
                "B_search_grid",
                X,
                y,
                _cv_xgb(X, y, **SEARCH_XGB, **params),
                note=str(params),
            )
        )
    best = max(rows, key=lambda r: r["cv_mean"])
    print(f"B search best: {best['note']}  R2 {best['cv_mean']:.4f}", flush=True)
    return rows


def main() -> None:
    print("=== prep: outlier-dropped (baseline / A / B / D) ===", flush=True)
    X_mi, y_drop = _prepare(drop_outliers=True, mi_cut=True)
    X_all, _ = _prepare(drop_outliers=True, mi_cut=False)

    print("=== prep: no row drop (C) ===", flush=True)
    X_nodrop, y_full = _prepare(drop_outliers=False, mi_cut=True)

    rows = []
    print("=== A leftovers / C outliers / D ensemble / baseline ===", flush=True)
    rows.append(_record("baseline_mi_stock", X_mi, y_drop, _cv_xgb(X_mi, y_drop, **STOCK_XGB)))
    rows.append(_record("A_all_post_cull", X_all, y_drop, _cv_xgb(X_all, y_drop, **STOCK_XGB)))
    rows.append(_record("C_no_row_drop", X_nodrop, y_full, _cv_xgb(X_nodrop, y_full, **STOCK_XGB)))
    rows.append(_record("D_seed_mean", X_mi, y_drop, _cv_seed_mean(X_mi, y_drop), note=f"seeds={SEEDS}"))
    rows.append(
        _record(
            "D_xgb_ridge",
            X_mi,
            y_drop,
            _cv_xgb_ridge(X_mi, y_drop),
            note=f"mean(stock XGB, Ridge top-{RIDGE_K} |corr|)",
        )
    )

    print("=== B XGB search ===", flush=True)
    search_rows = _search(X_mi, y_drop)
    rows.extend(search_rows)

    base = next(r["cv_mean"] for r in rows if r["arm"] == "baseline_mi_stock")
    summary = [
        r
        for r in rows
        if r["arm"]
        in {
            "baseline_mi_stock",
            "A_all_post_cull",
            "C_no_row_drop",
            "D_seed_mean",
            "D_xgb_ridge",
            "B_search_prior",
        }
        or (r["arm"] == "B_search_grid" and r["cv_mean"] == max(s["cv_mean"] for s in search_rows))
    ]
    print("\n=== isolated deltas vs baseline ===", flush=True)
    print(f"{'arm':22s} {'R2':>8} {'Δ':>7}  n  p  note")
    for r in sorted(summary, key=lambda x: -x["cv_mean"]):
        print(
            f"{r['arm']:22s} {r['cv_mean']:8.4f} {r['cv_mean'] - base:+7.4f}  "
            f"{r['n_train']} {r['n_features']}  {r.get('note', '')}"
        )


if __name__ == "__main__":
    main()
