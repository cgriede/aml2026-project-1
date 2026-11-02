"""In-fold feature-selector comparison.

Decision: keep the selector with the best outer-fold XGB R². The accepted
`|corr|` top-150 set stays unless the challenger beats it by >= 0.01 mean
paired R² and that gap exceeds the standard error of the five fold deltas.
`all` can win, which means selection itself failed.

lib/ is imported, not edited. Selection, scaling, and the correlation cull
fit on the outer training fold only. Defect columns are dropped once.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import (  # noqa: E402
    CV_FOLDS,
    FS_CORR_MAX,
    FS_N_ROUNDS,
    FS_N_TRIALS,
    FS_SCORE_SPLIT,
    FS_SUBSET_SIZE,
    FS_TOP_FRAC,
    RANDOM_STATE,
    XGB_SEEDS,
)
from clean import _dead_columns, _insane_columns  # noqa: E402
from load import load  # noqa: E402
from select_features.drop_high_corr import drop_high_corr  # noqa: E402
from train import Trainer  # noqa: E402

SELECTORS = ("all", "corr", "mi", "monte_carlo", "elastic_net", "xgb_perm")
FIXED_K = ("corr", "mi", "monte_carlo", "elastic_net", "xgb_perm")
K = 150
FLOOR = 0.01
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "001" / "results" / "k150_matrix.json"

XGB_KW = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=7,
    n_jobs=4,
)


def _check_decision_math() -> None:
    mean, se = _paired(np.array([0.02, 0.02, 0.02, 0.02, 0.02]))
    assert _clears(mean, se)
    mean, se = _paired(np.array([0.005, 0.005, 0.005, 0.005, 0.005]))
    assert not _clears(mean, se)
    assert abs(_jaccard(["a", "b"], ["b", "c"]) - 1 / 3) < 1e-9


def _paired(delta: np.ndarray) -> tuple[float, float]:
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return mean, se


def _clears(mean: float, se: float) -> bool:
    return bool(mean >= FLOOR and mean > se)


def _jaccard(a, b) -> float:
    left, right = set(a), set(b)
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _metrics(y, pred) -> dict:
    y = np.ravel(y)
    pred = np.ravel(pred)
    return {
        "r2": float(r2_score(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
    }


def _top(scores: pd.Series, k: int) -> list[str]:
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index[: min(k, len(order))]]


def _median_scale(X_tr, X_te):
    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
    return X_tr, X_te


def _lifetime(X: pd.DataFrame, y, rounds: int, trials: int, subset: int) -> pd.Series:
    """Same Ridge Monte Carlo weights as lib/select_features/monte_carlo.py."""
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.ravel(y)
    active = list(X.columns)
    lifetime = {c: 0.0 for c in active}
    X_fit, X_sc, y_fit, y_sc = train_test_split(X, y, test_size=FS_SCORE_SPLIT, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_fit = pd.DataFrame(scaler.fit_transform(X_fit), index=X_fit.index, columns=X_fit.columns)
    X_sc = pd.DataFrame(scaler.transform(X_sc), index=X_sc.index, columns=X_sc.columns)
    for _rnd in range(rounds):
        width = min(subset, len(active))
        if width < 2 or len(active) <= width:
            break
        trial_rows = []
        for _ in range(trials):
            chosen = rng.choice(active, size=width, replace=False)
            model = Ridge(solver="svd")
            model.fit(X_fit[chosen], y_fit)
            trial_rows.append((r2_score(y_sc, model.predict(X_sc[chosen])), chosen))
        trial_rows.sort(key=lambda t: t[0], reverse=True)
        band = max(1, int(trials * FS_TOP_FRAC))
        delta = {c: 0.0 for c in active}
        for score, chosen in trial_rows[:band]:
            for c in chosen:
                delta[c] += score * 100
        for score, chosen in trial_rows[-band:]:
            for c in chosen:
                delta[c] -= score * 100
        for c, weight in delta.items():
            lifetime[c] += weight
        nxt = [c for c in active if delta[c] >= 0]
        if len(nxt) < width:
            nxt = sorted(active, key=lambda c: delta[c], reverse=True)[: max(width, len(active) // 2)]
        active = nxt
    return pd.Series(lifetime)


def _xgb(random_state: int, smoke: bool) -> XGBRegressor:
    if smoke:
        return XGBRegressor(n_estimators=20, max_depth=2, random_state=random_state, n_jobs=2)
    return XGBRegressor(**XGB_KW, early_stopping_rounds=50, random_state=random_state)


def _select(name: str, X: pd.DataFrame, y, k: int, smoke: bool) -> list[str]:
    yv = np.ravel(y)
    if name == "corr":
        scores = X.corrwith(pd.Series(yv, index=X.index)).abs()
        return _top(scores, k)
    if name == "mi":
        scores = mutual_info_regression(X, yv, random_state=RANDOM_STATE, n_jobs=4)
        return _top(pd.Series(scores, index=X.columns), k)
    if name == "monte_carlo":
        scores = _lifetime(
            X,
            yv,
            rounds=1 if smoke else FS_N_ROUNDS,
            trials=4 if smoke else FS_N_TRIALS,
            subset=20 if smoke else FS_SUBSET_SIZE,
        )
        return _top(scores, k)
    if name == "elastic_net":
        model = ElasticNetCV(
            l1_ratio=0.5,
            cv=2 if smoke else 3,
            alphas=5 if smoke else 25,
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        model.fit(X, yv)
        return _top(pd.Series(np.abs(model.coef_), index=X.columns), k)
    if name == "xgb_perm":
        X_fit, X_va, y_fit, y_va = train_test_split(X, yv, test_size=0.2, random_state=RANDOM_STATE)
        model = _xgb(RANDOM_STATE, smoke)
        if smoke:
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_fit, y_fit, eval_set=[(X_va, y_va)], verbose=False)
        imp = permutation_importance(
            model,
            X_va,
            y_va,
            n_repeats=1 if smoke else 5,
            random_state=RANDOM_STATE,
            scoring="r2",
            n_jobs=1,
        )
        return _top(pd.Series(imp.importances_mean, index=X.columns), k)
    raise ValueError(name)


def _score_set(X_tr, y_tr, X_te, y_te, cols, smoke: bool) -> dict:
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr_v, y_te_v = np.ravel(y_tr), np.ravel(y_te)
    ridge = Trainer(X_tr, y_tr_v).train("ridge")
    xgb = _xgb(RANDOM_STATE, smoke)
    if smoke:
        xgb.fit(X_tr, y_tr_v)
    else:
        xgb.fit(X_tr, y_tr_v, eval_set=[(X_te, y_te_v)], verbose=False)
    svr = SVR()
    svr.fit(X_tr, y_tr_v)
    return {
        "n_features": len(cols),
        "ridge": _metrics(y_te_v, ridge.predict(X_te)),
        "xgb": _metrics(y_te_v, xgb.predict(X_te)),
        "svr": _metrics(y_te_v, svr.predict(X_te)),
    }


def _score_bag(X_tr, y_tr, X_te, y_te, cols) -> dict:
    X_tr, X_te = X_tr[cols], X_te[cols]
    y_tr_v, y_te_v = np.ravel(y_tr), np.ravel(y_te)
    preds = []
    for seed in XGB_SEEDS:
        model = _xgb(seed, smoke=False)
        model.fit(X_tr, y_tr_v, eval_set=[(X_te, y_te_v)], verbose=False)
        preds.append(model.predict(X_te))
    return _metrics(y_te_v, np.mean(preds, axis=0))


def _fold_frames(X, y, train_idx, test_idx):
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    X_tr, X_te = _median_scale(X_tr, X_te)
    keep = drop_high_corr(X_tr, y_tr, FS_CORR_MAX)
    return X_tr, y_tr, X_te, y_te, keep


def _mean_table(folds: list[dict]) -> dict:
    table = {}
    for name in SELECTORS:
        table[name] = {}
        for model in ("ridge", "xgb", "svr"):
            r2 = np.array([fold["scores"][name][model]["r2"] for fold in folds])
            rmse = np.array([fold["scores"][name][model]["rmse"] for fold in folds])
            table[name][model] = {
                "r2_mean": float(r2.mean()),
                "r2_std": float(r2.std(ddof=1)),
                "r2_folds": [float(v) for v in r2],
                "rmse_mean": float(rmse.mean()),
            }
        table[name]["n_features"] = [fold["scores"][name]["n_features"] for fold in folds]
    return table


def _stability(folds: list[dict]) -> dict:
    out = {}
    for name in FIXED_K:
        cols = [fold["columns"][name] for fold in folds]
        vals = [_jaccard(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]
        out[name] = float(np.mean(vals))
    return out


def _overlap_with_corr(folds: list[dict]) -> dict:
    out = {}
    for name in FIXED_K:
        if name == "corr":
            continue
        vals = [_jaccard(fold["columns"][name], fold["columns"]["corr"]) for fold in folds]
        out[name] = float(np.mean(vals))
    return out


def _vs(folds: list[dict], left: str, right: str, model: str) -> dict:
    delta = np.array(
        [
            fold["scores"][left][model]["r2"] - fold["scores"][right][model]["r2"]
            for fold in folds
        ]
    )
    mean, se = _paired(delta)
    return {"mean": mean, "se": se, "clears": _clears(mean, se), "folds": [float(v) for v in delta]}


def decide(folds: list[dict]) -> dict:
    table = _mean_table(folds)
    xgb_rank = sorted(SELECTORS, key=lambda n: table[n]["xgb"]["r2_mean"], reverse=True)
    ridge_rank = sorted(SELECTORS, key=lambda n: table[n]["ridge"]["r2_mean"], reverse=True)
    best = xgb_rank[0]
    best_fixed = max(FIXED_K, key=lambda n: table[n]["xgb"]["r2_mean"])
    vs_all = _vs(folds, best_fixed, "all", "xgb")
    vs_corr = None if best == "corr" else _vs(folds, best, "corr", "xgb")
    if best == "corr" or not vs_corr["clears"]:
        keep = "corr"
    else:
        keep = best
    return {
        "keep": keep,
        "best_xgb": best,
        "best_fixed_k": best_fixed,
        "xgb_rank": xgb_rank,
        "ridge_rank": ridge_rank,
        "h1a": "supported" if vs_all["clears"] else "not_supported",
        "h1a_detail": {"selector": best_fixed, **vs_all},
        "h1b": "supported" if xgb_rank[0] != ridge_rank[0] else "not_supported",
        "vs_corr": vs_corr,
        "table": table,
        "stability_jaccard": _stability(folds),
        "jaccard_vs_corr": _overlap_with_corr(folds),
    }


def _print_table(verdict: dict) -> None:
    print(f"keep={verdict['keep']}  H1a={verdict['h1a']}  H1b={verdict['h1b']}", flush=True)
    print(f"{'set':<14} {'p':>5} {'ridge':>8} {'xgb':>8} {'svr':>8} {'xgb rmse':>9}", flush=True)
    for name in verdict["xgb_rank"]:
        row = verdict["table"][name]
        width = int(np.median(row["n_features"]))
        print(
            f"{name:<14} {width:5d} {row['ridge']['r2_mean']:8.3f} "
            f"{row['xgb']['r2_mean']:8.3f} {row['svr']['r2_mean']:8.3f} "
            f"{row['xgb']['rmse_mean']:9.3f}",
            flush=True,
        )


def _acquire() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            holder = json.loads(LOCK.read_text())
            pid = int(holder.get("pid", -1))
        except (json.JSONDecodeError, TypeError, ValueError):
            pid = -1
        if pid > 0 and _alive(pid):
            raise SystemExit(f"pipeline run held by pid {pid}: {LOCK}")
        LOCK.unlink()
    fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(
        fd,
        json.dumps(
            {"slug": "feature_selector_comparison", "pid": os.getpid(), "script": str(Path(__file__))}
        ).encode(),
    )
    os.close(fd)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool) -> dict:
    _check_decision_math()
    os.chdir(ROOT)
    X, y, _ = load()
    drop = _dead_columns(X) + _insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    if smoke:
        X, y = X.iloc[:200].copy(), y.iloc[:200].copy()
    k = 8 if smoke else K
    folds_n = 2 if smoke else CV_FOLDS
    splitter = KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = _fold_frames(X, y, tr, te)
        culled = X_tr[keep]
        columns = {"all": list(X_tr.columns)}
        scores = {}
        scores["all"] = _score_set(X_tr, y_tr, X_te, y_te, columns["all"], smoke)
        print(f"  all xgb {scores['all']['xgb']['r2']:.3f} p={scores['all']['n_features']}", flush=True)
        for name in FIXED_K:
            columns[name] = _select(name, culled, y_tr, k, smoke)
            scores[name] = _score_set(X_tr, y_tr, X_te, y_te, columns[name], smoke)
            print(f"  {name} xgb {scores[name]['xgb']['r2']:.3f} p={scores[name]['n_features']}", flush=True)
        folds.append({"fold": fold_id, "scores": scores, "columns": columns})
        _dump({"smoke": smoke, "folds": _scores_only(folds)}, partial=True, smoke=smoke)
    verdict = decide(folds)
    _print_table(verdict)
    if not smoke and verdict["h1a"] == "supported":
        winner = verdict["best_fixed_k"]
        print(f"3-seed confirm {winner} and corr", flush=True)
        bag = {winner: [], "corr": []}
        for fold_id, (tr, te) in enumerate(splitter.split(X)):
            X_tr, y_tr, X_te, y_te, _keep = _fold_frames(X, y, tr, te)
            for name in bag:
                bag[name].append(_score_bag(X_tr, y_tr, X_te, y_te, folds[fold_id]["columns"][name]))
        verdict["xgb_bag"] = {
            name: {
                "r2_mean": float(np.mean([row["r2"] for row in rows])),
                "rmse_mean": float(np.mean([row["rmse"] for row in rows])),
                "r2_folds": [row["r2"] for row in rows],
            }
            for name, rows in bag.items()
        }
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": k,
        "folds_n": folds_n,
        "verdict": verdict,
        "folds": folds,
    }
    _dump(payload, partial=False, smoke=smoke)
    return payload


def _scores_only(folds: list[dict]) -> list[dict]:
    return [{"fold": fold["fold"], "scores": fold["scores"]} for fold in folds]


def _dump(payload: dict, partial: bool, smoke: bool) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    path = OUT.with_name("k150_matrix.smoke.json") if smoke else OUT
    if partial:
        path = path.with_suffix(".partial.json")
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}", flush=True)


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
