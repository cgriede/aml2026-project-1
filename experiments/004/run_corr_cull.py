"""|r| >= 0.95 cull on or off, same |corr| top-150.

The accepted pipeline drops one of each near-duplicate pair, then keeps the
top 150 by |corr(x, y)|. This arm skips that drop and still keeps 150, so
two features at |r| = 0.95 with each other can both stay.

The cull stays unless the no-cull set beats the bag by >= 0.01 mean paired
R² and that gap exceeds the standard error of the five fold deltas.
Ridge is scored on the same columns. The keep rule uses the bag.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, FS_CORR_MAX, RANDOM_STATE  # noqa: E402
from select_features.drop_high_corr import drop_high_corr  # noqa: E402

ARMS = ("cull", "nocull")
MODELS = ("ridge", "xgboost_bag")
INCUMBENT = "cull"
K = 150
FLOOR = 0.01
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "004" / "results" / "corr_cull.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rank(X: pd.DataFrame, y) -> list[str]:
    scores = X.corrwith(pd.Series(np.ravel(y), index=X.index)).abs()
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index]


def _pairs(X: pd.DataFrame, cols: list[str], tau: float) -> tuple[int, int]:
    corr = X[cols].corr().abs()
    names = list(corr.columns)
    n = 0
    touched: set[str] = set()
    for i, a in enumerate(names):
        row = corr.loc[a]
        for b in names[i + 1 :]:
            if row[b] >= tau:
                n += 1
                touched.add(a)
                touched.add(b)
    return n, len(touched)


def _jaccard(a: list[str], b: list[str]) -> float:
    left, right = set(a), set(b)
    return len(left & right) / len(left | right)


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


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
    os.write(fd, json.dumps({"slug": "corr_cull", "pid": os.getpid()}).encode())
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool) -> dict:
    k150 = _load("k150_matrix", ROOT / "experiments" / "001" / "run_k150_matrix.py")
    models = _load("run_models", ROOT / "experiments" / "003" / "run_models.py")
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    if smoke:
        X, y = X.iloc[:180].copy(), y.iloc[:180].copy()
    k = 12 if smoke else K
    folds_n = 2 if smoke else CV_FOLDS
    splitter = k150.KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        X_tr, X_te = k150._median_scale(X_tr, X_te)
        culled = drop_high_corr(X_tr, y_tr, FS_CORR_MAX)
        sets = {
            "cull": _rank(X_tr[culled], y_tr)[:k],
            "nocull": _rank(X_tr, y_tr)[:k],
        }
        scores = {}
        for arm, cols in sets.items():
            pairs, touched = _pairs(X_tr, cols, FS_CORR_MAX)
            pred = {
                "ridge": models._predict("ridge", X_tr[cols], y_tr, X_te[cols], smoke),
                "xgboost_bag": models._predict("xgboost_bag", X_tr[cols], y_tr, X_te[cols], smoke),
            }
            scores[arm] = {
                "n_features": len(cols),
                "redundant_pairs": pairs,
                "columns_in_a_pair": touched,
                "ridge": models._metrics(y_te, pred["ridge"]),
                "xgboost_bag": models._metrics(y_te, pred["xgboost_bag"]),
            }
            print(
                f"  {arm:<8} bag {scores[arm]['xgboost_bag']['r2']:.3f}"
                f"  ridge {scores[arm]['ridge']['r2']:.3f}"
                f"  pairs {pairs}",
                flush=True,
            )
        folds.append(
            {
                "fold": fold_id,
                "n_after_cull": len(culled),
                "jaccard": _jaccard(sets["cull"], sets["nocull"]),
                "scores": scores,
            }
        )
    verdict = _decide(folds)
    _print(verdict)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "k": k,
        "tau": FS_CORR_MAX,
        "folds_n": folds_n,
        "verdict": verdict,
        "folds": folds,
    }
    out = OUT.with_name("corr_cull.smoke.json") if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out}", flush=True)
    return payload


def _decide(folds: list[dict]) -> dict:
    table = {}
    for arm in ARMS:
        table[arm] = {}
        for model in MODELS:
            r2 = np.array([fold["scores"][arm][model]["r2"] for fold in folds])
            rmse = np.array([fold["scores"][arm][model]["rmse"] for fold in folds])
            table[arm][model] = {
                "r2_mean": float(r2.mean()),
                "r2_std": float(r2.std(ddof=1)),
                "r2_folds": [float(v) for v in r2],
                "rmse_mean": float(rmse.mean()),
            }
        pairs = np.array([fold["scores"][arm]["redundant_pairs"] for fold in folds])
        table[arm]["redundant_pairs_mean"] = float(pairs.mean())
    gaps = {}
    for model in MODELS:
        delta = np.array(
            [
                fold["scores"]["nocull"][model]["r2"] - fold["scores"]["cull"][model]["r2"]
                for fold in folds
            ]
        )
        mean, se = _paired(delta)
        gaps[model] = {"mean": mean, "se": se, "clears": bool(mean >= FLOOR and mean > se)}
    return {
        "keep": "nocull" if gaps["xgboost_bag"]["clears"] else "cull",
        "incumbent": INCUMBENT,
        "jaccard_mean": float(np.mean([fold["jaccard"] for fold in folds])),
        "gaps": gaps,
        "table": table,
    }


def _print(verdict: dict) -> None:
    print(f"keep={verdict['keep']}  jaccard={verdict['jaccard_mean']:.3f}", flush=True)
    print(f"{'arm':<8} {'bag':>8} {'ridge':>8} {'pairs':>8}", flush=True)
    for arm in ARMS:
        row = verdict["table"][arm]
        print(
            f"{arm:<8} {row['xgboost_bag']['r2_mean']:8.3f}"
            f" {row['ridge']['r2_mean']:8.3f}"
            f" {row['redundant_pairs_mean']:8.1f}",
            flush=True,
        )
    for model, gap in verdict["gaps"].items():
        print(f"nocull - cull {model}: {gap['mean']:+.3f} se {gap['se']:.3f}", flush=True)


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
