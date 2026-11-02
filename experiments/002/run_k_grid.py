"""|corr| prefix sizes on the same outer folds as exp 001.

k=150 stays unless another prefix beats it by >= 0.01 mean paired XGB R²
and that gap exceeds the standard error of the five fold deltas.
`all` is the uncullled matrix, scored once per fold. A requested k larger
than the post-cull width uses every surviving column and records that width.
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
KS = (25, 50, 75, 100, 150, 200, 300, 400, 600, 825)
ZOOM = (110, 130, 140, 150, 160, 170, 190)
BASE = 150
FLOOR = 0.01
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "002" / "results" / "k_grid.json"


def _k150():
    path = ROOT / "experiments" / "001" / "run_k150_matrix.py"
    spec = importlib.util.spec_from_file_location("k150_matrix", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paired(delta: np.ndarray) -> tuple[float, float]:
    return float(delta.mean()), float(delta.std(ddof=1) / np.sqrt(len(delta)))


def _clears(mean: float, se: float) -> bool:
    return bool(mean >= FLOOR and mean > se)


def _rank(X: pd.DataFrame, y) -> list[str]:
    scores = X.corrwith(pd.Series(np.ravel(y), index=X.index)).abs()
    order = scores.fillna(-np.inf).sort_values(ascending=False, kind="mergesort")
    return [str(c) for c in order.index]


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
    os.write(fd, json.dumps({"slug": "k_grid", "pid": os.getpid(), "script": str(Path(__file__))}).encode())
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool, ks: tuple[int, ...] | None = None, out: Path | None = None, score_all: bool = True) -> dict:
    k150 = _k150()
    os.chdir(ROOT)
    X, y, _ = k150.load()
    drop = k150._dead_columns(X) + k150._insane_columns(X)
    print(f"drop {len(drop)} defect columns", flush=True)
    X = X.drop(columns=drop)
    if smoke:
        X, y = X.iloc[:200].copy(), y.iloc[:200].copy()
        ks = (4, 8)
    else:
        ks = ks or KS
    out = out or OUT
    base = 8 if smoke else BASE
    folds_n = 2 if smoke else k150.CV_FOLDS
    splitter = k150.KFold(n_splits=folds_n, shuffle=True, random_state=k150.RANDOM_STATE)
    folds = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, y_tr, X_te, y_te, keep = k150._fold_frames(X, y, tr, te)
        ranked = _rank(X_tr[keep], y_tr)
        print(f"  culled {len(ranked)}  uncullled {X_tr.shape[1]}", flush=True)
        cache: dict[int, dict] = {}
        scores = {}
        for k in ks:
            take = min(k, len(ranked))
            if take not in cache:
                cols = ranked[:take]
                cache[take] = k150._score_set(X_tr, y_tr, X_te, y_te, cols, smoke)
                print(f"  k={take} xgb {cache[take]['xgb']['r2']:.3f}", flush=True)
            scores[str(k)] = {**cache[take], "k_requested": k, "k_effective": take}
        if score_all:
            scores["all"] = k150._score_set(X_tr, y_tr, X_te, y_te, list(X_tr.columns), smoke)
            print(f"  all xgb {scores['all']['xgb']['r2']:.3f} p={scores['all']['n_features']}", flush=True)
        folds.append({"fold": fold_id, "n_culled": len(ranked), "scores": scores})
    verdict = _decide(folds, ks, base, score_all)
    _print(verdict)
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "ks": list(ks),
        "base": base,
        "folds_n": folds_n,
        "verdict": verdict,
        "folds": folds,
    }
    _dump(payload, out)
    return payload


def _decide(folds: list[dict], ks: tuple[int, ...], base: int, score_all: bool) -> dict:
    names = [str(k) for k in ks] + (["all"] if score_all else [])
    table = {}
    for name in names:
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
        table[name]["k_effective"] = [fold["scores"][name]["k_effective"] for fold in folds] if name != "all" else [
            fold["scores"]["all"]["n_features"] for fold in folds
        ]
    challengers = {}
    for k in ks:
        if k == base:
            continue
        delta = np.array(
            [
                fold["scores"][str(k)]["xgb"]["r2"] - fold["scores"][str(base)]["xgb"]["r2"]
                for fold in folds
            ]
        )
        mean, se = _paired(delta)
        challengers[str(k)] = {"mean": mean, "se": se, "clears": _clears(mean, se), "folds": [float(v) for v in delta]}
    clearing = [k for k, row in challengers.items() if row["clears"]]
    keep = max(clearing, key=lambda k: table[k]["xgb"]["r2_mean"]) if clearing else str(base)
    return {"keep": keep, "base": str(base), "challengers": challengers, "table": table}


def _print(verdict: dict) -> None:
    print(f"keep={verdict['keep']}  base={verdict['base']}", flush=True)
    print(f"{'k':<8} {'p':>5} {'ridge':>8} {'xgb':>8} {'svr':>8} {'xgb rmse':>9}  vs {verdict['base']}", flush=True)
    order = sorted(verdict["table"], key=lambda n: (n == "all", int(n) if n != "all" else 10**9))
    for name in order:
        row = verdict["table"][name]
        width = int(np.median(row["k_effective"]))
        extra = ""
        if name in verdict["challengers"]:
            gap = verdict["challengers"][name]
            extra = f"  {gap['mean']:+.3f} (SE {gap['se']:.3f})"
        print(
            f"{name:<8} {width:5d} {row['ridge']['r2_mean']:8.3f} "
            f"{row['xgb']['r2_mean']:8.3f} {row['svr']['r2_mean']:8.3f} "
            f"{row['xgb']['rmse_mean']:9.3f}{extra}",
            flush=True,
        )


def _dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}", flush=True)


def main() -> None:
    smoke = "--smoke" in sys.argv
    zoom = "--zoom" in sys.argv
    if not smoke:
        _acquire()
    try:
        if zoom:
            run(smoke, ks=ZOOM, out=OUT.with_name("k_grid_zoom.json"), score_all=False)
        else:
            run(smoke, out=OUT.with_name("k_grid.smoke.json") if smoke else OUT)
    finally:
        if not smoke:
            _release()


if __name__ == "__main__":
    main()
