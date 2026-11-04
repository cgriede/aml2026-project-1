"""Selector × k grid scored by SVR R² inside outer CV folds.

Each fold: train-only median fill, z-score, `|r| >= 0.95` cull, then each
selector at k in {50, 100, 150}. SVR is the kernel model. Pick by mean R².
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lib"))

from GLOBAL import CV_FOLDS, FS_CORR_MAX, RANDOM_STATE  # noqa: E402
from clean import clean  # noqa: E402
from load import load  # noqa: E402
from select_features import cluster as cluster_fs  # noqa: E402
from select_features import corr as corr_fs  # noqa: E402
from select_features import mi as mi_fs  # noqa: E402
from select_features import monte_carlo as monte_carlo_fs  # noqa: E402
from select_features import mrmr as mrmr_fs  # noqa: E402
from select_features import rrelieff as rrelieff_fs  # noqa: E402
from select_features.drop_high_corr import drop_high_corr  # noqa: E402

SELECTORS = ("corr", "mi", "monte_carlo", "cluster", "mrmr", "rrelieff")
KS = (50, 100, 150)
LOCK = ROOT / "experiments" / ".pipeline_run.lock"
OUT = ROOT / "experiments" / "008" / "results" / "selector_k_grid.json"


def _median_scale(X_tr, X_te):
    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=X_te.columns)
    return X_tr, X_te


def _rankings(X: pd.DataFrame, y) -> dict[str, list[str]]:
    """One ranking per selector; prefixes give k=50/100/150."""
    max_k = max(KS)
    return {
        "corr": corr_fs.select(X, y, k=max_k),
        "mi": mi_fs.select(X, y, k=max_k),
        "monte_carlo": monte_carlo_fs.select(X, y, k=max_k),
        "cluster": cluster_fs.select(X, y, k=max_k),
        "mrmr": mrmr_fs.select(X, y, k=max_k),
        "rrelieff": rrelieff_fs.select(X, y, k=max_k),
    }


def _score_svr(X_tr, y_tr, X_te, y_te, cols) -> dict:
    cols = list(cols)
    model = SVR()
    model.fit(X_tr[cols], np.ravel(y_tr))
    pred = model.predict(X_te[cols])
    return {
        "n_features": len(cols),
        "r2": float(r2_score(np.ravel(y_te), pred)),
        "rmse": float(np.sqrt(np.mean((np.ravel(y_te) - pred) ** 2))),
    }


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
        json.dumps({"slug": "selector_k_grid", "pid": os.getpid(), "script": str(Path(__file__))}).encode(),
    )
    os.close(fd)


def _release() -> None:
    if LOCK.exists():
        LOCK.unlink()


def run(smoke: bool = False) -> dict:
    os.chdir(ROOT)
    X, y, _ = load()
    X, _ = clean(X)
    y = pd.Series(np.ravel(y), index=X.index)
    ks = (8, 16) if smoke else KS
    folds_n = 2 if smoke else CV_FOLDS
    if smoke:
        X, y = X.iloc[:200].copy(), y.iloc[:200].copy()

    splitter = KFold(n_splits=folds_n, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for fold_id, (tr, te) in enumerate(splitter.split(X)):
        print(f"fold {fold_id}", flush=True)
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        X_tr, X_te = _median_scale(X_tr, X_te)
        keep = drop_high_corr(X_tr, y_tr, FS_CORR_MAX)
        X_tr, X_te = X_tr[keep], X_te[keep]
        print(f"  culled {len(keep)}", flush=True)
        rankings = _rankings(X_tr, y_tr) if not smoke else {
            "corr": corr_fs.select(X_tr, y_tr, k=max(ks)),
            "mi": mi_fs.select(X_tr, y_tr, k=max(ks)),
            "monte_carlo": monte_carlo_fs.select(X_tr, y_tr, k=max(ks)),
            "cluster": cluster_fs.select(X_tr, y_tr, k=max(ks)),
            "mrmr": mrmr_fs.select(X_tr, y_tr, k=max(ks)),
            "rrelieff": rrelieff_fs.select(X_tr, y_tr, k=max(ks)),
        }
        scores = {}
        for name, ranked in rankings.items():
            for k in ks:
                take = min(k, len(ranked))
                key = f"{name}@k{k}"
                scores[key] = {
                    **_score_svr(X_tr, y_tr, X_te, y_te, ranked[:take]),
                    "method": name,
                    "k_requested": k,
                    "k_effective": take,
                }
                print(f"  {key} svr {scores[key]['r2']:.3f} p={take}", flush=True)
        folds.append({"fold": fold_id, "n_culled": len(keep), "scores": scores})

    keys = [f"{m}@k{k}" for m in SELECTORS for k in ks]
    table = {}
    for key in keys:
        r2 = np.array([fold["scores"][key]["r2"] for fold in folds])
        rmse = np.array([fold["scores"][key]["rmse"] for fold in folds])
        table[key] = {
            "method": fold_method(key),
            "k": int(key.rsplit("@k", 1)[1]),
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std(ddof=1)),
            "r2_folds": [float(v) for v in r2],
            "rmse_mean": float(rmse.mean()),
            "k_effective": [fold["scores"][key]["k_effective"] for fold in folds],
        }
    best = max(table, key=lambda n: table[n]["r2_mean"])
    base = "corr@k150"
    vs_base = None
    if base in table and best != base:
        delta = np.array(table[best]["r2_folds"]) - np.array(table[base]["r2_folds"])
        mean = float(delta.mean())
        se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        vs_base = {"mean": mean, "se": se, "folds": [float(v) for v in delta]}

    verdict = {"best": best, "base": base, "vs_base": vs_base, "table": table}
    _print(verdict)
    out = OUT.with_name("selector_k_grid.smoke.json") if smoke else OUT
    payload = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "smoke": smoke,
        "model": "svr",
        "selectors": list(SELECTORS),
        "ks": list(ks),
        "folds_n": folds_n,
        "verdict": verdict,
        "folds": folds,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)
    return payload


def fold_method(key: str) -> str:
    return key.rsplit("@k", 1)[0]


def _print(verdict: dict) -> None:
    print(f"best={verdict['best']}  base={verdict['base']}", flush=True)
    if verdict["vs_base"] is not None:
        gap = verdict["vs_base"]
        print(f"vs {verdict['base']}: {gap['mean']:+.3f} (SE {gap['se']:.3f})", flush=True)
    print(f"{'arm':<18} {'p':>5} {'svr r2':>8} {'std':>8} {'rmse':>8}", flush=True)
    order = sorted(verdict["table"], key=lambda n: verdict["table"][n]["r2_mean"], reverse=True)
    for name in order:
        row = verdict["table"][name]
        width = int(np.median(row["k_effective"]))
        print(
            f"{name:<18} {width:5d} {row['r2_mean']:8.3f} {row['r2_std']:8.3f} {row['rmse_mean']:8.3f}",
            flush=True,
        )


def main() -> None:
    smoke = "--smoke" in sys.argv
    if not smoke:
        _acquire()
    try:
        run(smoke=smoke)
    finally:
        if not smoke:
            _release()


if __name__ == "__main__":
    main()
