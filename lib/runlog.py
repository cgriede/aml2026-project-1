import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import GLOBAL as G

RUNS_DIR = Path("runs")
RESIDUALS_DIR = RUNS_DIR / "residuals"
JSONL = RUNS_DIR / "log.jsonl"
MD = RUNS_DIR / "log.md"
BIN_WIDTH = 5.0
INK = "#1c1c1c"
LINE = "#c4552a"

_HEAD = ("ts", "NOTE")


def append_run(metrics: dict, models: list, oof: dict | None = None) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {
        "ts": ts,
        **_globals(),
        **metrics,
        "models": {
            m.__class__.__name__: _jsonable(m.get_params())
            for m in models
        },
    }
    if oof is not None:
        plot_path = _save_residual_plot(ts, oof)
        row["residual_plot"] = str(plot_path)
        print(f"residual plot {plot_path}")
    with JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    _write_md()


def _binned_mean(y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = BIN_WIDTH * np.floor(y.min() / BIN_WIDTH)
    hi = BIN_WIDTH * np.ceil(y.max() / BIN_WIDTH)
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2
    means = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            mask = (y >= edges[i]) & (y < edges[i + 1])
        else:
            mask = (y >= edges[i]) & (y <= edges[i + 1])
        if mask.any():
            means[i] = float(np.mean(values[mask]))
    return centers, means


def _save_residual_plot(ts: str, oof: dict) -> Path:
    """oof: {y, pred: {name: array}, scores?: {name: r2}}."""
    RESIDUALS_DIR.mkdir(parents=True, exist_ok=True)
    y = np.ravel(oof["y"]).astype(float)
    preds = oof["pred"]
    scores = oof.get("scores") or {}
    names = list(preds)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    colors = ("#1f6b4a", "#2a5f8f", "#c4552a", "#7a3e9d")
    for ax, name, color in zip(axes, names, colors):
        pred = np.ravel(preds[name]).astype(float)
        resid = pred - y
        ax.scatter(y, resid, s=10, alpha=0.35, color=color, linewidths=0, zorder=2)
        centers, means = _binned_mean(y, resid)
        ax.plot(centers, means, color=LINE, linewidth=1.4, zorder=3)
        ax.axhline(0, color="#888", linewidth=0.7, zorder=1)
        title = name
        if name in scores:
            title = f"{name}   R² {float(scores[name]):.3f}"
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=4)
        ax.set_xlabel("True age")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
        ax.grid(color="#f0f0f0", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Residual (pred − age)")
    fig.suptitle(
        "Out-of-fold residual against true age",
        color=INK,
        fontsize=13,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.015,
        "Orange is the mean residual in 5-year bins. Same CV folds as the logged scores.",
        color="#444",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    path = RESIDUALS_DIR / f"{ts.replace(':', '')}_residual_vs_age.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _globals() -> dict:
    out = {}
    for k, v in vars(G).items():
        if k.isupper():
            out[k] = str(v) if isinstance(v, Path) else v
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _write_md() -> None:
    rows = []
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    keys = []
    seen = set()
    for k in _HEAD:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    extras = sorted({k for r in rows for k, v in r.items() if k not in seen and _is_scalar(v)})
    keys.extend(extras)

    def cell(r, k):
        v = r.get(k, "")
        if v is None:
            return ""
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(cell(r, k) for k in keys) + " |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
