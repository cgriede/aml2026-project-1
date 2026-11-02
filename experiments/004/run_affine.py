"""NnEnsembler with weights in [-2, 2] and an additive offset in [-15, 15] years.

The softmax blend can only sit between the four predictions. This head can
down-weight a model that runs high and shift the result up or down.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "004"))

from run_weight_net import BASES, SRC, NnEnsembler, _metrics  # noqa: E402

OUT = ROOT / "experiments/004/results/affine.json"
PLOT = ROOT / "experiments/004/results/affine_offset.png"
INK = "#1c1c1c"
COLORS = {"ridge": "#2166ac", "random_forest": "#b85c38", "xgboost": "#1f6b4a", "mlp": "#6b4c9a"}
CONFIGS = (
    {"name": "softmax h16", "hidden": 16, "lr": 1e-2, "link": "softmax"},
    {"name": "affine h16", "hidden": 16, "lr": 1e-2, "link": "affine"},
    {"name": "affine h16, lr 1e-1", "hidden": 16, "lr": 1e-1, "link": "affine"},
    {"name": "affine h32", "hidden": 32, "lr": 1e-2, "link": "affine"},
)


def run() -> None:
    payload = json.loads(Path(SRC).read_text())
    y = np.array(payload["y"], dtype=float)
    columns = {name: np.array(payload["pred"][name], dtype=float) for name in BASES}
    bag = np.array(payload["pred"]["xgboost_bag"])
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(y))
    print(f"{'config':<22} {'R2':>8} {'MAE':>8} {'w min':>8} {'w max':>8} {'c min':>8} {'c max':>8}", flush=True)
    kept = None
    for cfg in CONFIGS:
        blend = np.full(len(y), np.nan)
        weight = np.full((len(y), 4), np.nan)
        const = np.full(len(y), np.nan)
        for fold_id, (tr, te) in enumerate(folds):
            train_pred = {name: values[tr] for name, values in columns.items()}
            test_pred = {name: values[te] for name, values in columns.items()}
            model = NnEnsembler(hidden=cfg["hidden"], lr=cfg["lr"], seed=42 + fold_id, link=cfg["link"])
            model.fit(train_pred, y[tr])
            named, hat, cache = model.forward(test_pred)
            blend[te] = hat
            weight[te] = np.column_stack([named[name] for name in BASES])
            const[te] = cache["const"]
        score = _metrics(y, blend)
        print(
            f"{cfg['name']:<22} {score['r2']:8.3f} {score['mae']:8.3f} "
            f"{np.nanmin(weight):8.2f} {np.nanmax(weight):8.2f} "
            f"{np.nanmin(const):8.1f} {np.nanmax(const):8.1f}",
            flush=True,
        )
        if cfg["name"] == "affine h32":
            kept = {"y": y, "blend": blend, "weight": weight, "const": const, "bag": bag, "score": score}
    _plot(kept)
    OUT.write_text(json.dumps({"score": kept["score"], "const_min": float(kept["const"].min()), "const_max": float(kept["const"].max())}))
    print(PLOT, flush=True)


def _plot(kept: dict) -> None:
    y, const, weight = kept["y"], kept["const"], kept["weight"]
    edges = np.quantile(y, [0, 0.2, 0.4, 0.6, 0.8, 1])
    bands = np.digitize(y, edges[1:-1], right=True)
    xs, labels = [], []
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for band in range(5):
        mask = bands == band
        xs.append(float(y[mask].mean()))
        labels.append(f"{y[mask].min():.0f}-{y[mask].max():.0f}")
    ax = axes[0]
    for k, name in enumerate(BASES):
        means = [float(weight[bands == band, k].mean()) for band in range(5)]
        ax.plot(xs, means, marker="o", color=COLORS[name], label=name.replace("_", " "), linewidth=1.6)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.set_ylim(-2.1, 2.1)
    ax.set_ylabel("Mean weight")
    ax.set_xlabel("True age")
    ax.set_title("Weights in [-2, 2]", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    means = [float(const[bands == band].mean()) for band in range(5)]
    ax.plot(xs, means, marker="o", color=INK, linewidth=1.6)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Mean added years")
    ax.set_xlabel("True age")
    ax.set_title("Learned offset", loc="left", color=INK)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#f0f0f0")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)


if __name__ == "__main__":
    run()
