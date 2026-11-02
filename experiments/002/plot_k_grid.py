"""Outer-fold R² at each |corr| prefix. Reads the wide grid and the 110–190 zoom."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIDE = json.loads((ROOT / "experiments/002/results/k_grid.json").read_text())
ZOOM = json.loads((ROOT / "experiments/002/results/k_grid_zoom.json").read_text())
OUT = ROOT / "experiments/002/results/k_grid_r2.png"
SUMMARY = ROOT / "experiments/002/results/k_grid_means.png"
SERIES = (
    ("ridge", "Ridge", "#2166ac"),
    ("xgb", "XGBoost", "#1f6b4a"),
    ("svr", "SVR", "#b85c38"),
)

ORDER = [
    "25", "50", "75", "100",
    "110", "130", "140", "150", "160", "170", "190",
    "200", "300", "400", "600", "825", "all",
]
ZOOM_ONLY = {"110", "130", "140", "160", "170", "190"}
MODELS = (
    ("ridge", "Ridge"),
    ("xgb", "XGBoost"),
    ("svr", "SVR"),
)
INK = "#1c1c1c"
MEAN = "#1f6b4a"
FOLD = "#8a8a8a"
BASE = "#c4552a"


def _row(name: str) -> dict:
    source = ZOOM if name in ZOOM_ONLY else WIDE
    return source["verdict"]["table"][name]


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 8.4), sharex=True)
    xs = np.arange(len(ORDER))
    for ax, (key, title) in zip(axes, MODELS):
        means, folds = [], []
        for name in ORDER:
            row = _row(name)[key]
            means.append(row["r2_mean"])
            folds.append(row["r2_folds"])
        for i, scores in enumerate(folds):
            jitter = np.linspace(-0.12, 0.12, len(scores))
            ax.scatter(xs[i] + jitter, scores, s=22, color=FOLD, zorder=2)
        ax.plot(xs, means, color=MEAN, marker="o", markersize=5.5, linewidth=1.6, zorder=3, label="mean")
        base = _row("150")[key]["r2_mean"]
        ax.axhline(base, color=BASE, linewidth=0.8, linestyle="--", zorder=1)
        ax.axvline(ORDER.index("150"), color=BASE, linewidth=0.6, alpha=0.45)
        ax.set_ylabel("R²")
        ax.set_title(title, loc="left", color=INK, fontsize=12, pad=4)
        ax.tick_params(colors=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#ececec", zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylim(-3.0, 0.55)
    axes[1].set_ylim(0.40, 0.62)
    axes[2].set_ylim(0.20, 0.50)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(ORDER, fontsize=8)
    axes[-1].set_xlabel("k    (all = uncullled 825 columns)")
    fig.suptitle("Outer-fold R² by |corr| prefix", color=INK, fontsize=14, x=0.02, ha="left")
    fig.text(
        0.02,
        0.012,
        "Gray points are the five folds. Green is the mean. Orange marks k=150. 110 to 190 are the zoom.",
        color="#444",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT, dpi=140)
    print(OUT)
    _summary()


def _summary() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    xs = np.arange(len(ORDER))
    for key, label, color in SERIES:
        means = [_row(name)[key]["r2_mean"] for name in ORDER]
        ax.plot(xs, means, color=color, marker="o", markersize=5, linewidth=1.8, label=label, zorder=3)
    ax.axhline(0, color="#d0d0d0", linewidth=0.8, zorder=1)
    ax.axvline(ORDER.index("150"), color="#c4552a", linewidth=0.7, alpha=0.7, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER, fontsize=8)
    ax.set_xlabel("k    (all = uncullled 825 columns)")
    ax.set_ylabel("Mean outer-fold R²")
    ax.set_title("Mean R² on one scale", loc="left", color=INK, fontsize=14)
    ax.tick_params(colors=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#ececec", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(SUMMARY, dpi=140)
    print(SUMMARY)


if __name__ == "__main__":
    main()
