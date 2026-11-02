"""Predicted age against true age, and MAE inside target quintiles."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = json.loads((ROOT / "experiments/003/results/models.json").read_text())
OUT_SCATTER = ROOT / "experiments/003/results/response_vs_age.png"
OUT_BANDS = ROOT / "experiments/003/results/mae_by_age.png"
OUT_DIST = ROOT / "experiments/003/results/age_distribution.png"
INK = "#1c1c1c"
LINE = "#c4552a"
DOT = "#1f6b4a"

LABELS = {
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "knn": "kNN",
    "svr": "SVR",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "hist_gbr": "Hist. boosting",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "xgboost_bag": "XGB bag",
    "age_mixture": "Age mixture",
    "age_residual": "Age residual",
}


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)


def scatter() -> None:
    y = np.array(SRC["y"])
    names = list(LABELS)
    lo, hi = float(y.min()) - 2, float(y.max()) + 2
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.6), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        pred = np.array(SRC["pred"][name])
        ax.scatter(y, pred, s=8, alpha=0.35, color=DOT, linewidths=0, zorder=2)
        ax.plot([lo, hi], [lo, hi], color=LINE, linewidth=0.9, zorder=3)
        r2 = SRC["verdict"]["table"][name]["r2_mean"]
        ax.set_title(f"{LABELS[name]}   {r2:.3f}", loc="left", color=INK, fontsize=10, pad=3)
        _style(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted age")
    for ax in axes[-1, :]:
        ax.set_xlabel("True age")
    fig.suptitle("Out-of-fold prediction against true age", color=INK, fontsize=14, x=0.01, ha="left")
    fig.text(0.01, 0.012, "Each point is one training person, scored from the fold that held them out. Orange is the identity line. The number is mean outer-fold R².", color="#444", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(OUT_SCATTER, dpi=140)
    print(OUT_SCATTER)


def distributions() -> None:
    y = np.array(SRC["y"])
    names = list(LABELS)
    bins = np.linspace(40, 100, 25)
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.6), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        pred = np.array(SRC["pred"][name])
        ax.hist(y, bins=bins, density=True, color="#d9d9d9", label="True age", zorder=1)
        ax.hist(pred, bins=bins, density=True, histtype="step", linewidth=1.6, color=DOT, label="Predicted", zorder=2)
        _style(ax)
        ax.set_title(LABELS[name], loc="left", color=INK, fontsize=10, pad=3)
        ax.set_xlim(40, 100)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    for ax in axes[-1, :]:
        ax.set_xlabel("Age")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("True age distribution against each model's predictions", color=INK, fontsize=14, x=0.01, ha="left")
    fig.text(0.01, 0.012, "Gray is the same true-age distribution in every panel. The green outline is that model's out-of-fold predictions.", color="#444", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(OUT_DIST, dpi=140)
    print(OUT_DIST)


def bands() -> None:
    rows = SRC["age_bands"]
    names = list(LABELS)
    mae = np.array([[row[name]["mae"] for row in rows] for name in names])
    labels = [f"{row['age_min']:.0f}-{row['age_max']:.0f}" for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    image = ax.imshow(mae, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(rows)), labels)
    ax.set_yticks(range(len(names)), [LABELS[name] for name in names])
    best = mae.argmin(axis=0)
    for i in range(mae.shape[0]):
        for j in range(mae.shape[1]):
            weight = "bold" if i == best[j] else "normal"
            ax.text(j, i, f"{mae[i, j]:.1f}", ha="center", va="center", fontsize=8, color=INK, fontweight=weight)
    ax.set_xlabel("True age")
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_title("MAE by true-age quintile. Bold is the lowest error in that range.", loc="left", color=INK, fontsize=12, pad=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("MAE (years)", color=INK)
    fig.tight_layout()
    fig.savefig(OUT_BANDS, dpi=140)
    print(OUT_BANDS)


if __name__ == "__main__":
    scatter()
    distributions()
    bands()
