"""Residual diagnostics for the two specialists and their blends."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "009" / "results"
INK = "#1c1c1c"
LINE = "#c4552a"
DOT = "#1f6b4a"
COLORS = {
    "corr_xgb": "#1f6b4a",
    "rrelieff_svr": "#2a5f8f",
    "mean": "#c4552a",
    "nnls": "#7a3e9d",
}
LABELS = {
    "corr_xgb": "Corr XGB",
    "rrelieff_svr": "RReliefF SVR",
    "mean": "Mean blend",
    "nnls": "NNLS blend",
}
SERIES = ("corr_xgb", "rrelieff_svr", "mean", "nnls")
BIN_WIDTH = 5.0


def _load() -> dict:
    smoke = "--smoke" in sys.argv
    path = RESULTS / ("union.smoke.json" if smoke else "union.json")
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/009/run_union.py first")
    return json.loads(path.read_text())


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color="#f0f0f0", zorder=0)
    ax.set_axisbelow(True)


def _bins(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = BIN_WIDTH * np.floor(y.min() / BIN_WIDTH)
    hi = BIN_WIDTH * np.ceil(y.max() / BIN_WIDTH)
    edges = np.arange(lo, hi + BIN_WIDTH, BIN_WIDTH)
    centers = edges[:-1] + BIN_WIDTH / 2
    return edges, centers


def _binned_mean(y: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        mask = (y >= edges[i]) & (y < edges[i + 1])
        if i == len(edges) - 2:
            mask = (y >= edges[i]) & (y <= edges[i + 1])
        if mask.any():
            out[i] = float(np.mean(values[mask]))
    return out


def residuals(src: dict) -> Path:
    y = np.array(src["y"], dtype=float)
    edges, centers = _bins(y)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), SERIES):
        pred = np.array(src["pred"][name], dtype=float)
        resid = pred - y
        ax.scatter(y, resid, s=10, alpha=0.35, color=COLORS[name], linewidths=0, zorder=2)
        binned = _binned_mean(y, resid, edges)
        ax.plot(centers, binned, color=LINE, linewidth=1.4, zorder=3)
        ax.axhline(0, color="#888", linewidth=0.7, zorder=1)
        r2 = src["verdict"]["table"][name]["r2_mean"]
        ax.set_title(f"{LABELS[name]}   R² {r2:.3f}", loc="left", color=INK, fontsize=10, pad=3)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Residual (pred − age)")
    for ax in axes[-1, :]:
        ax.set_xlabel("True age")
    fig.suptitle(
        "Out-of-fold residual against true age",
        color=INK,
        fontsize=14,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.012,
        "Orange is the mean residual in 5-year bins. A flatter line handles the age extremes better.",
        color="#444",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = RESULTS / ("residual_vs_age.smoke.png" if src.get("smoke") else "residual_vs_age.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return out


def residual_by_nan(src: dict) -> Path:
    y = np.array(src["y"], dtype=float)
    nan_count = np.array(src["nan_count"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for ax, name in zip(axes, ("corr_xgb", "rrelieff_svr")):
        pred = np.array(src["pred"][name], dtype=float)
        resid = pred - y
        pts = ax.scatter(
            y,
            resid,
            c=nan_count,
            s=12,
            alpha=0.75,
            cmap="viridis",
            linewidths=0,
            zorder=2,
        )
        ax.axhline(0, color="#888", linewidth=0.7, zorder=1)
        ax.set_title(LABELS[name], loc="left", color=INK, fontsize=10, pad=3)
        _style(ax)
        ax.set_xlabel("True age")
    axes[0].set_ylabel("Residual (pred − age)")
    fig.suptitle(
        "Specialist residuals coloured by missingness",
        color=INK,
        fontsize=14,
        x=0.01,
        ha="left",
    )
    fig.subplots_adjust(left=0.08, right=0.88, top=0.86, bottom=0.12, wspace=0.18)
    cbar = fig.colorbar(pts, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Raw NaN count", color=INK)
    out = RESULTS / ("residual_by_nan.smoke.png" if src.get("smoke") else "residual_by_nan.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return out


def binned_mae(src: dict) -> Path:
    y = np.array(src["y"], dtype=float)
    edges, centers = _bins(y)
    counts = np.array(
        [
            ((y >= edges[i]) & (y < edges[i + 1] if i < len(edges) - 2 else y <= edges[i + 1])).sum()
            for i in range(len(edges) - 1)
        ],
        dtype=float,
    )
    # Twin-axis bars sit on top of the lines in matplotlib. Put n in its own strip.
    fig, (ax, ax_n) = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.06},
    )
    for name in SERIES:
        pred = np.array(src["pred"][name], dtype=float)
        abs_err = np.abs(pred - y)
        mae = _binned_mean(y, abs_err, edges)
        # Sparse bins stay visible but thinner, so the middle does not look authoritative by accident.
        sizes = 3.0 + 5.0 * np.sqrt(counts / max(counts.max(), 1.0))
        ax.plot(
            centers,
            mae,
            color=COLORS[name],
            linewidth=1.6,
            label=LABELS[name],
            zorder=2,
        )
        ax.scatter(
            centers,
            mae,
            s=sizes**2,
            color=COLORS[name],
            linewidths=0,
            zorder=3,
        )
    _style(ax)
    ax.set_ylabel("Mean absolute error")
    ax.set_title(
        "Binned MAE by age. Crossing lines would justify an age-dependent blend.",
        loc="left",
        color=INK,
        fontsize=12,
        pad=18,
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=4,
        borderaxespad=0.2,
        columnspacing=1.2,
        handlelength=1.6,
    )
    ax.tick_params(labelbottom=False)

    ax_n.bar(centers, counts, width=BIN_WIDTH * 0.8, color="#c8c8c8", edgecolor="none", zorder=1)
    _style(ax_n)
    ax_n.set_ylabel("n")
    ax_n.set_xlabel("True age (bin centre)")
    ax_n.set_ylim(0, max(counts.max() * 1.15, 1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.1, hspace=0.08)
    out = RESULTS / ("mae_by_age.smoke.png" if src.get("smoke") else "mae_by_age.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return out


def residual_scatter(src: dict) -> Path:
    y = np.array(src["y"], dtype=float)
    a = np.array(src["pred"]["corr_xgb"], dtype=float)
    b = np.array(src["pred"]["rrelieff_svr"], dtype=float)
    ra, rb = a - y, b - y
    pred_corr = float(np.corrcoef(a, b)[0, 1])
    resid_corr = float(np.corrcoef(ra, rb)[0, 1])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lim = float(np.nanmax(np.abs(np.concatenate([ra, rb])))) * 1.05
    ax.scatter(ra, rb, s=12, alpha=0.4, color=DOT, linewidths=0, zorder=2)
    ax.plot([-lim, lim], [-lim, lim], color=LINE, linewidth=0.9, zorder=3)
    ax.axhline(0, color="#888", linewidth=0.6)
    ax.axvline(0, color="#888", linewidth=0.6)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Corr XGB residual")
    ax.set_ylabel("RReliefF SVR residual")
    ax.set_title(
        f"Residual scatter. Pred corr {pred_corr:.3f}, residual corr {resid_corr:.3f}",
        loc="left",
        color=INK,
        fontsize=11,
        pad=8,
    )
    _style(ax)
    fig.text(
        0.01,
        0.012,
        "Points far from the diagonal are samples where one specialist is right and the other wrong.",
        color="#444",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = RESULTS / ("residual_scatter.smoke.png" if src.get("smoke") else "residual_scatter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return out


def hard_samples(src: dict) -> Path:
    y = np.array(src["y"], dtype=float)
    nan_count = np.array(src["nan_count"], dtype=float)
    fold_id = np.array(src["fold_id"], dtype=int)
    a = np.array(src["pred"]["corr_xgb"], dtype=float)
    b = np.array(src["pred"]["rrelieff_svr"], dtype=float)
    mean_abs = 0.5 * (np.abs(a - y) + np.abs(b - y))
    n_hard = max(1, int(np.ceil(0.05 * len(y))))
    order = np.argsort(-mean_abs)[:n_hard]
    rows = []
    for i in order:
        rows.append(
            {
                "idx": int(i),
                "age": float(y[i]),
                "nan_count": float(nan_count[i]),
                "fold": int(fold_id[i]),
                "corr_xgb_resid": float(a[i] - y[i]),
                "rrelieff_svr_resid": float(b[i] - y[i]),
                "mean_abs_err": float(mean_abs[i]),
            }
        )
    out_json = RESULTS / ("hard_samples.smoke.json" if src.get("smoke") else "hard_samples.json")
    out_json.write_text(json.dumps({"n": n_hard, "rows": rows}, indent=2))
    print(out_json)

    fig, ax = plt.subplots(figsize=(9.2, max(3.5, 0.28 * n_hard + 1.2)))
    ypos = np.arange(n_hard)
    ax.barh(ypos - 0.15, [r["corr_xgb_resid"] for r in rows], height=0.28, color=COLORS["corr_xgb"], label="Corr XGB")
    ax.barh(ypos + 0.15, [r["rrelieff_svr_resid"] for r in rows], height=0.28, color=COLORS["rrelieff_svr"], label="RReliefF SVR")
    labels = [f"age {r['age']:.0f}  nan {r['nan_count']:.0f}" for r in rows]
    ax.set_yticks(ypos, labels)
    ax.axvline(0, color="#888", linewidth=0.7)
    ax.invert_yaxis()
    ax.set_xlabel("Residual (pred − age)")
    ax.set_title(
        f"Hardest {n_hard} samples by mean absolute specialist error",
        loc="left",
        color=INK,
        fontsize=12,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    out = RESULTS / ("hard_samples.smoke.png" if src.get("smoke") else "hard_samples.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return out


def main() -> None:
    src = _load()
    RESULTS.mkdir(parents=True, exist_ok=True)
    residuals(src)
    residual_by_nan(src)
    binned_mae(src)
    residual_scatter(src)
    hard_samples(src)


if __name__ == "__main__":
    main()
