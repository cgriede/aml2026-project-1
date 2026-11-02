"""MAE-weighted blend of ridge, random forest, XGBoost, and MLP.

The network reads the four out-of-fold predictions and returns a softmax
weight for each. The blend is the weighted sum. Loss is MAE against age.
True age is not an input. Base predictions come from experiment 3, so the
base models are not refit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments/003/results/models.json"
OUT = ROOT / "experiments/004/results/weight_net.json"
PLOT = ROOT / "experiments/004/results/weights_by_age.png"
BASES = ("ridge", "random_forest", "xgboost", "mlp")
LABELS = {"ridge": "Ridge", "random_forest": "Random forest", "xgboost": "XGBoost", "mlp": "MLP"}
HIDDEN = 16
LR = 1e-2
WEIGHT_SPAN = 2.0
OFFSET_SPAN = 15.0
EPOCHS = 800
PATIENCE = 40
SEED = 42
INK = "#1c1c1c"
COLORS = {"ridge": "#2166ac", "random_forest": "#b85c38", "xgboost": "#1f6b4a", "mlp": "#6b4c9a"}


def _stack(model_predictions: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.ravel(model_predictions[name]) for name in BASES])


class NnEnsembler:
    """Softmax weights over ridge, random forest, XGBoost, and MLP.

    HIDDEN and LR are the first-run settings. They were not tuned.
    `features` is an optional matrix aligned with the predictions, chosen
    outside this class. The blend still uses only the four raw predictions.
    """

    def __init__(self, hidden: int = HIDDEN, lr: float = LR, seed: int = SEED, link: str = "softmax"):
        self.hidden = hidden
        self.lr = lr
        self.seed = seed
        self.link = link
        self.history: dict[str, list[float]] = {"train_mae": [], "val_mae": [], "train_r2": [], "val_r2": []}
        self.best_epoch = 0

    def _init(self, n_in: int) -> None:
        rng = np.random.default_rng(self.seed)
        n_out = 4 if self.link == "softmax" else 5
        self.W1 = rng.normal(0, 0.05, size=(n_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.05, size=(self.hidden, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, model_predictions: dict[str, np.ndarray], features: np.ndarray | None = None):
        P = _stack(model_predictions)
        raw = P if features is None else np.column_stack([P, np.asarray(features, dtype=float)])
        X = (raw - self.mu) / self.sd
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        if self.link == "softmax":
            logits = logits - logits.max(axis=1, keepdims=True)
            weight = np.exp(logits)
            weight /= weight.sum(axis=1, keepdims=True)
            const = np.zeros(len(P))
            blended = (weight * P).sum(axis=1)
        else:
            weight = WEIGHT_SPAN * np.tanh(logits[:, :4])
            const = OFFSET_SPAN * np.tanh(logits[:, 4])
            blended = (weight * P).sum(axis=1) + const
        named = {name: weight[:, i] for i, name in enumerate(BASES)}
        return named, blended, {"P": P, "h": h, "X": X, "weight": weight, "const": const, "logits": logits}

    def fit(self, model_predictions: dict[str, np.ndarray], y, features: np.ndarray | None = None):
        y = np.ravel(y)
        P = _stack(model_predictions)
        raw = P if features is None else np.column_stack([P, np.asarray(features, dtype=float)])
        idx = np.arange(len(y))
        fit_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=self.seed)
        self.mu = raw[fit_idx].mean(axis=0)
        self.sd = raw[fit_idx].std(axis=0) + 1e-6
        self._init(raw.shape[1])
        self.history = {key: [] for key in ("train_mae", "val_mae", "train_r2", "val_r2")}
        moment = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        velocity = {k: np.zeros_like(v) for k, v in moment.items()}
        best = {k: getattr(self, k).copy() for k in moment}
        best_mae, wait, step = np.inf, 0, 0

        def _slice(rows):
            pred = {name: np.ravel(values)[rows] for name, values in model_predictions.items()}
            extra = None if features is None else np.asarray(features)[rows]
            return pred, extra

        train_pred, train_x = _slice(fit_idx)
        val_pred, val_x = _slice(val_idx)
        for _ in range(EPOCHS):
            _, blended, cache = self.forward(train_pred, train_x)
            P_fit, h, X, weight = cache["P"], cache["h"], cache["X"], cache["weight"]
            d_blend = np.sign(blended - y[fit_idx]) / len(fit_idx)
            if self.link == "softmax":
                d_logits = d_blend[:, None] * weight * (P_fit - blended[:, None])
            else:
                dw = WEIGHT_SPAN * (1 - (weight / WEIGHT_SPAN) ** 2)
                d_w = d_blend[:, None] * P_fit * dw
                const = cache["const"]
                dc = OFFSET_SPAN * (1 - (const / OFFSET_SPAN) ** 2)
                d_logits = np.column_stack([d_w, d_blend * dc])
            d_h = d_logits @ self.W2.T
            d_h[h <= 0.0] = 0.0
            grads = {"W1": X.T @ d_h, "b1": d_h.sum(0), "W2": h.T @ d_logits, "b2": d_logits.sum(0)}
            step += 1
            for key, grad in grads.items():
                moment[key] = 0.9 * moment[key] + 0.1 * grad
                velocity[key] = 0.999 * velocity[key] + 0.001 * grad**2
                mhat = moment[key] / (1 - 0.9**step)
                vhat = velocity[key] / (1 - 0.999**step)
                setattr(self, key, getattr(self, key) - self.lr * mhat / (np.sqrt(vhat) + 1e-8))
            _, train_hat, _ = self.forward(train_pred, train_x)
            _, val_hat, _ = self.forward(val_pred, val_x)
            train_mae = float(np.mean(np.abs(train_hat - y[fit_idx])))
            val_mae = float(np.mean(np.abs(val_hat - y[val_idx])))
            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)
            self.history["train_r2"].append(float(r2_score(y[fit_idx], train_hat)))
            self.history["val_r2"].append(float(r2_score(y[val_idx], val_hat)))
            if val_mae + 1e-5 < best_mae:
                best_mae = val_mae
                best = {k: getattr(self, k).copy() for k in moment}
                self.best_epoch = len(self.history["val_mae"]) - 1
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        for key, value in best.items():
            setattr(self, key, value)
        return self


def predict_weights(model_predictions, model=None, features=None):
    model = NnEnsembler() if model is None else model
    predicted_weight, y_ensembled, _ = model.forward(model_predictions, features)
    return predicted_weight, y_ensembled


def _metrics(y, pred) -> dict:
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(np.mean(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
    }


def _sample_rows(y, P, weight, blended) -> list[dict]:
    order = np.argsort(y)
    picks = [order[0], order[len(order) // 5], order[len(order) // 2], order[4 * len(order) // 5], order[-1]]
    rows = []
    for i in picks:
        rows.append(
            {
                "y": float(y[i]),
                "blend": float(blended[i]),
                "pred": {name: float(P[i, k]) for k, name in enumerate(BASES)},
                "weight": {name: float(weight[i, k]) for k, name in enumerate(BASES)},
            }
        )
    return rows


def run() -> dict:
    payload = json.loads(SRC.read_text())
    y = np.array(payload["y"], dtype=float)
    P = np.column_stack([np.array(payload["pred"][name], dtype=float) for name in BASES])
    stored = payload["verdict"]["table"]["xgboost"]["r2_folds"]
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    check = []
    for _, te in splitter.split(y):
        check.append(float(r2_score(y[te], P[te, BASES.index("xgboost")])))
    if max(abs(a - b) for a, b in zip(check, stored)) > 1e-6:
        raise SystemExit("experiment 3 predictions are not on these folds")

    blend = np.full(len(y), np.nan)
    weight = np.full((len(y), 4), np.nan)
    fold_rows = []
    columns = {name: P[:, i] for i, name in enumerate(BASES)}
    for fold_id, (tr, te) in enumerate(splitter.split(y)):
        train_pred = {name: values[tr] for name, values in columns.items()}
        test_pred = {name: values[te] for name, values in columns.items()}
        model = NnEnsembler(seed=SEED + fold_id).fit(train_pred, y[tr])
        predicted_weight, y_ensembled = predict_weights(test_pred, model)
        hat = y_ensembled
        w = np.column_stack([predicted_weight[name] for name in BASES])
        blend[te] = hat
        weight[te] = w
        row = _metrics(y[te], hat)
        row["fold"] = fold_id
        fold_rows.append(row)
        print(f"fold {fold_id}  blend R2 {row['r2']:.3f}  MAE {row['mae']:.3f}", flush=True)

    mean4 = P.mean(axis=1)
    mean3 = P[:, :3].mean(axis=1)
    scores = {
        "weight_net": _metrics(y, blend),
        "mean_of_4": _metrics(y, mean4),
        "mean_of_ridge_rf_xgb": _metrics(y, mean3),
    }
    for name in BASES:
        scores[name] = _metrics(y, P[:, BASES.index(name)])
    scores["xgboost_bag"] = _metrics(y, np.array(payload["pred"]["xgboost_bag"]))
    fold_r2 = [row["r2"] for row in fold_rows]
    samples = _sample_rows(y, P, weight, blend)
    print(f"{'model':<22} {'R2':>8} {'MAE':>8}", flush=True)
    for name, row in sorted(scores.items(), key=lambda item: -item[1]["r2"]):
        print(f"{name:<22} {row['r2']:8.3f} {row['mae']:8.3f}", flush=True)
    print("y   blend   weights ridge, forest, xgb, mlp    preds", flush=True)
    for row in samples:
        w = row["weight"]
        pred = row["pred"]
        print(
            f"{row['y']:5.1f}  {row['blend']:5.1f}   "
            f"{w['ridge']:.2f} {w['random_forest']:.2f} {w['xgboost']:.2f} {w['mlp']:.2f}    "
            f"{pred['ridge']:.1f} {pred['random_forest']:.1f} {pred['xgboost']:.1f} {pred['mlp']:.1f}",
            flush=True,
        )
    result = {
        "bases": list(BASES),
        "fold_r2": fold_r2,
        "fold_r2_mean": float(np.mean(fold_r2)),
        "fold_r2_std": float(np.std(fold_r2, ddof=1)),
        "scores": scores,
        "samples": samples,
        "y": y.tolist(),
        "blend": blend.tolist(),
        "weight": weight.tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result))
    _plot(y, weight)
    print(f"wrote {OUT}", flush=True)
    return result


def _plot(y, weight) -> None:
    edges = np.quantile(y, [0, 0.2, 0.4, 0.6, 0.8, 1])
    bands = np.digitize(y, edges[1:-1], right=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs, labels = [], []
    for band in range(5):
        mask = bands == band
        xs.append(float(y[mask].mean()))
        labels.append(f"{y[mask].min():.0f}-{y[mask].max():.0f}")
    for k, name in enumerate(BASES):
        means = [float(weight[bands == band, k].mean()) for band in range(5)]
        ax.plot(xs, means, marker="o", color=COLORS[name], label=LABELS[name], linewidth=1.6)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean softmax weight")
    ax.set_xlabel("True age")
    ax.set_title("Learned blend weights by true age", loc="left", color=INK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)
    print(PLOT, flush=True)


if __name__ == "__main__":
    run()
