import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GLOBAL import (
    FS_CORR_MAX,
    FS_N_ROUNDS,
    FS_N_TRIALS,
    FS_SCORE_SPLIT,
    FS_SUBSET_SIZE,
    FS_TOP_FRAC,
    FS_TRACK,
    RANDOM_STATE,
)


def select_features(
    X: pd.DataFrame,
    y: pd.DataFrame | None,
    selected_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    if selected_cols is None:
        X = X[_drop_high_corr(X, y, FS_CORR_MAX)]
        selected_cols = _monte_carlo_select(X, y)
    return X[selected_cols], y, selected_cols


def _drop_high_corr(X: pd.DataFrame, y, tau: float) -> list[str]:
    y = pd.Series(np.ravel(y), index=X.index)
    corr = X.corr().abs()
    ycorr = X.corrwith(y).abs()
    drop = set()
    cols = list(X.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            if c1 in drop or c2 in drop:
                continue
            if corr.loc[c1, c2] >= tau:
                drop.add(c2 if ycorr[c1] >= ycorr[c2] else c1)
    if drop:
        print(f"High-corr drop |r|>={tau}: {sorted(drop)}")
    else:
        print(f"High-corr drop |r|>={tau}: none")
    return [c for c in cols if c not in drop]


def _monte_carlo_select(X, y):
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.ravel(y)
    all_cols = list(X.columns)
    active = list(all_cols)
    lifetime = {c: 0.0 for c in all_cols}
    track = list(rng.choice(all_cols, size=min(FS_TRACK, len(all_cols)), replace=False))
    history = {c: [0] for c in track}

    X_tr, X_sc, y_tr, y_sc = train_test_split(
        X, y, test_size=FS_SCORE_SPLIT, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
    X_sc = pd.DataFrame(scaler.transform(X_sc), index=X_sc.index, columns=X_sc.columns)

    results = []
    for rnd in range(FS_N_ROUNDS):
        k = min(FS_SUBSET_SIZE, len(active))
        if k < 2 or len(active) <= k:
            print(f"Round {rnd}: stop, {len(active)} features left (k={k})")
            break

        results = []
        for _ in range(FS_N_TRIALS):
            chosen = rng.choice(active, size=k, replace=False)
            model = Ridge(solver="svd")
            model.fit(X_tr[chosen], y_tr)
            score = r2_score(y_sc, model.predict(X_sc[chosen]))
            results.append((score, chosen))

        results.sort(key=lambda t: t[0], reverse=True)
        n_band = max(1, int(FS_N_TRIALS * FS_TOP_FRAC))
        # R2-weighted ±points this round. Untested stay (0). Only net-negative drop.
        delta = {c: 0.0 for c in active}
        for r2, chosen in results[:n_band]:
            w = r2 * 100
            for c in chosen:
                delta[c] += w
        for r2, chosen in results[-n_band:]:
            w = r2 * 100
            for c in chosen:
                delta[c] -= w
        for c, w in delta.items():
            lifetime[c] += w

        next_active = [c for c in active if delta[c] >= 0]
        if len(next_active) < k:
            ranked = sorted(active, key=lambda c: delta[c], reverse=True)
            next_active = ranked[: max(k, len(active) // 2)]
        r2s = np.array([s for s, _ in results])
        print(
            f"Round {rnd}: {len(active)}→{len(next_active)}  "
            f"best {r2s.max():.3f}  mean {r2s.mean():.3f}  worst {r2s.min():.3f}"
        )
        active = next_active

        for c in track:
            history[c].append(lifetime[c])

    rows = []
    n_hist = len(history[track[0]])
    for c in track:
        row = {"feature": c}
        for i in range(n_hist):
            row[f"r{i}"] = round(history[c][i], 1)
        row["alive"] = c in active
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Selected {len(active)} features after {FS_N_ROUNDS} rounds")
    return active
