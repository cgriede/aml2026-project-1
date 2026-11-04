import numpy as np
from sklearn.feature_selection import mutual_info_regression

from GLOBAL import RANDOM_STATE


def select(X, y, k: int | None = None):
    y = np.ravel(y)
    mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    scores = dict(zip(X.columns, mi))
    if k is None:
        cutoff = float(np.median(mi))
        keep = [c for c, v in scores.items() if v >= cutoff]
        print(
            f"MI select: {len(keep)}/{X.shape[1]}  "
            f"cutoff=median={cutoff:.4f}  "
            f"mi[min,med,max]={mi.min():.4f},{np.median(mi):.4f},{mi.max():.4f}"
        )
        return keep
    k = min(k, X.shape[1])
    order = sorted(scores, key=scores.get, reverse=True)[:k]
    kept = np.array([scores[c] for c in order])
    print(
        f"MI select: top {k}/{X.shape[1]}  "
        f"mi[min,med,max]={kept.min():.4f},{np.median(kept):.4f},{kept.max():.4f}"
    )
    return order
