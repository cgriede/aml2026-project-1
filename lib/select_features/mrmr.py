import numpy as np
import pandas as pd

from GLOBAL import FS_K


def select(X, y, k: int | None = None):
    y = pd.Series(np.ravel(y), index=X.index)
    relevance = X.corrwith(y).abs().fillna(0)
    corr = X.corr().abs().fillna(0)
    k = min(FS_K if k is None else k, X.shape[1])
    remaining = set(X.columns)
    keep = []
    while len(keep) < k and remaining:
        if not keep:
            pick = relevance.loc[list(remaining)].idxmax()
        else:
            red = corr.loc[list(remaining), keep].mean(axis=1)
            score = relevance.loc[list(remaining)] - red
            pick = score.idxmax()
        keep.append(pick)
        remaining.remove(pick)
    kept = relevance[keep]
    print(
        f"mrmr select: top {k}/{X.shape[1]}  "
        f"|r|[min,med,max]={kept.min():.4f},{kept.median():.4f},{kept.max():.4f}"
    )
    return keep
