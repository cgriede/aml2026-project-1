import numpy as np
import pandas as pd

from GLOBAL import FS_K


def select(X, y, k: int | None = None):
    y = pd.Series(np.ravel(y), index=X.index)
    scores = X.corrwith(y).abs()
    k = min(FS_K if k is None else k, X.shape[1])
    keep = scores.nlargest(k).index.tolist()
    kept = scores[keep]
    print(
        f"corr select: top {k}/{X.shape[1]}  "
        f"|r|[min,med,max]={kept.min():.4f},{kept.median():.4f},{kept.max():.4f}"
    )
    return keep
