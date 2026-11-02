import numpy as np
from sklearn.feature_selection import mutual_info_regression

from GLOBAL import RANDOM_STATE


def select(X, y):
    """Keep columns with MI(y) at or above the median. No k=100 cap."""
    y = np.ravel(y)
    mi = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    cutoff = float(np.median(mi))
    keep = [c for c, v in zip(X.columns, mi) if v >= cutoff]
    print(
        f"MI select: {len(keep)}/{X.shape[1]}  "
        f"cutoff=median={cutoff:.4f}  "
        f"mi[min,med,max]={mi.min():.4f},{np.median(mi):.4f},{mi.max():.4f}"
    )
    return keep
