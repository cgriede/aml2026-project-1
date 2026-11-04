import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from GLOBAL import FS_K

# Neighbors used for each hit/miss update.
FS_RRELIEFF_NEIGHBORS = 10


def select(X, y, k: int | None = None, n_neighbors: int = FS_RRELIEFF_NEIGHBORS):
    y = np.ravel(y).astype(float)
    values = X.to_numpy(dtype=float)
    n, p = values.shape
    n_neighbors = min(n_neighbors, n - 1)
    if n_neighbors < 1:
        return list(X.columns[: min(FS_K if k is None else k, p)])

    y_range = float(np.ptp(y))
    if y_range <= 0:
        y_range = 1.0
    feat_range = np.ptp(values, axis=0)
    feat_range[feat_range <= 0] = 1.0

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(values)
    _, idx = nn.kneighbors(values)
    neighbors = idx[:, 1:]

    # Robnik-Sikonja & Kononenko RReliefF for regression.
    m = float(n * n_neighbors)
    ndc = 0.0
    nda = np.zeros(p)
    ndcda = np.zeros(p)
    for i in range(n):
        diff_y = np.abs(y[i] - y[neighbors[i]]) / y_range
        diff_x = np.abs(values[i] - values[neighbors[i]]) / feat_range
        ndc += diff_y.sum()
        nda += diff_x.sum(axis=0)
        ndcda += (diff_y[:, None] * diff_x).sum(axis=0)

    if ndc <= 0 or m <= ndc:
        weights = ndcda
    else:
        weights = ndcda / ndc - (nda - ndcda) / (m - ndc)
    scores = pd.Series(weights, index=X.columns)
    k = min(FS_K if k is None else k, p)
    keep = scores.nlargest(k).index.tolist()
    kept = scores[keep]
    print(
        f"rrelieff select: top {k}/{p}  neighbors={n_neighbors}  "
        f"w[min,med,max]={kept.min():.4f},{kept.median():.4f},{kept.max():.4f}"
    )
    return keep
