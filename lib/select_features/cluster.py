import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Features with |r| above about 0.7 land in the same cluster.
FS_CLUSTER_DIST = 0.3


def select(X, y, k: int | None = None, threshold: float = FS_CLUSTER_DIST):
    y = pd.Series(np.ravel(y), index=X.index)
    ycorr = X.corrwith(y).abs().fillna(0)
    corr = X.corr().abs().fillna(0)
    dist = squareform(np.clip(1.0 - corr.to_numpy(), 0, None), checks=False)
    labels = fcluster(linkage(dist, method="average"), t=threshold, criterion="distance")
    keep = []
    for lab in np.unique(labels):
        members = X.columns[labels == lab]
        keep.append(ycorr.loc[members].idxmax())
    if k is not None:
        keep = ycorr.loc[keep].nlargest(min(k, len(keep))).index.tolist()
    print(
        f"cluster select: {len(keep)}/{X.shape[1]}  "
        f"cut={threshold}  "
        f"|r|[min,med,max]={ycorr[keep].min():.4f},{ycorr[keep].median():.4f},{ycorr[keep].max():.4f}"
    )
    return keep
