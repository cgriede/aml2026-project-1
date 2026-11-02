import numpy as np
import pandas as pd


def drop_high_corr(X: pd.DataFrame, y, tau: float) -> list[str]:
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
