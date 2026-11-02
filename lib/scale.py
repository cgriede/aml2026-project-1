import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale(X, y, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        arr = scaler.fit_transform(X)
    else:
        arr = scaler.transform(X)
    X = pd.DataFrame(arr, index=X.index, columns=X.columns)
    return X, y, scaler
