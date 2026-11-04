import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from GLOBAL import RANDOM_STATE


def _isoforest_outliers(X_train: pd.DataFrame) -> set[int]:
    outlier_model = IsolationForest(contamination=0.01, random_state=RANDOM_STATE)
    print(f"contamination: {outlier_model.contamination}")
    outlier_model.fit(X_train)
    outlier_preds = outlier_model.predict(X_train)
    return set(np.where(outlier_preds == -1)[0])

def _svm_outliers(X_train: pd.DataFrame) -> set[int]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    svm_model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    print(f"nu: {svm_model.nu}, kernel: {svm_model.kernel}, gamma: {svm_model.gamma}")
    svm_model.fit(X_scaled)
    svm_preds = svm_model.predict(X_scaled)
    return set(np.where(svm_preds == -1)[0])

METHODS = {
    "isoforest": _isoforest_outliers,
    "svm": _svm_outliers,
}


def remove_outliers(X_train: pd.DataFrame, y_train: pd.DataFrame, methods: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if methods is None:
        methods = list(METHODS)
    if len(methods) == 0:
        return X_train, y_train
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown outlier method {unknown}. Known: {list(METHODS)}")
    seen: set[int] = set()
    for name in methods:
        found = METHODS[name](X_train)
        new = found - seen
        seen |= found
        print(f"{name}: {len(found)} flagged, {len(new)} new")
    print(f"dropped {len(seen)} rows")
    drop = X_train.index[sorted(seen)]
    return X_train.drop(index=drop), y_train.drop(index=drop)
