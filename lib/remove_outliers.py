import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from GLOBAL import RANDOM_STATE


def remove_outliers(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Isolation Forest, contamination guessed at 5%
    outlier_model = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    outlier_model.fit(X_train)
    outlier_preds = outlier_model.predict(X_train)
    isoforest_outlier_indices = set(np.where(outlier_preds == -1)[0])

    print(f"Number of outliers detected: {len(isoforest_outlier_indices)}")

    # OneClassSVM on a local scale (pipeline scale() runs later)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    svm_model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    svm_model.fit(X_scaled)
    svm_preds = svm_model.predict(X_scaled)

    svm_outlier_indices = set(np.where(svm_preds == -1)[0])

    print(f"Number of outliers detected by SVM: {np.sum(svm_preds == -1)}")

    common_outliers = isoforest_outlier_indices & svm_outlier_indices
    print(f"Number of common outliers detected by both methods: {len(common_outliers)}")

    drop_idx = X_train.index[list(common_outliers)]
    X_train = X_train.drop(index=drop_idx)
    y_train = y_train.drop(index=drop_idx)
    print(f"Cleaned X_train shape: {X_train.shape}, Cleaned y_train shape: {y_train.shape}")

    return X_train, y_train
