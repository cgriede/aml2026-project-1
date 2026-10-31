import numpy as np

def clean(X: np.ndarray) -> np.ndarray:
    imputation_function = np.median
    X.fillna(imputation_function(X), inplace=True)

    return X