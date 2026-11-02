import numpy as np
from sklearn.svm import SVR


def fit(X, y):
    """SVR defaults from exp 001: RBF, C=1, epsilon=0.1."""
    model = SVR()
    model.fit(X, np.ravel(y))
    return model
