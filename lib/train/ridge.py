from sklearn.linear_model import Ridge

from GLOBAL import RIDGE_ALPHA


def fit(X, y):
    """Use a simple Ridge model to get a baseline score"""
    model = Ridge(alpha=RIDGE_ALPHA, solver="svd")
    model.fit(X, y)
    return model
