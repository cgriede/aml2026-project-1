from sklearn.linear_model import Ridge


def fit(X, y):
    """Use a simple Ridge model to get a baseline score"""
    model = Ridge(alpha=1.0, solver="svd")
    model.fit(X, y)
    return model
