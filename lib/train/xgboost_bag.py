import numpy as np

from GLOBAL import XGB_SEEDS


class MeanRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def score(self, X, y):
        from sklearn.metrics import r2_score

        return r2_score(np.ravel(y), self.predict(X))

    def get_params(self):
        return {"n_models": len(self.models), **self.models[0].get_params()}


def fit(X, y, X_val=None, y_val=None, n_estimators=2000):
    from .xgboost import fit as fit_xgb

    models = [
        fit_xgb(X, y, X_val, y_val, n_estimators=n_estimators, random_state=s)
        for s in XGB_SEEDS
    ]
    return MeanRegressor(models)
