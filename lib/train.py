import numpy as np
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from GLOBAL import RANDOM_STATE, XGB_SEEDS


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


class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_baseline(self):
        """Use a simple Ridge model to get a baseline score"""
        model = Ridge(alpha=1.0, solver="svd")
        model.fit(self.X, self.y)
        return model

    def train_xgboost(self, X_val=None, y_val=None, n_estimators=2000, random_state=RANDOM_STATE):
        """XGB. Pass a fold as eval_set to early-stop; omit it to fit a locked tree count."""
        kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=7,
            random_state=random_state,
        )
        if X_val is not None:
            kwargs["early_stopping_rounds"] = 50
            model = XGBRegressor(**kwargs)
            model.fit(self.X, self.y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = XGBRegressor(**kwargs)
            model.fit(self.X, self.y)
        return model

    def train_xgboost_bag(self, X_val=None, y_val=None, n_estimators=2000):
        models = [
            self.train_xgboost(X_val, y_val, n_estimators=n_estimators, random_state=s)
            for s in XGB_SEEDS
        ]
        return MeanRegressor(models)
