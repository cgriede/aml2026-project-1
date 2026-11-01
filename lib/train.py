from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from GLOBAL import RANDOM_STATE


class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_baseline(self):
        """Use a simple Ridge model to get a baseline score"""
        model = Ridge(alpha=1.0, solver="svd")
        model.fit(self.X, self.y)
        return model

    def train_xgboost(self, X_val=None, y_val=None, n_estimators=100):
        """XGB. Pass a fold as eval_set to early-stop; omit it to fit a locked tree count."""
        kwargs = dict(
            n_estimators=n_estimators,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
        if X_val is not None:
            kwargs["early_stopping_rounds"] = 20
            model = XGBRegressor(**kwargs)
            model.fit(self.X, self.y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model = XGBRegressor(**kwargs)
            model.fit(self.X, self.y)
        return model
