from xgboost import XGBRegressor

from GLOBAL import RANDOM_STATE


def fit(X, y, X_val=None, y_val=None, n_estimators=2000, random_state=RANDOM_STATE):
    """XGB. Pass a fold as X_val to early-stop; omit it to fit a locked tree count."""
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
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = XGBRegressor(**kwargs)
        model.fit(X, y)
    return model
