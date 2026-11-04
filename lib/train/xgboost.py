from xgboost import XGBRegressor

from GLOBAL import (
    RANDOM_STATE,
    XGB_COLSAMPLE_BYTREE,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_N_ESTIMATORS,
    XGB_SUBSAMPLE,
)


def fit(X, y, X_val=None, y_val=None, n_estimators=XGB_N_ESTIMATORS, random_state=RANDOM_STATE):
    """XGB. Pass a fold as X_val to early-stop; omit it to fit a locked tree count."""
    kwargs = dict(
        n_estimators=n_estimators,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        random_state=random_state,
    )
    if X_val is not None:
        kwargs["early_stopping_rounds"] = XGB_EARLY_STOPPING_ROUNDS
        model = XGBRegressor(**kwargs)
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = XGBRegressor(**kwargs)
        model.fit(X, y)
    return model
