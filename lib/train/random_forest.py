from sklearn.ensemble import RandomForestRegressor

from GLOBAL import RANDOM_STATE


def fit(X, y, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model
