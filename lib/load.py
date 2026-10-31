import pandas as pd

from GLOBAL import DATA_DIR

def load():
    X_test = DATA_DIR / "X_test.csv"
    y_train = DATA_DIR / "y_train.csv"
    X_train = DATA_DIR / "X_train.csv"

    X_test = pd.read_csv(X_test, index_col="id")
    y_train = pd.read_csv(y_train, index_col="id")
    X_train = pd.read_csv(X_train, index_col="id")

    return X_train, y_train, X_test
