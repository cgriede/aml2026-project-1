import pandas as pd

from GLOBAL import DATA_DIR


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="id")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="id")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="id")
    return X_train, y_train, X_test
