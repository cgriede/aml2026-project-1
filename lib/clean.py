import pandas as pd


def _dead_columns(X: pd.DataFrame) -> list[str]:
    all_nan = X.isna().all()
    all_zero = X.fillna(0).eq(0).all()
    return X.columns[all_nan | all_zero].tolist()


def clean(X: pd.DataFrame, dead_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if dead_cols is None:
        dead_cols = _dead_columns(X)
        print(f"Dropping {len(dead_cols)} dead columns: {dead_cols}")
    X = X.drop(columns=dead_cols, errors="ignore")
    X = X.fillna(X.median())
    return X, dead_cols
