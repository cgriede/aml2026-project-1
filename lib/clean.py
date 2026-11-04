import pandas as pd

from GLOBAL import CLEAN_ABSMAX, DROP_INSANE_ABSMAX


def _dead_columns(X: pd.DataFrame) -> list[str]:
    all_nan = X.isna().all()
    all_zero = X.fillna(0).eq(0).all()
    return X.columns[all_nan | all_zero].tolist()


def _insane_columns(X: pd.DataFrame) -> list[str]:
    return X.columns[X.abs().max() > CLEAN_ABSMAX].tolist()


def clean(X: pd.DataFrame, drop_cols: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if drop_cols is None:
        dead = _dead_columns(X)
        insane = _insane_columns(X) if DROP_INSANE_ABSMAX else []
        print(f"Dropping {len(dead)} dead columns: {dead}")
        if DROP_INSANE_ABSMAX:
            print(f"Dropping {len(insane)} insane-absmax (>{CLEAN_ABSMAX:g}) columns: {insane}")
        else:
            print("Keeping insane-absmax columns")
        drop_cols = dead + insane
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.fillna(X.median())
    return X, drop_cols
