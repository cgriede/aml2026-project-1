from clean import clean
from evaluate import Evaluator
from load import load
from predict import DataOrchestrator
from remove_outliers import remove_outliers
from scale import scale
from select_features import select_features
from train import Trainer
from runlog import append_run

import numpy as np
from sklearn.model_selection import KFold

from GLOBAL import *


def _cv_scores(X, y):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    ridge_scores, xgb_scores, xgb_best = [], [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        trainer = Trainer(X_tr, y_tr)
        ridge = trainer.train_baseline()
        xgb = trainer.train_xgboost(X_te, y_te)
        ridge_scores.append(Evaluator(ridge, X_te, y_te).evaluate())
        xgb_scores.append(Evaluator(xgb, X_te, y_te).evaluate())
        xgb_best.append(int(xgb.best_iteration))
        print(
            f"Fold {fold}: Ridge {ridge_scores[-1]:.4f}  "
            f"XGB {xgb_scores[-1]:.4f}  best_iteration {xgb_best[-1]}"
        )
    return np.array(ridge_scores), np.array(xgb_scores), np.array(xgb_best)


def main():
    X_train, y_train, X_test = load()

    X_train, drop_cols = clean(X_train)
    X_test, _ = clean(X_test, drop_cols)

    if DROP_OUTLIERS:
        X_train, y_train = remove_outliers(X_train, y_train)

    X_train, y_train, scaler = scale(X_train, y_train)
    X_test, _, _ = scale(X_test, None, scaler)

    X_train, y_train, selected_cols = select_features(X_train, y_train)
    X_test, _, _ = select_features(X_test, None, selected_cols)

    print(f"shape after selection ({FS_METHOD}): train {X_train.shape}, test {X_test.shape}")

    ridge_scores, xgb_scores, xgb_best = _cv_scores(X_train, y_train)
    n_trees = int(np.median(xgb_best)) + 1
    print(
        f"Ridge CV: {ridge_scores.mean():.4f} ± {ridge_scores.std():.4f}  {np.round(ridge_scores, 4)}"
    )
    print(
        f"XGB  CV: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}  {np.round(xgb_scores, 4)}  "
        f"best_iteration {xgb_best}  n_estimators={n_trees}"
    )

    trainer = Trainer(X_train, y_train)
    baseline_model = trainer.train_baseline()
    xgb = trainer.train_xgboost_bag(n_estimators=n_trees)
    ridge_train_r2 = float(Evaluator(baseline_model, X_train, y_train).evaluate())
    xgb_train_r2 = float(Evaluator(xgb, X_train, y_train).evaluate())
    print(f"Ridge full-fit train R2: {ridge_train_r2:.4f}")
    print(f"XGB  full-fit train R2: {xgb_train_r2:.4f}")

    data_orchestrator = DataOrchestrator(X_test, xgb)
    data_orchestrator.predict()
    data_orchestrator.store_predictions()
    kaggle_public = data_orchestrator.submit_predictions() if KAGGLE_SUBMIT else None

    append_run(
        {
            "n_train": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "ridge_cv_mean": float(ridge_scores.mean()),
            "ridge_cv_std": float(ridge_scores.std()),
            "xgb_cv_mean": float(xgb_scores.mean()),
            "xgb_cv_std": float(xgb_scores.std()),
            "ridge_cv_folds": [float(s) for s in ridge_scores],
            "xgb_cv_folds": [float(s) for s in xgb_scores],
            "xgb_best_iteration": [int(b) for b in xgb_best],
            "n_trees": n_trees,
            "ridge_train_r2": ridge_train_r2,
            "xgb_train_r2": xgb_train_r2,
            "kaggle_public": kaggle_public,
        },
        models=[baseline_model, xgb],
    )


if __name__ == "__main__":
    main()
