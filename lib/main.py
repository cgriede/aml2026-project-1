from clean import clean
from evaluate import Evaluator
from load import load
from predict import DataOrchestrator
from remove_outliers import remove_outliers
from scale import scale
from select_features import select_features
from train import Trainer

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

    X_train, dead_cols = clean(X_train)
    X_test, _ = clean(X_test, dead_cols)

    X_train, y_train = remove_outliers(X_train, y_train)

    X_train, y_train = scale(X_train, y_train)
    X_test, _ = scale(X_test, None)

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
    xgb = trainer.train_xgboost(n_estimators=n_trees)
    print(f"Ridge full-fit train R2: {Evaluator(baseline_model, X_train, y_train).evaluate():.4f}")
    print(f"XGB  full-fit train R2: {Evaluator(xgb, X_train, y_train).evaluate():.4f}")

    #NOTE skip this for now first find useful training method
    #return None
    #store the predictions on test setand submit them
    data_orchestrator = DataOrchestrator(X_test, xgb)
    y_pred = data_orchestrator.predict()
    data_orchestrator.store_predictions()

    #NOTE max 10 predictions per day
    #data_orchestrator.submit_predictions()


if __name__ == "__main__":
    main()
