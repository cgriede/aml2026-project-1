from clean import clean
from evaluate import Evaluator
from load import load
from predict import DataOrchestrator
from remove_outliers import remove_outliers
from scale import scale
from select_features import select_features
from train import Trainer

from sklearn.model_selection import train_test_split
import pandas as pd

from GLOBAL import *


def main():
    X_train, y_train, X_test = load()

    X_train, dead_cols = clean(X_train)
    X_test, _ = clean(X_test, dead_cols)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE,
    )

    X_train, y_train = remove_outliers(X_train, y_train)

    X_train, y_train = scale(X_train, y_train)
    X_val, y_val = scale(X_val, y_val)
    X_test, _ = scale(X_test, None)

    X_train, y_train, selected_cols = select_features(X_train, y_train)
    X_val, y_val, _ = select_features(X_val, y_val, selected_cols)
    X_test, _, _ = select_features(X_test, None, selected_cols)

    print(f"""shape after selection:\n
    train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}""")

    trainer = Trainer(X_train, y_train)
    baseline_model = trainer.train_baseline()
    #check the internal results on the validation and training set
    train_score = Evaluator(baseline_model, X_train, y_train).evaluate()
    val_score = Evaluator(baseline_model, X_val, y_val).evaluate()
    print(f"Train Score:      {train_score:.4f}")
    print(f"Validation score: {val_score:.4f}")

    #train on the full dataset for submission
    X_train_f = pd.concat([X_train, X_val])
    y_train_f = pd.concat([y_train, y_val])
    trainer = Trainer(X_train_f, y_train_f)
    baseline_model = trainer.train_baseline()
    #check the internal results on the full dataset
    full_score = Evaluator(baseline_model, X_train_f, y_train_f).evaluate()
    print(f"Full Score:        {full_score:.4f}")

    #store the predictions on test setand submit them
    data_orchestrator = DataOrchestrator(X_test, baseline_model)
    y_pred = data_orchestrator.predict()
    data_orchestrator.store_predictions()
    
    #NOTE max 10 predictions per day
    #data_orchestrator.submit_predictions()


if __name__ == "__main__":
    main()
