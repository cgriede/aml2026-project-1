from clean import clean
from evaluate import evaluate
from load import load
from predict import predict
from remove_outliers import remove_outliers
from scale import scale
from select_features import select_features
from train import train

from sklearn.model_selection import train_test_split

from GLOBAL import *

def main():
    X_train, y_train, X_test = load()

    X_train = clean(X_train)
    X_test = clean(X_test)

    X_train, y_train, X_val, y_val = train_test_split(
    X_train, y_train,
    test_size=VAL_RATIO,
    random_state=RANDOM_STATE,
    )

    X_train, y_train = remove_outliers(X_train, y_train)

    X_train, y_train, scaler = scale(X_train, y_train)
    X_val, y_val, _ = scale(X_val, y_val, scaler)
    X_test, _, _ = scale(X_test, None, scaler)

    X_train, y_train, selected_cols = select_features(X_train, y_train)
    X_val, y_val, _ = select_features(X_val, y_val, selected_cols)
    X_test, _, _ = select_features(X_test, None, selected_cols)

    model = train(X_train, y_train)
    evaluate(model, X_train, y_train)
    evaluate(model, X_val, y_val)
    predict(model, X_test)


if __name__ == "__main__":
    main()
