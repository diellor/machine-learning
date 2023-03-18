#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(dataset.data, dataset.target,
                                                                                test_size=args.test_size,
                                                                                random_state=args.seed)
    miniMax = sklearn.preprocessing.MinMaxScaler()
    polyFeautres = sklearn.preprocessing.PolynomialFeatures()
    model = sklearn.linear_model.LogisticRegression(random_state=args.seed)
    # TODO: Create a pipeline, which
    pipe = sklearn.pipeline.Pipeline(steps=[('scaler', miniMax), ('poly', polyFeautres), ('logistic_regression', model)])

    parameters = {
        'poly__degree': [1, 2],
        'logistic_regression__C': [0.01, 1, 100],
        'logistic_regression__solver': ['lbfgs', 'sag']
    }
    gs = sklearn.model_selection.GridSearchCV(estimator=pipe, param_grid=parameters, cv=sklearn.model_selection.StratifiedKFold(n_splits=5), scoring='accuracy')
    gs.fit(X_train, Y_train)

    test_accuracy = gs.score(X_test, Y_test)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))