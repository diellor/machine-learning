#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.model_selection import GridSearchCV


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

def getIntColumns(train_data):
    i = 0
    categoricalColumns = []
    print(train_data)
    while i < train_data.shape[1]:
        column = train_data[:, i]
        if all(item.is_integer() for item in column):
            categoricalColumns.append(i)
        i = i + 1
    return categoricalColumns

class Dataset:
    """Rental Dataset.
    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)
    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        integerColumns = getIntColumns(train.data)

        ones = np.ones([train.data.shape[0]])
        train.data = np.column_stack((train.data, ones))

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.15, random_state=1)


        col_transformer = make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), integerColumns),
            remainder=StandardScaler()
        )

        polyFeatures = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

        pipe = sklearn.pipeline.Pipeline([
            ('col_transformer', col_transformer),
            ('polyFeatures', polyFeatures),
        ])
        pipe.fit(train.data)
        train.data = pipe.transform(train.data)
        X_val = pipe.transform(X_val)

        # generate random weights
        generator = np.random.RandomState(args.seed)
        weights = generator.uniform(size=train.data.shape[1], low=-0.1, high=0.1)

        for epoch in range(100):
            # TODO: Process the data in the order of `permutation`. For every
            # `args.batch_size` of them, average their gradient, and update the weights.
            # A gradient for example `(x_i, t_i)` is `(x_i^T weights - t_i) * x_i`,
            # and the SGD update is
            #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
            # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

            permutation = generator.permutation(train.data.shape[0])
            X_shuffled = train.data[permutation]
            y_shuffled = train.target[permutation]

            for i in range(0, train.data.shape[0], 10):
                xi = X_shuffled[i:i + 10]
                yi = y_shuffled[i:i + 10]

                gradients = (1 / 20 * xi.T.dot(xi.dot(weights) - yi))
                # compute gradients, to know where are the w's currently, then just move by the learning rate.
                weights = weights - 0.07 * (gradients + 0.02  * weights)

        # targetVals = X_val @ weights
        # rmse = np.sqrt(np.square(np.subtract(y_val, targetVals)).mean())

        # print(rmse)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = [weights, pipe]

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        ones = np.ones([test.data.shape[0]])
        test.data = np.column_stack((test.data, ones))

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        test.data = model[1].transform(test.data)
        predictions = test.data @ model[0]

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)