#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=40, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--range", default=3, type=int, help="Feature order range")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.
# MITCHEL 7fee082a-251c-11ec-986f-f39926f24a9c


def main(args: argparse.Namespace) -> list[float]:
    # Create the data
    xs = np.linspace(0, 7, num=args.data_size)
    ys = np.sin(xs) + np.random.RandomState(args.seed).normal(0, 0.2, size=args.data_size)

    rmses = []
    polyData = []
    copyxs = xs.copy()
    # xs = np.asmatrix(xs).T
    # t = np.asmatrix(t).T
    xs = np.array(xs).reshape(-1,1)
    copyxs = np.array(copyxs).reshape(-1,1)
    for order in range(1, args.range + 1):
        # TODO: Create features `(x^1, x^2, ..., x^order)`, preferably in this ordering.
        if order >= 2:
            xs = np.append(xs, copyxs ** order, axis=1)

        # TODO: Split the data into a train set and a test set.
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(xs,ys.reshape(-1, 1),
                                                                                    test_size=args.test_size,
                                                                                    random_state=args.seed)
        # TODO: Fit a linear regression model using `sklearn.linear_model.LinearRegression`;
        # consult the documentation and see especially the `fit` method.
        model = LinearRegression()
        model.fit(X_train, Y_train)

        # TODO: Predict targets on the test set using the `predict` method of the trained model.
        prediction_on_test_data = model.predict(X_test)
        # TODO: Compute root mean square error on the test set predictions.
        # You can either do it manually or look at `sklearn.metrics.mean_squared_error` method
        # and its `squared` parameter.
        rmse = np.sqrt(np.square(np.subtract(Y_test,prediction_on_test_data)).mean())

        rmses.append(rmse)

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.plot(train_data[:, 0], train_target, "go")
            plt.plot(test_data[:, 0], test_target, "ro")
            plt.plot(np.linspace(xs[0], xs[-1], num=100),
                     model.predict(np.power.outer(np.linspace(xs[0], xs[-1], num=100), np.arange(1, order + 1))), "b")
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmses = main(args)
    for order, rmse in enumerate(rmses):
        print("Maximum feature order {}: {:.2f} RMSE".format(order + 1, rmse))