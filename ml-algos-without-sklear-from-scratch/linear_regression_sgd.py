#!/usr/bin/env python3
import argparse

import math
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size") # number of examples we use to estimate the gradient
parser.add_argument("--data_size", default=100, type=int, help="Data size") # how many n data points we have
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
# how long does the training will be, number of passes through the data, in each pass you will have this
# batch sizes (b's) and do as many updates as the the number of training set / batch size
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data
    ones = np.ones([args.data_size])
    data = np.column_stack((data, ones))

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target\
        = sklearn.model_selection.train_test_split(data,target,test_size=args.test_size,random_state=args.seed)


    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # A gradient for example `(x_i, t_i)` is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is
        #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        permutation = generator.permutation(train_data.shape[0])
        X_shuffled = train_data[permutation]
        y_shuffled = train_target[permutation]
        for i in range(0, train_data.shape[0], args.batch_size):
            xi = X_shuffled[i:i + args.batch_size]
            yi = y_shuffled[i:i + args.batch_size]

            #(x_i^T weights - t_i) * x_i`,
            gradients = (1 / args.batch_size * xi.T.dot(xi.dot(weights) - yi))
            weights =  weights - args.learning_rate * (gradients + args.l2 * weights)

        targetVals = test_data @ weights

        # TODO: Manually compute root mean square error on the test set predictions.
        rmse = np.sqrt(np.square(np.subtract(test_target, targetVals)).mean())

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

        train_rmses.append(rmse)
        test_rmses.append(rmse)

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    model = LinearRegression()
    model.fit(train_data, train_target)
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    prediction_on_test_data = model.predict(test_data)

    explicit_rmse =  np.sqrt(np.square(np.subtract(test_target,prediction_on_test_data)).mean())
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.2f}".format(weight) for weight in weights[:12]), "...")