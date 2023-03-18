#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.metrics
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.




def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    def gradient(X, Y, W):

        Y_onehot = np.zeros(args.classes)
        Y_onehot[Y] = 1

        Z = X.T @ W
        P = softmax(Z)
        err = (P - Y_onehot)
        gd = err[np.newaxis].T @ X[np.newaxis]
        return gd.T
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for i in range(0, len(permutation), args.batch_size):
            batch = permutation[i:i + args.batch_size]
            grad = 0
            for i in range(args.batch_size):
                x_data = train_data[batch[i]]
                grad += gradient(x_data, train_target[batch[i]], weights)
            grad /= args.batch_size
            weights -= args.learning_rate * grad


        train_pred = np.array([softmax(np.dot(x.T, weights)) for x in train_data])
        test_pred = np.array([softmax(np.dot(x.T, weights)) for x in test_data])

        train_accuracy = accuracy_score(train_target , np.argmax(train_pred, axis=1))
        test_accuracy = accuracy_score(test_target , np.argmax(test_pred, axis=1))

        train_pred = np.array([softmax(np.dot(x.T, weights)) for x in train_data])
        test_pred = np.array([softmax(np.dot(x.T, weights)) for x in test_data])
        train_loss = sklearn.metrics.log_loss(train_target, train_pred)
        test_loss = sklearn.metrics.log_loss(test_target, test_pred)

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")