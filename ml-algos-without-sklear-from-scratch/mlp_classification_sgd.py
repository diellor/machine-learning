#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    def acc(preds, tars):
        return (1 / len(tars)) * np.sum([tars[j] == np.argmax(preds, axis=1)[j] for j in range(len(tars))])

    def relu(Z):
        return np.maximum(0,Z)

    def dRelu(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_encode = onehot_encoder.fit_transform(target.reshape(-1, 1))

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    t_d, te_d, tr_t, te_t = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, y_encode, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden

        z1 = np.dot(inputs, weights[0]) + biases[0]
        a1 = relu(z1)
        z2 = a1.dot(weights[1]) + biases[1]
        o1 = softmax(z2)
        return a1,o1

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for x in range(0, len(permutation), args.batch_size):
            batch = permutation[x:x + args.batch_size]
            gradw1 = 0
            gradw2 = 0
            b_2 = 0
            b_1 = 0
            for i in range(args.batch_size):
                x_data = train_data[batch[i]]
                y_target = train_target[batch[i]]

                a1, o1, = forward(x_data)
                d = (o1 - y_target)

                b_2+=d
                gradw2 += np.dot(a1.reshape(-1, 1), d.reshape(-1, 1).T)

                d2 = (d @ weights[1].T) * dRelu(np.dot(x_data, weights[0]) + biases[0])
                b_1 +=d2

                gradw1 += np.dot(x_data.reshape(-1, 1), d2.reshape(-1, 1).T)

            weights[0] = weights[0] - (gradw1 / args.batch_size) * args.learning_rate
            biases[0] = biases[0] - (b_1 / args.batch_size) * args.learning_rate
            weights[1] = weights[1] - (gradw2 / args.batch_size) * args.learning_rate
            biases[1] = biases[1] - (b_2 / args.batch_size) * args.learning_rate


        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
        # and gradient of `biases[0]` and `biases[1]`.
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of $-log P(target | data)$, or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer,
        # - compute the derivative with respect to `weights[1]` and `biases[1]`,
        # - compute the derivative with respect to the hidden layer output,
        # - compute the derivative with respect to the hidden layer input,
        # - compute the derivative with respect to `weights[0]` and `biases[0]`.

        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.

        z1 = np.dot(t_d, weights[0]) + biases[0]
        a1 = relu(z1)
        z2 = a1.dot(weights[1]) + biases[1]
        o1 = softmax(z2)

        train_accuracy = acc(o1, tr_t)

        z1 = np.dot(te_d, weights[0]) + biases[0]
        a1 = relu(z1)
        z2 = a1.dot(weights[1]) + biases[1]
        o1 = softmax(z2)

        test_accuracy = acc(o1, te_t)


        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")