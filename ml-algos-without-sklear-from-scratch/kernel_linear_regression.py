#!/usr/bin/env python3
import argparse
import numpy as np
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--epochs", default=200, type=int, help="Number of SGD training epochs")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[np.ndarray, float, list[float], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = (np.linspace(-1.2, 1.2, 2 * args.data_size))
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size)
    bias = 0

    train_rmses, test_rmses = [], []

    def kernel_rb(x, y):
        return  np.exp(-args.kernel_gamma * np.power(np.linalg.norm(x - y), 2))

    def kernel_p(x, y):
        return (np.dot(args.kernel_gamma , np.dot(x , y)) + 1) ** args.kernel_degree

    kernel_train = np.zeros((train_data.shape[0], train_data.shape[0]))

    if args.kernel == "poly":
        for i, row in enumerate(kernel_train):
            for j, col in enumerate(kernel_train.T):
                kernel_train[i, j] = kernel_p(train_data[i], train_data[j])

    else:
        for i, row in enumerate(kernel_train):
            for j, col in enumerate(kernel_train.T):
                kernel_train[i, j] = kernel_rb(train_data[i], train_data[j])


    kernel_test = np.zeros((test_data.shape[0], train_data.shape[0]))

    if args.kernel == "poly":
        for i, row in enumerate(kernel_test):
            for j, col in enumerate(kernel_test.T):
                kernel_test[i, j] = kernel_p(test_data[i], train_data[j])

    else:
        for i, row in enumerate(kernel_test):
            for j, col in enumerate(kernel_test.T):
                kernel_test[i, j] = kernel_rb(test_data[i], train_data[j])

    kernel_train = kernel_train.reshape(50,50)

    constant = args.learning_rate / args.batch_size
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for x in range(0, len(permutation), args.batch_size):
            batch = permutation[x:x + args.batch_size]
            multiplication = np.dot(kernel_train[batch],betas)
            gradient = multiplication + bias - train_target[batch]
            betas[batch] = betas[batch] - constant * gradient
            betas = betas - constant * args.l2 * betas
            bias = bias - constant * np.sum(gradient)

        y_pred = np.dot(kernel_train,betas) + bias
        rmse_y = np.sqrt(np.square(np.subtract(train_target,y_pred)).mean())

        test_y_pred = np.dot(kernel_test, betas) + bias
        rmse_y_test = np.sqrt(np.square(np.subtract(test_target, test_y_pred)).mean())

        train_rmses.append(rmse_y)
        test_rmses.append(rmse_y_test)

        if (epoch + 1) % 10 == 0:
            print("After epoch: {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                epoch + 1, train_rmses[-1], test_rmses[-1]))

    if args.plot:
        import matplotlib.pyplot as plt
        test_predictions = test_preds
        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return betas, bias, train_rmses, test_rmses


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    betas, bias, train_rmses, test_rmses = main(args)
    print("Learned betas", *("{:.2f}".format(beta) for beta in betas[:15]), "...")
    print("Learned bias", bias)
