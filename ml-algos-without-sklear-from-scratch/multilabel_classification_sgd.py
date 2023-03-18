#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

#e193d757-a74a-4cc3-9e6a-e6b8cb5422a9
#0119bc9b-9be0-4a86-8c30-95aaa58e9235
def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:


    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))


    def f_1(TP, FP, FN):
        if((TP + FP) != 0):
            precision = TP / (TP + FP)
        else:
            precision = 0
        recall = TP / (TP + FN)
        if precision != 0 and recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        return f1

    def gradient(X, Y, W):
        # Y_onehot = np.zeros(args.classes)
        # Y_onehot[Y] = 1
        Z = X.T @ W
        P = sigmoid(Z)
        err = (P - Y)
        gd = err[np.newaxis].T @ X[np.newaxis]
        return gd.T
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    one_hot = MultiLabelBinarizer()

    target = one_hot.fit_transform(target_list)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        for x in range(0, len(permutation), args.batch_size):
            batch = permutation[x:x + args.batch_size]
            grad = 0
            for i in range(args.batch_size):
                # This goes for each row in a batch
                x_data = train_data[batch[i]]
                grad += gradient(x_data, train_target[batch[i]], weights)
            grad /= args.batch_size
            weights -= args.learning_rate * grad

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.

        train_pred = np.array([sigmoid(np.dot(x.T, weights)) for x in train_data])
        test_pred = np.array([sigmoid(np.dot(x.T, weights)) for x in test_data])

        train_pred = np.rint(train_pred)
        test_pred = np.rint(test_pred)

        def confusion_matrix(y_true, y_pred):
            tp = np.logical_and(y_pred == 1, y_true == 1).sum()
            tn = np.logical_and(y_pred == 0, y_true == 0).sum()
            fp = np.logical_and(y_pred == 1, y_true == 0).sum()
            fn = np.logical_and(y_pred == 0, y_true == 1).sum()

            return tp, tn, fp, fn

        def multilabel_confusion_matrix_nosum(Y_test, Y_pred):
            tp = np.multiply(Y_test,Y_pred).sum(axis=0)
            fp = Y_pred.sum(axis=0) - tp
            fn = Y_test.sum(axis=0) - tp
            tn = Y_test.shape[0] - tp - fp - fn

            return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

        tp, tn, fp, fn = confusion_matrix(train_target,train_pred)


        confusion_array_train = multilabel_confusion_matrix_nosum(train_target, train_pred)
        result_train = pd.DataFrame(confusion_array_train.reshape(-1, 4),
                                    columns=["true_negative", "false_positive", "false_negative", "true_positive"])


        confusion_array_test = multilabel_confusion_matrix_nosum(test_target, test_pred)
        result_test = pd.DataFrame(confusion_array_test.reshape(-1, 4),
                                   columns=["true_negative", "false_positive", "false_negative", "true_positive"])


        train_f1_micro = f_1(tp,fp,fn)

        tp, tn, fp, fn = confusion_matrix(test_target,test_pred)
        test_f1_micro = f_1(tp,fp,fn)

        macroTrain = 0
        for i in range(train_pred.shape[1]):
            macroTrain += f_1(result_train["true_positive"][i], result_train["false_positive"][i],
                             result_train["false_negative"][i]) / train_pred.shape[1]

        macroTrain = macroTrain
        train_f1_macro = macroTrain

        macroTest = 0
        for i in range(test_pred.shape[1]):
            macroTest += f_1(result_test["true_positive"][i], result_test["false_positive"][i],
                              result_test["false_negative"][i])  / test_pred.shape[1]

        macroTest = macroTest

        test_f1_macro = macroTest

        '''
        MCM = multilabel_confusion_matrix(train_target, y_pred_train_class_ohe, labels=[3],
                                      sample_weight=None)
        tn, fp, fn, tp = MCM.ravel()
        print(tn, fp, fn, tp)
        '''

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")