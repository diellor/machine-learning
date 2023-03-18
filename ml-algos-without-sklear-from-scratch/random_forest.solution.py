#!/usr/bin/env python3
import argparse
import subprocess

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")


class DecisionTree:
    class Node:
        def __init__(self, instances, prediction):
            self.is_leaf = True
            self.instances = instances
            self.prediction = prediction

        def split(self, feature, value, left, right):
            self.is_leaf = False
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right

    def __init__(self, subsample_features, max_depth):
        self._subsample_features = subsample_features
        self._max_depth = max_depth

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._create_leaf(np.arange(len(self._data)))
        self._split_recursively(self._root, 0)

        return self

    def predict(self, data):
        results = np.zeros(len(data), dtype=np.int32)
        for i in range(len(data)):
            node = self._root
            while not node.is_leaf:
                node = node.left if data[i][node.feature] <= node.value else node.right
            results[i] = node.prediction

        return results

    def _split_recursively(self, node, depth):
        if not self._can_split(node, depth):
            return

        feature, value, left, right = self._best_split(node)
        node.split(feature, value, self._create_leaf(left), self._create_leaf(right))
        self._split_recursively(node.left, depth + 1)
        self._split_recursively(node.right, depth + 1)

    def _can_split(self, node, depth):
        return (
            (self._max_depth is None or depth < self._max_depth) and
            not np.array_equiv(self._targets[node.instances], node.prediction)
        )

    def _best_split(self, node):
        best_criterion = None
        for feature in np.where(self._subsample_features(self._data.shape[1]))[0]:
            sorted_indices = node.instances[np.argsort(self._data[node.instances, feature])]
            for i in range(len(sorted_indices) - 1):
                if self._data[sorted_indices[i], feature] == self._data[sorted_indices[i + 1], feature]:
                    continue
                value = (self._data[sorted_indices[i], feature] + self._data[sorted_indices[i + 1], feature]) / 2
                left, right = sorted_indices[:i + 1], sorted_indices[i + 1:]
                criterion = self._criterion(left) + self._criterion(right)
                if best_criterion is None or criterion < best_criterion:
                    best_criterion, best_feature, best_value, best_left, best_right = \
                        criterion, feature, value, left, right

        return best_feature, best_value, best_left, best_right

    def _criterion(self, instances):
        # We use the entropy criterion.
        bins = np.bincount(self._targets[instances])
        bins = bins[np.nonzero(bins)]
        return -np.sum(bins * np.log(bins / len(instances)))

    def _create_leaf(self, instances):
        # Create a new leaf, together with its prediction (the most frequent class).
        return self.Node(instances, np.argmax(np.bincount(self._targets[instances])))


class RandomForest:
    def __init__(self, bootstrap_dataset, subsample_features, trees, max_depth):
        self._bootstrap_dataset = bootstrap_dataset
        self._subsample_features = subsample_features
        self._num_trees = trees
        self._max_depth = max_depth

    def fit(self, data, targets):
        self._classes = np.max(targets) + 1
        self._trees = []
        for i in range(self._num_trees):
            data_indices = self._bootstrap_dataset(data)
            self._trees.append(
                DecisionTree(self._subsample_features, self._max_depth).fit(
                    data[data_indices], targets[data_indices]
                )
            )

    def predict(self, data):
        results = np.zeros((len(data), self._classes), np.int32)
        for tree in self._trees:
            for index, prediction in enumerate(tree.predict(data)):
                results[index, prediction] += 1
        return np.argmax(results, axis=1)


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        if not args.bagging: return np.arange(len(train_data))
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating a feature mask using
    #     subsample_features(number_of_features)
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not
    #   (i.e., when `feature_subsampling == 1`, all features are used).
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.
    random_forest = RandomForest(bootstrap_dataset, subsample_features, args.trees, args.max_depth)
    random_forest.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, random_forest.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, random_forest.predict(test_data))

    if args.plot:
        # Plot the final tree using graphviz.
        classes = np.max(target) + 1
        feature_names = getattr(sklearn.datasets, "load_{}".format(args.dataset))().feature_names
        dot = ["digraph Tree {node [shape=box]; bgcolor=invis;"]
        def plot(index, tree, node, parent):
            if parent is not None: dot.append("{} -> {}".format(parent, index))
            dot.append("{} [fontname=\"serif\"; label=\"{}c_entropy = {:.2f}\\ninstances = {}\\ncounts = [{}]\"];".format(
                index, "f. {} ({}) <= {:.3f}\\n".format(node.feature, feature_names[node.feature], node.value) if not node.is_leaf else "",
                tree._criterion(node.instances), len(node.instances),
                ",".join(map(str, np.bincount(tree._targets[node.instances], minlength=classes)))))
            if not node.is_leaf:
                index = plot(plot(index + 1, tree, node.left, index), tree, node.right, index)
            return index + 1
        index = 0
        for tree in random_forest._trees:
            index = plot(index, tree, tree._root, None)
        dot.append("}")
        subprocess.run(["dot", "-Txlib"] if args.plot is True else ["dot", "-Tsvg", "-o{}".format(args.plot)],
                       input="\n".join(dot), encoding="utf-8")

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
