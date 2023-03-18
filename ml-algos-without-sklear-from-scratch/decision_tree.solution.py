#!/usr/bin/env python3
import argparse
import heapq
import subprocess

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
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

    def __init__(self, criterion, max_depth, min_to_split, max_leaves):
        self._criterion = getattr(self, "_criterion_" + criterion)
        self._max_depth = max_depth
        self._min_to_split = min_to_split
        self._max_leaves = max_leaves

    def fit(self, data, targets):
        self._data = data
        self._targets = targets

        self._root = self._leaf(np.arange(len(self._data)))
        if self._max_leaves is None:
            self._split_recursively(self._root, 0)
        else:
            self._split_adaptively()

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

        _, feature, value, left, right = self._best_split(node)
        node.split(feature, value, self._leaf(left), self._leaf(right))
        self._split_recursively(node.left, depth + 1)
        self._split_recursively(node.right, depth + 1)

    def _split_adaptively(self):
        def split_value(node, index, depth):
            best_split = self._best_split(node)
            return (best_split[0], index, depth, node, *best_split[1:])

        heap = [split_value(self._root, 0, 0)]
        for i in range(self._max_leaves - 1):
            _, _, depth, node, feature, value, left, right = heapq.heappop(heap)
            node.split(feature, value, self._leaf(left), self._leaf(right))
            if self._can_split(node.left, depth + 1):
                heapq.heappush(heap, split_value(node.left, 2 * i + 1, depth + 1))
            if self._can_split(node.right, depth + 1):
                heapq.heappush(heap, split_value(node.right, 2 * i + 2, depth + 1))
            if not heap:
                break

    def _can_split(self, node, depth):
        return (
            (self._max_depth is None or depth < self._max_depth) and
            len(node.instances) >= self._min_to_split and
            not np.array_equiv(self._targets[node.instances], node.prediction)
        )

    def _best_split(self, node):
        best_criterion = None
        for feature in range(self._data.shape[1]):
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

        return best_criterion - self._criterion(node.instances), best_feature, best_value, best_left, best_right

    def _leaf(self, instances):
        return self.Node(instances, np.argmax(np.bincount(self._targets[instances])))

    def _criterion_gini(self, instances):
        bins = np.bincount(self._targets[instances])
        return np.sum(bins * (1 - bins / len(instances)))

    def _criterion_entropy(self, instances):
        bins = np.bincount(self._targets[instances])
        bins = bins[np.nonzero(bins)]
        return 0-np.sum(bins * np.log(bins / len(instances)))


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`
    #     (depth of the root node is zero);
    #   - when `args.max_leaves` is not `None`, there are less than `args.max_leaves` leaves
    #     (a leaf is a tree node without children);
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).
    decision_tree = DecisionTree(args.criterion, args.max_depth, args.min_to_split, args.max_leaves)
    decision_tree.fit(train_data, train_target)

    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy = sklearn.metrics.accuracy_score(train_target, decision_tree.predict(train_data))
    test_accuracy = sklearn.metrics.accuracy_score(test_target, decision_tree.predict(test_data))

    if args.plot:
        # Plot the final tree using graphviz.
        classes = np.max(target) + 1
        feature_names = getattr(sklearn.datasets, "load_{}".format(args.dataset))().feature_names
        dot = ["digraph Tree {node [shape=box]; bgcolor=invis;"]
        def plot(index, node, parent):
            if parent is not None: dot.append("{} -> {}".format(parent, index))
            dot.append("{} [fontname=\"serif\"; label=\"{}c_{} = {:.2f}\\ninstances = {}\\ncounts = [{}]\"];".format(
                index, "f. {} ({}) <= {:.3f}\\n".format(node.feature, feature_names[node.feature], node.value) if not node.is_leaf else "",
                args.criterion, decision_tree._criterion(node.instances), len(node.instances),
                ", ".join(map(str, np.bincount(decision_tree._targets[node.instances], minlength=classes)))))
            if not node.is_leaf:
                index = plot(plot(index + 1, node.left, index), node.right, index)
            return index + 1
        plot(0, decision_tree._root, None)
        dot.append("}")
        subprocess.run(["dot", "-Txlib"] if args.plot is True else ["dot", "-Tsvg", "-o{}".format(args.plot)],
                       input="\n".join(dot), encoding="utf-8")

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
