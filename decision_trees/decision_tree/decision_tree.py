import abc

import numpy as np
import numpy.typing as npt

from decision_trees.decision_tree.criterion import impurity_func


class DecisionTreeNode:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_samples = data.shape[0]
        self.num_classes = len(set(labels))
        if self.num_classes > 0 and isinstance(labels[0], (int, np.integer)):
            # Use np.bincount for faster counting of integer labels
            self.counts = np.bincount(self.labels)
        self.feature_index = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.is_leaf = False


class DecisionTree(abc.ABC):
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        self.criterion = criterion
        self.max_depth = float("inf") if max_depth is None else max_depth
        self.max_features = max_features
        self.tree = None
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree = DecisionTreeNode(data=X, labels=y)
        self._build_tree(self.tree, 0)

    def _find_best_criteria(self, feature_idx, node):
        # sort the thresholds in ascending order, find split where left and right have different labels
        # Get the indices that would sort the array of the selected feature
        raise NotImplementedError(
            "Find_best_criteria method is not implemented for the base class. Use a subclass instead."
        )

    def _build_tree(self, node, depth):
        try:
            assert depth < self.max_depth
            assert node
            assert node.num_samples >= self.min_samples_split

            best_impurity = float("inf")
            best_criteria = None

            assert isinstance(self.max_features, int)

            if self.random_state:
                np.random.seed(self.random_state)
            feature_indices = np.random.choice(
                node.data.shape[1], self.max_features, replace=False
            )

            for feature_idx in feature_indices:
                impurity, criteria = self._find_best_criteria(feature_idx, node)

                # Check if the found impurity is less than the current best
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_criteria = criteria

            if best_criteria:
                node.feature_index, node.threshold, left_mask, right_mask = (
                    best_criteria
                )
                left_node = DecisionTreeNode(
                    data=node.data[left_mask], labels=node.labels[left_mask]
                )
                if left_node.num_samples > 0:
                    node.left = left_node
                    self._build_tree(node.left, depth + 1)
                right_node = DecisionTreeNode(
                    data=node.data[right_mask], labels=node.labels[right_mask]
                )
                if right_node.num_samples > 0:
                    node.right = right_node
                    self._build_tree(node.right, depth + 1)

        except AssertionError:
            node.is_leaf = True

    def predict(self, X):
        raise NotImplementedError(
            "Predict method is not implemented for the base class. Use a subclass instead."
        )

    def score(self, X, y):
        raise NotImplementedError(
            "Score method is not implemented for the base class. Use a subclass instead."
        )

    def _predict_single(self, x):
        raise NotImplementedError(
            "Predict_single method is not implemented for the base class. Use a subclass instead."
        )


class DecisionTreeClassifier(DecisionTree):
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
        )

    def _find_best_criteria(self, feature_idx, node):
        # sort the thresholds in ascending order, find split where left and right have different labels
        # Get the indices that would sort the array of the selected feature
        feature_values = node.data[:, feature_idx]
        sorted_indices = np.argsort(feature_values)

        # Use these indices to sort both the data and the labels
        thresholds = feature_values[sorted_indices]
        labels = node.labels[sorted_indices]

        best_impurity = float("inf")
        best_criteria = None

        left_counts: npt.NDArray[np.intp] = np.zeros(
            node.counts.shape[0], dtype=np.intp
        )
        right_counts: npt.NDArray[np.intp] = np.copy(node.counts)

        for idx, threshold in enumerate(thresholds):
            left_mask, right_mask = (
                feature_values <= threshold,
                feature_values > threshold,
            )

            left_label = labels[idx]
            right_label = labels[idx + 1] if idx + 1 < labels.shape[0] else left_label
            # only split if both sides have samples and different labels
            left_counts[left_label] += 1
            right_counts[left_label] -= 1
            if left_label != right_label:
                left_y_size, right_y_size = (
                    node.data[left_mask].shape[0],
                    node.data[right_mask].shape[0],
                )
                left_input = (left_y_size, left_counts)
                right_input = (right_y_size, right_counts)

                left_y_size, left_input = left_input
                right_y_size, right_input = right_input
                weighted_left_impurity = (
                    left_y_size / node.num_samples
                ) * impurity_func(self.criterion, (left_y_size, left_input))
                weighted_right_impurity = (
                    right_y_size / node.num_samples
                ) * impurity_func(self.criterion, (right_y_size, right_input))

                impurity = weighted_left_impurity + weighted_right_impurity
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_criteria = (
                        feature_idx,
                        threshold,
                        left_mask.copy(),
                        right_mask.copy(),
                    )

        return best_impurity, best_criteria

    def predict(self, X):
        predictions = np.apply_along_axis(self._predict_single, 1, X)
        return predictions

    def _predict_single(self, x):
        node = self.tree
        while (
            node and not node.is_leaf and not (node.left is None and node.right is None)
        ):
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        assert node
        return np.argmax(node.counts)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


class DecisionTreeRegressor(DecisionTree):
    def __init__(
        self,
        criterion="mse",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        random_state=None,
    ):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
        )

    def _find_best_criteria(self, feature_idx, node):
        # sort the thresholds in ascending order, find split where left and right have different labels
        # Get the indices that would sort the array of the selected feature
        feature_values = node.data[:, feature_idx]
        sorted_indices = np.argsort(feature_values)

        # Use these indices to sort both the data and the labels
        thresholds = feature_values[sorted_indices]
        labels = node.labels[sorted_indices]

        best_impurity = float("inf")
        best_criteria = None

        # mse = (sum_of_squares - sum_of_values^2) / num_samples
        left_count, right_count = 0, node.num_samples
        left_sum, right_sum = 0, np.sum(node.labels)
        left_sum_sq, right_sum_sq = 0, np.sum(node.labels**2)

        for idx, threshold in enumerate(thresholds):
            left_mask, right_mask = (
                feature_values <= threshold,
                feature_values > threshold,
            )

            left_label = labels[idx]
            right_label = labels[idx + 1] if idx + 1 < labels.shape[0] else left_label

            left_count += 1
            right_count -= 1
            left_sum += left_label
            left_sum_sq += left_label**2
            right_sum -= left_label
            right_sum_sq -= left_label**2

            # only split if both sides have samples and different labels
            if left_label != right_label and left_count > 0 and right_count > 0:
                left_impurity = (left_sum_sq - (left_sum**2) / left_count) / left_count
                right_impurity = (
                    right_sum_sq - (right_sum**2) / right_count
                ) / right_count

                impurity = (left_count / node.num_samples) * left_impurity + (
                    right_count / node.num_samples
                ) * right_impurity
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_criteria = (
                        feature_idx,
                        threshold,
                        left_mask.copy(),
                        right_mask.copy(),
                    )

        return best_impurity, best_criteria

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.tree
        while (
            node and not node.is_leaf and not (node.left is None and node.right is None)
        ):
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        assert node
        return np.mean(node.labels)

    def score(self, X, y):
        # calculate the coefficient of determination
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
