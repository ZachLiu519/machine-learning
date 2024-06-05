import multiprocessing

import numpy as np
from scipy.stats import mode

from decision_trees.decision_tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        criterion,
        n_estimators=100,
        max_depth=None,
        max_features=None,
        min_samples_split=2,
        random_state=None,
        n_jobs=-1,
    ):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []
        self.random_state = random_state
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

    def _single_tree_fit(self, X, y):
        tree = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )
        tree.fit(X, y)
        return tree

    def fit(self, X, y):
        if self.max_features is None:
            self.max_features = int(X.shape[1] ** 0.5)
        # Parallelize the fitting process
        with multiprocessing.Pool(self.n_jobs) as pool:
            self.trees = pool.starmap(
                self._single_tree_fit, [(X, y)] * self.n_estimators
            )

    def _single_tree_predict(self, tree, X):
        return tree.predict(X)

    def predict(self, X):
        # voting on the predictions from each tree
        predictions = []

        # Parallelize the prediction process
        with multiprocessing.Pool(self.n_jobs) as pool:
            predictions = pool.starmap(
                self._single_tree_predict, [(tree, X) for tree in self.trees]
            )

        # select the most common prediction using stats.mode
        predictions = np.array(predictions).T

        return mode(predictions, axis=1)[0]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
