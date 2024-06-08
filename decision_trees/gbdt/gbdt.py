import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from decision_trees.decision_tree.decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        max_features=None,
        random_state=None,
        min_samples_split=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.max_features = max_features
        self.random_state = random_state
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        residuals = np.copy(y)
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            residuals -= self.learning_rate * tree.predict(X)
            if i % 10 == 0:
                print("Residuals:", min(residuals), max(residuals), np.mean(residuals))

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def score(self, X, y):
        # calculate the coefficient of determination
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


def softmax(z):
    # Stable softmax function
    exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


class GradientBoostingClassifier:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        max_features=None,
        random_state=None,
        min_samples_split=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.max_features = max_features
        self.random_state = random_state
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.le = LabelEncoder()
        labels = self.le.fit_transform(y)
        # one-hot encode the labels
        self.ohe = OneHotEncoder()
        onehot_labels = self.ohe.fit_transform(labels.reshape(-1, 1)).toarray()

        y_proba = np.mean(onehot_labels, axis=0)  # Better initial probabilities
        self.trees = [[] for _ in range(onehot_labels.shape[1])]
        residuals = onehot_labels - y_proba

        for _ in range(self.n_estimators):
            for j in range(onehot_labels.shape[1]):
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    max_features=self.max_features,
                    random_state=self.random_state,
                    min_samples_split=self.min_samples_split,
                )
                tree.fit(X, residuals[:, j])
                self.trees[j].append(tree)
                residuals[:, j] -= self.learning_rate * tree.predict(X)

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return self.le.inverse_transform(np.argmax(y_prob, axis=1))

    def predict_proba(self, X):
        y_raw = np.zeros((X.shape[0], len(self.le.classes_)))

        for c, trees in enumerate(self.trees):
            for tree in trees:
                y_raw[:, c] += self.learning_rate * tree.predict(X)

        return softmax(y_raw)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
