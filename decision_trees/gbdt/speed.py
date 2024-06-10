import cProfile
import pstats
import time

from gbdt import *  # Change to your specific import
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGradientBoostingClassifier,
)
from sklearn.ensemble import (
    GradientBoostingRegressor as SklearnGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split


def main():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=3
    )

    # Instantiate your Cython class
    tree_classifier = GradientBoostingClassifier(
        max_depth=3,
        min_samples_split=4,
        max_features=2,
    )

    # Profile the fitting process
    profiler = cProfile.Profile()
    profiler.enable()
    tree_classifier.fit(X_train, y=y_train)
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()

    print("Training score:", tree_classifier.score(X_train, y_train))
    print("Testing score:", tree_classifier.score(X_test, y_test))

    # Instantiate the sklearn class
    sklearn_tree_classifier = SklearnGradientBoostingClassifier(
        max_depth=3,
        min_samples_split=4,
        max_features=2,
    )

    time_start = time.time()
    sklearn_tree_classifier.fit(X_train, y_train)
    time_end = time.time()
    print("Time taken predicting fitting:", time_end - time_start)

    print("Training score:", sklearn_tree_classifier.score(X_train, y_train))
    print("Testing score:", sklearn_tree_classifier.score(X_test, y_test))

    # test against regressors

    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=3
    )

    # Instantiate your Cython class
    tree_regressor = GradientBoostingRegressor(
        max_depth=3, max_features=2, min_samples_split=4
    )

    # Profile the fitting process
    profiler = cProfile.Profile()
    profiler.enable()
    tree_regressor.fit(X_train, y=y_train)
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()

    print("Training score:", tree_regressor.score(X_train, y_train))
    print("Testing score:", tree_regressor.score(X_test, y_test))

    # Instantiate the sklearn class
    sklearn_tree_regressor = SklearnGradientBoostingRegressor(
        max_depth=3, max_features=2, min_samples_split=4
    )

    time_start = time.time()
    sklearn_tree_regressor.fit(X_train, y_train)
    time_end = time.time()
    print("Time taken fitting:", time_end - time_start)

    print("Training score:", sklearn_tree_regressor.score(X_train, y_train))
    print("Testing score:", sklearn_tree_regressor.score(X_test, y_test))


if __name__ == "__main__":
    main()
