import cProfile
import pstats
import time

from random_forests import *  # Change to your specific import
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=3
    )

    # Instantiate your Cython class
    tree_classifier = RandomForestClassifier(
        n_estimators=100,
        criterion="log_loss",
        max_depth=3,
        min_samples_split=4,
        random_state=3,
        # max_features=5,
    )

    # Profile the fitting process
    profiler = cProfile.Profile()
    profiler.enable()
    time_start = time.time()
    tree_classifier.fit(X_train, y=y_train)
    time_end = time.time()
    profiler.disable()

    # Print the profiling results
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()

    print("Time taken fitting:", time_end - time_start)

    time_start = time.time()
    print("Training score:", tree_classifier.score(X_train, y_train))
    time_end = time.time()
    print("Time taken predicting training:", time_end - time_start)
    print("Testing score:", tree_classifier.score(X_test, y_test))

    # Instantiate the sklearn class
    sklearn_tree_classifier = SklearnRandomForestClassifier(
        n_estimators=20,
        criterion="log_loss",
        max_depth=3,
        min_samples_split=4,
        random_state=3,
        # max_features=5,
    )

    time_start = time.time()
    sklearn_tree_classifier.fit(X_train, y_train)
    time_end = time.time()
    print("Time taken fitting:", time_end - time_start)

    time_start = time.time()
    print("Training score:", sklearn_tree_classifier.score(X_train, y_train))
    time_end = time.time()
    print("Time taken predicting training:", time_end - time_start)
    print("Testing score:", sklearn_tree_classifier.score(X_test, y_test))


if __name__ == "__main__":
    main()
