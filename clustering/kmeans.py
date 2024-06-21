import numpy as np


class KMeans:
    def __init__(self, num_clusters: int, max_iter: int = 1000):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.centroids = []
        self.converged = False
        self.clusters = []

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _update_clusters(self, X: np.ndarray):
        self.clusters = [[] for _ in range(self.num_clusters)]
        for i, x in enumerate(X):
            distances = np.apply_along_axis(
                lambda centroid: self._euclidean_distance(x, centroid),
                1,
                self.centroids,
            )
            cluster = np.argmin(distances)
            self.clusters[cluster].append(x)

    def _update_centroids(self):
        for i, cluster in enumerate(self.clusters):
            if not cluster:
                continue
            self.centroids[i] = np.mean(cluster, axis=0)

    def fit(self, X: np.ndarray):
        self.centroids = X[
            np.random.choice(X.shape[0], self.num_clusters, replace=False)
        ]
        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            self._update_clusters(X)
            self._update_centroids()
            if np.allclose(old_centroids, self.centroids, atol=1e-4):
                self.converged = True
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.array(
            [
                [self._euclidean_distance(x, centroid) for centroid in self.centroids]
                for x in X
            ]
        )
        return np.argmin(distances, axis=1)
