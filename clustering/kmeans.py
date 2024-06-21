import numpy as np


class KMeans:
    def __init__(
        self, num_clusters: int, max_iter: int = 10000, random_state: int = 42
    ):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.centroids = []
        self.converged = False
        self.clusters = {i: [] for i in range(self.num_clusters)}
        self.random_state = random_state

    def _euclidean_distance(self, X):
        return np.sqrt(((X - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))

    def _update_clusters(self, X):
        """Assign each point to the nearest centroid using indices."""
        distances = self._euclidean_distance(X)
        clusters = np.argmin(distances, axis=0)
        self.clusters = {
            i: np.where(clusters == i)[0] for i in range(self.num_clusters)
        }

    def _update_centroids(self, X):
        """Update centroids to the mean of the points in each cluster."""
        for i, cluster_indices in self.clusters.items():
            if len(cluster_indices) == 0:
                # If a cluster is empty, reinitialize the centroid to a random data point
                self.centroids[i] = X[np.random.randint(0, X.shape[0])]
            else:
                # Calculate the mean of all points indexed by cluster_indices
                self.centroids[i] = np.mean(X[cluster_indices], axis=0)

    def fit(self, X: np.ndarray):
        np.random.seed(self.random_state)
        self.centroids = X[
            np.random.choice(X.shape[0], self.num_clusters, replace=False)
        ]
        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)
            self._update_clusters(X)
            self._update_centroids(X)
            if np.allclose(old_centroids, self.centroids, atol=1e-4):
                self.converged = True
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self._euclidean_distance(X)
        return np.argmin(distances, axis=0)
