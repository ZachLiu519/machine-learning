import argparse
import time

from kmeans import KMeans
from sklearn.cluster import KMeans as KMeansSKL
from sklearn.datasets import make_blobs


def main(
    n_samples: int, n_features: int, centers: int, random_state: int, n_clusters: int
):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state,
    )

    kmeans = KMeans(num_clusters=n_clusters, random_state=random_state)
    time_start = time.time()
    kmeans.fit(X)
    time_end = time.time()
    print(f"KMeans: {time_end - time_start:.4f} seconds")

    kmeans_skl = KMeansSKL(
        n_clusters=n_clusters, init="random", random_state=random_state
    )
    time_start = time.time()
    kmeans_skl.fit(X)
    time_end = time.time()
    print(f"KMeansSKL: {time_end - time_start:.4f} seconds")

    print(kmeans.centroids)

    print(kmeans_skl.cluster_centers_)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument("--n_samples", type=int, default=5000)
    argParser.add_argument("--n_features", type=int, default=4)
    argParser.add_argument("--centers", type=int, default=3)
    argParser.add_argument("--random_state", type=int, default=42)

    argParser.add_argument("--n_clusters", type=int, default=3)

    args = argParser.parse_args()

    main(
        args.n_samples,
        args.n_features,
        args.centers,
        args.random_state,
        args.n_clusters,
    )
