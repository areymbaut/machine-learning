import numpy as np
from numpy.typing import NDArray


class KMeans:
    """
    The k-means clustering algorithm partitions samples into k clusters
    within which each sample belongs to the cluster with the nearest mean
    (also called "cluster centers" or "cluster centroid").
    """

    def __init__(self,
                 k: int = 3,
                 n_iters: int = 100) -> None:
        """
        Args:
        - k (int): number of clusters.
        - n_iters (int): maximal number of iterations.
        """
        self.k = k
        self.n_iters = n_iters

        # List of sample indices for each cluster        
        self.clusters = np.array([[] for _ in range(self.k)],
                                 dtype=np.int32)

        # List of cluster-centroid coordinates
        self.centroids = np.array([[] for _ in range(self.k)],
                                  dtype=np.float64)

    def predict(self, X: NDArray) -> NDArray:
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize cluster centroids randomly (without replacement)
        random_sample_idx = np.random.choice(self.n_samples,
                                             self.k,
                                             replace=False)
        self.centroids = np.array([self.X[idx] for idx in random_sample_idx])

        # Optimize
        for _ in range(self.n_iters):
            # Assign samples to closest centroids
            self.clusters = self._generate_clusters(self.centroids)
            
            # Get new centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Check for convergence
            if self._is_converged(centroids_old, self.centroids):
                break
        
        return self._get_cluster_labels(self.clusters)

    def _generate_clusters(self, centroids: NDArray) -> NDArray:
        # Assign the samples to the closest centroids to form clusters
        clusters: list[list[int]] = [[] for _ in range(self.k)]
        for sample_idx, sample in enumerate(self.X):
            centroid_idx = self._get_closest_centroid(sample, centroids)
            clusters[centroid_idx].append(sample_idx)
        return clusters

    def _get_closest_centroid(self,
                              sample: NDArray,
                              centroids: NDArray) -> int:
        # Distance of the current sample to each centroid
        distances = [self._euclidian_distance(sample, c) for c in centroids]
        return np.argmin(distances)

    def _euclidian_distance(self,
                            sample: NDArray,
                            centroid: NDArray) -> NDArray:
        return np.sqrt(np.sum((sample - centroid)**2))

    def _get_centroids(self, clusters: NDArray) -> NDArray:
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _get_cluster_labels(self, clusters: NDArray) -> NDArray:
        # All samples within a cluster get assigned the cluster index
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _is_converged(self,
                      centroids_old: NDArray,
                      centroids_new: NDArray) -> bool:
        # Check distances between old and new centroids
        distances = [self._euclidian_distance(centroids_old[i],
                                              centroids_new[i])
                     for i in range(self.k)]
        return (sum(distances) == 0)


def main():
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Set random seed
    np.random.seed(0)

    # Load data
    X, y = datasets.make_blobs(n_samples=500,
                               n_features=2,
                               centers=3,
                               cluster_std=1.05,
                               shuffle=True,
                               random_state=0)
    n_clusters = len(np.unique(y))

    # Infer
    clustering = KMeans(k=n_clusters, n_iters=150)
    clustering.predict(X)

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    for _, idx in enumerate(clustering.clusters):
        samples = clustering.X[idx].T
        ax.scatter(*samples)
    for c in clustering.centroids:
        ax.scatter(*c, marker='x', color='k', linewidth=2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'k-means clustering (k = {n_clusters})')
    plt.show()


if __name__ == '__main__':
    main()
