from typing import *
import numpy as np
from sklearn.cluster import KMeans
from clustering.constraints import *
from clustering.optimizations import KOptimization, SilhouetteKOptimization


class ClusteringModel:
    def cluster(self, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__


class KMeans(ClusteringModel):
    def cluster(self, X, n_clusters: int, random_state: int = 42):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        return labels


class HardLabelsKMeans(KMeans):
    def cluster(
            self, 
            X, 
            constraint: HardLabelsClusteringContraints, 
            k_optimization: KOptimization, 
            min_k: int = 2,
            max_k: int = 10, 
            max_iter: int = 300, 
            tol: float = 1e-4
        ):
        """
        Finds the best number of clusters using a customizable score function after applying hard constraints.

        Parameters:
        X: np.ndarray
            The data array of shape (n_samples, n_features).
        hard_labels: dict
            A dictionary where keys are the indices of constrained instances and values are the assigned cluster.
        max_k: int
            The maximum number of clusters to consider.
        k_optimization: KOptimization
            A KOptimization class that takes (X, labels) and returns a score. Higher scores indicate better clustering.

        Returns:
        best_k: int
            The best number of clusters.
        best_score: float
            The score corresponding to the best k.
        """
        best_k = min_k
        best_score = -1
        hard_labels = constraint.instances
        
        for k in range(min_k, max_k + 1):
            labels, _ = self._hard_constrained_kmeans(
                X=X, 
                hard_labels=hard_labels, 
                n_clusters=k, 
                max_iter=max_iter, 
                tol=tol
            )
            score = k_optimization.score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
        return best_k, best_score

    def _hard_constrained_kmeans(self, X, hard_labels: dict, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        n_samples, n_features = X.shape
        
        # Initialize centroids using k-means++ or random initialization
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        
        # Force initial assignment based on hard constraints
        labels = np.zeros(n_samples, dtype=int)
        labels[list(hard_labels.keys())] = list(hard_labels.values())
        
        for i in range(max_iter):
            # Assignment step
            for j in range(n_samples):
                if j not in hard_labels:
                    distances = np.linalg.norm(X[j] - centroids, axis=1)
                    labels[j] = np.argmin(distances)
            
            # Update step
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    # Handle empty clusters by reinitializing centroids
                    new_centroids[k] = X[np.random.choice(n_samples)]
            
            # Check for convergence
            if np.all(np.abs(new_centroids - centroids) < tol):
                break
            
            centroids = new_centroids
        
        return labels, centroids
