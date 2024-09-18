from typing import *
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from clustering.constraints import *
from clustering.optimizations import KOptimization
from clustering.models.base_models import BaseKMeans


class HardLabelsKMeans(BaseKMeans):
    def _cluster(self, X, n_clusters: int, constraint: HardLabelsClusteringContraints, k_optimization: KOptimization, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        hard_labels = constraint.instances
        labels, _, kmeans, score = self._hard_constrained_kmeans(
            X=X,
            hard_labels=hard_labels, 
            n_clusters=n_clusters, 
            k_optimization=k_optimization,
            max_iter=max_iter, 
            tol=tol,
            random_state=random_state
        )
        return labels, kmeans, score

    def _hard_constrained_kmeans(self, X, hard_labels: dict, n_clusters: int, k_optimization: KOptimization, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        n_samples, n_features = X.shape
        
        # Initialize centroids using k-means++ or random initialization
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        
        # Force initial assignment based on hard constraints
        labels = np.full(n_samples, -1, dtype=int)

        # Assign hard constraint labels
        labels[list(hard_labels.keys())] = list(hard_labels.values())
        
        for i in range(max_iter):
            # Assignment step
            for j in range(n_samples):
                if j not in hard_labels: # Only assign points without hard labels
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

            score = k_optimization.score(X, labels)
        
        return labels, centroids, kmeans, score

    def _hard_constrained_kmeans2(self, X, hard_labels: dict, k_optimization: KOptimization, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        n_samples, n_features = X.shape

        init_clusters = []

        
        # Initialize centroids using k-means++ or random initialization
        kmeans = KMeans(n_clusters=n_clusters, init=init_clusters, random_state=random_state)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        
        # Force initial assignment based on hard constraints
        labels = np.full(n_samples, -1, dtype=int)

        # Assign hard constraint labels
        labels[list(hard_labels.keys())] = list(hard_labels.values())
        
        for i in range(max_iter):
            # Assignment step
            for j in range(n_samples):
                if j not in hard_labels: # Only assign points without hard labels
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
        
        return labels, centroids, kmeans
    