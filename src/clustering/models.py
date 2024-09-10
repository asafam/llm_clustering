from typing import *
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from clustering.constraints import *
from clustering.optimizations import KOptimization
import logging


class ClusteringModel:
    def cluster(self, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__


class BaseKMeans(ClusteringModel):
    def cluster(
            self, 
            X,
            k_optimization: KOptimization, 
            n_clusters: Optional[int] = None,
            min_k: int = 2,
            max_k: int = 10, 
            k_optimization_coarse_step_size: int = 10,
            k_optimization_fine_range: int = 10,
            random_state: int = 42,
            **kwargs
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
        
        logger = logging.getLogger('default')

        if n_clusters:
            start_datetime = datetime.now()
            labels = self._cluster(X, n_clusters=n_clusters, random_state=random_state, **kwargs)
            score = k_optimization.score(X, labels)
            end_datetime = datetime.now()
            elapsed_seconds = (end_datetime - start_datetime).total_seconds()
            k_labels.append(dict(
                k=k, 
                labels=labels,
                score=score,
                k_optimization=str(k_optimization),
                mode='n_clusters',
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                elapsed_seconds=elapsed_seconds
            ))
            return dict(
                labels=labels, 
                n_clusters=n_clusters,
                index=0,
                k_labels=k_labels
            ) 
        
        # Find the best k in the coarse search
        k_labels = [] # Store the iteration results

        # Coarse search over a large range
        logger.debug(f"Optimizing clustering for coarse range of k ({min_k}, {max_k}, {k_optimization_coarse_step_size})")
        coarse_k_values = range(min_k, max_k + 1, k_optimization_coarse_step_size)  # Every 10th value
        for k in coarse_k_values:
            k_start_datetime = datetime.now()
            labels = self._cluster(X, n_clusters=k, random_state=random_state, **kwargs)
            score = k_optimization.score(X, labels)
            k_end_datetime = datetime.now()
            elapsed_seconds=(k_end_datetime - k_start_datetime).total_seconds()
            logger.debug(f"Optimizing clustering for k = {k} returned a score of {score} after {elapsed_seconds} seconds")
            k_labels.append(dict(
                k=k, 
                labels=labels,
                score=score,
                k_optimization=str(k_optimization),
                mode='coarse',
                start_datetime=k_start_datetime,
                end_datetime=k_end_datetime,
                elapsed_seconds=elapsed_seconds
            ))

        # Find the best k in the coarse search
        max_k_labels = max(k_labels, key=lambda x: x['score'])
        best_k = max_k_labels['k']

        if k_optimization_coarse_step_size > 1 and k_optimization_fine_range > 0:
            # Fine search around the best coarse k
            logger.debug(f"Optimizing clustering for fine range of k: ({max(min_k, best_k - k_optimization_fine_range + 1)}, {min(max_k, best_k + k_optimization_fine_range)})")
            fine_k_values = range(max(min_k, best_k - k_optimization_fine_range + 1), min(max_k, best_k + k_optimization_fine_range))  # ±k_optimization_fine_range around best coarse k

            for k in fine_k_values:
                labels = self._cluster(X, n_clusters=k, random_state=random_state, **kwargs)
                score = k_optimization.score(X, labels)
                elapsed_seconds=(k_end_datetime - k_start_datetime).total_seconds()
                logger.debug(f"Optimizing clustering for k = {k} returned a score of {score} after {elapsed_seconds} seconds")
                k_labels.append(dict(
                    k=k,
                    labels=labels,
                    score=score,
                    k_optimization=str(k_optimization),
                    mode='fine',
                    start_datetime=k_start_datetime,
                    end_datetime=k_end_datetime,
                    elapsed_seconds=elapsed_seconds
                ))

        # Find the best k
        max_k_labels = max(k_labels, key=lambda x: x['score'])
        best_labels = max_k_labels['labels']
        best_k = max_k_labels['k']
        best_index = k_labels.index(max_k_labels)

        logger.debug(f"Optimization with {str(k_optimization)} identified optimal k = {best_k}")
        
        return dict(
            labels=best_labels, 
            n_clusters=best_k,
            index=best_index,
            k_labels=k_labels
        )
    
    def _cluster(self, X, n_clusters: int, random_state: int):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        return labels



class HardLabelsKMeans(BaseKMeans):
    def _cluster(self, X, n_clusters: int, constraint: HardLabelsClusteringContraints, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        hard_labels = constraint.instances
        labels, _ = self._hard_constrained_kmeans(
            X=X, 
            hard_labels=hard_labels, 
            n_clusters=n_clusters, 
            max_iter=max_iter, 
            tol=tol,
            random_state=random_state
        )
        return labels

    def _hard_constrained_kmeans(self, X, hard_labels: dict, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
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
        
        return labels, centroids
