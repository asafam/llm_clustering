from typing import *
import numpy as np
from datetime import datetime
from metric_learn import ITML_Supervised
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustering.constraints import *
from clustering.optimizations import KOptimization
from clustering.models.base_models import *


class HardLabelsKMeans(KHyperparamClusteringModel):
    def __init__(self, init_centroids: bool = False, enforce_labels: bool = True) -> None:
        super().__init__()
        self.init_centroids = init_centroids
        self.enforce_labels = enforce_labels

    def _cluster(self, X, ids: list, n_clusters: int, constraint: HardLabelsClusteringContraints, k_optimization: KOptimization, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        hard_labels = constraint.sentences_labels
        labels, centroids = self._hard_constrained_kmeans(
            X=X,
            hard_labels=hard_labels, 
            n_clusters=n_clusters, 
            k_optimization=k_optimization,
            max_iter=max_iter, 
            tol=tol,
            random_state=random_state
        )
        return labels

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
        
        return labels, centroids

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
    

class HardLabelsCentroidsKMeans(KHyperparamClusteringModel):
    def _cluster(
            self, 
            X, 
            ids: list,
            n_clusters: int, 
            constraint: HardLabelsClusteringContraints, 
            n_init: int = 300, 
            tol: float = 1e-4, 
            random_state: int = 42,
            verbose: bool = True,
            **kwargs
        ):
        sentences_labels = constraint.sentences_labels
        labels = list(sentences_labels.values())
        centroids = compute_centroids(X, labels)
        kmeans = KMeans(n_clusters=n_clusters, init=centroids, n_init=n_init, random_state=random_state, verbose=verbose)
        labels = kmeans.fit_predict(X)

        return labels


class HardLabelsMahalanobisKMeans(KHyperparamClusteringModel):
    def __init__(self, n_constraints: int) -> None:
        super().__init__()
        self.n_constraints = n_constraints

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n_constraints={self.n_constraints})"

    def _cluster(self, X, ids: list, n_clusters: int, constraint: HardLabelsClusteringContraints, random_state: int = 42, **kwargs):
        # Labels for the labeled data (-1 for unlabeled points)
        labels_pred_map = {id: -1 for id in ids}
        print("ids = ", ids)
        print("sentences_labels = ", constraint.sentences_labels)
        for id, label in list(constraint.sentences_labels.items()):
            if id in labels_pred_map:
                labels_pred_map[id] = label
            else:
                print(f"predicted {id} not in ids")
        y = np.array(list(labels_pred_map.values()))

        # Step 1: Select only the labeled data for metric learning
        labeled_mask = y != -1  # Mask to select only labeled points
        y_labeled = y[labeled_mask]

        # Step 2: Standardize the data (applies to both labeled and unlabeled points)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  # Fit on all data

        # Step 3: Learn the Mahalanobis distance using only the labeled data
        n_constraints = len(constraint.must_link) if self.n_constraints == 0 else self.n_constraints
        itml = ITML_Supervised(n_constraints=n_constraints, verbose=True)
        itml.fit(X_scaled[labeled_mask], y_labeled)

        # Step 4: Transform the entire dataset (including unlabeled points)
        X_transformed = itml.transform(X_scaled)

        # Step 5: Perform KMeans clustering on the transformed data
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X_transformed)

        return labels