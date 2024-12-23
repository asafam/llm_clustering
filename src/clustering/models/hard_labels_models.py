from typing import *
import numpy as np
from datetime import datetime
import random
from metric_learn import ITML_Supervised, MMC_Supervised
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from clustering.constraints import *
from clustering.optimizations import KOptimization
from clustering.models.base_models import *
import logging


class HardLabelsKMeans(KHyperparamClusteringModel):
    def __init__(self, init_centroids: bool = False, enforce_labels: bool = True) -> None:
        super().__init__()
        self.init_centroids = init_centroids
        self.enforce_labels = enforce_labels

    def _cluster(self, X, ids: list, n_clusters: int, constraint: HardLabelsClusteringContraints, k_optimization: KOptimization, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        hard_labels = constraint.get_ids_labels()
        labels, centroids = self._hard_constrained_kmeans(
            X=X,
            hard_labels=hard_labels, 
            n_clusters=n_clusters, 
            k_optimization=k_optimization,
            max_iter=max_iter, 
            tol=tol,
            random_state=random_state
        )
        return labels, centroids

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
    

class HardLabelsCentroidsInitialization(KHyperparamClusteringModel):
    def _cluster(
            self, 
            X, 
            n_clusters: int, 
            constraint: HardLabelsClusteringContraints, 
            random_state: int = 42,
            verbose: bool = False,
            **kwargs
        ):
        sentences_labels = constraint.get_ids_labels()
        embeddings = np.vstack([X[id] for id in sentences_labels.keys()])
        labels = list(sentences_labels.values())
        init_centroids = compute_centroids(embeddings, labels)
        labels, centroids = self._base_cluster(X, n_clusters=n_clusters, init=init_centroids, random_state=random_state, verbose=verbose)
        return labels, centroids
    

class HardLabelsCentroidsInitializationKMeansClustering(BaseKMeans, HardLabelsCentroidsInitialization):
    pass
    

class HardLabelsCentroidsSubstitutionClustering(KHyperparamClusteringModel):
    def _cluster(
            self, 
            X, 
            n_clusters: int, 
            constraint: HardLabelsClusteringContraints, 
            random_state: int = 42,
            verbose: bool = False,
            **kwargs
        ):
        sentences_labels = constraint.get_ids_labels()
        labels = list(sentences_labels.values())
        X_centroid_reduced, ids_map = create_centroid_reduced_X(X, constraint.get_ids_labels())
        labels_centroid_reduced, centroids = self._base_cluster(X_centroid_reduced, n_clusters=n_clusters, random_state=random_state, verbose=verbose)
        labels = [labels_centroid_reduced[ids_map[i]] for i in range(X.shape[0])]
        return labels, centroids
    

class HardLabelsCentroidsSubstitutionKMeansClustering(BaseKMeans, HardLabelsCentroidsSubstitutionClustering):
    pass


class HardLabelsCentroidsSubstitutionAgglomerativeClustering(BaseAgglomerativeClustering, HardLabelsCentroidsSubstitutionClustering):
    pass
    

class HardLabelsMahalanobisClustering(KHyperparamClusteringModel):
    def __init__(self, n_constraints: int, algo: str) -> None:
        super().__init__()
        self.n_constraints = n_constraints
        self.algo = algo or 'MMC'

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(n_constraints={self.n_constraints}, algo={self.algo})"

    def _cluster(self, 
                 X, 
                 ids: list, 
                 n_clusters: int, 
                 constraint: HardLabelsClusteringContraints, 
                 random_state: int = 42, 
                 verbose: bool = False, 
                 **kwargs):
        logger = logging.getLogger('default')

        # Labels for the labeled data (-1 for unlabeled points)
        labels_pred_map = {id: -1 for id in ids}
        for id, label in list(constraint.get_ids_labels().items()):
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
        X_normalized = scaler.fit_transform(X)  # Fit on all data
        X_constraints = X_normalized[labeled_mask]

        # Sample hard labels constraints
        if self.n_constraints == 0:
            n_constraints = len(X_constraints)
        # Raise an error if the requested number of constraints is higher than the actual generated contraints
        elif n_constraints is None or len(X_constraints) >= self.n_constraints:
            n_constraints = self.n_constraints
        else:
            err_msgs = f"Number of actual constraints ({n_constraints}) is smaller than the number of the requested contraints ({self.n_constraints})"
            logger.error(err_msgs)
            raise ValueError(f"Number of actual constraints ({n_constraints}) is smaller than the number of the requested contraints ({self.n_constraints})")
        
        if len(X_constraints) < n_constraints:
            random.seed(random_state)
            indices = random.sample(range(len(y_labeled)), n_constraints)
            X_constraints = X_constraints[indices, :]
            y_labeled = [y_labeled[i] for i in indices]

        # Step 3: Learn the Mahalanobis distance using only the labeled data
        if self.algo == 'ITML':
            model = ITML_Supervised(n_constraints=n_constraints, verbose=verbose, random_state=random_state)
        elif self.algo == 'MMC':
            model = MMC_Supervised(n_constraints=n_constraints, verbose=verbose, random_state=random_state)
        else:
            raise ValueError("No valid Mahalanobix approximation algorithm was specified. Value provided: {self.algo}")

        model.fit(X_constraints, y_labeled)

        # Step 4: Transform the entire dataset (including unlabeled points)
        X_transformed = model.transform(X_normalized)

        # Step 5: Perform clustering on the transformed data
        labels, centroids = self._base_cluster(X_transformed, n_clusters=n_clusters, random_state=random_state)

        return labels, centroids
        

class HardLabelsMahalanobisKMeansClustering(BaseKMeans, HardLabelsMahalanobisClustering):
    pass
    

class HardLabelsMahalanobisAgglomerativeClustering(BaseAgglomerativeClustering, HardLabelsMahalanobisClustering):
    pass


class HardLabelsGMMClustering(KHyperparamClusteringModel):
    def __init__(self, fit_hard_labeled_set: bool = False):
        super().__init__()
        self.fit_hard_labeled_set = fit_hard_labeled_set

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(fit_hard_labeled_set={self.fit_hard_labeled_set})"

    def _cluster(self, 
                X, 
                ids: list, 
                n_clusters: int, 
                constraint: HardLabelsClusteringContraints, 
                random_state: int = 42, 
                verbose: bool = False, 
                **kwargs):
        logger = logging.getLogger('default')

        # Steps 1: Labels for the labeled data
        y_labeled = constraint.get_labels()

        # Step 2: Standardize the data (applies to both labeled and unlabeled points)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)  # Fit on all data
        X_constraints = X[constraint.get_ids()]

        # Step 3: Compute GMM initialization parameters from the labeled subset
        unique_labels = np.unique(y_labeled)  # Correctly extract unique labels as an iterable
        centroids = np.array([X_constraints[y_labeled == label].mean(axis=0) for label in unique_labels])
        weights_init = np.array([np.sum(y_labeled == label) / len(y_labeled) for label in unique_labels])
        # covariances_init = np.array([np.cov(X_constraints[y_labeled == label].T) for label in unique_labels])
        epsilon = 1e-6  # Small constant for regularization
        covariances_init = np.array([
            np.cov(X_constraints[y_labeled == label].T) + epsilon * np.eye(X_constraints.shape[1])
            for label in unique_labels
        ])

        # Step 4: Initialize and fit GMM with custom parameters
        gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            # covariance_type='full',  # Use 'full' because covariances are full matrices
            means_init=centroids,
            weights_init=weights_init
        )

        # Set the precisions manually based on covariances
        # gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances_init))  # Convert covariances to precisions
        if self.fit_hard_labeled_set:
            gmm.fit(X_constraints)
        else:
            gmm.fit(X_normalized)

        # Step 5: Predict cluster assignments
        labels = gmm.predict(X_normalized)

        return labels, centroids

