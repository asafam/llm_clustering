from typing import *
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from clustering.constraints import *
from clustering.optimizations import KOptimization
from tqdm.notebook import tqdm
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans


class ClusteringModel:
    def cluster(self, **kwargs):
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
    

class KHyperparamClusteringModel(ClusteringModel):
    def cluster(
            self, 
            X,
            ids: list,
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
            A dictionary where keys are the indices of constrained sentences_labels and values are the assigned cluster.
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
        logger = logging.getLogger('default')
        logger.info(f"Clustering model: {type(self).__name__}")

        k_labels = [] # Store the iteration results

        if n_clusters:
            logger.debug(f"Clustering with n_clusters = {n_clusters}")
            start_datetime = datetime.now()
            labels = self._cluster(X, ids=ids, n_clusters=n_clusters, random_state=random_state, **kwargs)
            score = k_optimization.score(X, labels)
            end_datetime = datetime.now()
            elapsed_seconds = (end_datetime - start_datetime).total_seconds()
            k_labels.append(dict(
                k=n_clusters, 
                labels=labels,
                score=score,
                k_optimization=str(k_optimization),
                mode='n_clusters',
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                elapsed_seconds=elapsed_seconds
            ))
            logger.debug(f"Clustering for n_clusters = {n_clusters} returned a score of {score} after {elapsed_seconds} seconds")
            return dict(
                labels=labels, 
                n_clusters=n_clusters,
                index=0,
                k_labels=k_labels
            )
        
        # Find the best k in the coarse search
        # Coarse search over a large range
        logger.debug(f"Optimizing clustering for coarse range of k ({min_k}, {max_k}, {k_optimization_coarse_step_size})")
        coarse_k_values = range(min_k, max_k + 1, k_optimization_coarse_step_size)  # Every 10th value
        for k in coarse_k_values:
            k_start_datetime = datetime.now()
            labels = self._cluster(X, ids=ids, n_clusters=k, random_state=random_state, **kwargs)
            score = k_optimization.score(X, labels)
            k_end_datetime = datetime.now()
            elapsed_seconds=(k_end_datetime - k_start_datetime).total_seconds()
            logger.debug(f"Optimizing clustering for k = {k} returned a score of {score} after {elapsed_seconds} seconds")
            k_labels.append(dict(
                k=k, 
                labels=labels,
                score=score,
                k_optimization=str(k_optimization),
                wcss=compute_inertia(X, labels),
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
                labels = self._cluster(X, ids=ids, n_clusters=k, random_state=random_state, **kwargs)
                score = k_optimization.score(X, labels)
                elapsed_seconds=(k_end_datetime - k_start_datetime).total_seconds()
                logger.debug(f"Optimizing clustering for k = {k} returned a score of {score} after {elapsed_seconds} seconds")
                k_labels.append(dict(
                    k=k,
                    labels=labels,
                    score=score,
                    k_optimization=str(k_optimization),
                    wcss=compute_inertia(X, labels),
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
    
    def _cluster(self, **kwargs):
        raise NotImplementedError()
    
    
class BaseKMeans(KHyperparamClusteringModel):
    
    def _cluster(
            self, 
            X, 
            ids: list,
            n_clusters: int, 
            random_state: int, 
            n_init: int = 1, 
            verbose: bool = True,
            **kwargs
        ):
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state, verbose=verbose)
        labels = kmeans.fit_predict(X)
        
        return labels
