from typing import *
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score
from scipy.spatial import distance
from kneed import KneeLocator
import numpy as np

class KOptimization:
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def score(self, **kwargs) -> int:
        raise NotImplementedError()
    
    def find_best_k(self, k_values: list, scores: list) -> dict:
        raise NotImplementedError()
    

class SilhouetteKOptimization(KOptimization):    
    def score(self, X, labels, **kwargs) -> int:
        score = silhouette_score(X, labels)
        return score
    
    def find_best_k(self, k_values: list, scores: list) -> dict:
        optimal_index = int(np.argmax(scores))
        optimal_k = k_values[optimal_index]
        return dict(
            optimal_k=optimal_k,
            optimal_index=optimal_index
        )
    

class InteriaKneeKOptimization(KOptimization):    
    def score(self, X, labels, centroids, **kwargs) -> int:
        score = sum(
              distance.euclidean(X[i], centroids[labels[i]])**2 for i in range(X.shape[0])
        )
        return score
    
    def find_best_k(self, k_values: list, scores: list) -> dict:
        knee_locator = KneeLocator(k_values, scores, curve="convex", direction="decreasing")
        optimal_k = knee_locator.knee
        optimal_index = k_values.index(optimal_k)
        return dict(
            optimal_k=optimal_k,
            optimal_index=optimal_index
        )
