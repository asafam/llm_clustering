from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score
from scipy.spatial import distance

class KOptimization:
    def score(self, **kwargs) -> int:
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
    

class SilhouetteKOptimization(KOptimization):    
    def score(self, X, labels, **kwargs) -> int:
        score = silhouette_score(X, labels)
        return score
    

class InteriaMinimizationKOptimization(KOptimization):    
    def score(self, X, labels, centroids, **kwargs) -> int:
        score = -sum(
              distance.euclidean(X[i], centroids[labels[i]]) for i in range(X.shape[0])
        )
        return score