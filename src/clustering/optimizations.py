from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score


class KOptimization:
    def score(self, **kwargs) -> int:
        raise NotImplementedError()
    
    def __str__(self) -> str:
        return self.__class__.__name__
    

class SilhouetteKOptimization(KOptimization):    
    def score(self, X, labels) -> int:
        score = silhouette_score(X, labels)
        return score