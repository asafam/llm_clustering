from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score
from clustering.models import ClusteringModel
import logging


class KOptimization:
    def score(self, **kwargs) -> int:
        raise NotImplementedError()
    

class SilhouetteKOptimization(KOptimization):
    def score(self, **kwargs) -> int:
        return silhouette_score(**kwargs)