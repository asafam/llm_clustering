from enum import Enum
from clustering.utils import SilhouetteKDetemination


class ClusteringModel:
    pass


class KMeans(ClusteringModel):
    def cluster(X, k: int = 0, max_k: int = 10, visualize: bool = False):
        if k == 0:
            k_determiner = SilhouetteKDetemination()
            k = k_determiner.find_best_k(X=X, max_k=max_k)
        
        kmeans = KMeans(n_clusters=k, random_state=0)
        pred_labels = kmeans.fit_predict(X)
            
        return pred_labels