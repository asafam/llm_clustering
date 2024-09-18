from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score
import logging


def visualize_clusters(all_embeddings, kmeans_labels):
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-means Clustering of Sentence Embeddings (PCA)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=kmeans_labels, cmap='viridis')
    plt.title('K-means Clustering of Sentence Embeddings (t-SNE)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()


def evaluate_clustering(labels_true, labels_pred, X=None):
    logger = logging.getLogger('default')
    silhouette_avg = silhouette_score(X, labels_pred) if X is not None else None
    davies_bouldin = davies_bouldin_score(X, labels_pred) if X is not None else None
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    v_measure = v_measure_score(labels_true, labels_pred)
    
    logger.debug(f"Silhouette Score: {silhouette_avg}")
    logger.debug(f"Davies-Bouldin Index: {davies_bouldin}")
    logger.debug(f"Adjusted Rand Index (ARI): {ari}")
    logger.debug(f"Normalized Mutual Information (NMI): {nmi}")
    logger.debug(f"V-measure: {v_measure}")
    
    return dict(
        silhouette_avg = silhouette_avg,
        davies_bouldin = davies_bouldin,
        ari = ari,
        nmi = nmi,
        v_measure = v_measure,
    )
