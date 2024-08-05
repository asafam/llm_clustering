from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score, davies_bouldin_score
import logging


def find_best_k(X: list, min_k=2, max_k=10):
    """
    Function to find the best k using silhouette score
    """
    best_k = min_k
    best_score = -1
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
#         print(f'K: {k}, Silhouette Score: {score}')
        if score > best_score:
            best_score = score
            best_k = k
            
#     print(f'best_k = {best_k}')
    return best_k, best_score


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


def cluster(X, k: int = 0, cluster_model='KMeans', visualize: bool = False):
    if k == 0:
        k, _ = find_best_k(X)
    
    pred_labels = None
    if cluster_model == 'KMeans':
        kmeans = KMeans(n_clusters=k, random_state=0)
        pred_labels = kmeans.fit_predict(X)

    if visualize:
        visualize_clusters(X, pred_labels)
        
    return pred_labels


def assess_clustering(X, gold_labels, pred_labels):
    logger = logging.getLogger()
    silhouette_avg = silhouette_score(X, gold_labels)
    davies_bouldin = davies_bouldin_score(X, gold_labels)
    ari = adjusted_rand_score(gold_labels, pred_labels)
    nmi = normalized_mutual_info_score(gold_labels, pred_labels)
    v_measure = v_measure_score(gold_labels, pred_labels)
    
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