import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import *
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


def evaluate_k(k_true: int, k_pred: int) -> dict:
    return dict(
        k_true = k_true,
        k_pred = k_pred,
        absolute_difference = abs(k_pred - k_true),
        relative_error = abs(k_pred - k_true) / k_true * 100,
        normalized_absolute_error = abs(k_pred - k_true) / k_true,
        squared_error = (k_pred - k_true) ** 2,
    )


def evaluate_clustering(labels_true, labels_pred, X=None):
    logger = logging.getLogger('default')
    silhouette_avg = silhouette_score(X, labels_pred) if X is not None else None
    davies_bouldin = davies_bouldin_score(X, labels_pred) if X is not None else None
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    v_measure = v_measure_score(labels_true, labels_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    fmi = fowlkes_mallows_score(labels_true, labels_pred)
    cm = confusion_matrix(labels_true, labels_pred)
    
    logger.debug(f"Silhouette Score: {silhouette_avg}")
    logger.debug(f"Davies-Bouldin Index: {davies_bouldin}")
    logger.debug(f"Adjusted Rand Index (ARI): {ari}")
    logger.debug(f"Normalized Mutual Information (NMI): {nmi}")
    logger.debug(f"V-measure: {v_measure}")
    logger.debug(f"Homogeneity: {homogeneity}")
    logger.debug(f"Completeness: {completeness}")
    
    return dict(
        silhouette_avg = silhouette_avg,
        davies_bouldin = davies_bouldin,
        ari = ari,
        nmi = nmi,
        v_measure = v_measure,
        homogeneity=homogeneity,
        completeness=completeness,
        fmi=fmi,
        cm=cm
    )


def evaluate_must_link_cannot_link(must_link_true: list, cannot_link_true: list, must_link_pred: list, cannot_link_pred: list) -> dict:
    result = dict(
        must_link = _evaluate_pairs(must_link_true, must_link_pred),
        cannot_link = _evaluate_pairs(cannot_link_true, cannot_link_pred),
    )
    return result


def _evaluate_pairs(pairs_true: list, pairs_pred: list) -> dict:
    # Normalize pairs to ensure order consistency
    pairs_true = set(tuple(sorted(pair)) for pair in pairs_true)
    pairs_pred = set(tuple(sorted(pair)) for pair in pairs_pred)

     # Create a unified set of all possible pairs from both lists
    all_pairs = pairs_true.union(pairs_pred)
    
    # Convert sets to lists of 1 (must-link) or 0 (no link)
    y_true = [1 if pair in pairs_true else 0 for pair in all_pairs]
    y_pred = [1 if pair in pairs_pred else 0 for pair in all_pairs]

    # Calculate precision, recall, accuracy, and f1 scores
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return dict(
        precision=precision, 
        recall=recall, 
        accuracy=accuracy, 
        f1=f1
    )


def transform_hard_labels_to_ml_cl(sentences_labels):
    must_link = []
    cannot_link = []
    for id1, label1 in sentences_labels.items():
        for id2, label2 in sentences_labels.items():
            if id1 != id2:
                if label1 == label2 and (id2, id1) not in must_link:
                    must_link.append((id1, id2))
                elif (id2, id1) not in cannot_link:
                    cannot_link.append((id1, id2))
    return must_link, cannot_link


def get_true_ml_cl(ids_true: list, labels_true: list):
    # Create ground truth pairs based on the true labels
    must_link_true = []
    cannot_link_true = []

    # Compare all pairs and decide if they should be must-link or cannot-link based on true labels
    for i in range(len(ids_true)):
        for j in range(i + 1, len(ids_true)):
            if labels_true[i] == labels_true[j]:
                must_link_true.append((ids_true[i], ids_true[j]))
            else:
                cannot_link_true.append((ids_true[i], ids_true[j]))

    return must_link_true, cannot_link_true


def count_singletons(sentences_labels: dict) -> int:
    label_to_ids = defaultdict(list)
    for id_, label in sentences_labels.items():
        label_to_ids[label].append(id_)

    # Count how many labels have only one ID associated with them
    count_single_id_labels = sum(1 for ids in label_to_ids.values() if len(ids) == 1)
    return count_single_id_labels


def compute_inertia(X, labels):
    """
    Compute inertia as the sum of squared distances between each point
    and the centroid of its assigned cluster.

    Parameters:
    - X: Data points (array of shape [n_samples, n_features]).
    - labels: Cluster labels for each point.

    Returns:
    - inertia: The sum of squared distances to cluster centroids.
    """
    n_clusters = len(np.unique(labels))  # Number of clusters
    inertia = 0

    # Compute the centroid of each cluster and the inertia
    for cluster in range(n_clusters):
        # Get the points belonging to the current cluster
        cluster_points = X[labels == cluster]
        
        # Compute the centroid of the current cluster
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            
            # Compute the sum of squared distances to the centroid
            inertia += np.sum((cluster_points - centroid) ** 2)

    return inertia


def compute_centroids(embeddings, labels):
    """
    Compute the centroids of clusters based on given vectors and labels.

    Parameters:
    vectors (array-like): List or array of vectors (points).
    labels (array-like): Corresponding labels for each vector.

    Returns:
    centroids (numpy array): Centroids of the clusters.
    """
    # Group vectors by their labels
    clusters = defaultdict(list)
    for vector, label in zip(embeddings, labels):
        clusters[label].append(vector)

    # Compute the centroid for each cluster
    centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters.values()])

    return centroids