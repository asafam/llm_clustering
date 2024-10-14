from typing import *
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from clustering.utils import *
import logging

class ClusteringConstraints:
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def get_labels(self) -> list:
        raise NotImplementedError()


class PartitionsLevelClusteringConstraints(ClusteringConstraints):
    pass


class HardLabelsClusteringContraints(PartitionsLevelClusteringConstraints):
    def __init__(self, sentences_labels: dict) -> None:
        """
        Initialize the SubsetHardLabelsClusteringContraints class.

        This class is used to define clustering constraints based on hard labels assigned to a subset of sentences_labels.
        Each instance is associated with a specific cluster, which is represented by a tuple of (instance_id, cluster_id).

        Parameters:
        -----------
        sentences_labels : list of tuples
            A list where each element is a tuple (instance_id: int, cluster_id: int).
            - instance_id: An integer that uniquely identifies an instance.
            - cluster_id: An integer that represents the cluster ID to which the instance is constrained.

        Examples:
        ---------
        >>> constraints = SubsetHardLabelsClusteringContraints(sentences_labels={0: 22, 1: 22, 2: 13, 3: 13, 4: 22})
        >>> constraints.sentences_labels
        {0: 1, 1: 1, 2: 0, 3: 0, 4: 1}
        
        """
        super().__init__()
        
        self.labels = sorted(set(sentences_labels.values())) # Get the unique labels and sort them
        label_to_index = {label: index for index, label in enumerate(self.labels)} # Create a mapping from label to its index
        self.sentences_labels = {id: label_to_index[label] for id, label in sentences_labels.items()} # Translate the labels to their indices
        self.instances = self.sentences_labels

    def __str__(self) -> str:
        return str(self.sentences_labels)
    
    def get_labels(self) -> list:
        return self.labels
    
    def get_k(self) -> int:
        return len(self.labels)
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        labels_pred_dict = {}
        for id in ids_true:
            labels_pred_dict[id] = -1

        for id, label in self.sentences_labels.items():
            if id in labels_pred_dict:
                labels_pred_dict[id] = label
        labels_pred = list(labels_pred_dict.values())

        must_link_pred, cannot_link_pred = transform_hard_labels_to_ml_cl(self.sentences_labels)
        must_link_true, cannot_link_true = get_true_ml_cl(ids_true=ids_true, labels_true=labels_true)

        result = dict(
            ari = adjusted_rand_score(labels_true, labels_pred),
            nmi = normalized_mutual_info_score(labels_true, labels_pred),
            v_measure = v_measure_score(labels_true, labels_pred),
            number_of_singletons = count_singletons(self.sentences_labels),
        )

        # Add must-link/cannot-link evaluation metrics 
        result.update(
            evaluate_must_link_cannot_link(
                must_link_true=must_link_true, 
                must_link_pred=must_link_pred, 
                cannot_link_true=cannot_link_true, 
                cannot_link_pred=cannot_link_pred
            )
        )

        # Add K predictions evaluation metrics
        result.update(
            k = evaluate_k(k_true=len(set(labels_true)), k_pred=len(set(self.sentences_labels.values())))
        )
        return result
    

class FuzzyLabelsClusteringContraints(PartitionsLevelClusteringConstraints):
    def __init__(self, sentences_labels: dict) -> None:
        """
        Initialize the SubsetFuzzyLabelsClusteringContraints class.

        This class is used to define clustering constraints based on fuzzy labels assigned to a subset of sentences_labels.
        Each instance is associated with a list of cluster memberships, where each membership is represented by 
        a tuple (cluster_id, membership_degree). This allows for soft clustering, where an instance can belong 
        to multiple clusters with varying degrees of membership.

        Parameters:
        -----------
        sentences_labels : list of tuples
            A list where each element is a tuple (instance_id: int, memberships: list of tuples).
            - instance_id: An integer that uniquely identifies an instance.
            - memberships: A list of tuples (cluster_id: int, membership_degree: float).
              - cluster_id: An integer representing the cluster ID.
              - membership_degree: A float between 0 and 1 representing the degree of membership in the cluster.
                A value of 1 indicates full membership, while a value closer to 0 indicates partial membership.

        Examples:
        ---------
        >>> constraints = SubsetFuzzyLabelsClusteringContraints({
        ...     1: [(2, 0.8), (3, 0.2)], 
        ...     2: [(1, 0.5), (2, 0.5)]
        ... })
        >>> constraints.sentences_labels
        {1: [(2, 0.8), (3, 0.2)], 2: [(1, 0.5), (2, 0.5)]}

        """
        super().__init__()
        self.sentences_labels = sentences_labels

        try:
            labels = []
            for value in sentences_labels.values():
                for l in value:
                    labels.append(l[0])
            self.labels = set(labels)
        except:
            self.labels = None


class InstancesLevelClusteringConstraints(ClusteringConstraints):
    pass


class PairwiseInstanceLevelClusteringConstraints(InstancesLevelClusteringConstraints):
    pass


class MustLinkCannotLinkInstanceLevelClusteringConstraints(PairwiseInstanceLevelClusteringConstraints):
    def __init__(self, must_link: list, cannot_link: list, sentences_labels: Optional[dict] = None) -> None:
        """
        Initialize the MustLinkCannotLinkInstanceLevelClusteringConstraints class.

        This class defines pairwise clustering constraints at the instance level, where specific pairs of sentences_labels 
        are either required to be in the same cluster (must-link) or required to be in different clusters (cannot-link).

        Parameters:
        -----------
        must_link : list of tuples
            A list of tuples where each tuple (instance_id_1: int, instance_id_2: int) specifies that the two sentences_labels 
            identified by instance_id_1 and instance_id_2 must be in the same cluster.

        cannot_link : list of tuples
            A list of tuples where each tuple (instance_id_1: int, instance_id_2: int) specifies that the two sentences_labels 
            identified by instance_id_1 and instance_id_2 must be in different clusters.

        Examples:
        ---------
        >>> constraints = MustLinkCannotLinkInstanceLevelClusteringConstraints(
        ...     must_link=[(1, 2), (3, 4)], 
        ...     cannot_link=[(1, 3), (2, 5)],
        ...     labels_true=[0, 0, 1, 1, 2]
        ... )
        >>> constraints.must_link
        [(1, 2), (3, 4)]
        >>> constraints.cannot_link
        [(1, 3), (2, 5)]
        
        """
        super().__init__()
        self.must_link = must_link
        self.cannot_link = cannot_link
        self.sentences_labels = sentences_labels

        if sentences_labels:
            self.labels = sorted(set(sentences_labels.values())) # Get the unique labels and sort them
            label_to_index = {label: index for index, label in enumerate(self.labels)} # Create a mapping from label to its index
            self.sentences_labels = {id: label_to_index[label] for id, label in sentences_labels.items()} # Translate the labels to their indices

    def get_labels(self) -> list:
        return self.labels
    
    def get_k(self) -> int:
        return 0
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        must_link_true, cannot_link_true = get_true_ml_cl(ids_true=ids_true, labels_true=labels_true)

        result = evaluate_must_link_cannot_link(
            must_link_true=must_link_true,
            must_link_pred=self.must_link,
            cannot_link_true=cannot_link_true,
            cannot_link_pred=self.cannot_link,
        )

        if self.sentences_labels:
            number_of_singletons = count_singletons(self.sentences_labels)
            result['number_of_singletons'] = number_of_singletons
        return result

        
class KClusteringContraints(ClusteringConstraints):
    def __init__(self, k: Optional[int] = None, label_names: Optional[list] = None) -> None:
        super().__init__()
        if k:
            self.k = k
        elif label_names:
            self.k = len(set(label_names))
        
        self.label_names = label_names

    def __str__(self) -> str:
        return str(self.k)
    
    def get_labels(self) -> list:
        return None
    
    def get_k(self) -> int:
        return self.k
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        k_pred = self.k
        k_true = len(set(labels_true))
        return dict(
            k = evaluate_k(k_true=k_true, k_pred=k_pred)
        )
