from sklearn.metrics import precision_score, recall_score, accuracy_score, adjusted_rand_score, normalized_mutual_info_score, v_measure_score
import logging

class ClusteringConstraints:
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def get_labels(self) -> list:
        raise NotImplementedError()


class PartitionsLevelClusteringConstraints(ClusteringConstraints):
    pass


class HardLabelsClusteringContraints(PartitionsLevelClusteringConstraints):
    def __init__(self, instances: list) -> None:
        """
        Initialize the SubsetHardLabelsClusteringContraints class.

        This class is used to define clustering constraints based on hard labels assigned to a subset of instances.
        Each instance is associated with a specific cluster, which is represented by a tuple of (instance_id, cluster_id).

        Parameters:
        -----------
        instances : list of tuples
            A list where each element is a tuple (instance_id: int, cluster_id: int).
            - instance_id: An integer that uniquely identifies an instance.
            - cluster_id: An integer that represents the cluster ID to which the instance is constrained.

        Examples:
        ---------
        >>> constraints = SubsetHardLabelsClusteringContraints(instances={0: 22, 1: 22, 2: 13, 3: 13, 4: 22})
        >>> constraints.instances
        {0: 1, 1: 1, 2: 0, 3: 0, 4: 1}
        
        """
        super().__init__()
        
        self.labels = sorted(set(instances.values())) # Get the unique labels and sort them
        label_to_index = {label: index for index, label in enumerate(self.labels)} # Create a mapping from label to its index
        self.instances = {id: label_to_index[label] for id, label in instances.items()} # Translate the labels to their indices

    def __repr__(self) -> str:
        return str(self.instances)
    
    def get_labels(self) -> list:
        return self.labels
    
    def get_k(self) -> int:
        return len(self.labels)
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        labels_pred_dict = {}
        for id in ids_true:
            labels_pred_dict[id] = -1

        for id, label in self.instances.items():
            if id in labels_pred_dict:
                labels_pred_dict[id] = label
        labels_pred = list(labels_pred_dict.values())

        return dict(
            ari = adjusted_rand_score(labels_true, labels_pred),
            nmi = normalized_mutual_info_score(labels_true, labels_pred),
            v_measure = v_measure_score(labels_true, labels_pred),
        )
    

class FuzzyLabelsClusteringContraints(PartitionsLevelClusteringConstraints):
    def __init__(self, instances: list) -> None:
        """
        Initialize the SubsetFuzzyLabelsClusteringContraints class.

        This class is used to define clustering constraints based on fuzzy labels assigned to a subset of instances.
        Each instance is associated with a list of cluster memberships, where each membership is represented by 
        a tuple (cluster_id, membership_degree). This allows for soft clustering, where an instance can belong 
        to multiple clusters with varying degrees of membership.

        Parameters:
        -----------
        instances : list of tuples
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
        >>> constraints.instances
        {1: [(2, 0.8), (3, 0.2)], 2: [(1, 0.5), (2, 0.5)]}

        """
        super().__init__()
        self.instances = instances

        try:
            labels = []
            for value in instances.values():
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
    def __init__(self, must_link: list, cannot_link: list) -> None:
        """
        Initialize the MustLinkCannotLinkInstanceLevelClusteringConstraints class.

        This class defines pairwise clustering constraints at the instance level, where specific pairs of instances 
        are either required to be in the same cluster (must-link) or required to be in different clusters (cannot-link).

        Parameters:
        -----------
        must_link : list of tuples
            A list of tuples where each tuple (instance_id_1: int, instance_id_2: int) specifies that the two instances 
            identified by instance_id_1 and instance_id_2 must be in the same cluster.

        cannot_link : list of tuples
            A list of tuples where each tuple (instance_id_1: int, instance_id_2: int) specifies that the two instances 
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

    def get_labels(self) -> list:
        return []
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        # Create ground truth pairs based on the true labels
        true_must_link = []
        true_cannot_link = []

        # Compare all pairs and decide if they should be must-link or cannot-link based on true labels
        for i in range(len(ids_true)):
            for j in range(i + 1, len(ids_true)):
                if labels_true[i] == labels_true[j]:
                    true_must_link.append((ids_true[i], ids_true[j]))
                else:
                    true_cannot_link.append((ids_true[i], ids_true[j]))

        # Convert must-link and cannot-link predictions to binary labels for evaluation
        # We will check if the predicted pairs match the true must-link or cannot-link pairs
        y_true = []
        y_pred = []

        # Evaluate must-link predictions
        print("self.must_link = " , self.must_link)
        for i, j in self.must_link:
            y_true.append(1 if (i, j) in true_must_link else 0)
            y_pred.append(1)

        # Evaluate cannot-link predictions
        for i, j in self.cannot_link:
            y_true.append(0 if (i, j) in true_cannot_link else 1)
            y_pred.append(0)

        # Compute precision, recall, and accuracy
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        result = dict(
            precision = precision,
            recall = recall,
            accuracy = accuracy
        )
        return result


class KClusteringContraints(ClusteringConstraints):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def __repr__(self) -> str:
        return str(self.k)
    
    def get_labels(self) -> list:
        return None
    
    def get_k(self) -> int:
        return self.k
    
    def evaluate(self, ids_true: list, labels_true: list) -> dict:
        predicted_k = self.k
        true_k = len(set(labels_true))

        return dict(
            absolute_difference = abs(predicted_k - true_k),
            relative_error = abs(predicted_k - true_k) / true_k * 100,
            normalized_absolute_error = abs(predicted_k - true_k) / true_k,
            squared_error = (predicted_k - true_k) ** 2,
        )
    

