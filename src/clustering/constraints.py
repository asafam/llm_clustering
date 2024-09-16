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
        [{0: 1, 1: 1, 2: 0, 3: 0, 4: 1}]
        
        """
        super().__init__()
        
        self.labels = sorted(set(instances.values())) # Get the unique labels and sort them
        label_to_index = {label: index for index, label in enumerate(self.labels)} # Create a mapping from label to its index
        self.instances = {id: label_to_index[label] for id, label in instances.items()} # Translate the labels to their indices

    def __repr__(self) -> str:
        return str(self.instances)
    
    def get_labels(self) -> list:
        return self.labels
    

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
        ...     cannot_link=[(1, 3), (2, 5)]
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