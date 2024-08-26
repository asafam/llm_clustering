class ClusteringConstraints:
    pass


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
        >>> constraints = SubsetHardLabelsClusteringContraints([(1, 2), (3, 4), (5, 2)])
        >>> constraints.instances
        [(1, 2), (3, 4), (5, 2)]
        
        """
        super().__init__()
        self.instances = instances
        self.labels = list(set(instances.values()))


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