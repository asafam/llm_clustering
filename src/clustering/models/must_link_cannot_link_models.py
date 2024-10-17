from typing import *
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel
import torch
from torch import optim, nn
from clustering.constraints import *
from clustering.optimizations import KOptimization
from clustering.models.base_models import *
from clustering.models.lib.cop_kmeans import cop_kmeans
import logging

class MustLinkCannotLinkClusteringModel(KHyperparamClusteringModel):
    def __init__(self, ml_cl_constraint_mask: Optional[dict] = None) -> None:
        super().__init__()
        self.ml_cl_constraint_mask = ml_cl_constraint_mask if ml_cl_constraint_mask is not None else {}

    def __str__(self) -> str:
        if self.ml_cl_constraint_mask: 
            return f"{self.__class__.__name__}({self.ml_cl_constraint_mask})"
        else:
            return super().__str__()

    def _cluster(
            self, 
            X,
            ids: list,
            constraint: ClusteringConstraints,
            n_clusters: Optional[int] = None,
            random_state: int = 42, 
            gamma=1.0,
            verbose: bool = True,
            **kwargs
        ):
        logger = logging.getLogger('default')
        logger.info(f"Clustering model: {type(self).__name__}")

        # Get the ML and CL constraints
        must_link, cannot_link = constraint.get_ml_cl()
        must_link = must_link if self.ml_cl_constraint_mask.get('must_link', True) else []
        cannot_link = cannot_link if self.ml_cl_constraint_mask.get('cannot_link', True) else []

        # Step 1: Compute the RBF (Gaussian) affinity matrix
        affinity = rbf_kernel(X, gamma=gamma)

        # Step 2: Apply must-link and cannot-link constraints to the affinity matrix
        for i, j in must_link:
            affinity[i, j] = affinity[j, i] = 1.0  # set similarity to 1 between must-linked points

        for i, j in cannot_link:
            affinity[i, j] = affinity[j, i] = 0.0  # set similarity to 0 between cannot-linked points

        # Step 3: Perform spectral clustering with the modified affinity matrix
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=random_state, verbose=verbose)
        labels = spectral.fit_predict(affinity)

        return labels
        

class MustLinkCannotLinkKMeans(BaseKMeans):
    def __init__(self, ml_cl_constraint_mask: Optional[dict] = None) -> None:
        super().__init__()
        self.ml_cl_constraint_mask = ml_cl_constraint_mask if ml_cl_constraint_mask is not None else {}
    
    def __str__(self) -> str:
        if self.ml_cl_constraint_mask: 
            return f"{self.__class__.__name__}(ml_cl_constraint_mask={self.ml_cl_constraint_mask})"
        else:
            return super().__str__()

    def cluster(
            self, 
            X,
            ids: list,
            constraint: ClusteringConstraints,
            k_optimization: KOptimization, 
            n_clusters: Optional[int] = None,
            min_k: int = 2,
            max_k: int = 10, 
            k_optimization_coarse_step_size: int = 10,
            k_optimization_fine_range: int = 10,
            num_epochs: int = 200,
            random_state: int = 42,
            **kwargs
        ):
        logger = logging.getLogger('default')
        logger.info(f"Clustering model: {type(self).__name__}")

        must_link, cannot_link = constraint.get_ml_cl()
        must_link = must_link if self.ml_cl_constraint_mask.get('must_link', True) else []
        cannot_link = cannot_link if self.ml_cl_constraint_mask.get('cannot_link', True) else []

        # Train similarity weight matrix
        P = self._train(X, must_link=must_link, cannot_link=cannot_link, num_epochs=num_epochs)

        # After training, apply the learned projection matrix P
        X_projected = np.dot(X, P.detach().cpu().numpy())

        return super().cluster(
            X=X_projected,
            k_optimization=k_optimization, 
            n_clusters=n_clusters,
            min_k=min_k,
            max_k=max_k, 
            k_optimization_coarse_step_size=k_optimization_coarse_step_size,
            k_optimization_fine_range=k_optimization_fine_range,
            random_state=random_state,
            **kwargs
        )
    
    def _train(self, X, must_link, cannot_link, num_epochs = 200, lr=1e-3):
        logger = logging.getLogger('default')
        # Projection matrix P to be learned (D x D)
        X = torch.from_numpy(X).float()
        embedding_dim = X.shape[1]
        P = torch.randn(embedding_dim, embedding_dim, requires_grad=True)
        
        contrastive_loss = ContrastiveLoss(margin=1.0) # Loss function
        optimizer = optim.Adam([P], lr=lr) # Optimizer

        # Training loop       
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Compute the projected embeddings
            X_projected = torch.matmul(X, P)

            # Compute distances for must-link and cannot-link pairs
            must_link_distances = torch.stack([torch.norm(X_projected[i] - X_projected[j]) for i, j in must_link])
            cannot_link_distances = torch.stack([torch.norm(X_projected[i] - X_projected[j]) for i, j in cannot_link])

            # Labels: 0 for must-link (should be closer), 1 for cannot-link (should be farther)
            must_link_labels = torch.zeros(len(must_link))
            cannot_link_labels = torch.ones(len(cannot_link))

            # Combine distances and labels
            distances = torch.cat([must_link_distances, cannot_link_distances])
            labels = torch.cat([must_link_labels, cannot_link_labels])

            # Compute contrastive loss
            loss = contrastive_loss(distances, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.debug(f"Must-link cannot-link train at epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

        return P


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss: minimize distance for must-link pairs and maximize for cannot-link pairs.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        # label=1 for must-link (should be closer), label=0 for cannot-link (should be farther)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +  # Must-link: minimize distance
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))  # Cannot-link: maximize distance
        return loss        


class MustLinkCannotLinkCOPKMeans(BaseKMeans):
    def __init__(self, ml_cl_constraint_mask: dict = {}) -> None:
        super().__init__()
        self.ml_cl_constraint_mask = ml_cl_constraint_mask

    def _cluster(
            self,
            X, 
            n_clusters: int, 
            constraint: ClusteringConstraints, 
            k_optimization: KOptimization,
            tol: float = 1e-4, 
            random_state: int = 42,
            **kwargs
        ):
        must_link = constraint.must_link if self.ml_cl_constraint_mask.get('must_link', True) else []
        cannot_link = constraint.cannot_link if self.ml_cl_constraint_mask.get('cannot_link', True) else []
        labels, centers = cop_kmeans(X, k=n_clusters, ml=must_link, cl=cannot_link, tol=tol, random_state=random_state)
        score = k_optimization.score(X=X, labels=labels, centers=centers)
        return labels, None, score