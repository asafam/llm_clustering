from typing import *
import numpy as np
from clustering.constraints import *
from clustering.optimizations import KOptimization
import logging
import torch
from torch import optim, nn
from clustering.models.base_models import BaseKMeans


class MustLinkCannotLinkKMeans(BaseKMeans):
    def cluster(
            self, 
            X,
            constraint: MustLinkCannotLinkInstanceLevelClusteringConstraints,
            k_optimization: KOptimization, 
            n_clusters: Optional[int] = None,
            min_k: int = 2,
            max_k: int = 10, 
            k_optimization_coarse_step_size: int = 10,
            k_optimization_fine_range: int = 10,
            random_state: int = 42,
            **kwargs
        ):
        P = self._train(X)

        # After training, apply the learned projection matrix P
        X_projected = np.dot(X, P.detach().cpu().numpy())

        return super().cluster(
            X=X_projected.detach().numpy(),
            k_optimization=k_optimization, 
            n_clusters=n_clusters,
            min_k=min_k,
            max_k=max_k, 
            k_optimization_coarse_step_size=k_optimization_coarse_step_size,
            k_optimization_fine_range=k_optimization_fine_range,
            random_state=random_state,
            **kwargs
        )
    
    def _train(self, X, num_epochs = 100, lr=1e-3):
        logger = logging.getLogger('default')
        # Projection matrix P to be learned (D x D)
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
            must_link_distances = torch.stack([torch.norm(X_projected[i] - X_projected[j]) for i, j in self.must_link])
            cannot_link_distances = torch.stack([torch.norm(X_projected[i] - X_projected[j]) for i, j in self.cannot_link])

            # Labels: 0 for must-link (should be closer), 1 for cannot-link (should be farther)
            must_link_labels = torch.zeros(len(self.must_link))
            cannot_link_labels = torch.ones(len(self.cannot_link))

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

