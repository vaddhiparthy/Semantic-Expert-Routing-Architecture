"""Centroid manager.

This module provides a simple utility to maintain centroid vectors in
MeaningSpace. It updates centroids using exponential moving averages
(EMA) and provides accessors for reading the current centroid matrix.

Classes
-------
CentroidManager
    Maintains and updates centroids via EMA.
"""

from __future__ import annotations

import torch


class CentroidManager:
    """Maintain centroids and update them with EMA.

    Parameters
    ----------
    n_experts: int
        Number of centroids/experts.
    dim: int
        Dimensionality of the centroids.
    momentum: float
        EMA momentum factor. Values closer to 0 yield faster updates.
    """

    def __init__(self, n_experts: int, dim: int, momentum: float = 0.1) -> None:
        self.centroids = torch.zeros(n_experts, dim)
        self.momentum = momentum

    def update(self, assignments: torch.Tensor, vectors: torch.Tensor) -> None:
        """Update centroids using EMA.

        Parameters
        ----------
        assignments: torch.Tensor
            Long tensor of shape ``(n_vectors,)`` containing the index
            of the centroid for each vector.
        vectors: torch.Tensor
            Tensor of shape ``(n_vectors, dim)`` containing the vectors
            used to update the centroids.
        """
        for idx in range(self.centroids.size(0)):
            mask = assignments == idx
            if mask.any():
                avg = vectors[mask].mean(dim=0)
                self.centroids[idx] = (
                    self.momentum * avg + (1 - self.momentum) * self.centroids[idx]
                )