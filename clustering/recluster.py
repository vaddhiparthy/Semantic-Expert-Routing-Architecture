"""Reclustering utilities.

This module exposes a helper function to reassign centroids when
distribution of MeaningSpace vectors drifts significantly. The
implementation here is intentionally simple: it performs a single
round of K‑means on a provided batch of vectors and assigns new
centroids accordingly.

Functions
---------
reassign_centroids
    Compute new centroids from a set of vectors using K‑means.
"""

from __future__ import annotations

import torch


def reassign_centroids(vectors: torch.Tensor, n_clusters: int) -> torch.Tensor:
    """Compute new centroids via K‑means clustering.

    Parameters
    ----------
    vectors: torch.Tensor
        Tensor of shape ``(n_samples, d)`` containing data points.
    n_clusters: int
        Desired number of centroids.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_clusters, d)`` containing the new
        centroids.
    """
    # Randomly initialise centroids by sampling from vectors
    indices = torch.randperm(vectors.size(0))[:n_clusters]
    centroids = vectors[indices].clone()
    # Perform a small number of K‑means iterations
    for _ in range(5):
        # Assign vectors to nearest centroid
        distances = (
            (vectors[:, None, :] - centroids[None, :, :]) ** 2
        ).sum(dim=2)
        assignments = distances.argmin(dim=1)
        # Update centroids
        for c in range(n_clusters):
            mask = assignments == c
            if mask.any():
                centroids[c] = vectors[mask].mean(dim=0)
    return centroids