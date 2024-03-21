"""Semantic teacher router.

This router implements routing based on distances between token
embeddings in the MeaningSpace and a set of cluster centroids. The
router does not learn during training; instead, it relies on the
pretrained semantic encoder and the current centroids maintained by
the clustering subsystem.

Classes
-------
TeacherRouter
    Computes semantic routing scores and selects top‑K experts.
"""

from __future__ import annotations

from typing import Tuple, List
import torch


class TeacherRouter:
    """Compute routing based on semantic distances.

    Parameters
    ----------
    centroids: torch.Tensor
        Tensor of shape ``(n_experts, d)`` representing the current
        centroids in MeaningSpace. The dimension ``d`` must match the
        output dimension of the MeaningSpace encoder.
    top_k: int
        Number of experts to return for each token. Typically 1 or 2.

    Notes
    -----
    This class does not modify the centroids; updating centroids is
    handled by the clustering subsystem. Routing outputs are indices
    into the centroids tensor.
    """

    def __init__(self, centroids: torch.Tensor, top_k: int = 1) -> None:
        self.centroids = centroids
        self.top_k = top_k

    def __call__(self, semantic_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute nearest experts for each semantic vector.

        Parameters
        ----------
        semantic_vectors: torch.Tensor
            Tensor of shape ``(batch_size, seq_len, d)`` containing
            MeaningSpace vectors for each token.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple ``(indices, scores)`` where ``indices`` is a
            ``(batch_size, seq_len, top_k)`` tensor of expert indices and
            ``scores`` contains the corresponding negative squared
            distances.
        """
        # Flatten tokens for vectorised computation
        bsz, seqlen, dim = semantic_vectors.shape
        flat = semantic_vectors.view(bsz * seqlen, dim)

        # Compute squared distances to centroids
        # Use (a - b)^2 = a^2 - 2ab + b^2; precompute squared norms
        centroid_norms = (self.centroids ** 2).sum(dim=1)
        vector_norms = (flat ** 2).sum(dim=1, keepdim=True)
        scores = -(
            vector_norms - 2 * flat @ self.centroids.t() + centroid_norms
        )  # higher is closer

        # Top‑k selection
        top_scores, top_indices = scores.topk(self.top_k, dim=1)
        top_indices = top_indices.view(bsz, seqlen, self.top_k)
        top_scores = top_scores.view(bsz, seqlen, self.top_k)
        return top_indices, top_scores