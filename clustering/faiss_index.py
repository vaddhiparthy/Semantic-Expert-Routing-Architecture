"""FAISS index wrapper.

This module provides a thin wrapper around FAISS (Facebook AI Similarity
Search) to build and query an approximate nearest neighbor index. For
deployment or small‑scale experiments without FAISS installed, the
implementation can fall back to brute‑force nearest neighbour
computation using PyTorch.

Classes
-------
FaissIndex
    Build and query a vector index for MeaningSpace centroids.
"""

from __future__ import annotations

from typing import Tuple
import torch

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FAISS_AVAILABLE = False


class FaissIndex:
    """FAISS index for nearest neighbour search.

    If FAISS is unavailable, the class falls back to a simple
    brute‑force search using matrix multiplication.
    """

    def __init__(self, d: int, nlist: int = 100) -> None:
        self.d = d
        self.nlist = nlist
        self.index = None

    def build(self, vectors: torch.Tensor) -> None:
        """Build the index from a set of centroids.

        Parameters
        ----------
        vectors: torch.Tensor
            Tensor of shape ``(n_vectors, d)`` representing the
            centroids.
        """
        if _FAISS_AVAILABLE:
            quantizer = faiss.IndexFlatL2(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
            self.index.train(vectors.cpu().numpy())
            self.index.add(vectors.cpu().numpy())
        else:
            # fallback: store the vectors for brute force
            self.index = vectors

    def search(self, queries: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the index for nearest neighbours.

        Parameters
        ----------
        queries: torch.Tensor
            Tensor of shape ``(n_queries, d)`` containing query
            vectors.
        k: int, optional
            Number of nearest neighbours to retrieve, by default 1.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(indices, distances)`` where indices is a tensor of shape
            ``(n_queries, k)`` and distances has the same shape.
        """
        if _FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(queries.cpu().numpy(), k)
            return (
                torch.from_numpy(indices).to(queries.device),
                torch.from_numpy(distances).to(queries.device),
            )
        else:
            # brute force: compute distances manually
            # queries: [nq, d], index: [nv, d]
            index_vectors = self.index  # type: ignore
            dist = (
                (queries[:, None, :] - index_vectors[None, :, :]) ** 2
            ).sum(dim=2)
            scores, idx = dist.topk(k, largest=False)  # smaller distance is better
            return idx, scores