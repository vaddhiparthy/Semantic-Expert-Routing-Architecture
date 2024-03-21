"""Clustering utilities for ASTRA‑X‑BASE.

The ``clustering`` package provides utilities for maintaining and
querying clusters in the MeaningSpace. These include:

* A wrapper around FAISS for indexing and retrieving nearest
  centroids.
* A centroid manager for updating cluster centroids based on
  incoming data.
* Functions to trigger reclustering when cluster quality degrades.
"""

from .faiss_index import FaissIndex
from .centroid_manager import CentroidManager
from .recluster import reassign_centroids

__all__ = [
    "FaissIndex",
    "CentroidManager",
    "reassign_centroids",
]