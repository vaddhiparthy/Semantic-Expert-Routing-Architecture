"""Model snapshotting utilities.

This module provides a simple interface for saving and loading model
weights and centroid states during training. Snapshots can be used to
resume training or for inspection.

Classes
-------
Snapshotter
    Save and load PyTorch model and centroid state.
"""

from __future__ import annotations

from typing import Dict, Any
import torch


class Snapshotter:
    """Save and restore model and centroid state.

    Parameters
    ----------
    model: torch.nn.Module
        The model whose parameters should be saved or loaded.
    centroids: torch.Tensor
        Tensor of shape ``(n_experts, d)`` containing centroid vectors.
    """

    def __init__(self, model: torch.nn.Module, centroids: torch.Tensor) -> None:
        self.model = model
        self.centroids = centroids

    def save(self, path: str) -> None:
        """Save model parameters and centroids to a file."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "centroids": self.centroids.clone(),
        }
        torch.save(state, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """Load model parameters and centroids from a file."""
        state: Dict[str, Any] = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state["model_state_dict"])  # type: ignore
        self.centroids.copy_(state["centroids"])  # type: ignore