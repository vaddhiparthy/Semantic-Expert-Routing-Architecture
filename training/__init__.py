"""Training utilities for ASTRA‑X‑BASE.

This package contains functions and classes for orchestrating
model training, including data loading, optimisation, configuration
defaults, and the main training loop. It is intentionally simple and
serves as a reference implementation.
"""

from .train_loop import train
from .configs import TrainingConfig

__all__ = [
    "train",
    "TrainingConfig",
]