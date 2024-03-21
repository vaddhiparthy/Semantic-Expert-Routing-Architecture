"""Configuration structures for training.

This module defines dataclasses capturing default values for training
hyperparameters and dataset configuration. Users may instantiate
``TrainingConfig`` and customise attributes before passing it to
``train()``.

Classes
-------
TrainingConfig
    Encapsulates common training hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Simple container for training hyperparameters."""

    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 1
    clip_grad: float = 1.0
    alpha_start: float = 1.0
    alpha_end: float = 0.0
    alpha_decay_steps: int = 10_000
    top_k: int = 1
    beta: float = 1.0