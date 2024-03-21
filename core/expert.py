"""Expert module definitions.

This module defines the expert neural network used in the MoE layer.
Each expert is implemented as a simple feedforward neural network with
two linear layers and an activation function. The experts are
intentionally lightweight and designed to be specialised on a
particular subset of the input space.

Classes
-------
ExpertMLP
    A basic two‑layer feedforward network used as an expert.
"""

import torch
import torch.nn as nn


class ExpertMLP(nn.Module):
    """A simple MLP expert.

    Parameters
    ----------
    input_dim: int
        Dimension of the input features.
    hidden_dim: int
        Dimension of the hidden layer. Typically a multiple of ``input_dim``.
    output_dim: int
        Dimension of the output features. Should match the input dim when
        used in a transformer layer.

    Notes
    -----
    The activation function used is GELU, which is common in transformer
    architectures. This class is intentionally simple; in practice,
    additional normalisation and dropout may be beneficial.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x