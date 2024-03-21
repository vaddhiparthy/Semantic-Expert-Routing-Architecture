"""Student router learning to mimic semantic routing.

This module defines the student router, a learnable network that maps
transformer hidden states to routing scores over experts. During
training, the student router is encouraged to approximate the teacher
router via a distillation loss. At inference time, the student router
handles routing once maturity conditions are met.

Classes
-------
StudentRouter
    A simple feedforward network producing expert scores from hidden
    states.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StudentRouter(nn.Module):
    """A learnable router for expert selection.

    Parameters
    ----------
    d_model: int
        Dimension of the transformer hidden states.
    n_experts: int
        Number of experts to produce scores for.

    Notes
    -----
    The router is implemented as a two‑layer MLP with a ReLU activation
    and a final linear layer producing logits over experts. The softmax
    of these logits yields routing probabilities.
    """

    def __init__(self, d_model: int, n_experts: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_model, n_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute routing logits for each token.

        Parameters
        ----------
        hidden_states: torch.Tensor
            Tensor of shape ``(batch_size, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch_size, seq_len, n_experts)``.
        """
        x = self.fc1(hidden_states)
        x = self.act(x)
        logits = self.fc2(x)
        return logits