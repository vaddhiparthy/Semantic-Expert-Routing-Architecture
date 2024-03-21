"""Mixture‑of‑Experts (MoE) layer.

This module defines a generic mixture‑of‑experts (MoE) layer that can
be plugged into a transformer model. It supports routing tokens to a
set of expert modules based on routing scores. The routing logic itself
is provided externally (via the routing package); this layer simply
dispatches the inputs to the selected experts and combines their
outputs.

Classes
-------
MoELayer
    Implements a basic MoE mechanism for a single transformer layer.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

from .expert import ExpertMLP


class MoELayer(nn.Module):
    """A simple MoE layer.

    Parameters
    ----------
    experts: List[ExpertMLP]
        A list of expert modules. Each expert must expose a callable
        interface returning a tensor of the same shape as its input.

    Notes
    -----
    This layer assumes that routing decisions have already been
    computed. The ``forward`` method expects two tensors: the input
    hidden states and a tensor of expert indices indicating which
    expert should handle each token in the batch.
    """

    def __init__(self, experts: List[ExpertMLP]) -> None:
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_states: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Dispatch hidden states to experts and combine outputs.

        Parameters
        ----------
        hidden_states: torch.Tensor
            Tensor of shape ``(batch_size, seq_length, d_model)``.
        expert_indices: torch.Tensor
            Long tensor of shape ``(batch_size, seq_length)`` with values in
            ``[0, len(experts) - 1]`` indicating which expert processes each
            token.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, seq_length, d_model)`` containing
            the combined outputs of the experts.
        """
        # Flatten the first two dims for easier indexing
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(bsz * seq_len, hidden_dim)
        flat_indices = expert_indices.view(-1)

        # Preallocate output tensor
        out = torch.zeros_like(flat)

        # Dispatch tokens to each expert
        for idx, expert in enumerate(self.experts):
            mask = flat_indices == idx
            if mask.any():
                subset = flat[mask]
                out[mask] = expert(subset)

        return out.view(bsz, seq_len, hidden_dim)