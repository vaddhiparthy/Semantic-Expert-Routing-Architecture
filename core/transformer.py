"""Tiny transformer backbone.

This module defines a minimal transformer architecture designed to
support the ASTRA‑X‑BASE mixture‑of‑experts system. It omits many of
the bells and whistles of large‑scale transformers and focuses on
providing a simple encoder/decoder interface with a configurable
number of layers and hidden dimensions.

Classes
-------
TinyTransformer
    A minimal transformer supporting token embedding and forward
    propagation through stacked self‑attention and feedforward layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    """Configuration for the TinyTransformer.

    Parameters
    ----------
    vocab_size: int
        Size of the vocabulary for token embeddings.
    d_model: int
        Hidden dimension of the model.
    n_layers: int
        Number of transformer layers.
    n_heads: int
        Number of attention heads.
    dropout: float
        Dropout probability.
    """

    vocab_size: int
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1


class TinyTransformer(nn.Module):
    """A small transformer model suitable for experimentation.

    This class provides just enough infrastructure to encode a sequence
    of token indices into hidden states. It deliberately avoids more
    advanced features like adaptive softmax, caching, or weight tying.
    Use this as a starting point for experimentation; for production
    systems, consider swapping in a more robust implementation.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dropout=config.dropout,
            dim_feedforward=4 * config.d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the forward pass of the model.

        Parameters
        ----------
        input_ids: torch.Tensor
            Tensor of shape ``(batch_size, seq_length)`` containing token
            indices.

        Returns
        -------
        torch.Tensor
            Hidden states of shape ``(batch_size, seq_length, d_model)``.
        """
        x = self.embed_tokens(input_ids)
        x = self.encoder(x)
        return self.layer_norm(x)