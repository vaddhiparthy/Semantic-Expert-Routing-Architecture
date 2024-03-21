"""Minimal data loader utilities.

This module defines placeholder functions for preparing batches of
tokenised inputs and targets. In practice, you would use a dataset
object or an external library such as HuggingFace Datasets to supply
training data. The functions here are intentionally minimal for
demonstration purposes.

Functions
---------
create_dummy_dataloader
    Create an iterable over dummy token IDs and targets.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import torch


def create_dummy_dataloader(batch_size: int, seq_len: int, vocab_size: int, num_batches: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Return a generator yielding dummy inputs and targets.

    Parameters
    ----------
    batch_size: int
        Number of samples per batch.
    seq_len: int
        Length of each input sequence.
    vocab_size: int
        Maximum token index (exclusive).
    num_batches: int
        Number of batches to yield.

    Returns
    -------
    Iterable[Tuple[torch.Tensor, torch.Tensor]]
        A generator producing ``(input_ids, targets)`` tuples.
    """
    for _ in range(num_batches):
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        yield inputs, targets