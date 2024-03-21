"""Hybrid router combining semantic and student scores.

This module implements the logic for combining the teacher (semantic)
router scores with the student router's logits and a load penalty to
produce final routing decisions. It supports a gradually decreasing
``alpha`` parameter controlling the blend between teacher and student.

Classes
-------
HybridRouter
    Combines teacher and student outputs with a load penalty.
"""

from __future__ import annotations

from typing import Tuple
import torch


class HybridRouter:
    """Blend semantic and student scores for routing.

    Parameters
    ----------
    alpha: float
        Weight for the teacher router; ``1`` means full teacher,
        ``0`` means full student.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(
        self,
        teacher_scores: torch.Tensor,
        student_logits: torch.Tensor,
        load_penalty: torch.Tensor,
    ) -> torch.Tensor:
        """Combine teacher, student, and load penalty into final scores.

        Parameters
        ----------
        teacher_scores: torch.Tensor
            Tensor of shape ``(batch, seq, n_experts)`` containing
            negative distances (higher is better) from the semantic router.
        student_logits: torch.Tensor
            Tensor of shape ``(batch, seq, n_experts)`` with logits from
            the student router.
        load_penalty: torch.Tensor
            Tensor of shape ``(n_experts,)`` with penalty values.

        Returns
        -------
        torch.Tensor
            Combined scores of shape ``(batch, seq, n_experts)``.
        """
        # Ensure penalty has correct shape
        penalty = load_penalty.view(1, 1, -1)
        combined = self.alpha * teacher_scores + (1 - self.alpha) * student_logits - penalty
        return combined