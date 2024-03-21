"""Load balancer for expert routing.

This component computes a penalty term for each expert based on its
current usage. The penalty discourages routing too many tokens to the
same expert, thus encouraging balanced expert utilisation.

Classes
-------
LoadBalancer
    Computes a load penalty vector given expert usage statistics.
"""

from __future__ import annotations

import torch


class LoadBalancer:
    """Compute load penalties for experts.

    Parameters
    ----------
    n_experts: int
        Number of experts.
    beta: float
        Scaling factor for the penalty; higher values enforce more
        aggressive balancing.

    Notes
    -----
    The load penalty is computed as ``beta * (usage / ideal)`` where
    ``usage`` is the number of tokens currently routed to each expert
    and ``ideal`` is the number of tokens divided by the number of
    experts.
    """

    def __init__(self, n_experts: int, beta: float = 1.0) -> None:
        self.n_experts = n_experts
        self.beta = beta

    def __call__(self, usage: torch.Tensor) -> torch.Tensor:
        """Compute the load penalty vector.

        Parameters
        ----------
        usage: torch.Tensor
            Tensor of shape ``(n_experts,)`` containing the current
            token counts per expert.

        Returns
        -------
        torch.Tensor
            Penalty vector of shape ``(n_experts,)``.
        """
        total = usage.sum().item() if usage.numel() > 0 else 0
        ideal = total / max(1, self.n_experts)
        # Avoid division by zero when no tokens are routed
        if total == 0:
            return torch.zeros_like(usage)
        return self.beta * (usage.float() / ideal)