"""Main training loop for ASTRA‑X‑BASE.

This function wires together the components from ``core``, ``routing``,
``clustering``, and ``observe`` to perform a simple training run. It
illustrates how to combine the base transformer, expert modules,
routers, clustering, and load balancing into an end‑to‑end system.

Functions
---------
train
    Run a training epoch over a dataset.
"""

from __future__ import annotations

from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

from ..core import TinyTransformer, MoELayer, ExpertMLP
from ..routing import TeacherRouter, StudentRouter, HybridRouter, LoadBalancer
from ..clustering import CentroidManager
from ..observe import MetricsLogger
from .configs import TrainingConfig


def train(
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    n_experts: int,
    config: TrainingConfig,
) -> MetricsLogger:
    """Train the ASTRA‑X‑BASE system on a dataset.

    Parameters
    ----------
    dataloader: Iterable
        An iterable yielding ``(input_ids, target_ids)`` batches.
    vocab_size: int
        Vocabulary size for the transformer.
    n_experts: int
        Number of experts.
    config: TrainingConfig
        Hyperparameters controlling the training process.

    Returns
    -------
    MetricsLogger
        Contains logged training metrics.
    """
    metrics = MetricsLogger()

    # Build tiny transformer
    model_cfg = TinyTransformer(
        config=__import__("astra_x_base.core.transformer", fromlist=["TransformerConfig"]).TransformerConfig(vocab_size=vocab_size)
    )
    backbone = model_cfg

    # Create experts
    experts = [ExpertMLP(backbone.config.d_model, 4 * backbone.config.d_model, backbone.config.d_model) for _ in range(n_experts)]
    moe = MoELayer(experts)

    # Centroids initialised to zero
    centroid_manager = CentroidManager(n_experts, dim=backbone.config.d_model)
    teacher_router = TeacherRouter(centroid_manager.centroids, top_k=config.top_k)
    student_router = StudentRouter(backbone.config.d_model, n_experts)
    load_balancer = LoadBalancer(n_experts, beta=config.beta)
    hybrid_router = HybridRouter(alpha=config.alpha_start)

    # Loss functions and optimiser
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(
        list(backbone.parameters())
        + list(moe.parameters())
        + list(student_router.parameters()),
        lr=config.learning_rate,
    )

    total_tokens = 0
    for step, (input_ids, target_ids) in enumerate(dataloader):
        optimizer.zero_grad()
        total_tokens += input_ids.numel()
        # Forward through backbone
        hidden_states = backbone(input_ids)
        # Get semantic vectors for each token (use hidden_states mean as fallback)
        semantic_vecs = hidden_states  # placeholder for real semantic encoder
        # Teacher routing
        teacher_idx, teacher_scores = teacher_router(semantic_vecs)
        # Student routing logits
        student_logits = student_router(hidden_states)
        # Compute usage counts for load penalty (flatten tokens)
        flat_idx = teacher_idx.view(-1)
        usage = torch.bincount(flat_idx, minlength=n_experts)
        load_penalty = load_balancer(usage)
        # Combine scores
        final_scores = hybrid_router(
            teacher_scores.squeeze(-1), student_logits, load_penalty
        )
        # Select experts (greedy top‑k)
        chosen_experts = final_scores.argmax(dim=-1)
        # Dispatch to experts
        expert_output = moe(hidden_states, chosen_experts)
        # Compute cross entropy over vocabulary (language modelling)
        logits = expert_output @ backbone.embed_tokens.weight.t()
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], config.clip_grad)
        optimizer.step()
        metrics.log("loss", loss.item())
        # Update alpha schedule
        if config.alpha_decay_steps > 0:
            step_frac = min(1.0, step / config.alpha_decay_steps)
            hybrid_router.alpha = config.alpha_start + step_frac * (config.alpha_end - config.alpha_start)
    return metrics