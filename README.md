# Semantic Mixture-of-Experts

This repository contains a PyTorch research scaffold for experimenting with semantic routing in a small mixture-of-experts language-model architecture.

## Scope

The code is a compact experimental implementation. It includes a tiny transformer backbone, MLP experts, semantic and student routers, load balancing, centroid management, observability helpers, and a reference training loop over dummy token batches.

It is not a production inference server and does not include pretrained weights, benchmark results, or a full dataset pipeline.

## Implemented Components

| Area | Implementation |
| --- | --- |
| Backbone | `core/transformer.py` defines a small Transformer encoder |
| Experts | `core/expert.py` and `core/moe_layer.py` define MLP experts and token dispatch |
| Teacher router | `routing/teacher_router.py` routes by distance to semantic centroids |
| Student router | `routing/student_router.py` predicts expert logits from hidden states |
| Hybrid router | `routing/hybrid_router.py` blends teacher scores, student logits, and load penalties |
| Load balancing | `routing/load_balancer.py` penalizes overloaded experts |
| Clustering | `clustering/centroid_manager.py`, `clustering/faiss_index.py`, and `clustering/recluster.py` manage centroids and nearest-neighbor lookup |
| Training loop | `training/train_loop.py` wires the backbone, routers, experts, and loss function together |
| Observability | `observe/metrics.py`, `observe/trace.py`, `observe/snapshots.py`, and `observe/visualizer.py` capture training and routing state |

## Architecture

```text
token ids
  -> tiny transformer
  -> hidden states
  -> teacher router from semantic centroids
  -> student router from hidden states
  -> load penalty
  -> hybrid routing scores
  -> selected expert per token
  -> expert output
  -> language-model loss
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Minimal Example

```python
from astra_x_base.training import TrainingConfig, create_dummy_dataloader, train

config = TrainingConfig(batch_size=8, num_epochs=1, top_k=1)
dataloader = create_dummy_dataloader(
    batch_size=8,
    seq_len=16,
    vocab_size=100,
    num_batches=10,
)
metrics = train(dataloader, vocab_size=100, n_experts=4, config=config)
print(metrics.summary())
```

## Validation

Static validation:

```powershell
python -m compileall .
```

Training smoke validation after installing PyTorch:

```powershell
python - <<'PY'
from astra_x_base.training import TrainingConfig, create_dummy_dataloader, train

config = TrainingConfig(batch_size=2, num_epochs=1, top_k=1)
dataloader = create_dummy_dataloader(batch_size=2, seq_len=4, vocab_size=32, num_batches=1)
metrics = train(dataloader, vocab_size=32, n_experts=2, config=config)
print(metrics.summary())
PY
```

## Current Limits

- The semantic encoder is represented by hidden states in the reference loop.
- Expert split/merge policies are not implemented as a complete lifecycle.
- FAISS is optional; the vector index falls back to brute-force PyTorch distance search.
- The reference dataloader uses dummy token IDs.
- No benchmark numbers are claimed in this repository.
