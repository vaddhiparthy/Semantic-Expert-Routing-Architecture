# ASTRA‑X‑BASE

ASTRA‑X‑BASE is a research‑grade library implementing a semantic mixture‑of‑experts architecture designed for small‑compute environments. The system combines a lightweight transformer backbone with a teacher–student router and dynamic clustering to achieve specialised expert routing on modest hardware.

## Features

- **Tiny Transformer Backbone** – A minimal transformer implementation that processes input tokens into hidden states.
- **Mixture‑of‑Experts Layer** – Dispatches tokens to specialised MLP experts according to routing decisions.
- **MeaningSpace Encoder** – Produces semantic vectors used by the teacher router for early, stable routing.
- **Teacher/Student Routers** – The teacher router uses semantic distances, while the student router learns to mimic the teacher and eventually takes over routing.
- **Hybrid Routing** – Combines teacher and student scores with load penalties to avoid expert overload.
- **Dynamic Clustering** – Maintains centroids of MeaningSpace clusters via FAISS and reclustering.
- **Split/Merge Experts** – Supports splitting overloaded experts and merging underutilised ones (not fully implemented here but planned).
- **Observability** – Logs metrics, traces routing decisions, and snapshots model and centroids for analysis.
- **Simple Training Loop** – A reference training loop demonstrating how to wire together the components.

## Usage

This repository is a skeleton designed for experimentation. To train the system on your own data, instantiate the components from `astra_x_base` and call the training loop:

```python
from astra_x_base.training import create_dummy_dataloader, TrainingConfig, train

config = TrainingConfig(batch_size=8, num_epochs=1)
dataloader = create_dummy_dataloader(batch_size=8, seq_len=16, vocab_size=100, num_batches=10)
metrics = train(dataloader, vocab_size=100, n_experts=4, config=config)
print(metrics.summary())
```

For real training, replace `create_dummy_dataloader` with a dataloader that yields tokenised inputs and targets from your dataset.

## Development

This codebase is modular. Key subpackages:

- `astra_x_base/core` – Core model components: transformer, MoE layer and experts.
- `astra_x_base/routing` – Teacher, student and hybrid routers plus the load balancer.
- `astra_x_base/clustering` – Centroid management and FAISS wrapper.
- `astra_x_base/observe` – Logging and snapshotting utilities.
- `astra_x_base/training` – Training loop, configuration and data loading utilities.

### Note

This skeleton is intended for research and educational purposes. Certain features (e.g. full FAISS integration, sophisticated scheduling policies and expert split/merge logic) are simplified or stubbed. Use this as a starting point for your own experiments; feel free to extend, refactor and improve the components.