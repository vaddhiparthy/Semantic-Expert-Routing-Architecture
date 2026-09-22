# Semantic Expert Routing Architecture

A compact PyTorch research scaffold for experimenting with semantic routing in a
mixture-of-experts language model. It contains a small transformer backbone, MLP
experts, a teacher/student/hybrid router stack, centroid management, load
balancing, and observability helpers, wired together by a reference training
loop over dummy token batches.

The importable package namespace is `astra_x_base`; `pyproject.toml` maps it
onto the repository root.

There are no pretrained weights, no dataset pipeline, and no benchmark results
in this repository.

## How It Works

```text
token ids
  -> TinyTransformer backbone            -> hidden states
  -> TeacherRouter (distance to centroids) -> per-token expert scores
  -> StudentRouter (learned MLP)           -> per-token expert logits
  -> LoadBalancer (usage penalty)          -> per-expert penalty
  -> HybridRouter (alpha blend - penalty)  -> final routing scores
  -> MoELayer dispatch to the argmax expert
  -> tied-embedding projection -> cross-entropy loss
```

**Routing.** `TeacherRouter` scores each token by the negative squared distance
from its semantic vector to every centroid and returns the top-k experts. It has
no parameters and never updates the centroids. `StudentRouter` is a two-layer
MLP (`Linear -> ReLU -> Linear`) that predicts expert logits from hidden states
and is trained jointly with the rest of the model. `HybridRouter` blends the two
with a scalar `alpha` and subtracts the load penalty; the training loop decays
`alpha` linearly from `alpha_start` to `alpha_end` over `alpha_decay_steps`, so
routing shifts from teacher-driven to student-driven during a run.

**Load balancing.** `LoadBalancer` converts per-expert token counts into
`beta * (usage / ideal)`, where `ideal` is total tokens divided by expert count,
and returns zeros when no tokens have been routed.

**Clustering.** `CentroidManager` holds an `(n_experts, dim)` centroid tensor
initialised to zeros and updates it with an exponential moving average over
assigned vectors. `reassign_centroids` runs five k-means iterations from a random
sample for periodic re-clustering. `FaissIndex` wraps an IVF-flat FAISS index and
falls back to brute-force PyTorch distance search when FAISS is not importable.

**Experts.** `ExpertMLP` is a feedforward block; `MoELayer` takes precomputed
per-token expert indices, masks the flattened token batch per expert, and
reassembles the outputs.

**Observability.** `MetricsLogger` accumulates named scalar series with JSON
export, `TraceLogger` records arbitrary per-step dictionaries, and `Snapshotter`
saves and restores model state plus centroids. Nothing in `observe` draws
figures. The reference loop uses `MetricsLogger` only; the trace and snapshot
helpers are standalone and must be called directly.

## Configuration

`TrainingConfig` (`training/configs.py`) holds the hyperparameters:

| Field | Default | Purpose |
| --- | --- | --- |
| `batch_size` | `16` | Batch size for the dummy dataloader |
| `learning_rate` | `3e-4` | Adam learning rate |
| `num_epochs` | `1` | Epoch count |
| `clip_grad` | `1.0` | Gradient-norm clip |
| `alpha_start` | `1.0` | Initial teacher weight in the hybrid router |
| `alpha_end` | `0.0` | Final teacher weight |
| `alpha_decay_steps` | `10000` | Steps over which `alpha` decays |
| `top_k` | `1` | Experts returned per token by the teacher router |
| `beta` | `1.0` | Load-penalty scale |

`TransformerConfig` (`core/transformer.py`) defaults to `d_model=256`,
`n_layers=2`, `n_heads=4`, `dropout=0.1`, with feedforward width `4 * d_model`.
The training loop constructs the backbone with these defaults and only overrides
`vocab_size`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`faiss-cpu` is declared only for non-Windows platforms. On Windows the index
falls back to brute-force search.

## Run

The reference loop currently requires `top_k` to equal the number of experts
(see Limitations). A run with four experts:

```python
from astra_x_base.training import TrainingConfig, create_dummy_dataloader, train

config = TrainingConfig(batch_size=8, num_epochs=1, top_k=4)
dataloader = create_dummy_dataloader(
    batch_size=8,
    seq_len=16,
    vocab_size=100,
    num_batches=10,
)
metrics = train(dataloader, vocab_size=100, n_experts=4, config=config)
print(metrics.summary())
```

`create_dummy_dataloader` yields random `(input_ids, targets)` token tensors, so
the loss value carries no meaning beyond confirming that the graph runs.

## Testing

There is no automated test suite. The available checks are a syntax pass and the
training smoke run above:

```powershell
python -m compileall .
```

## Repository Layout

| Path | Contents |
| --- | --- |
| `core/transformer.py` | `TransformerConfig` and the `TinyTransformer` encoder |
| `core/expert.py` | `ExpertMLP` feedforward expert |
| `core/moe_layer.py` | `MoELayer` token dispatch and recombination |
| `routing/teacher_router.py` | Centroid-distance routing |
| `routing/student_router.py` | Learned routing MLP |
| `routing/hybrid_router.py` | Teacher/student blend with load penalty |
| `routing/load_balancer.py` | Per-expert usage penalty |
| `clustering/centroid_manager.py` | EMA centroid state |
| `clustering/recluster.py` | K-means centroid reassignment |
| `clustering/faiss_index.py` | FAISS index with brute-force fallback |
| `observe/` | Metrics, traces, and model/centroid snapshots |
| `training/` | `TrainingConfig`, dummy dataloader, reference `train()` |

## Limitations

- **`top_k < n_experts` fails.** The training loop passes
  `teacher_scores.squeeze(-1)` into `HybridRouter`. With `top_k=1` that collapses
  to `(batch, seq)` while the student logits are `(batch, seq, n_experts)`, and
  the blend raises a shape error. Runs only complete when `top_k == n_experts`,
  which defeats sparse routing. This is a known defect in the reference loop, not
  a property of the router components.
- No semantic encoder is wired in. The loop uses the backbone hidden states
  directly as the "semantic vectors" fed to the teacher router.
- The centroids are created at zero and are not updated during `train()`;
  `CentroidManager.update` and `reassign_centroids` exist but are not called from
  the reference loop, so teacher scores are uninformative in a default run.
- No distillation loss between the student and teacher routers is implemented,
  despite the student router being described as learning to mimic the teacher.
- Expert split and merge lifecycle policies are not implemented.
- The dataloader emits random token IDs, not real text.
- No benchmark numbers are claimed.
