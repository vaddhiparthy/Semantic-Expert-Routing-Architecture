# Architecture

The repository implements a compact semantic-routing mixture-of-experts scaffold.

```text
token ids
  -> TinyTransformer
  -> TeacherRouter / StudentRouter
  -> LoadBalancer
  -> HybridRouter
  -> MoELayer
  -> expert MLP output
  -> language-model loss
```

## Component Boundaries

| Component | Path | Role |
| --- | --- | --- |
| Tiny transformer | `core/transformer.py` | Produces token hidden states |
| Expert MLP | `core/expert.py` | Processes routed token subsets |
| MoE layer | `core/moe_layer.py` | Dispatches tokens to selected experts |
| Teacher router | `routing/teacher_router.py` | Uses centroid distance as semantic routing signal |
| Student router | `routing/student_router.py` | Learns expert logits from hidden states |
| Hybrid router | `routing/hybrid_router.py` | Combines teacher, student, and load signals |
| Centroid manager | `clustering/centroid_manager.py` | Maintains expert centroids |
| Metrics logger | `observe/metrics.py` | Records scalar training metrics |
| Training loop | `training/train_loop.py` | Wires the reference training path |

## Research Boundary

This project is a scaffold for studying routing behavior. It does not claim trained performance, production serving, or dataset-specific benchmark results.
