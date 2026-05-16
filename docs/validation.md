# Validation

## Static Validation

```powershell
python -m compileall .
```

## Training Smoke

After installing PyTorch:

```powershell
python - <<'PY'
from astra_x_base.training import TrainingConfig, create_dummy_dataloader, train

config = TrainingConfig(batch_size=2, num_epochs=1, top_k=1)
dataloader = create_dummy_dataloader(batch_size=2, seq_len=4, vocab_size=32, num_batches=1)
metrics = train(dataloader, vocab_size=32, n_experts=2, config=config)
print(metrics.summary())
PY
```

## Known Gaps

- There are no committed benchmark datasets.
- There are no automated unit tests yet.
- The reference training loop uses dummy token batches.
- Full split/merge expert lifecycle testing is not implemented.
