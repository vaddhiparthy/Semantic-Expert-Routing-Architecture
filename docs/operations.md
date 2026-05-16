# Operations

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Artifacts

Do not commit generated checkpoints, runs, or experiment outputs. The repository ignores:

- `outputs/`;
- `checkpoints/`;
- `runs/`.

## FAISS

`faiss-cpu` is optional and is not generally available on Windows through the same wheel path as Linux. The code has a PyTorch brute-force fallback for nearest-neighbor lookup when FAISS is unavailable.
