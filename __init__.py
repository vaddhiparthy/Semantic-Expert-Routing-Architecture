"""Top-level package for the ASTRA‑X‑BASE library.

This package exposes a modular, research‑grade implementation of a
semantic‑aware mixture‑of‑experts model. The design follows the
architecture outlined in the accompanying documentation, including:

* A lightweight base transformer backbone.
* A semantic router comprised of a teacher and student module.
* A dynamic clustering engine built on FAISS.
* Expert MLP layers with split/merge capability.
* Observability tooling for monitoring routing decisions.
* A training loop engineered for small compute environments.

Modules are organised into subpackages: ``core``, ``routing``,
``clustering``, ``observe``, and ``training``. See the README for
detailed usage and design rationale.
"""

from importlib import metadata as _metadata  # pragma: no cover

__all__ = [
    "__version__",
]

try:
    __version__ = _metadata.version(__name__)
except Exception:
    __version__ = "unknown"