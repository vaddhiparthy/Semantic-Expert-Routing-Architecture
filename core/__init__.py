"""Core building blocks of the ASTRA‑X‑BASE model.

The ``core`` package contains the low‑level components used by
ASTRA‑X‑BASE. These include the tiny transformer backbone, the
mixture‑of‑experts layers, and the definition of individual expert
modules. Each component is designed to be lightweight and composable.
"""

from .transformer import TinyTransformer
from .moe_layer import MoELayer
from .expert import ExpertMLP

__all__ = [
    "TinyTransformer",
    "MoELayer",
    "ExpertMLP",
]