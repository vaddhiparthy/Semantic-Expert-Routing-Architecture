"""Routing modules for ASTRA‑X‑BASE.

The ``routing`` package contains classes implementing different
components of the routing mechanism:

* ``TeacherRouter`` provides stable early routing based on semantic
  distances in MeaningSpace.
* ``StudentRouter`` learns to route according to the LM hidden states.
* ``HybridRouter`` combines teacher and student scores with load
  penalties to produce final routing decisions.
* ``LoadBalancer`` calculates load penalties to avoid expert collapse.

These components interact closely with the clustering and core
subpackages.
"""

from .teacher_router import TeacherRouter
from .student_router import StudentRouter
from .hybrid_router import HybridRouter
from .load_balancer import LoadBalancer

__all__ = [
    "TeacherRouter",
    "StudentRouter",
    "HybridRouter",
    "LoadBalancer",
]