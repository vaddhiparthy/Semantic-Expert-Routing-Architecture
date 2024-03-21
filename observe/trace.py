"""Trace logging utilities.

This module defines a simple trace logger for recording routing
decisions and other per‑token information during training or
inference. The recorded traces can be used to debug or visualise how
tokens flow through the model.

Classes
-------
TraceLogger
    Collects and stores token‑level trace information.
"""

from __future__ import annotations

from typing import List, Dict, Any
import json


class TraceLogger:
    """Collect per‑token traces.

    This simple logger accumulates trace records in memory. For large
    training runs, consider streaming traces to disk instead.
    """

    def __init__(self) -> None:
        self.traces: List[Dict[str, Any]] = []

    def add(self, record: Dict[str, Any]) -> None:
        """Add a new trace record.

        Parameters
        ----------
        record: Dict[str, Any]
            A dictionary containing trace information for one or more
            tokens.
        """
        self.traces.append(record)

    def save_json(self, path: str) -> None:
        """Write all traces to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.traces, f, indent=2)