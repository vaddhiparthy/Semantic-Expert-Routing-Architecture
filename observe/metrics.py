"""Metrics logging utilities.

This module defines a simple logger for tracking metrics during
training. Metrics are stored in a dictionary and can be periodically
written to disk or visualised. The logger supports adding scalar
values and computing moving averages.

Classes
-------
MetricsLogger
    Record and aggregate scalar metrics.
"""

from __future__ import annotations

from typing import Dict, List
import json


class MetricsLogger:
    """Store and aggregate scalar metrics.

    Attributes
    ----------
    data: Dict[str, List[float]]
        Collected metric values keyed by metric name.
    """

    def __init__(self) -> None:
        self.data: Dict[str, List[float]] = {}

    def log(self, name: str, value: float) -> None:
        """Record a metric value.

        Parameters
        ----------
        name: str
            Name of the metric.
        value: float
            Value to log.
        """
        self.data.setdefault(name, []).append(value)

    def get_latest(self, name: str) -> float:
        """Get the most recent value for a metric."""
        return self.data.get(name, [0.0])[-1]

    def summary(self) -> Dict[str, float]:
        """Compute the mean of all logged values for each metric."""
        return {k: sum(v) / len(v) for k, v in self.data.items() if v}

    def save_json(self, path: str) -> None:
        """Save metrics to a JSON file.

        Parameters
        ----------
        path: str
            File path to save the metrics.
        """
        with open(path, "w") as f:
            json.dump(self.data, f)