"""Evaluation metrics tracker for debate quality assessment."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


class MetricsTracker:
    """Tracks and aggregates evaluation metrics across episodes.

    Metrics tracked (per paper Section 6.1):
      - decision accuracy  (task success rate)
      - consistency        (variance across runs)
      - consensus quality  (final disagreement, devil advocate pass rate)
      - convergence speed  (average steps to consensus)
      - rule adaptability  (recovery time after rule changes)
    """

    def __init__(self) -> None:
        self._buffers: Dict[str, List[float]] = defaultdict(list)

    def record(self, key: str, value: float) -> None:
        self._buffers[key].append(value)

    def get_mean(self, key: str) -> float:
        buf = self._buffers.get(key, [])
        return float(np.mean(buf)) if buf else 0.0

    def get_std(self, key: str) -> float:
        buf = self._buffers.get(key, [])
        return float(np.std(buf)) if buf else 0.0

    def get_last(self, key: str) -> float:
        buf = self._buffers.get(key, [])
        return buf[-1] if buf else 0.0

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for key, buf in self._buffers.items():
            arr = np.array(buf)
            result[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": len(buf),
            }
        return result

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._buffers.clear()
        else:
            self._buffers.pop(key, None)

    def __repr__(self) -> str:
        lines = []
        for key in sorted(self._buffers.keys()):
            m, s = self.get_mean(key), self.get_std(key)
            lines.append(f"  {key}: {m:.4f} ± {s:.4f}  (n={len(self._buffers[key])})")
        return "MetricsTracker(\n" + "\n".join(lines) + "\n)"
