"""Rolling latency / throughput aggregation.

A small, allocation-light registry for summarizing a stream of per-request observations into
the percentile-style numbers operators actually watch (p50/p95 latency, mean throughput).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Series:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.values.append(value)

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        idx = min(len(ordered) - 1, max(0, math.ceil(q / 100 * len(ordered)) - 1))
        return ordered[idx]

    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0


class MetricsRegistry:
    """Accumulate named series and emit a flat summary dict for logging."""

    def __init__(self) -> None:
        self._series: dict[str, _Series] = {}

    def observe(self, name: str, value: float) -> None:
        self._series.setdefault(name, _Series()).add(value)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, series in self._series.items():
            out[f"{name}.mean"] = series.mean()
            out[f"{name}.p50"] = series.percentile(50)
            out[f"{name}.p95"] = series.percentile(95)
            out[f"{name}.count"] = float(len(series.values))
        return out
