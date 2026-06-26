"""A tiny ZenML-style pipeline scheduler with content-hash step caching.

Steps are plain functions decorated with :func:`step`. The :class:`Pipeline` wires them into
a DAG by matching each step's parameter names to upstream step names, runs them in
topological order, and caches each step's output keyed by a content hash of *its source code
plus its resolved inputs*. Re-running an unchanged step is a cache hit — the same property
that lets ZenML/MLflow skip expensive recomputation.

Caching is in-memory by default; pass ``cache_dir`` to persist across processes.
"""

from __future__ import annotations

import hashlib
import inspect
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    name: str
    fn: Callable[..., Any]
    deps: list[str] = field(default_factory=list)
    cache: bool = True

    def source_hash(self) -> str:
        try:
            src = inspect.getsource(self.fn)
        except (OSError, TypeError):  # e.g. lambdas defined in a REPL
            src = repr(self.fn)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


def step(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    cache: bool = True,
) -> Callable[..., Any]:
    """Decorator turning a function into a cacheable pipeline :class:`Step`.

    Usable bare (``@step``) or parameterized (``@step(name="load", cache=False)``).
    Parameter names become upstream dependencies resolved by the pipeline.
    """

    def wrap(fn: Callable[..., Any]) -> Step:
        params = list(inspect.signature(fn).parameters)
        return Step(name=name or fn.__name__, fn=fn, deps=params, cache=cache)

    return wrap(_fn) if _fn is not None else wrap


@dataclass
class RunReport:
    outputs: dict[str, Any]
    executed: list[str]
    cached: list[str]
    order: list[str]


class Pipeline:
    def __init__(
        self,
        steps: list[Step],
        inputs: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.steps = {s.name: s for s in steps}
        self.inputs = inputs or {}
        self._mem_cache: dict[str, Any] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _topo_order(self) -> list[str]:
        """Kahn topological sort; raises on cycles or unknown dependencies."""
        indeg = {name: 0 for name in self.steps}
        adj: dict[str, list[str]] = {name: [] for name in self.steps}
        for name, s in self.steps.items():
            for dep in s.deps:
                if dep in self.inputs:
                    continue
                if dep not in self.steps:
                    raise ValueError(f"Step {name!r} depends on unknown input {dep!r}")
                adj[dep].append(name)
                indeg[name] += 1
        ready = sorted(n for n, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
            ready.sort()
        if len(order) != len(self.steps):
            raise ValueError("Pipeline has a dependency cycle")
        return order

    def _cache_key(self, s: Step, kwargs: dict[str, Any]) -> str:
        payload = pickle.dumps((s.source_hash(), sorted(kwargs.items())), protocol=4)
        return f"{s.name}-{hashlib.sha256(payload).hexdigest()[:16]}"

    def _cache_get(self, key: str) -> tuple[bool, Any]:
        if key in self._mem_cache:
            return True, self._mem_cache[key]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.pkl"
            if path.exists():
                value = pickle.loads(path.read_bytes())
                self._mem_cache[key] = value
                return True, value
        return False, None

    def _cache_put(self, key: str, value: Any) -> None:
        self._mem_cache[key] = value
        if self.cache_dir:
            (self.cache_dir / f"{key}.pkl").write_bytes(pickle.dumps(value, protocol=4))

    def run(self) -> RunReport:
        order = self._topo_order()
        results: dict[str, Any] = {}
        executed: list[str] = []
        cached: list[str] = []
        for name in order:
            s = self.steps[name]
            kwargs = {
                dep: (self.inputs[dep] if dep in self.inputs else results[dep])
                for dep in s.deps
            }
            if s.cache:
                key = self._cache_key(s, kwargs)
                hit, value = self._cache_get(key)
                if hit:
                    results[name] = value
                    cached.append(name)
                    continue
            value = s.fn(**kwargs)
            if s.cache:
                self._cache_put(key, value)
            results[name] = value
            executed.append(name)
        return RunReport(outputs=results, executed=executed, cached=cached, order=order)
