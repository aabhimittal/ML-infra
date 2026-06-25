"""Tests for the tracking layer: experiment tracker, metrics registry, DAG scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlinfra.tracking.metrics import MetricsRegistry
from mlinfra.tracking.scheduler import Pipeline, step
from mlinfra.tracking.tracker import ExperimentTracker


def test_tracker_persists_params_and_metrics(tmp_path: Path):
    tracker = ExperimentTracker(db_path=tmp_path / "mlruns.db")
    with tracker.start_run(experiment="exp", name="r1") as run:
        run.log_param("lr", 0.01)
        run.log_metric("loss", 0.5, step=0)
        run.log_metric("loss", 0.3, step=1)
        run_id = run.run_id

    fetched = tracker.get_run(run_id)
    assert fetched.name == "r1"
    assert fetched.params["lr"] == "0.01"
    assert fetched.metrics["loss"] == 0.3  # latest by step
    assert tracker.list_runs("exp") == [run_id]


def test_tracker_marks_failed_run(tmp_path: Path):
    tracker = ExperimentTracker(db_path=tmp_path / "mlruns.db")
    with pytest.raises(RuntimeError):
        with tracker.start_run(experiment="exp") as run:
            run_id = run.run_id
            raise RuntimeError("boom")
    # Run is recorded even though it failed; status is captured internally.
    assert run_id in tracker.list_runs("exp")


def test_metrics_registry_summary():
    reg = MetricsRegistry()
    for v in [0.1, 0.2, 0.3, 0.4]:
        reg.observe("latency", v)
    summary = reg.summary()
    assert summary["latency.count"] == 4
    assert abs(summary["latency.mean"] - 0.25) < 1e-9
    assert summary["latency.p50"] in (0.2, 0.3)


def test_scheduler_runs_in_topo_order_and_caches(tmp_path: Path):
    calls: list[str] = []

    @step
    def load() -> int:
        calls.append("load")
        return 2

    @step
    def double(load: int) -> int:
        calls.append("double")
        return load * 2

    @step
    def square(double: int) -> int:
        calls.append("square")
        return double * double

    cache_dir = tmp_path / "cache"
    report1 = Pipeline([square, double, load], cache_dir=cache_dir).run()
    assert report1.order == ["load", "double", "square"]
    assert report1.outputs["square"] == 16
    assert report1.executed == ["load", "double", "square"]
    assert calls == ["load", "double", "square"]

    # Second pipeline with the same cache dir: everything is a cache hit, nothing re-runs.
    calls.clear()
    report2 = Pipeline([square, double, load], cache_dir=cache_dir).run()
    assert report2.cached == ["load", "double", "square"]
    assert report2.executed == []
    assert calls == []


def test_scheduler_detects_cycle():
    @step
    def a(b: int) -> int:
        return b

    @step
    def b(a: int) -> int:
        return a

    with pytest.raises(ValueError, match="cycle"):
        Pipeline([a, b]).run()


def test_scheduler_unknown_dependency():
    @step
    def needs_missing(ghost: int) -> int:
        return ghost

    with pytest.raises(ValueError, match="unknown input"):
        Pipeline([needs_missing]).run()
