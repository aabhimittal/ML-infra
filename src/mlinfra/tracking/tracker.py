"""A minimal MLflow-style experiment tracker backed by sqlite.

Records *runs*, each with *params* (config) and time-series *metrics*. sqlite keeps it
zero-infrastructure and inspectable (``sqlite3 mlruns.db``) while still modelling the real
data layout: experiments contain runs, runs contain params + metric points.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    name TEXT,
    status TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL
);
CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL
);
"""


@dataclass
class Run:
    run_id: str
    experiment: str
    name: str | None = None
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class ExperimentTracker:
    def __init__(self, db_path: str | Path = "mlruns.db") -> None:
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def start_run(self, experiment: str = "default", name: str | None = None) -> Iterator[Run]:
        """Context manager that opens a run and finalizes its status on exit."""
        run = Run(run_id=uuid.uuid4().hex, experiment=experiment, name=name)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, experiment, name, status, start_time) "
                "VALUES (?, ?, ?, 'RUNNING', ?)",
                (run.run_id, experiment, name, time.time()),
            )
        self._current = run
        status = "FINISHED"
        try:
            yield _RunHandle(self, run)
        except Exception:
            status = "FAILED"
            raise
        finally:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?",
                    (status, time.time(), run.run_id),
                )

    def log_param(self, run_id: str, key: str, value: object) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value) if not isinstance(value, str) else value),
            )

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (run_id, key, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                (run_id, key, float(value), step, time.time()),
            )

    def get_run(self, run_id: str) -> Run:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            params = {
                r["key"]: r["value"]
                for r in conn.execute("SELECT key, value FROM params WHERE run_id = ?", (run_id,))
            }
            # Latest value per metric key.
            metrics = {
                r["key"]: r["value"]
                for r in conn.execute(
                    "SELECT key, value FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
                )
            }
        return Run(
            run_id=row["run_id"],
            experiment=row["experiment"],
            name=row["name"],
            params=params,
            metrics=metrics,
        )

    def list_runs(self, experiment: str | None = None) -> list[str]:
        query = "SELECT run_id FROM runs"
        args: tuple = ()
        if experiment is not None:
            query += " WHERE experiment = ?"
            args = (experiment,)
        query += " ORDER BY start_time"
        with self._connect() as conn:
            return [r["run_id"] for r in conn.execute(query, args)]


class _RunHandle:
    """Convenience wrapper so callers can ``run.log_metric(...)`` inside a ``with`` block."""

    def __init__(self, tracker: ExperimentTracker, run: Run) -> None:
        self._tracker = tracker
        self.run_id = run.run_id
        self.experiment = run.experiment
        self.name = run.name

    def log_param(self, key: str, value: object) -> None:
        self._tracker.log_param(self.run_id, key, value)

    def log_params(self, params: dict[str, object]) -> None:
        for k, v in params.items():
            self._tracker.log_param(self.run_id, k, v)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._tracker.log_metric(self.run_id, key, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        for k, v in metrics.items():
            self._tracker.log_metric(self.run_id, k, v, step)
