"""A small correctness + performance harness for comparing kernel implementations.

The engine (:func:`benchmark_impls`) is backend-agnostic — it works on numpy arrays or torch
tensors and is unit-tested on CPU. :func:`run_softmax_benchmark` wires the real GPU kernels
(Triton, numba, and torch/cuBLAS as the reference) into that engine and logs every result to
the MLflow-style :class:`~mlinfra.tracking.tracker.ExperimentTracker`, closing the loop between
the CUDA layer and the tracking layer.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchResult:
    name: str
    correct: bool
    max_abs_err: float
    mean_ms: float
    p50_ms: float
    throughput_gitems_s: float  # billion work-items / second (0 if work_items not given)
    iters: int


def _to_numpy(x: Any):
    """Coerce a torch tensor or array-like to numpy for error comparison."""
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    import numpy as np

    return np.asarray(x)


def _max_abs_err(a: Any, b: Any) -> float:
    import numpy as np

    return float(np.max(np.abs(_to_numpy(a) - _to_numpy(b))))


def benchmark_impls(
    impls: dict[str, Callable[..., Any]],
    reference: Callable[..., Any],
    inputs: Sequence[Any],
    *,
    warmup: int = 3,
    iters: int = 20,
    atol: float = 1e-4,
    sync: Callable[[], None] | None = None,
    work_items: int | None = None,
) -> list[BenchResult]:
    """Time and validate each implementation against ``reference`` on the same ``inputs``.

    ``sync`` is called after each invocation before stopping the timer — pass
    ``torch.cuda.synchronize`` for GPU timing; the default no-op is correct for CPU work.
    Returns results sorted fastest-first.
    """
    sync = sync or (lambda: None)
    expected = reference(*inputs)
    sync()

    results: list[BenchResult] = []
    for name, fn in impls.items():
        for _ in range(warmup):
            fn(*inputs)
        sync()

        latencies: list[float] = []
        out = None
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(*inputs)
            sync()
            latencies.append((time.perf_counter() - t0) * 1e3)

        err = _max_abs_err(out, expected)
        mean_ms = statistics.fmean(latencies)
        thr = (work_items / (mean_ms / 1e3) / 1e9) if (work_items and mean_ms > 0) else 0.0
        results.append(
            BenchResult(
                name=name,
                correct=err <= atol,
                max_abs_err=err,
                mean_ms=mean_ms,
                p50_ms=statistics.median(latencies),
                throughput_gitems_s=thr,
                iters=iters,
            )
        )
    results.sort(key=lambda r: r.mean_ms)
    return results


def run_softmax_benchmark(
    rows: int = 4096,
    cols: int = 2048,
    *,
    iters: int = 50,
    db_path: str = "mlruns.db",
):  # pragma: no cover - requires a GPU
    """Compare Triton / numba / torch row-softmax on a GPU and log to the tracker.

    Returns the list of :class:`BenchResult`. Raises ``NoGpuError`` without a CUDA device.
    """
    from mlinfra.cuda.runtime import NoGpuError
    from mlinfra.cuda.triton_kernels import softmax as triton_softmax
    from mlinfra.cuda.triton_kernels import triton_gpu_ready
    from mlinfra.tracking.tracker import ExperimentTracker

    if not triton_gpu_ready():
        raise NoGpuError("Softmax benchmark needs triton + torch + a CUDA device.")

    import torch

    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    impls: dict[str, Callable[..., Any]] = {
        "torch_cublas": lambda t: torch.softmax(t, dim=1),
        "triton": lambda t: triton_softmax(t),
    }

    # numba is optional — include it only if its softmax launcher is usable here.
    try:
        from mlinfra.cuda.numba_kernels import launch_softmax, numba_available

        if numba_available():
            impls["numba"] = lambda t: launch_softmax(t)
    except Exception:
        pass

    results = benchmark_impls(
        impls,
        reference=lambda t: torch.softmax(t, dim=1),
        inputs=(x,),
        warmup=5,
        iters=iters,
        atol=1e-4,
        sync=torch.cuda.synchronize,
        work_items=rows * cols,
    )

    tracker = ExperimentTracker(db_path=db_path)
    with tracker.start_run(experiment="kernel-bench", name=f"softmax-{rows}x{cols}") as run:
        run.log_params({"op": "softmax", "rows": rows, "cols": cols, "iters": iters,
                        "device": torch.cuda.get_device_name(0)})
        for r in results:
            run.log_metric(f"{r.name}.mean_ms", r.mean_ms)
            run.log_metric(f"{r.name}.throughput_gitems_s", r.throughput_gitems_s)
            run.log_metric(f"{r.name}.max_abs_err", r.max_abs_err)
    return results


def format_results(results: list[BenchResult]) -> str:
    """Render a compact table for console/CI output."""
    header = f"{'impl':<16}{'ok':<5}{'mean_ms':>10}{'p50_ms':>10}{'Gitem/s':>10}{'max_err':>12}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.name:<16}{('yes' if r.correct else 'NO'):<5}"
            f"{r.mean_ms:>10.4f}{r.p50_ms:>10.4f}{r.throughput_gitems_s:>10.2f}"
            f"{r.max_abs_err:>12.2e}"
        )
    return "\n".join(lines)
