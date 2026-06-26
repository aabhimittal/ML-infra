"""GPU softmax shootout: Triton vs numba vs torch/cuBLAS, logged to the tracker.

    pip install -e ".[cuda,numba,triton]"
    python examples/kernel_bench.py            # requires a GPU

On a CPU-only host it prints a clear message and exits 0 (nothing to benchmark).
"""

from __future__ import annotations

from mlinfra.cuda import format_results, run_softmax_benchmark
from mlinfra.cuda.triton_kernels import triton_gpu_ready


def main() -> None:
    if not triton_gpu_ready():
        print("No GPU (triton + torch + CUDA device) available — skipping kernel benchmark.")
        print("Run this on a GPU host, e.g. via the `GPU` CI workflow.")
        return

    results = run_softmax_benchmark(rows=4096, cols=2048, iters=50, db_path="mlruns.db")
    print(format_results(results))
    print("\nLogged to experiment 'kernel-bench' in mlruns.db")


if __name__ == "__main__":
    main()
