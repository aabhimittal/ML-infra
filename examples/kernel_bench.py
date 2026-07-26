"""GPU kernel shootout: Triton vs numba vs torch/cuBLAS, logged to the tracker.

Two tracked ops:
  * softmax  — memory-bound, row-wise reduction (the attention softmax).
  * GEMM     — compute-bound, shared-memory tiled matmul (the dense projections).

    pip install -e ".[cuda,numba,triton]"
    python examples/kernel_bench.py            # requires a GPU

On a CPU-only host it prints a clear message and exits 0 (nothing to benchmark).
"""

from __future__ import annotations

from mlinfra.cuda import format_results, run_gemm_benchmark, run_softmax_benchmark
from mlinfra.cuda.triton_kernels import triton_gpu_ready


def main() -> None:
    if not triton_gpu_ready():
        print("No GPU (triton + torch + CUDA device) available — skipping kernel benchmark.")
        print("Run this on a GPU host, e.g. via the `GPU` CI workflow.")
        return

    print("== softmax (4096 x 2048) ==")
    softmax_results = run_softmax_benchmark(rows=4096, cols=2048, iters=50, db_path="mlruns.db")
    print(format_results(softmax_results, unit="Gitem/s"))

    print("\n== tiled GEMM (1024 x 1024 x 1024) ==")
    gemm_results = run_gemm_benchmark(m=1024, n=1024, k=1024, iters=20, db_path="mlruns.db")
    print(format_results(gemm_results, unit="GFLOP/s"))

    print("\nLogged both runs to experiment 'kernel-bench' in mlruns.db")


if __name__ == "__main__":
    main()
