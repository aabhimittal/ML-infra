"""Tests for the CUDA layer.

The compile tests run on CPU whenever the toolchain wheels are installed (``mlinfra[cuda]``)
and are skipped cleanly otherwise — so CI passes with or without the extra. The launch test is
gated on real hardware.
"""

from __future__ import annotations

import pytest

from mlinfra.cuda import (
    compile_kernel,
    gpu_available,
    list_kernels,
    nvrtc_available,
    ptx_to_cubin,
    ptxas_path,
)
from mlinfra.cuda.compile import CudaToolchainError

requires_nvrtc = pytest.mark.skipif(
    not nvrtc_available(), reason="CUDA toolchain not installed (pip install mlinfra[cuda])"
)
requires_ptxas = pytest.mark.skipif(
    ptxas_path() is None, reason="ptxas not installed (pip install mlinfra[cuda])"
)


def test_bundled_kernels_present():
    assert {"saxpy", "tiled_gemm", "fused_softmax"} <= set(list_kernels())


def test_compile_raises_clearly_without_toolchain():
    if nvrtc_available():
        pytest.skip("toolchain is installed")
    with pytest.raises(CudaToolchainError):
        compile_kernel("saxpy")


@requires_nvrtc
@pytest.mark.parametrize("stem", ["saxpy", "tiled_gemm", "fused_softmax"])
def test_kernels_compile_to_ptx(stem: str):
    result = compile_kernel(stem)
    assert result.log == "" or "error" not in result.log.lower()
    assert ".visible .entry" in result.ptx  # a kernel symbol was emitted
    assert result.num_instructions > 0


@requires_nvrtc
def test_compile_error_surfaces_log():
    from mlinfra.cuda.compile import compile_to_ptx

    with pytest.raises(CudaToolchainError):
        compile_to_ptx("__global__ void bad() { this is not c++ }", name="bad.cu")


@requires_nvrtc
@requires_ptxas
def test_ptx_assembles_to_cubin():
    ptx = compile_kernel("saxpy").ptx
    cubin = ptx_to_cubin(ptx, arch="sm_75")
    assert cubin[:4] == b"\x7fELF"  # a real ELF/cubin


def test_gpu_launch_or_skip():
    if not gpu_available():
        pytest.skip("no CUDA device on this host")
    from mlinfra.cuda.runtime import saxpy

    out = saxpy(2.0, [1.0, 2.0, 3.0], [10.0, 10.0, 10.0])
    assert out == pytest.approx([12.0, 14.0, 16.0])


# --- numba.cuda path (compiles to PTX on CPU) -----------------------------------------

from mlinfra.cuda import (  # noqa: E402
    compile_gemm_ptx,
    compile_saxpy_ptx,
    compile_softmax_ptx,
    numba_available,
    triton_available,
    triton_gpu_ready,
)

requires_numba = pytest.mark.skipif(
    not numba_available(), reason="numba CUDA toolchain not installed (pip install mlinfra[numba])"
)


@requires_numba
@pytest.mark.parametrize("compile_fn", [compile_saxpy_ptx, compile_softmax_ptx, compile_gemm_ptx])
def test_numba_compiles_to_ptx_on_cpu(compile_fn):
    ptx = compile_fn(cc=(7, 5))
    assert ".entry" in ptx
    assert ".target sm_75" in ptx


@requires_numba
def test_numba_gemm_allocates_shared_tiles():
    """The tiled GEMM must stage both operands in shared memory: 2 x TILE^2 x 4 bytes."""
    ptx = compile_gemm_ptx(cc=(8, 0), tile=16)
    shared = [line for line in ptx.splitlines() if ".shared" in line and "_cudapy_smem" in line]
    assert len(shared) >= 2
    assert "[1024]" in "".join(shared)  # 16 * 16 * 4 bytes per tile


@requires_numba
def test_numba_targets_requested_arch():
    assert ".target sm_80" in compile_saxpy_ptx(cc=(8, 0))


def test_numba_launch_or_skip():
    from mlinfra.cuda.numba_kernels import launch_saxpy
    from mlinfra.cuda.numba_kernels import numba_available as nb

    if not nb():
        pytest.skip("numba not installed")
    try:
        from numba import cuda
    except Exception:
        pytest.skip("numba not importable")
    if not cuda.is_available():
        pytest.skip("no CUDA device on this host")
    out = launch_saxpy(3.0, [1.0, 2.0], [1.0, 1.0])
    assert list(out) == pytest.approx([4.0, 7.0])


# --- triton path (GPU-gated) ----------------------------------------------------------

def test_triton_kernels_defined_or_skip():
    if not triton_available():
        pytest.skip("triton not installed")
    from mlinfra.cuda import triton_kernels as tk

    assert hasattr(tk, "_softmax_kernel") and hasattr(tk, "_add_kernel")


def test_triton_softmax_or_skip():
    if not triton_gpu_ready():
        pytest.skip("triton + torch + GPU required")
    import torch

    from mlinfra.cuda.triton_kernels import softmax

    x = torch.randn(4, 128, device="cuda")
    expected = torch.softmax(x, dim=1)
    assert torch.allclose(softmax(x), expected, atol=1e-5)


# --- benchmark harness (engine is CPU-tested with numpy) ------------------------------

def test_benchmark_engine_ranks_and_flags_correctness():
    import numpy as np

    from mlinfra.cuda import benchmark_impls, format_results

    def softmax_ref(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def softmax_good(x):
        return softmax_ref(x)

    def softmax_wrong(x):  # forgets to normalize -> should be flagged incorrect
        return np.exp(x - x.max(axis=1, keepdims=True))

    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 256)).astype(np.float32)

    results = benchmark_impls(
        {"good": softmax_good, "wrong": softmax_wrong},
        reference=softmax_ref,
        inputs=(x,),
        warmup=1,
        iters=5,
        atol=1e-5,
        work_items=x.size,
    )
    by_name = {r.name: r for r in results}
    assert by_name["good"].correct is True
    assert by_name["good"].max_abs_err <= 1e-5
    assert by_name["wrong"].correct is False
    assert all(r.mean_ms >= 0 for r in results)
    assert by_name["good"].throughput_gitems_s > 0
    assert "impl" in format_results(results)


def test_benchmark_engine_reports_flops_for_gemm():
    """Same engine, GEMM-shaped work: throughput should read as GFLOP/s when fed FLOPs."""
    import numpy as np

    from mlinfra.cuda import benchmark_impls, format_results

    rng = np.random.default_rng(1)
    m = k = n = 64
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    def naive(x, y):  # a genuinely different blocking strategy, same answer
        out = np.zeros((x.shape[0], y.shape[1]), dtype=np.float32)
        for i in range(0, x.shape[1], 16):
            out += x[:, i : i + 16] @ y[i : i + 16, :]
        return out

    results = benchmark_impls(
        {"numpy_matmul": lambda x, y: x @ y, "blocked": naive},
        reference=lambda x, y: x @ y,
        inputs=(a, b),
        warmup=1,
        iters=3,
        atol=1e-3,
        work_items=2 * m * n * k,
    )
    assert all(r.correct for r in results)
    assert all(r.throughput_gitems_s > 0 for r in results)
    assert "GFLOP/s" in format_results(results, unit="GFLOP/s")


def test_softmax_benchmark_or_skip():
    if not triton_gpu_ready():
        pytest.skip("triton + torch + GPU required")
    from mlinfra.cuda import run_softmax_benchmark

    results = run_softmax_benchmark(rows=512, cols=512, iters=5, db_path=":memory:")
    assert any(r.name == "triton" for r in results)
    assert all(r.correct for r in results)


def test_gemm_benchmark_or_skip():
    if not triton_gpu_ready():
        pytest.skip("triton + torch + GPU required")
    from mlinfra.cuda import run_gemm_benchmark

    results = run_gemm_benchmark(m=256, n=256, k=256, iters=5, db_path=":memory:")
    assert any(r.name == "triton" for r in results)
    assert all(r.correct for r in results)
    assert all(r.throughput_gitems_s > 0 for r in results)


def test_triton_matmul_or_skip():
    if not triton_gpu_ready():
        pytest.skip("triton + torch + GPU required")
    import torch

    from mlinfra.cuda.triton_kernels import matmul

    a = torch.randn(128, 96, device="cuda")
    b = torch.randn(96, 64, device="cuda")
    assert torch.allclose(matmul(a, b), a @ b, atol=1e-3)
