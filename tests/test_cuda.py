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
