"""High-level GPU kernels written in Python with ``numba.cuda``.

This is the "write a kernel without leaving Python" counterpart to the raw ``.cu`` files in
``kernels/``. Crucially, numba can lower these to PTX **on a CPU-only machine** (via NVVM from
the pip toolchain wheels), so :func:`compile_saxpy_ptx` / :func:`compile_softmax_ptx` are
exercised in CI without a GPU — the same property the NVRTC path has. Actually *launching*
them (:func:`launch_saxpy`) still needs real hardware and degrades gracefully.

Requires the ``numba`` extra:  ``pip install mlinfra[numba]``  (numba-cuda + cuda-python +
the nvcc wheel). numba-cuda auto-detects the wheels through the NVIDIA CUDA bindings.
"""

from __future__ import annotations

import os

# numba reads this at first import; enabling the NVIDIA binding lets numba-cuda find the
# pip-installed NVVM/nvcc wheels (no system CUDA toolkit needed). Set before importing numba.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")

from functools import lru_cache  # noqa: E402


class NumbaUnavailableError(RuntimeError):
    """Raised when the numba CUDA toolchain is not importable."""


@lru_cache(maxsize=1)
def numba_available() -> bool:
    try:
        from numba import cuda  # noqa: F401

        return True
    except Exception:
        return False


def _require_numba():
    try:
        from numba import cuda, float32

        return cuda, float32
    except Exception as exc:  # pragma: no cover - only when extra is absent
        raise NumbaUnavailableError(
            "numba CUDA toolchain unavailable. Install: pip install mlinfra[numba]"
        ) from exc


def _saxpy_kernel():
    cuda, _ = _require_numba()

    def saxpy(a, x, y, out):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a * x[i] + y[i]

    return saxpy


def _softmax_kernel():
    """One block per row, numerically-stable softmax — the attention-style kernel in Python."""
    cuda, float32 = _require_numba()

    def softmax_rows(x, out):
        row = cuda.blockIdx.x
        tid = cuda.threadIdx.x
        nthreads = cuda.blockDim.x
        cols = x.shape[1]
        sm = cuda.shared.array(shape=0, dtype=float32)  # dynamic shared memory

        # row max
        local = float32(-3.4e38)
        j = tid
        while j < cols:
            v = x[row, j]
            if v > local:
                local = v
            j += nthreads
        sm[tid] = local
        cuda.syncthreads()
        s = nthreads // 2
        while s > 0:
            if tid < s and sm[tid + s] > sm[tid]:
                sm[tid] = sm[tid + s]
            cuda.syncthreads()
            s //= 2
        row_max = sm[0]
        cuda.syncthreads()

        # exp and sum
        acc = float32(0.0)
        j = tid
        while j < cols:
            e = float32(2.718281828) ** (x[row, j] - row_max)
            out[row, j] = e
            acc += e
            j += nthreads
        sm[tid] = acc
        cuda.syncthreads()
        s = nthreads // 2
        while s > 0:
            if tid < s:
                sm[tid] += sm[tid + s]
            cuda.syncthreads()
            s //= 2
        inv = float32(1.0) / sm[0]

        j = tid
        while j < cols:
            out[row, j] *= inv
            j += nthreads

    return softmax_rows


def compile_saxpy_ptx(cc: tuple[int, int] = (7, 5)) -> str:
    """Compile the numba SAXPY kernel to PTX on CPU. Returns the PTX text."""
    cuda, float32 = _require_numba()
    sig = (float32, float32[:], float32[:], float32[:])
    ptx, _ = cuda.compile_ptx(_saxpy_kernel(), sig, cc=cc)
    return ptx


def compile_softmax_ptx(cc: tuple[int, int] = (7, 5)) -> str:
    """Compile the numba row-softmax kernel to PTX on CPU. Returns the PTX text."""
    cuda, float32 = _require_numba()
    sig = (float32[:, :], float32[:, :])
    ptx, _ = cuda.compile_ptx(_softmax_kernel(), sig, cc=cc)
    return ptx


def launch_saxpy(a: float, x, y):  # pragma: no cover - requires a GPU
    """Run the numba SAXPY kernel on a GPU. Requires hardware."""
    cuda, _ = _require_numba()
    if not cuda.is_available():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("No CUDA device available; run on a GPU host.")
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    out = np.empty_like(x)
    kernel = cuda.jit(_saxpy_kernel())
    threads = 256
    blocks = (x.size + threads - 1) // threads
    kernel[blocks, threads](np.float32(a), x, y, out)
    return out
