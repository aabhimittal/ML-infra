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

from functools import lru_cache


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


def _gemm_kernel(tile: int = 16):
    """Shared-memory tiled SGEMM — the Python twin of ``kernels/tiled_gemm.cu``.

    ``tile`` is captured as a closure constant so numba can size the shared-memory tiles at
    compile time (they must be compile-time constants, exactly as in CUDA C++).
    """
    cuda, float32 = _require_numba()

    TILE = tile

    def sgemm_tiled(A, B, C):
        sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
        sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        row = cuda.blockIdx.y * TILE + ty
        col = cuda.blockIdx.x * TILE + tx

        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[1]

        acc = float32(0.0)
        for t in range((K + TILE - 1) // TILE):
            a_col = t * TILE + tx
            b_row = t * TILE + ty
            if row < M and a_col < K:
                sA[ty, tx] = A[row, a_col]
            else:
                sA[ty, tx] = float32(0.0)
            if b_row < K and col < N:
                sB[ty, tx] = B[b_row, col]
            else:
                sB[ty, tx] = float32(0.0)
            cuda.syncthreads()

            for k in range(TILE):
                acc += sA[ty, k] * sB[k, tx]
            cuda.syncthreads()

        if row < M and col < N:
            C[row, col] = acc

    return sgemm_tiled


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


def compile_gemm_ptx(cc: tuple[int, int] = (7, 5), tile: int = 16) -> str:
    """Compile the numba tiled-GEMM kernel to PTX on CPU. Returns the PTX text."""
    cuda, float32 = _require_numba()
    sig = (float32[:, :], float32[:, :], float32[:, :])
    ptx, _ = cuda.compile_ptx(_gemm_kernel(tile), sig, cc=cc)
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


def launch_softmax(x, threads: int = 256):  # pragma: no cover - requires a GPU
    """Run the numba row-softmax kernel on a GPU. Accepts a numpy array or torch tensor.

    One block per row, with dynamic shared memory sized to the block. Returns the same type
    as the input (torch tensor in / torch tensor out) so it slots into the benchmark harness.
    """
    cuda, _ = _require_numba()
    if not cuda.is_available():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("No CUDA device available; run on a GPU host.")
    import numpy as np

    is_torch = hasattr(x, "detach")
    arr = x.detach().cpu().numpy() if is_torch else np.asarray(x, dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    rows, _cols = arr.shape
    out = np.empty_like(arr)

    kernel = cuda.jit(_softmax_kernel())
    shared_bytes = threads * 4  # float32 scratch for the block reduction
    kernel[rows, threads, 0, shared_bytes](arr, out)

    if is_torch:
        import torch

        return torch.as_tensor(out, device=x.device)
    return out


def launch_gemm(a, b, tile: int = 16):  # pragma: no cover - requires a GPU
    """Run the numba tiled SGEMM on a GPU. Accepts numpy arrays or torch tensors.

    Returns the same type as the input (torch in / torch out) so it slots into the benchmark
    harness alongside the Triton and cuBLAS implementations.
    """
    cuda, _ = _require_numba()
    if not cuda.is_available():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("No CUDA device available; run on a GPU host.")
    import numpy as np

    is_torch = hasattr(a, "detach")
    A = a.detach().cpu().numpy() if is_torch else np.asarray(a, dtype=np.float32)
    B = b.detach().cpu().numpy() if is_torch else np.asarray(b, dtype=np.float32)
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch for GEMM: {A.shape} @ {B.shape}")

    M, N = A.shape[0], B.shape[1]
    C = np.empty((M, N), dtype=np.float32)

    kernel = cuda.jit(_gemm_kernel(tile))
    blocks = ((N + tile - 1) // tile, (M + tile - 1) // tile)  # grid is (x=cols, y=rows)
    kernel[blocks, (tile, tile)](A, B, C)

    if is_torch:
        import torch

        return torch.as_tensor(C, device=a.device)
    return C
