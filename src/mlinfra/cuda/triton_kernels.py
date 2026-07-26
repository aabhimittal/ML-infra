"""Triton kernels — the language vLLM and HF TGI actually write custom kernels in.

Triton lowers to PTX through a GPU backend, so unlike the NVRTC and numba paths these can be
*defined* on CPU but only *compiled and launched* on real hardware. The host wrappers are
GPU-gated and raise :class:`~mlinfra.cuda.runtime.NoGpuError` (or skip in tests) otherwise.

Requires the ``triton`` extra:  ``pip install mlinfra[triton]``  (triton + torch).
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def triton_available() -> bool:
    try:
        import triton
        import triton.language  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def triton_gpu_ready() -> bool:
    """True only if triton, torch, and a CUDA device are all present."""
    if not triton_available():
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


# Define the kernels only when triton is importable, so the package loads without the extra.
if triton_available():
    import triton
    import triton.language as tl

    @triton.jit
    def _softmax_kernel(in_ptr, out_ptr, in_row_stride, out_row_stride, n_cols,
                        BLOCK_SIZE: tl.constexpr):
        """Numerically-stable row softmax, one program per row."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols

        x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        denom = tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_row_stride + cols, num / denom, mask=mask)

    @triton.jit
    def _gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """Tiled SGEMM: C = A @ B. One program computes a BLOCK_M x BLOCK_N tile of C.

        The K loop streams tiles of A and B through registers/shared memory and accumulates
        with ``tl.dot`` — the same blocking idea as the CUDA C++ kernel, but Triton handles
        the shared-memory staging and scheduling.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                        other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                        other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
                 tl.load(y_ptr + offs, mask=mask), mask=mask)


def softmax(x):  # pragma: no cover - requires a GPU
    """Row-wise softmax of a 2-D tensor via the Triton kernel. Requires a GPU."""
    if not triton_gpu_ready():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("Triton softmax needs triton + torch + a CUDA device.")
    import torch

    x = x.cuda() if not x.is_cuda else x
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    _softmax_kernel[(n_rows,)](
        x, out, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block
    )
    return out


def matmul(a, b, block_m: int = 64, block_n: int = 64,  # pragma: no cover - requires a GPU
           block_k: int = 32):
    """C = A @ B via the Triton tiled GEMM kernel. Requires a GPU."""
    if not triton_gpu_ready():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("Triton matmul needs triton + torch + a CUDA device.")
    import torch

    a = a.cuda() if not a.is_cuda else a
    b = b.cuda() if not b.is_cuda else b
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch for GEMM: {tuple(a.shape)} @ {tuple(b.shape)}")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c


def vector_add(x, y):  # pragma: no cover - requires a GPU
    """Elementwise add via the Triton kernel. Requires a GPU."""
    if not triton_gpu_ready():
        from mlinfra.cuda.runtime import NoGpuError

        raise NoGpuError("Triton vector_add needs triton + torch + a CUDA device.")
    import torch

    x, y = x.cuda(), y.cuda()
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, 1024),)
    _add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out
