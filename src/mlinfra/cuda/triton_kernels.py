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
        import triton  # noqa: F401
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
