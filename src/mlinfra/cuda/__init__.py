"""CUDA kernel layer: compile real ``.cu`` kernels to PTX/SASS on CPU; launch on a GPU.

The compile path (:mod:`~mlinfra.cuda.compile`) runs anywhere the toolchain wheels are
installed (``pip install mlinfra[cuda]``) — no GPU required. The launch path
(:mod:`~mlinfra.cuda.runtime`) needs real hardware and degrades gracefully otherwise.
"""

from mlinfra.cuda.compile import (
    CompileResult,
    CudaToolchainError,
    compile_file,
    compile_kernel,
    compile_to_ptx,
    list_kernels,
    nvrtc_available,
    ptx_to_cubin,
    ptxas_path,
)
from mlinfra.cuda.runtime import NoGpuError, gpu_available

__all__ = [
    "CompileResult",
    "CudaToolchainError",
    "compile_file",
    "compile_kernel",
    "compile_to_ptx",
    "list_kernels",
    "nvrtc_available",
    "ptx_to_cubin",
    "ptxas_path",
    "NoGpuError",
    "gpu_available",
]
