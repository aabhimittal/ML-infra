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
from mlinfra.cuda.numba_kernels import (
    NumbaUnavailableError,
    compile_saxpy_ptx,
    compile_softmax_ptx,
    numba_available,
)
from mlinfra.cuda.runtime import NoGpuError, gpu_available
from mlinfra.cuda.triton_kernels import triton_available, triton_gpu_ready

__all__ = [
    # raw CUDA C++ (NVRTC) path
    "CompileResult",
    "CudaToolchainError",
    "compile_file",
    "compile_kernel",
    "compile_to_ptx",
    "list_kernels",
    "nvrtc_available",
    "ptx_to_cubin",
    "ptxas_path",
    # numba (Python -> PTX, CPU-compilable) path
    "NumbaUnavailableError",
    "compile_saxpy_ptx",
    "compile_softmax_ptx",
    "numba_available",
    # triton (GPU-gated) path
    "triton_available",
    "triton_gpu_ready",
    # driver-API launch (GPU-gated)
    "NoGpuError",
    "gpu_available",
]
