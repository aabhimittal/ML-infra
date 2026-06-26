"""Launch compiled kernels on a real GPU via the CUDA driver API (ctypes).

This is the half that *cannot* run without hardware: it needs ``libcuda.so`` (shipped with the
NVIDIA driver, not pip) and an actual device. Every entry point degrades gracefully — call
:func:`gpu_available` first, and the launchers raise a clear :class:`NoGpuError` otherwise — so
the package and its tests stay green on CPU-only machines while the code path is ready for a
GPU box.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache

from mlinfra.cuda.compile import compile_kernel, ptx_to_cubin


class NoGpuError(RuntimeError):
    """Raised when a launch is attempted without a usable CUDA device."""


@lru_cache(maxsize=1)
def _libcuda() -> ctypes.CDLL | None:
    for name in ("libcuda.so", "libcuda.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True only if the driver loads *and* reports at least one device."""
    lib = _libcuda()
    if lib is None:
        return False
    if lib.cuInit(0) != 0:
        return False
    count = ctypes.c_int(0)
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return False
    return count.value > 0


def _check(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise NoGpuError(f"CUDA driver call failed ({what}): code {rc}")


def saxpy(a: float, x: list[float], y: list[float]) -> list[float]:
    """Compile, upload, and launch the SAXPY kernel on a GPU. Requires hardware.

    Mirrors what a serving engine's custom-kernel path looks like end to end:
    source -> PTX -> cubin -> module load -> launch -> copy back.
    """
    if not gpu_available():
        raise NoGpuError("No CUDA device available; run this on a GPU host.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    lib = _libcuda()
    assert lib is not None
    n = len(x)
    cubin = ptx_to_cubin(compile_kernel("saxpy").ptx)

    ctx = ctypes.c_void_p()
    module = ctypes.c_void_p()
    func = ctypes.c_void_p()
    dev = ctypes.c_int(0)
    _check(lib, lib.cuInit(0), "cuInit")
    _check(lib, lib.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    _check(lib, lib.cuCtxCreate(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    try:
        _check(lib, lib.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        _check(
            lib,
            lib.cuModuleGetFunction(ctypes.byref(func), module, b"saxpy"),
            "cuModuleGetFunction",
        )

        nbytes = n * ctypes.sizeof(ctypes.c_float)
        dx, dy, dout = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        for buf in (dx, dy, dout):
            _check(lib, lib.cuMemAlloc(ctypes.byref(buf), nbytes), "cuMemAlloc")
        arr = ctypes.c_float * n
        _check(lib, lib.cuMemcpyHtoD(dx, arr(*x), nbytes), "cuMemcpyHtoD x")
        _check(lib, lib.cuMemcpyHtoD(dy, arr(*y), nbytes), "cuMemcpyHtoD y")

        c_a = ctypes.c_float(a)
        c_n = ctypes.c_int(n)
        params = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(c_a), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dx), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dy), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dout), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(c_n), ctypes.c_void_p),
        )
        threads = 256
        blocks = (n + threads - 1) // threads
        _check(
            lib,
            lib.cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, None, params, None),
            "cuLaunchKernel",
        )
        _check(lib, lib.cuCtxSynchronize(), "cuCtxSynchronize")

        out = arr()
        _check(lib, lib.cuMemcpyDtoH(out, dout, nbytes), "cuMemcpyDtoH")
        for buf in (dx, dy, dout):
            lib.cuMemFree(buf)
        return list(out)
    finally:
        lib.cuCtxDestroy(ctx)
