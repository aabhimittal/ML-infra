"""Compile CUDA C++ to PTX (and optionally SASS/cubin) without a GPU.

This wraps NVIDIA's NVRTC runtime-compilation library via ``ctypes`` so we can turn ``.cu``
source into PTX entirely on CPU — ``nvcc`` is a compile-time tool and needs no device. PTX can
then be assembled to a ``.cubin`` (real SASS machine code) with ``ptxas``.

The compiler toolchain ships as pip wheels (the ``cuda`` extra):

    pip install mlinfra[cuda]   # nvidia-cuda-nvrtc-cu12 + nvidia-cuda-nvcc-cu12

Nothing here can *run* a kernel — see :mod:`mlinfra.cuda.runtime` for the (GPU-gated) launch
path. Everything in this module is exercisable in CI on a CPU-only machine.
"""

from __future__ import annotations

import ctypes
import glob
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

KERNELS_DIR = Path(__file__).parent / "kernels"


class CudaToolchainError(RuntimeError):
    """Raised when the NVRTC/ptxas wheels are not installed."""


# --- locating the wheels --------------------------------------------------------------

def _site_nvidia_dirs() -> list[Path]:
    """Directories under installed ``nvidia-*`` wheels, if present."""
    try:
        import nvidia  # type: ignore
    except ImportError:
        return []
    return [Path(p) for p in nvidia.__path__]  # namespace package


@lru_cache(maxsize=1)
def _find_libnvrtc() -> str | None:
    patterns = []
    for base in _site_nvidia_dirs():
        patterns.append(str(base / "cuda_nvrtc" / "lib" / "libnvrtc.so*"))
    # Fall back to a system-installed toolkit.
    patterns += ["/usr/local/cuda*/lib64/libnvrtc.so*", "/usr/lib/*/libnvrtc.so*"]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


@lru_cache(maxsize=1)
def ptxas_path() -> str | None:
    """Path to the ``ptxas`` assembler, if available."""
    for base in _site_nvidia_dirs():
        hit = base / "cuda_nvcc" / "bin" / "ptxas"
        if hit.exists():
            return str(hit)
    for pat in ("/usr/local/cuda*/bin/ptxas", "/usr/bin/ptxas"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def nvrtc_available() -> bool:
    return _find_libnvrtc() is not None


# --- NVRTC ctypes binding -------------------------------------------------------------

@lru_cache(maxsize=1)
def _nvrtc() -> ctypes.CDLL:
    path = _find_libnvrtc()
    if path is None:
        raise CudaToolchainError(
            "libnvrtc not found. Install the toolchain: pip install mlinfra[cuda]"
        )
    lib = ctypes.CDLL(path)
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    lib.nvrtcGetErrorString.argtypes = [ctypes.c_int]
    lib.nvrtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.nvrtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    for fn in ("nvrtcGetPTXSize", "nvrtcGetProgramLogSize"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    for fn in ("nvrtcGetPTX", "nvrtcGetProgramLog"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    return lib


def _check(lib: ctypes.CDLL, rc: int) -> None:
    if rc != 0:
        msg = lib.nvrtcGetErrorString(rc) or b"unknown NVRTC error"
        raise CudaToolchainError(msg.decode())


@dataclass
class CompileResult:
    name: str
    ptx: str
    log: str
    arch: str

    @property
    def num_instructions(self) -> int:
        """Rough PTX instruction count — a cheap proxy for regression diffing."""
        return sum(
            1
            for line in self.ptx.splitlines()
            if line.strip() and not line.strip().startswith(("//", ".", "$", "{", "}"))
        )


def compile_to_ptx(
    source: str,
    name: str = "kernel.cu",
    arch: str = "compute_75",
    options: list[str] | None = None,
) -> CompileResult:
    """Compile CUDA C++ ``source`` to PTX. Raises :class:`CudaToolchainError` on failure."""
    lib = _nvrtc()
    prog = ctypes.c_void_p()
    _check(lib, lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(),
                                       0, None, None))
    try:
        opts = [f"--gpu-architecture={arch}", *(options or [])]
        c_opts = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        rc = lib.nvrtcCompileProgram(prog, len(opts), c_opts)

        log_size = ctypes.c_size_t()
        lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log_buf = ctypes.create_string_buffer(log_size.value)
        lib.nvrtcGetProgramLog(prog, log_buf)
        log = log_buf.value.decode().strip()
        _check(lib, rc)  # after fetching the log, so failures surface diagnostics

        ptx_size = ctypes.c_size_t()
        lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        lib.nvrtcGetPTX(prog, ptx_buf)
        return CompileResult(name=name, ptx=ptx_buf.value.decode(), log=log, arch=arch)
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))


def compile_file(path: str | Path, **kwargs: object) -> CompileResult:
    """Compile a ``.cu`` file by path."""
    p = Path(path)
    return compile_to_ptx(p.read_text(), name=p.name, **kwargs)  # type: ignore[arg-type]


def compile_kernel(stem: str, **kwargs: object) -> CompileResult:
    """Compile one of the bundled kernels by stem, e.g. ``compile_kernel("tiled_gemm")``."""
    return compile_file(KERNELS_DIR / f"{stem}.cu", **kwargs)


def ptx_to_cubin(ptx: str, arch: str = "sm_75") -> bytes:
    """Assemble PTX to a cubin (SASS) with ``ptxas``. Requires the ``cuda`` extra."""
    exe = ptxas_path()
    if exe is None:
        raise CudaToolchainError("ptxas not found. Install: pip install mlinfra[cuda]")
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ptx_path = Path(d) / "k.ptx"
        cubin_path = Path(d) / "k.cubin"
        ptx_path.write_text(ptx)
        subprocess.run(
            [exe, f"--gpu-name={arch}", str(ptx_path), "-o", str(cubin_path)],
            check=True,
            capture_output=True,
        )
        return cubin_path.read_bytes()


def list_kernels() -> list[str]:
    return sorted(p.stem for p in KERNELS_DIR.glob("*.cu"))
