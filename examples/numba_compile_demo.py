"""Compile numba.cuda kernels (written in Python) to PTX on CPU — no GPU needed.

    pip install -e ".[numba]"
    python examples/numba_compile_demo.py
"""

from __future__ import annotations

from mlinfra.cuda import (
    compile_saxpy_ptx,
    compile_softmax_ptx,
    numba_available,
    triton_available,
    triton_gpu_ready,
)


def main() -> None:
    if not numba_available():
        print("numba toolchain not installed. Run: pip install -e '.[numba]'")
        return

    for name, fn in [("saxpy", compile_saxpy_ptx), ("softmax_rows", compile_softmax_ptx)]:
        for cc in [(7, 5), (8, 0), (9, 0)]:
            ptx = fn(cc=cc)
            target = next(line for line in ptx.splitlines() if line.startswith(".target"))
            print(f"{name:<14} cc={cc[0]}.{cc[1]}  PTX={len(ptx):>6}B  {target}")

    print(f"\ntriton available: {triton_available()}  |  triton GPU ready: {triton_gpu_ready()}")
    if triton_available() and not triton_gpu_ready():
        print("(triton kernels defined; compile/launch needs a GPU — see mlinfra.cuda.triton_kernels)")


if __name__ == "__main__":
    main()
