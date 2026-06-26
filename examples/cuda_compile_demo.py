"""Compile the bundled CUDA kernels to PTX (and cubin) — runs on CPU, no GPU needed.

    pip install -e ".[cuda]"
    python examples/cuda_compile_demo.py
"""

from __future__ import annotations

from mlinfra.cuda import (
    compile_kernel,
    gpu_available,
    list_kernels,
    nvrtc_available,
    ptx_to_cubin,
    ptxas_path,
)


def main() -> None:
    if not nvrtc_available():
        print("CUDA toolchain not installed. Run: pip install -e '.[cuda]'")
        return

    for stem in list_kernels():
        result = compile_kernel(stem)
        line = f"{stem:<14} arch={result.arch}  PTX={len(result.ptx):>5}B  " \
               f"~instrs={result.num_instructions}"
        if ptxas_path():
            cubin = ptx_to_cubin(result.ptx, arch="sm_75")
            line += f"  cubin={len(cubin)}B (SASS)"
        print(line)

    print(f"\nGPU available for launch: {gpu_available()}")
    if not gpu_available():
        print("(compile path verified; launch requires a GPU host — see mlinfra.cuda.runtime)")


if __name__ == "__main__":
    main()
