"""Hand-written CUDA kernels (and an optional CUTLASS GEMM), JIT-built as a torch extension.

The build happens on first use and is cached by torch in ~/.cache/torch_extensions.
Set CUTLASS_DIR to a CUTLASS checkout to also build the CUTLASS GEMM.
"""

import functools
import os
import shutil
from pathlib import Path

from .registry import register

CSRC = Path(__file__).parent / "csrc"
CUTLASS_DIR = os.environ.get("CUTLASS_DIR")


def _unavailable(device: str):
    if device != "cuda":
        return "needs CUDA"
    if not (shutil.which("nvcc") or os.environ.get("CUDA_HOME")):
        return "nvcc not found (set CUDA_HOME)"
    return None


def _cutlass_unavailable(device: str):
    return _unavailable(device) or (None if CUTLASS_DIR else "CUTLASS_DIR not set")


@functools.cache
def ext():
    from torch.utils.cpp_extension import load

    sources = [CSRC / "binding.cpp", CSRC / "vector_add.cu", CSRC / "softmax.cu"]
    flags, includes = [], []
    if CUTLASS_DIR:
        sources.append(CSRC / "cutlass_gemm.cu")
        flags.append("-DWITH_CUTLASS")
        includes += [f"{CUTLASS_DIR}/include", f"{CUTLASS_DIR}/tools/util/include"]
    return load(
        name="gpubench_cuda" + ("_cutlass" if CUTLASS_DIR else ""),
        sources=[str(s) for s in sources],
        extra_cflags=["-O3", *flags],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", *flags],
        extra_include_paths=includes,
        verbose=os.environ.get("GPUBENCH_VERBOSE_BUILD") == "1",
    )


@register("vector_add", "cuda", _unavailable)
def vector_add(x, y):
    return ext().vector_add(x, y)


@register("softmax", "cuda", _unavailable)
def softmax(x):
    return ext().softmax(x)


@register("matmul", "cutlass", _cutlass_unavailable)
def matmul(a, b):
    return ext().cutlass_gemm(a, b)
