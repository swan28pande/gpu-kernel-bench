"""gpubench: benchmark suite for custom CUDA / CUTLASS / Triton kernels."""

from . import cuda_kernels, triton_kernels, torch_baselines  # noqa: F401  (registers impls)
