"""Triton kernels.

With TRITON_INTERPRET=1 these run on CPU tensors through Triton's interpreter,
which is useful for correctness checks on machines without a GPU.
"""

import os

import torch

from .registry import register

try:
    import triton
    import triton.language as tl
except ImportError:  # e.g. macOS, where Triton isn't published
    triton = None

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _unavailable(device: str):
    if triton is None:
        return "triton not installed"
    if device != "cuda" and not INTERPRET:
        return "needs CUDA (or TRITON_INTERPRET=1)"
    return None


if triton is None:
    # Still list the backend so results show it as skipped rather than absent.
    for _op in ("vector_add", "softmax", "matmul"):
        register(_op, "triton", _unavailable)(lambda *args: None)

else:

    # ---------------------------------------------------------------- vector add
    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @register("vector_add", "triton", _unavailable)
    def vector_add(x, y):
        out = torch.empty_like(x)
        n = x.numel()
        _add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        return out

    # ------------------------------------------------------------------- softmax
    @triton.jit
    def _softmax_kernel(in_ptr, out_ptr, in_stride, out_stride, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        tl.store(out_ptr + row * out_stride + cols, num / tl.sum(num, axis=0), mask=mask)

    @register("softmax", "triton", _unavailable)
    def softmax(x):
        m, n = x.shape
        block = triton.next_power_of_2(n)  # one program holds a whole row
        warps = 4 if block <= 2048 else 8 if block <= 4096 else 16
        out = torch.empty_like(x)
        _softmax_kernel[(m,)](x, out, x.stride(0), out.stride(0), n, BLOCK=block, num_warps=warps)
        return out

    # -------------------------------------------------------------------- matmul
    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        # Grouped launch order so neighbouring programs reuse tiles of A in L2.
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BM)
        num_n = tl.cdiv(N, BN)
        per_group = GROUP_M * num_n
        first_m = (pid // per_group) * GROUP_M
        group_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % per_group) % group_m
        pid_n = (pid % per_group) // group_m

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            k_left = K - k * BK
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    _CONFIGS = [
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    ]
    # Autotuning benchmarks on the GPU, so the interpreter gets one fixed config.
    _matmul = _matmul_kernel if INTERPRET else triton.autotune(_CONFIGS, key=["M", "N", "K"])(_matmul_kernel)

    @register("matmul", "triton", _unavailable)
    def matmul(a, b):
        (m, k), (_, n) = a.shape, b.shape
        c = torch.empty((m, n), device=a.device, dtype=torch.float16)
        grid = lambda meta: (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)
        extra = {"BM": 32, "BN": 32, "BK": 32, "GROUP_M": 4} if INTERPRET else {}
        _matmul[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            **extra,
        )
        return c
