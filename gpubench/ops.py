"""Benchmark ops: input generation, reference result, cost model, tolerances."""

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Op:
    name: str
    make_inputs: Callable[[tuple, str], tuple]
    reference: Callable[..., torch.Tensor]
    flops: Callable[[tuple], float]
    bytes: Callable[[tuple], float]
    sizes: list[tuple]
    quick_sizes: list[tuple]   # small shapes for fast checks (e.g. Triton interpreter on CPU)
    atol: float
    rtol: float


def _vec_inputs(size, device):
    (n,) = size
    return torch.randn(n, device=device), torch.randn(n, device=device)


def _softmax_inputs(size, device):
    m, n = size
    return (torch.randn(m, n, device=device),)


def _matmul_inputs(size, device):
    m, n, k = size
    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)
    return a, b


OPS: dict[str, Op] = {
    op.name: op
    for op in [
        Op(
            name="vector_add",
            make_inputs=_vec_inputs,
            reference=lambda x, y: x + y,
            flops=lambda s: s[0],
            bytes=lambda s: 3 * 4 * s[0],
            sizes=[(1 << 20,), (1 << 24,)],
            quick_sizes=[(4099,)],
            atol=0.0,
            rtol=0.0,
        ),
        Op(
            name="softmax",
            make_inputs=_softmax_inputs,
            reference=lambda x: torch.softmax(x.double(), dim=-1).float(),
            flops=lambda s: 5 * s[0] * s[1],
            bytes=lambda s: 2 * 4 * s[0] * s[1],
            sizes=[(4096, 1024), (4096, 4096)],
            quick_sizes=[(7, 1000)],
            atol=1e-5,
            rtol=1e-4,
        ),
        Op(
            name="matmul",
            make_inputs=_matmul_inputs,
            # fp32 accumulate, fp16 output: what every fp16 GEMM backend here does.
            reference=lambda a, b: (a.float() @ b.float()).half(),
            flops=lambda s: 2 * s[0] * s[1] * s[2],
            bytes=lambda s: 2 * (s[0] * s[2] + s[2] * s[1] + s[0] * s[1]),
            sizes=[(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 1024)],
            quick_sizes=[(64, 96, 128)],
            atol=1e-2,
            rtol=1e-2,
        ),
    ]
}
