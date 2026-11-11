"""PyTorch baselines (cuBLAS / ATen kernels) to compare custom kernels against."""

import torch

from .registry import register


@register("vector_add", "torch")
def vector_add(x, y):
    return torch.add(x, y)


@register("softmax", "torch")
def softmax(x):
    return torch.softmax(x, dim=-1)


@register("matmul", "torch")
def matmul(a, b):
    return a @ b
