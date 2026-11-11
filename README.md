# gpu-kernel-bench

A small benchmarking suite for custom GPU kernels written in CUDA, CUTLASS and Triton. For every kernel and shape it:

1. **checks correctness** against a high-precision PyTorch reference (`torch.testing.assert_close`, with tolerances per op),
2. **times** it with L2 flushing between reps (Triton's `do_bench`, or CUDA events as a fallback) and reports the median, p20 and p80 in ms, plus TFLOP/s and GB/s,
3. **profiles** it (optionally) with `torch.profiler`, saving a Chrome trace and the top GPU kernels it launched.

Each kernel is compared against a PyTorch baseline (cuBLAS / ATen).

**Stack:** CUDA · CUTLASS · Triton · Python / PyTorch

## Kernels

| Op | CUDA | CUTLASS | Triton | Baseline |
|---|---|---|---|---|
| `vector_add` (fp32) | grid-stride, float4-vectorised | | blocked | `torch.add` |
| `softmax` (fp32, row-wise) | block per row, warp-shuffle reductions | | row per program | `torch.softmax` |
| `matmul` (fp16 in/out, fp32 acc) | | 2.x device GEMM, tensor cores, SM80 | tiled, grouped ordering, autotuned | cuBLAS |

## Usage

You need an NVIDIA GPU, a CUDA toolkit (`nvcc`) and Linux for Triton.

```bash
pip install -e .
git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/cutlass   # optional
export CUTLASS_DIR=~/cutlass

python -m gpubench list                                  # ops and backends
python -m gpubench run                                   # everything, default shapes
python -m gpubench run --ops matmul --backends triton,cutlass,torch --profile
python -m gpubench check --quick                         # correctness only
```

The CUDA sources are JIT-compiled through `torch.utils.cpp_extension` on first use, which takes about a minute, and cached after that. Results go to `results/results.{json,csv}`, and profiler traces go to `results/traces/`, where you can open them in Perfetto or `chrome://tracing`.

Backends that can't run on the current machine (no GPU, no `CUTLASS_DIR`, no Triton) show up as `skipped` with the reason. `check` exits non-zero if any kernel mismatches the reference or errors.

### Without a GPU

The Triton kernels can be checked on CPU through Triton's interpreter:

```bash
TRITON_INTERPRET=1 python -m gpubench check --quick
```

CI does this, and also compiles the `.cu` sources with `nvcc` against CUTLASS. GitHub runners have no GPU, so CI covers correctness and compilation, not performance.

## Adding a kernel

Register a function for an existing op and it gets checked, timed and profiled automatically:

```python
from gpubench.registry import register

@register("softmax", "my_backend", unavailable=lambda device: None if device == "cuda" else "needs CUDA")
def softmax(x):
    ...
```

A new op is one `Op(...)` entry in `gpubench/ops.py`: input generator, reference, FLOP/byte model, shapes and tolerances.

## Layout

```
gpubench/
  __main__.py        CLI (list / run / check)
  bench.py           correctness, timing, profiling, result export
  ops.py             op definitions
  registry.py        (op, backend) -> implementation
  triton_kernels.py  Triton kernels
  cuda_kernels.py    JIT build + bindings for csrc/
  torch_baselines.py PyTorch baselines
  csrc/              CUDA kernels, CUTLASS GEMM, PyTorch binding
```
