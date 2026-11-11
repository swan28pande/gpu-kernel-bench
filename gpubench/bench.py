"""Benchmark loop: correctness check -> timing -> (optional) profiler trace."""

import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .ops import OPS, Op
from .registry import Impl, impls_for


@dataclass
class Result:
    op: str
    backend: str
    size: str
    status: str                       # ok | mismatch | error | skipped
    detail: str = ""
    ms_median: Optional[float] = None
    ms_p20: Optional[float] = None
    ms_p80: Optional[float] = None
    tflops: Optional[float] = None
    gbps: Optional[float] = None
    top_kernels: list = field(default_factory=list)


def _time_cuda(fn, warmup_ms=25, rep_ms=100) -> tuple[float, float, float]:
    """(median, p20, p80) in ms. Uses Triton's do_bench (flushes L2 between reps) if present."""
    try:
        from triton.testing import do_bench

        p50, p20, p80 = do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.5, 0.2, 0.8])
        return p50, p20, p80
    except ImportError:
        pass
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        fn()
    times = []
    for _ in range(100):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    q = statistics.quantiles(times, n=10)
    return statistics.median(times), q[1], q[7]


def _profile(fn, trace_path: Path, iters: int = 5) -> list[dict]:
    """Record a short torch.profiler trace; return the top GPU kernels by time."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    def gpu_us(evt):  # attribute name changed across torch versions
        return getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0)

    kernels = [e for e in prof.key_averages() if gpu_us(e) > 0 and e.key[:4] not in ("aten", "cuda")]
    kernels.sort(key=gpu_us, reverse=True)
    return [{"kernel": e.key[:80], "us_per_call": gpu_us(e) / max(e.count, 1), "calls": e.count}
            for e in kernels[:3]]


def _run_one(op: Op, impl: Impl, size: tuple, inputs, ref, device, profile_dir) -> Result:
    r = Result(op.name, impl.backend, "x".join(map(str, size)), "ok")
    reason = impl.unavailable(device)
    if reason:
        r.status, r.detail = "skipped", reason
        return r

    fn = lambda: impl.fn(*inputs)  # noqa: E731
    try:
        out = fn()
        torch.testing.assert_close(out, ref, atol=op.atol, rtol=op.rtol)
    except AssertionError as e:
        r.status, r.detail = "mismatch", str(e).strip().splitlines()[0]
    except Exception as e:  # build failures, launch errors, unsupported shapes
        r.status, r.detail = "error", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200]}"
        return r

    if device == "cuda":
        r.ms_median, r.ms_p20, r.ms_p80 = _time_cuda(fn)
        r.tflops = op.flops(size) / (r.ms_median * 1e-3) / 1e12
        r.gbps = op.bytes(size) / (r.ms_median * 1e-3) / 1e9
        if profile_dir:
            r.top_kernels = _profile(fn, profile_dir / f"{op.name}_{impl.backend}_{r.size}.json")
    return r


def run(ops: list[str], backends: Optional[list[str]], quick: bool, profile_dir: Optional[Path],
        device: str, seed: int = 0) -> list[Result]:
    results = []
    for name in ops:
        op = OPS[name]
        for size in op.quick_sizes if quick else op.sizes:
            torch.manual_seed(seed)
            inputs = op.make_inputs(size, device)
            ref = op.reference(*inputs)
            for impl in impls_for(name, backends):
                results.append(_run_one(op, impl, size, inputs, ref, device, profile_dir))
                print(_fmt_row(results[-1]), flush=True)
    return results


HEADER = f"{'op':<11} {'backend':<8} {'size':<16} {'status':<9} {'ms':>9} {'TFLOP/s':>8} {'GB/s':>8}  detail"


def _fmt_row(r: Result) -> str:
    num = lambda v, f: format(v, f) if v is not None else "-"  # noqa: E731
    return (f"{r.op:<11} {r.backend:<8} {r.size:<16} {r.status:<9} {num(r.ms_median, '9.4f'):>9} "
            f"{num(r.tflops, '8.2f'):>8} {num(r.gbps, '8.1f'):>8}  {r.detail}")


def save(results: list[Result], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "top_kernels"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
