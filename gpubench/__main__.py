"""CLI.

  python -m gpubench list
  python -m gpubench run   [--ops matmul,softmax] [--backends triton,cutlass,torch] [--profile] [--out results]
  python -m gpubench check [--ops ...] [--quick]     # correctness only, no timing
"""

import argparse
import sys
from pathlib import Path

import torch

from . import bench
from .ops import OPS
from .registry import IMPLS


def _csv(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> int:
    p = argparse.ArgumentParser(prog="gpubench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list ops and backends")
    for name in ("run", "check"):
        sp = sub.add_parser(name)
        sp.add_argument("--ops", type=_csv, default=list(OPS))
        sp.add_argument("--backends", type=_csv, default=None)
        sp.add_argument("--quick", action="store_true", help="use small shapes")
        sp.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if name == "run":
            sp.add_argument("--profile", action="store_true", help="save torch.profiler traces")
            sp.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    if args.cmd == "list":
        for op in OPS:
            print(f"{op:<11} {', '.join(i.backend for i in IMPLS if i.op == op)}")
        return 0

    unknown = set(args.ops) - set(OPS)
    if unknown:
        p.error(f"unknown ops: {', '.join(sorted(unknown))}")

    if args.cmd == "run" and args.device != "cuda":
        print("note: no CUDA device, so results are correctness-only (no timings)", file=sys.stderr)

    print(bench.HEADER)
    profile_dir = args.out / "traces" if args.cmd == "run" and args.profile else None
    results = bench.run(args.ops, args.backends, args.quick, profile_dir, args.device)
    if args.cmd == "run":
        bench.save(results, args.out)
        print(f"\nwrote {args.out}/results.json and results.csv")

    failed = [r for r in results if r.status in ("mismatch", "error")]
    ran = [r for r in results if r.status != "skipped"]
    print(f"\n{len(ran) - len(failed)}/{len(ran)} passed, {len(results) - len(ran)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
