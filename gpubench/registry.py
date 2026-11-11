"""Registry of kernel implementations, keyed by (op, backend)."""

from dataclasses import dataclass
from typing import Callable, Optional

# Returns None when the implementation can run on `device`, else a reason it can't.
Availability = Callable[[str], Optional[str]]


@dataclass
class Impl:
    op: str
    backend: str
    fn: Callable
    unavailable: Availability


IMPLS: list[Impl] = []


def register(op: str, backend: str, unavailable: Availability = lambda device: None):
    def deco(fn):
        IMPLS.append(Impl(op, backend, fn, unavailable))
        return fn

    return deco


def impls_for(op: str, backends: Optional[list[str]] = None) -> list[Impl]:
    return [i for i in IMPLS if i.op == op and (not backends or i.backend in backends)]
