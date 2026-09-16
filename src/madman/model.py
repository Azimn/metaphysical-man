from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Iterable

Signature = tuple[float, ...]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def dot(a: Signature, b: Signature) -> float:
    if len(a) != len(b):
        raise ValueError("signature dimensions differ")
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: Signature) -> float:
    return sqrt(sum(x * x for x in a))


def cosine(a: Signature, b: Signature) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return clamp(dot(a, b) / (na * nb), -1.0, 1.0)


def blend(a: Signature, b: Signature, weight: float) -> Signature:
    if len(a) != len(b):
        raise ValueError("signature dimensions differ")
    w = clamp(weight, 0.0, 1.0)
    return tuple((1.0 - w) * x + w * y for x, y in zip(a, b, strict=True))


def normalize(values: Iterable[float]) -> Signature:
    raw = tuple(float(v) for v in values)
    n = norm(raw)
    if n == 0.0:
        return raw
    return tuple(v / n for v in raw)


@dataclass(slots=True)
class ResourcePool:
    aether: float = 0.0
    ichor: float = 0.0
    azoth: float = 0.0
    ash: float = 0.0

    def conserved_mass(self) -> float:
        return self.aether + self.ichor + self.azoth + self.ash


@dataclass(slots=True)
class Site:
    site_id: int
    receptivity: Signature
    resources: ResourcePool
    activation: float = 0.0
    active_signature: Signature = field(default_factory=tuple)
    morrow: float = 0.0
    source_flux: float = 0.0
    report_surface: bool = False


@dataclass(slots=True)
class Edge:
    source: int
    target: int
    conductance: float
    scar: float = 0.0
    last_flux: float = 0.0

    def effective_conductance(self, source_ash: float, target_ash: float) -> float:
        residue = 1.0 + max(0.0, source_ash) + max(0.0, target_ash)
        return max(0.0, self.conductance + self.scar) / residue


@dataclass(frozen=True, slots=True)
class Spark:
    spark_id: int
    site_id: int
    signature: Signature
    intensity: float
    parent_ids: tuple[int, ...]
    origin: str
    generation: int = 0


@dataclass(frozen=True, slots=True)
class TraceEvent:
    step: int
    kind: str
    site_id: int | None
    detail: dict[str, object]
