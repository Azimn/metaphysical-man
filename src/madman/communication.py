from __future__ import annotations

from dataclasses import dataclass
import random

from .crucible import Crucible
from .model import normalize


@dataclass(frozen=True, slots=True)
class Glyph:
    signature: tuple[float, ...]
    intensity: float
    origin: str


class VeilChannel:
    """Communication as perturbation rather than state transfer."""

    def __init__(self, dimension: int, seed: int = 1, distortion: float = 0.12):
        self.dimension = dimension
        self.distortion = max(0.0, distortion)
        rng = random.Random(seed)
        self._bias = tuple(rng.uniform(-self.distortion, self.distortion) for _ in range(dimension))

    def emit(self, crucible: Crucible, site_id: int, intensity: float = 0.6) -> Glyph:
        site = crucible.sites[site_id]
        signature = site.active_signature if sum(abs(v) for v in site.active_signature) > 1e-9 else site.receptivity
        return Glyph(signature=signature, intensity=max(0.0, intensity * max(0.1, site.activation)), origin=f"crucible:{crucible.seed}:site:{site_id}")

    def receive(self, crucible: Crucible, site_id: int, glyph: Glyph) -> int:
        transformed = normalize(v + b for v, b in zip(glyph.signature, self._bias, strict=True))
        return crucible.perturb(site_id, transformed, glyph.intensity, origin=f"glyph:{glyph.origin}")
