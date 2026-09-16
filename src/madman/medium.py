from __future__ import annotations

from dataclasses import dataclass

from .crucible import Crucible


@dataclass(frozen=True, slots=True)
class PhenomenalImpression:
    site_id: int
    strength: float
    familiarity: float
    coherence: float


class Medium:
    """A deliberately disadvantaged report surface.

    The Medium can inspect only sites marked as report surfaces. It has no
    access to scars, hidden resources, causal ancestry, source flux, or the
    observer trace. Its output is interpretation, not privileged introspection.
    """

    def perceive(self, crucible: Crucible) -> tuple[PhenomenalImpression, ...]:
        impressions: list[PhenomenalImpression] = []
        for site_id, activation, signature, familiarity in crucible.surface_view():
            coherence = min(1.0, sum(abs(v) for v in signature) / max(1, len(signature)))
            if activation > 0.03:
                impressions.append(PhenomenalImpression(site_id, activation, familiarity, coherence))
        return tuple(impressions)

    def report(self, crucible: Crucible) -> str:
        impressions = self.perceive(crucible)
        if not impressions:
            return "Nothing is distinct enough for me to put into words."
        strongest = max(impressions, key=lambda x: x.strength)
        if strongest.strength > 1.2 and strongest.familiarity > 0.72:
            return "Something very strong and strangely familiar has taken over my attention."
        if strongest.strength > 0.75:
            return "Something is pressing hard enough that it is difficult to hold anything else."
        if strongest.familiarity > 0.76:
            return "Something about this feels familiar, although I cannot tell you why."
        if len(impressions) >= 3:
            return "Several things are present at once, and I cannot make them settle into one clear account."
        return "Something is present, but I only have a partial sense of what it is."
