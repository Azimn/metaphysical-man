from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import log

from .crucible import Crucible
from .model import cosine


@dataclass(frozen=True, slots=True)
class Constellation:
    members: tuple[int, ...]
    closure: float
    activation: float
    resource_share: float


class GhostHunter:
    """Observer-only instrumentation. Nothing in the Crucible can call this."""

    @staticmethod
    def flow_entropy(crucible: Crucible) -> float:
        weights = [max(0.0, edge.last_flux) for edge in crucible.edges]
        total = sum(weights)
        if total <= 0.0:
            return 0.0
        probs = [w / total for w in weights if w > 0.0]
        return -sum(p * log(p + 1e-15) for p in probs)

    @staticmethod
    def metabolic_dominance(crucible: Crucible) -> tuple[int, float]:
        weighted = {
            site_id: site.activation * (1.0 + site.resources.ichor) / (1.0 + site.resources.ash)
            for site_id, site in crucible.sites.items()
        }
        winner = max(weighted, key=weighted.get)
        total = sum(max(0.0, v) for v in weighted.values())
        return winner, 0.0 if total == 0.0 else weighted[winner] / total

    @staticmethod
    def constellations(crucible: Crucible, threshold: float = 0.28) -> tuple[Constellation, ...]:
        adjacency: dict[int, set[int]] = defaultdict(set)
        for edge in crucible.edges:
            source = crucible.sites[edge.source]
            target = crucible.sites[edge.target]
            strength = edge.effective_conductance(source.resources.ash, target.resources.ash)
            if strength >= threshold and source.activation > 0.08 and target.activation > 0.08:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        remaining = set(adjacency)
        groups: list[Constellation] = []
        total_aether = sum(s.resources.aether + s.resources.ichor for s in crucible.sites.values()) or 1.0
        while remaining:
            start = min(remaining)
            q = deque([start])
            group: set[int] = set()
            while q:
                node = q.popleft()
                if node in group:
                    continue
                group.add(node)
                for nxt in adjacency[node]:
                    if nxt not in group:
                        q.append(nxt)
            remaining -= group
            if len(group) < 2:
                continue
            internal = 0.0
            external = 0.0
            for edge in crucible.edges:
                weight = edge.conductance + edge.scar
                if edge.source in group and edge.target in group:
                    internal += weight
                elif edge.source in group or edge.target in group:
                    external += weight
            closure = internal / (internal + external + 1e-12)
            activation = sum(crucible.sites[i].activation for i in group)
            resource_share = sum(
                crucible.sites[i].resources.aether + crucible.sites[i].resources.ichor for i in group
            ) / total_aether
            groups.append(Constellation(tuple(sorted(group)), closure, activation, resource_share))
        return tuple(sorted(groups, key=lambda c: (-c.closure, -c.activation, c.members)))

    @staticmethod
    def echo_similarity(before: tuple[tuple[float, ...], ...], crucible: Crucible, members: tuple[int, ...]) -> float:
        if not before or not members:
            return 0.0
        sims: list[float] = []
        for site_id in members:
            if site_id >= len(before):
                continue
            sims.append(0.5 + 0.5 * cosine(before[site_id], crucible.sites[site_id].active_signature))
        return 0.0 if not sims else sum(sims) / len(sims)
