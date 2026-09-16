from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
import random
from typing import Iterable

from .config import CrucibleConfig
from .model import Edge, ResourcePool, Site, Spark, TraceEvent, blend, clamp, cosine, normalize


class Crucible:
    """The MADMAN substrate.

    The Crucible is physics, not an agent. It owns sites, edges, resource flow,
    local transformation rules, and an observer trace. No site can query this
    object or inspect another site's state.
    """

    def __init__(self, sites: dict[int, Site], edges: list[Edge], config: CrucibleConfig, seed: int = 1):
        if not sites:
            raise ValueError("a Crucible needs at least one site")
        self.sites = sites
        self.edges = edges
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.step_index = 0
        self._spark_seq = 0
        self._inbox: dict[int, deque[Spark]] = {site_id: deque() for site_id in sites}
        self._outgoing: dict[int, list[Edge]] = defaultdict(list)
        self._incoming: dict[int, list[Edge]] = defaultdict(list)
        for edge in edges:
            if edge.source not in sites or edge.target not in sites:
                raise ValueError("edge endpoint missing from sites")
            self._outgoing[edge.source].append(edge)
            self._incoming[edge.target].append(edge)
        for collection in (self._outgoing, self._incoming):
            for edge_list in collection.values():
                edge_list.sort(key=lambda e: (e.source, e.target))
        self.trace: deque[TraceEvent] = deque(maxlen=config.trace_limit)
        self.total_injected = 0.0
        self.total_leaked = 0.0
        self.total_burned = 0.0
        self.total_initial_mass = self.resource_mass()

    @classmethod
    def random_lattice(
        cls,
        node_count: int = 24,
        extra_edges: int = 28,
        config: CrucibleConfig | None = None,
        seed: int = 1,
    ) -> "Crucible":
        cfg = config or CrucibleConfig()
        rng = random.Random(seed)
        sites: dict[int, Site] = {}
        for site_id in range(node_count):
            receptive = normalize(rng.uniform(-1.0, 1.0) for _ in range(cfg.signature_dim))
            sites[site_id] = Site(
                site_id=site_id,
                receptivity=receptive,
                resources=ResourcePool(
                    aether=cfg.initial_aether,
                    ichor=cfg.initial_ichor,
                    azoth=cfg.initial_azoth,
                ),
                active_signature=tuple(0.0 for _ in range(cfg.signature_dim)),
            )
        pairs: set[tuple[int, int]] = set()
        for site_id in range(node_count):
            pairs.add((site_id, (site_id + 1) % node_count))
            pairs.add(((site_id + 1) % node_count, site_id))
        while len(pairs) < min(node_count * (node_count - 1), 2 * node_count + extra_edges):
            source = rng.randrange(node_count)
            target = rng.randrange(node_count)
            if source != target:
                pairs.add((source, target))
        edges = [
            Edge(source=a, target=b, conductance=rng.uniform(0.12, 0.45))
            for a, b in sorted(pairs)
        ]
        return cls(sites=sites, edges=edges, config=cfg, seed=seed)

    def clone(self, seed: int | None = None) -> "Crucible":
        clone = Crucible(deepcopy(self.sites), deepcopy(self.edges), self.config, self.seed if seed is None else seed)
        clone.step_index = self.step_index
        clone._spark_seq = self._spark_seq
        clone._inbox = deepcopy(self._inbox)
        clone.total_injected = self.total_injected
        clone.total_leaked = self.total_leaked
        clone.total_burned = self.total_burned
        clone.total_initial_mass = self.total_initial_mass
        clone.trace = deepcopy(self.trace)
        return clone

    def fuse(self, other: "Crucible", weight: float = 0.5, seed: int | None = None) -> "Crucible":
        """Externally create a recombined history for branching experiments."""
        if set(self.sites) != set(other.sites):
            raise ValueError("fusion requires matching site sets")
        if [(e.source, e.target) for e in self.edges] != [(e.source, e.target) for e in other.edges]:
            raise ValueError("fusion requires matching topology")
        w = clamp(weight, 0.0, 1.0)
        child = self.clone(seed=self.seed if seed is None else seed)
        for site_id in child.sites:
            a = self.sites[site_id]
            b = other.sites[site_id]
            c = child.sites[site_id]
            c.activation = (1.0 - w) * a.activation + w * b.activation
            c.active_signature = normalize(blend(a.active_signature, b.active_signature, w))
            c.morrow = (1.0 - w) * a.morrow + w * b.morrow
            c.resources.aether = (1.0 - w) * a.resources.aether + w * b.resources.aether
            c.resources.ichor = (1.0 - w) * a.resources.ichor + w * b.resources.ichor
            c.resources.azoth = (1.0 - w) * a.resources.azoth + w * b.resources.azoth
            c.resources.ash = (1.0 - w) * a.resources.ash + w * b.resources.ash
        for idx, edge in enumerate(child.edges):
            left = self.edges[idx]
            right = other.edges[idx]
            edge.scar = (1.0 - w) * left.scar + w * right.scar
            edge.conductance = (1.0 - w) * left.conductance + w * right.conductance
        child.total_initial_mass = child.resource_mass()
        child.total_injected = 0.0
        child.total_leaked = 0.0
        child.total_burned = 0.0
        child.trace.clear()
        child._record("fusion", None, weight=w, left_seed=self.seed, right_seed=other.seed)
        return child

    def set_source(self, site_id: int, flux: float) -> None:
        self.sites[site_id].source_flux = max(0.0, flux)

    def set_report_surface(self, site_ids: Iterable[int]) -> None:
        for site in self.sites.values():
            site.report_surface = False
        for site_id in site_ids:
            self.sites[site_id].report_surface = True

    def perturb(self, site_id: int, signature: Iterable[float], intensity: float, origin: str = "world") -> int:
        sig = normalize(signature)
        if len(sig) != self.config.signature_dim:
            raise ValueError(f"expected signature dimension {self.config.signature_dim}")
        self._spark_seq += 1
        spark = Spark(self._spark_seq, site_id, sig, max(0.0, intensity), (), origin, 0)
        self._inbox[site_id].append(spark)
        self._record("perturbation", site_id, spark_id=spark.spark_id, origin=origin, intensity=spark.intensity)
        return spark.spark_id

    def run(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        for _ in range(steps):
            self.step()

    def step(self) -> None:
        self.step_index += 1
        self._inject_sources()
        self._relax_morrow()
        self._diffuse_aether()
        self._release_ichor()
        processed = self._process_sparks()
        self._spontaneous_ignition(processed)
        self._plasticity()
        self._alchemy()
        self._decay()

    def resource_mass(self) -> float:
        return sum(site.resources.conserved_mass() for site in self.sites.values())

    def conservation_error(self) -> float:
        expected = self.total_initial_mass + self.total_injected - self.total_leaked - self.total_burned
        return self.resource_mass() - expected

    def active_vector(self) -> tuple[float, ...]:
        return tuple(self.sites[i].activation for i in sorted(self.sites))

    def surface_view(self) -> tuple[tuple[int, float, tuple[float, ...], float], ...]:
        """Return only organism-relative report-surface consequences."""
        return tuple(
            (
                site.site_id,
                site.activation,
                site.active_signature,
                0.5 + 0.5 * cosine(site.active_signature, site.receptivity),
            )
            for site in sorted(self.sites.values(), key=lambda s: s.site_id)
            if site.report_surface
        )

    def topology_snapshot(self) -> tuple[tuple[int, int, float, float], ...]:
        return tuple((e.source, e.target, e.conductance, e.scar) for e in sorted(self.edges, key=lambda x: (x.source, x.target)))

    def _inject_sources(self) -> None:
        for site in self.sites.values():
            amount = site.source_flux
            if amount <= 0.0:
                continue
            site.resources.aether += amount
            self.total_injected += amount
            self._record("aether_source", site.site_id, amount=amount)

    def _relax_morrow(self) -> None:
        cfg = self.config
        next_values: dict[int, float] = {}
        for site_id, site in self.sites.items():
            neighbor_pressure = 0.0
            for edge in self._outgoing[site_id]:
                target = self.sites[edge.target]
                neighbor_pressure = max(
                    neighbor_pressure,
                    target.morrow * edge.effective_conductance(site.resources.ash, target.resources.ash),
                )
            target_value = site.source_flux + cfg.morrow_neighbor_gain * neighbor_pressure
            next_values[site_id] = (1.0 - cfg.morrow_relaxation) * site.morrow + cfg.morrow_relaxation * target_value
        for site_id, value in next_values.items():
            self.sites[site_id].morrow = max(0.0, value)

    def _diffuse_aether(self) -> None:
        cfg = self.config
        delta = defaultdict(float)
        for edge in self.edges:
            source = self.sites[edge.source]
            target = self.sites[edge.target]
            gradient = source.resources.aether - target.resources.aether
            if gradient <= 0.0:
                edge.last_flux = 0.0
                continue
            conductance = edge.effective_conductance(source.resources.ash, target.resources.ash)
            amount = min(source.resources.aether, gradient * cfg.diffusion_rate * conductance)
            if amount <= 0.0:
                edge.last_flux = 0.0
                continue
            delta[edge.source] -= amount
            delta[edge.target] += amount
            edge.last_flux = amount
        for site_id, amount in delta.items():
            self.sites[site_id].resources.aether += amount

    def _release_ichor(self) -> None:
        cfg = self.config
        for site in self.sites.values():
            if site.activation < cfg.reserve_release_threshold:
                continue
            release = min(site.resources.ichor, site.resources.ichor * cfg.reserve_release)
            if release > 0.0:
                site.resources.ichor -= release
                site.resources.aether += release
                self._record("ichor_release", site.site_id, amount=release)

    def _process_sparks(self) -> set[int]:
        processed_sites: set[int] = set()
        for site_id in sorted(self.sites):
            inbox = self._inbox[site_id]
            local_count = len(inbox)
            for _ in range(local_count):
                spark = inbox.popleft()
                if self._assimilate(spark):
                    processed_sites.add(site_id)
        return processed_sites

    def _assimilate(self, spark: Spark) -> bool:
        cfg = self.config
        site = self.sites[spark.site_id]
        resonance = 0.5 + 0.5 * cosine(site.receptivity, spark.signature)
        effective_intensity = spark.intensity * (0.25 + 0.75 * resonance)
        cost = cfg.base_assimilation_cost * (1.0 + site.resources.ash) * (0.65 + effective_intensity)
        if site.resources.aether < cost:
            self._record("spark_starved", site.site_id, spark_id=spark.spark_id, required=cost, available=site.resources.aether)
            return False
        site.resources.aether -= cost
        self.total_burned += cost * (1.0 - cfg.ash_yield)
        site.resources.ash += cost * cfg.ash_yield
        previous_activation = site.activation
        site.activation = clamp(site.activation + effective_intensity, 0.0, 3.0)
        if previous_activation <= 0.001:
            site.active_signature = spark.signature
        else:
            site.active_signature = normalize(blend(site.active_signature, spark.signature, min(0.8, effective_intensity / (1.0 + site.activation))))
        capture = min(site.resources.aether, effective_intensity * cfg.reserve_capture)
        site.resources.aether -= capture
        site.resources.ichor += capture
        self._record(
            "spark_assimilated",
            site.site_id,
            spark_id=spark.spark_id,
            origin=spark.origin,
            resonance=resonance,
            intensity=effective_intensity,
            activation=site.activation,
        )
        if site.activation >= cfg.propagation_threshold and spark.generation < cfg.max_generation:
            self._propagate(site, spark)
        return True

    def _propagate(self, site: Site, parent: Spark) -> None:
        cfg = self.config
        candidates: list[tuple[float, Edge]] = []
        for edge in self._outgoing[site.site_id]:
            target = self.sites[edge.target]
            conductance = edge.effective_conductance(site.resources.ash, target.resources.ash)
            morrow_multiplier = 1.0 + cfg.morrow_bias * target.morrow
            score = conductance * morrow_multiplier
            candidates.append((score, edge))
        candidates.sort(key=lambda item: (-item[0], item[1].target))
        for score, edge in candidates:
            if score <= 0.0:
                continue
            cost = cfg.propagation_cost * (1.0 + site.resources.ash)
            if site.resources.aether < cost:
                break
            site.resources.aether -= cost
            self.total_burned += cost * (1.0 - cfg.ash_yield)
            site.resources.ash += cost * cfg.ash_yield
            child_intensity = parent.intensity * min(1.15, score) * 0.72
            if child_intensity < 0.025:
                continue
            self._spark_seq += 1
            child = Spark(
                spark_id=self._spark_seq,
                site_id=edge.target,
                signature=site.active_signature,
                intensity=child_intensity,
                parent_ids=(parent.spark_id,),
                origin="propagation",
                generation=parent.generation + 1,
            )
            self._inbox[edge.target].append(child)
            self._record("spark_propagated", site.site_id, spark_id=child.spark_id, target=edge.target, score=score)

    def _spontaneous_ignition(self, processed_sites: set[int]) -> None:
        cfg = self.config
        for site_id in sorted(self.sites):
            if site_id in processed_sites:
                continue
            site = self.sites[site_id]
            incoming_scar = sum(e.scar for e in self._incoming[site_id])
            drive = site.resources.ichor * (0.35 + incoming_scar) + site.morrow * 0.05
            noise = self.rng.uniform(0.0, cfg.spontaneous_noise)
            if drive + noise < cfg.reconstruction_threshold:
                continue
            available = site.resources.aether + site.resources.ichor
            if available < cfg.base_assimilation_cost:
                continue
            signature = site.active_signature
            if not signature or sum(abs(v) for v in signature) < 1e-9:
                signature = site.receptivity
            intensity = min(0.5, 0.12 + drive * 0.22)
            reserve_cost = min(site.resources.ichor, cfg.reconstruction_reserve_cost * (1.0 + intensity))
            if reserve_cost <= 0.0:
                continue
            site.resources.ichor -= reserve_cost
            self.total_burned += reserve_cost
            self._spark_seq += 1
            spark = Spark(self._spark_seq, site_id, signature, intensity, (), "reconstruction", 0)
            self._inbox[site_id].append(spark)
            self._record("spontaneous_reconstruction", site_id, spark_id=spark.spark_id, drive=drive, reserve_cost=reserve_cost)

    def _plasticity(self) -> None:
        cfg = self.config
        for edge in self.edges:
            source = self.sites[edge.source]
            target = self.sites[edge.target]
            coactivation = min(source.activation, target.activation)
            if coactivation <= 0.05:
                edge.scar *= cfg.scar_decay
                continue
            potential = min(source.resources.azoth, target.resources.azoth)
            if potential <= cfg.plasticity_cost:
                edge.scar *= cfg.scar_decay
                continue
            similarity = 0.5 + 0.5 * cosine(source.active_signature, target.active_signature)
            delta = cfg.plasticity_rate * coactivation * similarity
            spend = min(potential, cfg.plasticity_cost * (1.0 + coactivation))
            source.resources.azoth -= spend * 0.5
            target.resources.azoth -= spend * 0.5
            self.total_burned += spend
            edge.scar = clamp(edge.scar * cfg.scar_decay + delta, 0.0, cfg.max_scar)
            self._record("scar_changed", edge.target, source=edge.source, delta=delta, scar=edge.scar)

    def _alchemy(self) -> None:
        """Local resource conversion. No process chooses this conversion."""
        cfg = self.config
        for site in self.sites.values():
            if site.activation > 0.12 or site.resources.ash > 0.35:
                continue
            room = max(0.0, cfg.azoth_cap - site.resources.azoth)
            conversion = min(site.resources.aether, room, site.resources.aether * cfg.azoth_recovery_rate)
            if conversion <= 0.0:
                continue
            site.resources.aether -= conversion
            site.resources.azoth += conversion
            self._record("aether_to_azoth", site.site_id, amount=conversion)

    def _decay(self) -> None:
        cfg = self.config
        for site in self.sites.values():
            site.activation *= cfg.activation_decay
            ash_loss = site.resources.ash * (1.0 - cfg.ash_decay)
            site.resources.ash -= ash_loss
            self.total_leaked += ash_loss
            ichor_loss = site.resources.ichor * (1.0 - cfg.ichor_decay)
            site.resources.ichor -= ichor_loss
            self.total_leaked += ichor_loss
            leak = site.resources.aether * cfg.aether_leak
            site.resources.aether -= leak
            self.total_leaked += leak
        for edge in self.edges:
            edge.scar *= cfg.scar_decay

    def _record(self, kind: str, site_id: int | None, **detail: object) -> None:
        self.trace.append(TraceEvent(self.step_index, kind, site_id, detail))

    def observer_state(self) -> dict[str, object]:
        """Full third-person state for instrumentation only."""
        return {
            "step": self.step_index,
            "sites": {site_id: asdict(site) for site_id, site in self.sites.items()},
            "edges": [asdict(edge) for edge in self.edges],
            "resource_mass": self.resource_mass(),
            "conservation_error": self.conservation_error(),
        }
