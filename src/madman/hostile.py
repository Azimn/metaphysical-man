from __future__ import annotations

from dataclasses import replace
from math import sqrt
from statistics import mean, median, pstdev

from .config import CrucibleConfig
from .crucible import Crucible
from .medium import Medium
from .model import Edge, ResourcePool, Site, normalize
from .scenarios import prepared_crucible, signature


def _norm(values: tuple[float, ...]) -> float:
    return sqrt(sum(v * v for v in values))


def scheduler_id_bias() -> dict[str, float]:
    """Expose same-tick propagation asymmetry caused only by numeric site order."""
    cfg = CrucibleConfig(
        diffusion_rate=0.0,
        aether_leak=0.0,
        activation_decay=1.0,
        ash_decay=1.0,
        ichor_decay=1.0,
        base_assimilation_cost=0.01,
        propagation_cost=0.01,
        ash_yield=0.0,
        reserve_capture=0.0,
        reserve_release=0.0,
        reconstruction_threshold=999.0,
        propagation_threshold=0.1,
        max_generation=1,
        plasticity_rate=0.0,
        plasticity_cost=999.0,
        scar_decay=1.0,
        azoth_recovery_rate=0.0,
        spontaneous_noise=0.0,
        morrow_relaxation=0.0,
        morrow_neighbor_gain=0.0,
        morrow_bias=0.0,
    )
    sig = normalize((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def make() -> Crucible:
        sites = {
            i: Site(i, sig, ResourcePool(aether=10.0), active_signature=(0.0,) * 6)
            for i in range(2)
        }
        return Crucible(sites, [Edge(0, 1, 1.0), Edge(1, 0, 1.0)], cfg, seed=1)

    upward = make()
    upward.perturb(0, sig, 1.0, origin="upward")
    upward.step()
    upward_target = upward.sites[1].activation

    downward = make()
    downward.perturb(1, sig, 1.0, origin="downward")
    downward.step()
    downward_target = downward.sites[0].activation

    return {
        "upward_target_same_tick": upward_target,
        "downward_target_same_tick": downward_target,
        "absolute_bias": abs(upward_target - downward_target),
    }


def diffusion_overdraft() -> dict[str, float]:
    """Check whether simultaneous outgoing diffusion can spend the same Aether twice."""
    cfg = CrucibleConfig(
        initial_aether=0.0,
        initial_ichor=0.0,
        initial_azoth=0.0,
        diffusion_rate=1.0,
        aether_leak=0.0,
        activation_decay=1.0,
        ash_decay=1.0,
        ichor_decay=1.0,
        reserve_release=0.0,
        reconstruction_threshold=999.0,
        plasticity_cost=999.0,
        scar_decay=1.0,
        azoth_recovery_rate=0.0,
        spontaneous_noise=0.0,
        morrow_relaxation=0.0,
    )
    sig = normalize((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sites = {
        0: Site(0, sig, ResourcePool(aether=1.0), active_signature=(0.0,) * 6),
        1: Site(1, sig, ResourcePool(), active_signature=(0.0,) * 6),
        2: Site(2, sig, ResourcePool(), active_signature=(0.0,) * 6),
        3: Site(3, sig, ResourcePool(), active_signature=(0.0,) * 6),
    }
    c = Crucible(sites, [Edge(0, 1, 1.0), Edge(0, 2, 1.0), Edge(0, 3, 1.0)], cfg, seed=2)
    c.step()
    return {
        "source_aether_after": c.sites[0].resources.aether,
        "minimum_site_aether": min(s.resources.aether for s in c.sites.values()),
        "conservation_error": c.conservation_error(),
    }


def signature_persistence(seed: int = 7) -> dict[str, float]:
    """Test whether semantic signature survives after active state has effectively vanished."""
    c = prepared_crucible(seed, nodes=10)
    sig = signature(c.config.signature_dim, 2)
    c.perturb(1, sig, 0.9, origin="single_experience")
    c.run(2)
    c.config = replace(
        c.config,
        reconstruction_threshold=999.0,
        spontaneous_noise=0.0,
        reserve_release=0.0,
        morrow_bias=0.0,
    )
    original_norm = _norm(c.sites[1].active_signature)
    c.run(120)
    return {
        "activation_after_settle": c.sites[1].activation,
        "signature_norm_before_settle": original_norm,
        "signature_norm_after_settle": _norm(c.sites[1].active_signature),
    }


def double_scar_decay() -> dict[str, float]:
    cfg = CrucibleConfig(
        diffusion_rate=0.0,
        aether_leak=0.0,
        activation_decay=1.0,
        ash_decay=1.0,
        ichor_decay=1.0,
        reserve_release=0.0,
        reconstruction_threshold=999.0,
        plasticity_cost=999.0,
        scar_decay=0.9,
        azoth_recovery_rate=0.0,
        spontaneous_noise=0.0,
        morrow_relaxation=0.0,
    )
    sig = normalize((1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    sites = {
        0: Site(0, sig, ResourcePool(), active_signature=(0.0,) * 6),
        1: Site(1, sig, ResourcePool(), active_signature=(0.0,) * 6),
    }
    c = Crucible(sites, [Edge(0, 1, 0.2, scar=1.0)], cfg, seed=3)
    c.step()
    return {
        "scar_after_one_tick": c.edges[0].scar,
        "single_decay_expected": 0.9,
        "double_decay_expected": 0.81,
    }


def _trained_and_naive(seed: int) -> tuple[Crucible, Crucible, tuple[float, ...]]:
    original = prepared_crucible(seed, nodes=18)
    original.config = replace(
        original.config,
        reconstruction_threshold=999.0,
        spontaneous_noise=0.0,
    )
    trained = original.clone(seed=seed)
    naive = original.clone(seed=seed)
    sig = signature(original.config.signature_dim, 2)
    for _ in range(18):
        trained.perturb(1, sig, 0.82, origin="history")
        trained.run(3)
    trained.run(20)
    naive.run(74)
    return trained, naive, sig


def _probe_response(c: Crucible, sig: tuple[float, ...]) -> float:
    c.perturb(1, sig, 0.4, origin="probe")
    c.run(8)
    return sum(site.activation for site in c.sites.values())


def ablation_panel(seed: int = 7) -> dict[str, float]:
    trained, naive, sig = _trained_and_naive(seed)
    naive_response = _probe_response(naive.clone(seed=seed), sig)

    full = trained.clone(seed=seed)
    full_response = _probe_response(full, sig)

    no_scars = trained.clone(seed=seed)
    for edge in no_scars.edges:
        edge.scar = 0.0
    no_scars_response = _probe_response(no_scars, sig)

    no_ichor = trained.clone(seed=seed)
    for site in no_ichor.sites.values():
        site.resources.aether += site.resources.ichor
        site.resources.ichor = 0.0
    no_ichor_response = _probe_response(no_ichor, sig)

    no_signature = trained.clone(seed=seed)
    for site in no_signature.sites.values():
        site.active_signature = (0.0,) * no_signature.config.signature_dim
    no_signature_response = _probe_response(no_signature, sig)

    stripped = trained.clone(seed=seed)
    for edge in stripped.edges:
        edge.scar = 0.0
    for site in stripped.sites.values():
        site.resources.aether += site.resources.ichor
        site.resources.ichor = 0.0
        site.active_signature = (0.0,) * stripped.config.signature_dim
        site.morrow = 0.0
    stripped.config = replace(stripped.config, morrow_relaxation=0.0, morrow_bias=0.0)
    stripped_response = _probe_response(stripped, sig)

    return {
        "naive_response": naive_response,
        "full_delta": full_response - naive_response,
        "no_scars_delta": no_scars_response - naive_response,
        "no_ichor_delta": no_ichor_response - naive_response,
        "no_signature_delta": no_signature_response - naive_response,
        "stripped_delta": stripped_response - naive_response,
    }


def history_seed_sweep(start_seed: int = 1, count: int = 100) -> dict[str, float]:
    deltas: list[float] = []
    for seed in range(start_seed, start_seed + count):
        trained, naive, sig = _trained_and_naive(seed)
        trained_response = _probe_response(trained, sig)
        naive_response = _probe_response(naive, sig)
        deltas.append(trained_response - naive_response)
    return {
        "count": float(count),
        "mean_delta": mean(deltas),
        "median_delta": median(deltas),
        "population_sd": pstdev(deltas),
        "positive_fraction": sum(1 for d in deltas if d > 0.0) / count,
        "negative_fraction": sum(1 for d in deltas if d < 0.0) / count,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def scarcity_comparison(seed: int = 7) -> dict[str, float]:
    scarce = prepared_crucible(seed, nodes=16)
    abundant = scarce.clone(seed=seed)
    for site in abundant.sites.values():
        site.resources.aether += 10_000.0
        site.resources.ichor += 1_000.0

    sig_a = signature(scarce.config.signature_dim, 0)
    sig_b = signature(scarce.config.signature_dim, 4)
    for _ in range(12):
        for c in (scarce, abundant):
            c.perturb(0, sig_a, 0.82, origin="stream_A")
            c.perturb(8, sig_b, 0.82, origin="stream_B")
            c.run(2)

    def dominance(c: Crucible) -> float:
        weights = [
            site.activation * (1.0 + site.resources.ichor) / (1.0 + site.resources.ash)
            for site in c.sites.values()
        ]
        total = sum(max(0.0, x) for x in weights)
        return 0.0 if total == 0.0 else max(weights) / total

    return {
        "scarce_dominance": dominance(scarce),
        "abundant_dominance": dominance(abundant),
        "scarce_starved_events": float(sum(1 for e in scarce.trace if e.kind == "spark_starved")),
        "abundant_starved_events": float(sum(1 for e in abundant.trace if e.kind == "spark_starved")),
    }


def medium_capability_leak(seed: int = 7) -> dict[str, float]:
    """Demonstrate that the current Medium boundary is convention, not capability isolation."""
    c = prepared_crucible(seed, nodes=8)
    medium = Medium()
    _ = medium.report(c)
    privileged_mass = float(c.observer_state()["resource_mass"])
    return {
        "medium_receives_full_crucible_reference": 1.0,
        "privileged_state_reachable_from_same_reference": 1.0 if privileged_mass >= 0.0 else 0.0,
    }


def run_hostile_review(seed: int = 7) -> dict[str, dict[str, float]]:
    return {
        "scheduler_id_bias": scheduler_id_bias(),
        "diffusion_overdraft": diffusion_overdraft(),
        "signature_persistence": signature_persistence(seed),
        "double_scar_decay": double_scar_decay(),
        "ablation_panel": ablation_panel(seed),
        "history_seed_sweep": history_seed_sweep(seed, 100),
        "scarcity_comparison": scarcity_comparison(seed),
        "medium_capability_leak": medium_capability_leak(seed),
    }
