from __future__ import annotations

import json
from statistics import mean, median, pstdev

from .hostile import _probe_response, _trained_and_naive, scarcity_comparison


def _zero_activation(c) -> None:
    for site in c.sites.values():
        site.activation = 0.0


def _copy_resources(dst, src) -> None:
    for site_id in dst.sites:
        a = dst.sites[site_id].resources
        b = src.sites[site_id].resources
        a.aether = b.aether
        a.ichor = b.ichor
        a.azoth = b.azoth
        a.ash = b.ash


def _copy_structure(dst, src) -> None:
    for left, right in zip(dst.edges, src.edges, strict=True):
        left.scar = right.scar
        left.conductance = right.conductance
    for site_id in dst.sites:
        a = dst.sites[site_id]
        b = src.sites[site_id]
        a.active_signature = b.active_signature
        a.morrow = b.morrow


def state_difference(seed: int = 7) -> dict[str, float]:
    trained, naive, _ = _trained_and_naive(seed)

    def total(name: str, c) -> float:
        return sum(getattr(site.resources, name) for site in c.sites.values())

    return {
        "activation_difference": sum(site.activation for site in trained.sites.values())
        - sum(site.activation for site in naive.sites.values()),
        "aether_difference": total("aether", trained) - total("aether", naive),
        "ichor_difference": total("ichor", trained) - total("ichor", naive),
        "azoth_difference": total("azoth", trained) - total("azoth", naive),
        "ash_difference": total("ash", trained) - total("ash", naive),
        "scar_mass_difference": sum(edge.scar for edge in trained.edges) - sum(edge.scar for edge in naive.edges),
    }


def cold_history_decomposition(seed: int = 7) -> dict[str, float]:
    trained, naive, sig = _trained_and_naive(seed)
    _zero_activation(trained)
    _zero_activation(naive)

    baseline = _probe_response(naive.clone(seed=seed), sig)

    full = trained.clone(seed=seed)
    full_delta = _probe_response(full, sig) - baseline

    structural_only = trained.clone(seed=seed)
    _copy_resources(structural_only, naive)
    structural_only_delta = _probe_response(structural_only, sig) - baseline

    metabolic_only = trained.clone(seed=seed)
    _copy_structure(metabolic_only, naive)
    metabolic_only_delta = _probe_response(metabolic_only, sig) - baseline

    wiped = trained.clone(seed=seed)
    _copy_resources(wiped, naive)
    _copy_structure(wiped, naive)
    _zero_activation(wiped)
    wiped_delta = _probe_response(wiped, sig) - baseline

    return {
        "naive_response": baseline,
        "cold_full_delta": full_delta,
        "structural_only_delta": structural_only_delta,
        "metabolic_only_delta": metabolic_only_delta,
        "wiped_delta": wiped_delta,
    }


def cold_history_seed_sweep(start_seed: int = 7, count: int = 50) -> dict[str, float]:
    full_deltas: list[float] = []
    structural_deltas: list[float] = []
    metabolic_deltas: list[float] = []
    for seed in range(start_seed, start_seed + count):
        result = cold_history_decomposition(seed)
        full_deltas.append(result["cold_full_delta"])
        structural_deltas.append(result["structural_only_delta"])
        metabolic_deltas.append(result["metabolic_only_delta"])

    return {
        "count": float(count),
        "full_mean": mean(full_deltas),
        "full_median": median(full_deltas),
        "full_sd": pstdev(full_deltas),
        "full_positive_fraction": sum(d > 0.0 for d in full_deltas) / count,
        "structural_mean": mean(structural_deltas),
        "structural_positive_fraction": sum(d > 0.0 for d in structural_deltas) / count,
        "metabolic_mean": mean(metabolic_deltas),
        "metabolic_positive_fraction": sum(d > 0.0 for d in metabolic_deltas) / count,
    }


def scarcity_seed_sweep(start_seed: int = 7, count: int = 50) -> dict[str, float]:
    differences: list[float] = []
    scarce_starved: list[float] = []
    abundant_starved: list[float] = []
    for seed in range(start_seed, start_seed + count):
        result = scarcity_comparison(seed)
        differences.append(result["scarce_dominance"] - result["abundant_dominance"])
        scarce_starved.append(result["scarce_starved_events"])
        abundant_starved.append(result["abundant_starved_events"])
    return {
        "count": float(count),
        "mean_dominance_increase_under_scarcity": mean(differences),
        "median_dominance_increase_under_scarcity": median(differences),
        "scarcity_more_concentrated_fraction": sum(d > 0.0 for d in differences) / count,
        "scarce_mean_starved_events": mean(scarce_starved),
        "abundant_mean_starved_events": mean(abundant_starved),
    }


def run_round2(seed: int = 7) -> dict[str, dict[str, float]]:
    return {
        "state_difference": state_difference(seed),
        "cold_history_decomposition": cold_history_decomposition(seed),
        "cold_history_seed_sweep": cold_history_seed_sweep(seed, 50),
        "scarcity_seed_sweep": scarcity_seed_sweep(seed, 50),
    }


def main() -> None:
    print(json.dumps(run_round2(7), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
