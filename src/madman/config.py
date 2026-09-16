from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CrucibleConfig:
    signature_dim: int = 6
    initial_aether: float = 2.0
    initial_ichor: float = 0.25
    initial_azoth: float = 1.0
    source_injection: float = 0.65
    diffusion_rate: float = 0.16
    aether_leak: float = 0.002
    activation_decay: float = 0.76
    ash_decay: float = 0.82
    ichor_decay: float = 0.997
    base_assimilation_cost: float = 0.18
    propagation_cost: float = 0.06
    ash_yield: float = 0.16
    reserve_capture: float = 0.08
    reserve_release: float = 0.16
    reserve_release_threshold: float = 0.34
    reconstruction_threshold: float = 0.115
    reconstruction_reserve_cost: float = 0.035
    propagation_threshold: float = 0.19
    max_generation: int = 6
    plasticity_rate: float = 0.035
    plasticity_cost: float = 0.018
    scar_decay: float = 0.9997
    max_scar: float = 1.8
    azoth_recovery_rate: float = 0.006
    azoth_cap: float = 2.5
    spontaneous_noise: float = 0.025
    morrow_relaxation: float = 0.18
    morrow_neighbor_gain: float = 0.72
    morrow_bias: float = 0.28
    min_conductance: float = 0.02
    max_conductance: float = 1.4
    trace_limit: int = 50_000
