from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .config import CrucibleConfig
from .crucible import Crucible
from .medium import Medium
from .model import normalize
from .observer import GhostHunter


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    metrics: dict[str, float | int | str]


def signature(dim: int, axis: int, sign: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * dim
    values[axis % dim] = sign
    if dim > 1:
        values[(axis + 1) % dim] = 0.35 * sign
    return normalize(values)


def prepared_crucible(seed: int = 7, nodes: int = 18) -> Crucible:
    cfg = CrucibleConfig()
    c = Crucible.random_lattice(node_count=nodes, extra_edges=18, config=cfg, seed=seed)
    c.set_source(nodes - 1, cfg.source_injection)
    c.set_source(nodes // 2, cfg.source_injection * 0.55)
    c.set_report_surface([0, 1, 2])
    return c


def echo_experiment(seed: int = 7) -> ScenarioResult:
    c = prepared_crucible(seed)
    sig = signature(c.config.signature_dim, 0)
    trained_members = (0, 1, 2, 3)
    for _ in range(16):
        c.perturb(0, sig, 0.9, origin="training")
        c.run(4)
    scars_after_training = sum(e.scar for e in c.edges if e.source in trained_members or e.target in trained_members)
    c.run(35)
    activation_before = sum(c.sites[i].activation for i in trained_members)
    c.perturb(0, sig, 0.22, origin="partial_cue")
    c.run(8)
    activation_after = sum(c.sites[i].activation for i in trained_members)
    reconstructions = sum(1 for event in c.trace if event.kind == "spontaneous_reconstruction")
    return ScenarioResult(
        "echo",
        {
            "scar_mass": scars_after_training,
            "activation_before_cue": activation_before,
            "activation_after_cue": activation_after,
            "spontaneous_reconstructions": reconstructions,
            "conservation_error": abs(c.conservation_error()),
        },
    )


def history_experiment(seed: int = 11) -> ScenarioResult:
    original = prepared_crucible(seed)
    trained = original.clone(seed=seed)
    naive = original.clone(seed=seed)
    sig = signature(original.config.signature_dim, 2)
    for _ in range(18):
        trained.perturb(1, sig, 0.82, origin="history")
        trained.run(3)
    trained.run(20)
    naive.run(74)
    trained.perturb(1, sig, 0.4, origin="probe")
    naive.perturb(1, sig, 0.4, origin="probe")
    trained.run(8)
    naive.run(8)
    trained_response = sum(site.activation for site in trained.sites.values())
    naive_response = sum(site.activation for site in naive.sites.values())
    return ScenarioResult(
        "history",
        {
            "trained_response": trained_response,
            "naive_response": naive_response,
            "response_delta": trained_response - naive_response,
        },
    )


def haunting_experiment(seed: int = 13) -> ScenarioResult:
    c = prepared_crucible(seed, nodes=12)
    sig = signature(c.config.signature_dim, 3)
    for _ in range(22):
        c.perturb(0, sig, 1.0, origin="haunting_seed")
        c.run(2)
    c.run(12)
    early = [sum(site.activation for site in c.sites.values())]
    for _ in range(60):
        c.step()
        early.append(sum(site.activation for site in c.sites.values()))
    first_half = mean(early[:30])
    second_half = mean(early[30:])
    reconstructions = sum(1 for event in c.trace if event.kind == "spontaneous_reconstruction")
    return ScenarioResult(
        "haunting",
        {
            "early_mean_activation": first_half,
            "late_mean_activation": second_half,
            "reconstruction_count": reconstructions,
            "eventual_decay": 1 if second_half < first_half else 0,
        },
    )


def attention_experiment(seed: int = 17) -> ScenarioResult:
    c = prepared_crucible(seed, nodes=16)
    sig_a = signature(c.config.signature_dim, 0)
    sig_b = signature(c.config.signature_dim, 4)
    for _ in range(10):
        c.perturb(0, sig_a, 0.82, origin="stream_A")
        c.perturb(8, sig_b, 0.82, origin="stream_B")
        c.run(2)
    winner, share = GhostHunter.metabolic_dominance(c)
    return ScenarioResult(
        "attention",
        {
            "dominant_site": winner,
            "dominance_share": share,
            "flow_entropy": GhostHunter.flow_entropy(c),
        },
    )


def fork_experiment(seed: int = 19) -> ScenarioResult:
    ancestor = prepared_crucible(seed, nodes=14)
    ancestor.run(10)
    a = ancestor.clone(seed=seed + 1)
    b = ancestor.clone(seed=seed + 2)
    sig_a = signature(a.config.signature_dim, 1)
    sig_b = signature(b.config.signature_dim, 5)
    for _ in range(15):
        a.perturb(2, sig_a, 0.75, origin="branch_A")
        b.perturb(2, sig_b, 0.75, origin="branch_B")
        a.run(2)
        b.run(2)
    probe = signature(a.config.signature_dim, 2)
    a.perturb(2, probe, 0.5, origin="shared_probe")
    b.perturb(2, probe, 0.5, origin="shared_probe")
    a.run(6)
    b.run(6)
    divergence = sum(abs(x - y) for x, y in zip(a.active_vector(), b.active_vector(), strict=True))
    scar_divergence = sum(abs(x[3] - y[3]) for x, y in zip(a.topology_snapshot(), b.topology_snapshot(), strict=True))
    return ScenarioResult("fork", {"activation_divergence": divergence, "scar_divergence": scar_divergence})


def morrow_experiment(seed: int = 23) -> ScenarioResult:
    c = prepared_crucible(seed, nodes=10)
    c.run(30)
    source_site = max(c.sites, key=lambda i: c.sites[i].source_flux)
    pressures = {i: c.sites[i].morrow for i in c.sites}
    source_pressure = pressures[source_site]
    mean_pressure = mean(pressures.values())
    return ScenarioResult("morrow", {"source_pressure": source_pressure, "mean_pressure": mean_pressure})


def speech_experiment(seed: int = 29) -> ScenarioResult:
    c = prepared_crucible(seed, nodes=10)
    sig = signature(c.config.signature_dim, 0)
    for _ in range(12):
        c.perturb(0, sig, 0.9, origin="hidden_cause")
        c.run(2)
    medium = Medium()
    report = medium.report(c)
    hidden_scar_mass = sum(e.scar for e in c.edges)
    return ScenarioResult("speech", {"report": report, "hidden_scar_mass": hidden_scar_mass})


def run_all(seed: int = 7) -> tuple[ScenarioResult, ...]:
    return (
        echo_experiment(seed),
        history_experiment(seed + 1),
        haunting_experiment(seed + 2),
        attention_experiment(seed + 3),
        fork_experiment(seed + 4),
        morrow_experiment(seed + 5),
        speech_experiment(seed + 6),
    )
