from madman.medium import Medium
from madman.scenarios import (
    echo_experiment,
    fork_experiment,
    history_experiment,
    prepared_crucible,
    signature,
)


def test_history_changes_later_response() -> None:
    result = history_experiment(seed=211)
    assert abs(float(result.metrics["response_delta"])) > 1e-4


def test_training_leaves_structural_scars() -> None:
    result = echo_experiment(seed=223)
    assert float(result.metrics["scar_mass"]) > 0.0
    assert float(result.metrics["conservation_error"]) < 1e-8


def test_forks_develop_different_geometry() -> None:
    result = fork_experiment(seed=227)
    assert float(result.metrics["scar_divergence"]) > 0.0


def test_medium_only_uses_report_surface() -> None:
    c = prepared_crucible(seed=229, nodes=10)
    medium = Medium()
    quiet_report = medium.report(c)
    sig = signature(c.config.signature_dim, 0)
    c.perturb(0, sig, 1.0, origin="cause_hidden_from_medium")
    c.run(4)
    active_report = medium.report(c)
    assert quiet_report != active_report
    assert "scar" not in active_report.lower()
    assert "aether" not in active_report.lower()
