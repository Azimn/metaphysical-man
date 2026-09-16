from madman.observer import GhostHunter
from madman.scenarios import prepared_crucible, signature


def test_observer_can_detect_activity_without_being_in_loop() -> None:
    c = prepared_crucible(seed=307, nodes=10)
    sig = signature(c.config.signature_dim, 1)
    c.perturb(0, sig, 0.9)
    c.run(6)
    winner, share = GhostHunter.metabolic_dominance(c)
    assert winner in c.sites
    assert 0.0 <= share <= 1.0
