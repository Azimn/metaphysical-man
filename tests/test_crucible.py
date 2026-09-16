from madman.crucible import Crucible
from madman.scenarios import prepared_crucible, signature


def test_deterministic_replay() -> None:
    a = prepared_crucible(seed=101, nodes=12)
    b = prepared_crucible(seed=101, nodes=12)
    sig = signature(a.config.signature_dim, 1)
    for _ in range(8):
        a.perturb(0, sig, 0.7)
        b.perturb(0, sig, 0.7)
        a.run(3)
        b.run(3)
    assert a.active_vector() == b.active_vector()
    assert a.topology_snapshot() == b.topology_snapshot()


def test_resource_accounting_is_close() -> None:
    c = prepared_crucible(seed=103, nodes=10)
    sig = signature(c.config.signature_dim, 0)
    for _ in range(12):
        c.perturb(0, sig, 0.8)
        c.run(2)
    assert abs(c.conservation_error()) < 1e-8


def test_clone_preserves_state_but_can_diverge() -> None:
    c = Crucible.random_lattice(node_count=10, seed=107)
    clone = c.clone(seed=108)
    assert c.active_vector() == clone.active_vector()
    sig = signature(c.config.signature_dim, 2)
    c.perturb(0, sig, 1.0)
    c.run(5)
    clone.run(5)
    assert c.active_vector() != clone.active_vector()
