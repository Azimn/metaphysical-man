from madman.scenarios import prepared_crucible, signature


def test_fusion_blends_histories_without_privileged_identity() -> None:
    ancestor = prepared_crucible(seed=501, nodes=10)
    left = ancestor.clone(seed=502)
    right = ancestor.clone(seed=503)
    left_sig = signature(left.config.signature_dim, 1)
    right_sig = signature(right.config.signature_dim, 4)
    for _ in range(8):
        left.perturb(0, left_sig, 0.9)
        right.perturb(0, right_sig, 0.9)
        left.run(2)
        right.run(2)
    fused = left.fuse(right, weight=0.5, seed=504)
    assert fused.resource_mass() > 0.0
    assert abs(fused.conservation_error()) < 1e-9
    assert fused.topology_snapshot() != left.topology_snapshot() or fused.topology_snapshot() != right.topology_snapshot()
