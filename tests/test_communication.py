from madman.communication import VeilChannel
from madman.scenarios import prepared_crucible, signature


def test_glyph_is_perturbation_not_shared_state() -> None:
    sender = prepared_crucible(seed=401, nodes=10)
    receiver = prepared_crucible(seed=402, nodes=10)
    sig = signature(sender.config.signature_dim, 0)
    sender.perturb(0, sig, 1.0)
    sender.run(4)
    channel = VeilChannel(sender.config.signature_dim, seed=7, distortion=0.2)
    glyph = channel.emit(sender, 0)
    before = receiver.active_vector()
    channel.receive(receiver, 0, glyph)
    receiver.run(4)
    after = receiver.active_vector()
    assert before != after
    assert sender.active_vector() != receiver.active_vector()
