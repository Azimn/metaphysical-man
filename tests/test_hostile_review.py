from madman.hostile import (
    diffusion_overdraft,
    scheduler_id_bias,
    signature_persistence,
)


def test_scheduler_bias_is_detectable() -> None:
    result = scheduler_id_bias()
    assert result["absolute_bias"] > 0.0


def test_diffusion_can_overdraw_local_aether() -> None:
    result = diffusion_overdraft()
    assert result["minimum_site_aether"] < 0.0


def test_signature_survives_after_activation_decays() -> None:
    result = signature_persistence(seed=31)
    assert result["activation_after_settle"] < 1e-6
    assert result["signature_norm_after_settle"] > 0.9
