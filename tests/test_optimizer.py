import pytest

from mvt.models.mvt import MVT
from mvt.training import OptimizerConfig, build_optimizer_scheduler


def test_optimizer_uses_upstream_backbone_multiplier():
    config = OptimizerConfig(
        learning_rate=4e-4,
        backbone_multiplier=0.1,
        epochs=100,
    )
    optimizer, scheduler = build_optimizer_scheduler(MVT(), config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-4)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(4e-5)
    assert scheduler.T_max == 100
    assert scheduler.eta_min == pytest.approx(1e-9)
