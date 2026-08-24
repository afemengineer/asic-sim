import pytest

from asic_sim.hardware import get_hardware
from asic_sim.models import get_model
from asic_sim.placement import balanced_placement, decompose_model


def test_glm_decomposition_matches_ffn_scale() -> None:
    model = get_model("glm-5.2")
    d = decompose_model(model)
    assert d.expert_shard_parameters == pytest.approx(37.55e6, rel=0.01)
    assert d.always_on_parameters > 0
    assert d.active_always_on_parameters > 0


def test_kimi_decomposition_does_not_assume_standard_ffn() -> None:
    d = decompose_model(get_model("kimi-k3"))
    assert d.routed_fraction > 0.95
    assert d.expert_shard_parameters is not None
    assert d.expert_shard_parameters > 0


def test_balanced_glm_fits_32_tile_fabric() -> None:
    report = balanced_placement(get_model("glm-5.2"), get_hardware("fabric-32x32"))
    assert report.resident_system
    assert report.resident_per_tile_balanced
    assert report.expert_shards_max_per_tile - report.expert_shards_min_per_tile <= 1
    assert report.expected_remote_expert_fraction == pytest.approx(31 / 32)


def test_balanced_kimi_needs_64_tiles_at_four_bit() -> None:
    model = get_model("kimi-k3")
    assert not balanced_placement(model, get_hardware("fabric-32x32")).resident_system
    assert balanced_placement(model, get_hardware("fabric-64x32")).resident_system
