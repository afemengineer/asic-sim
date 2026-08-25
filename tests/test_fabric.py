import pytest

from asic_sim.fabric import build_physical_placement, trace_traffic, xy_route
from asic_sim.hardware import get_hardware
from asic_sim.models import get_model


def test_xy_route_matches_manhattan_distance() -> None:
    hw = get_hardware("fabric-32x32")
    path = xy_route(hw, 0, 31)
    assert path[0] == 0
    assert path[-1] == 31
    assert len(path) - 1 == 10  # (0,0) -> (3,7)


def test_kimi_physical_placement_conserves_storage_and_fits_64_tiles() -> None:
    model = get_model("kimi-k3")
    hw = get_hardware("fabric-64x32")
    placement = build_physical_placement(model, hw, bits_per_weight=4.0, overhead_fraction=0.05)
    assert sum(tile.storage_bytes for tile in placement.tiles) == pytest.approx(placement.total_storage_bytes)
    assert placement.total_storage_bytes == pytest.approx(model.storage_bytes(4.0, 0.05))
    assert placement.system_resident
    assert placement.per_tile_resident
    assert max(tile.expert_shards for tile in placement.tiles) - min(tile.expert_shards for tile in placement.tiles) <= 1


def test_qwen_single_tile_has_no_noc_traffic() -> None:
    model = get_model("qwen3.8-27b")
    hw = get_hardware("raptor-like")
    placement = build_physical_placement(model, hw, bits_per_weight=4.0)
    traffic = trace_traffic(placement, tokens=8)
    assert placement.per_tile_resident
    assert traffic.network_payload_bytes_per_token == 0.0
    assert traffic.hop_bytes_per_token == 0.0
    assert traffic.hottest_link is None


def test_kimi_balanced_trace_conserves_link_and_tile_bytes() -> None:
    placement = build_physical_placement(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        bits_per_weight=4.0,
    )
    traffic = trace_traffic(placement, tokens=4, profile="balanced")
    assert traffic.remote_expert_fraction > 0.95
    assert traffic.network_payload_bytes_per_token > 0
    assert traffic.hop_bytes_per_token > traffic.network_payload_bytes_per_token
    assert sum(value for _, value in traffic.link_bytes_per_token) == pytest.approx(traffic.hop_bytes_per_token)
    assert sum(traffic.tile_tx_bytes_per_token) == pytest.approx(traffic.network_payload_bytes_per_token)
    assert sum(traffic.tile_rx_bytes_per_token) == pytest.approx(traffic.network_payload_bytes_per_token)
    assert traffic.hottest_link is not None
    assert traffic.hotspot_ratio >= 1.0


def test_routing_profiles_are_deterministic_for_same_seed() -> None:
    placement = build_physical_placement(
        get_model("glm-5.2"),
        get_hardware("fabric-32x32"),
        bits_per_weight=4.0,
    )
    a = trace_traffic(placement, tokens=4, profile="zipf", seed=7)
    b = trace_traffic(placement, tokens=4, profile="zipf", seed=7)
    assert a.link_bytes_per_token == b.link_bytes_per_token
    assert a.remote_expert_fraction == b.remote_expert_fraction
