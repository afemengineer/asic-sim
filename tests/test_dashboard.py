from asic_sim.dashboard import build_snapshot, interpretation


def test_qwen_q4_is_single_tile_control() -> None:
    snapshot = build_snapshot("qwen3.8-27b", "raptor-like", bits_per_weight=4.0)
    assert snapshot.result.resident
    assert snapshot.physical_placement.per_tile_resident
    assert snapshot.traffic.network_payload_bytes_per_token == 0.0
    assert snapshot.capacity_utilization < 0.5


def test_kimi_k3_4bit_fits_two_tb_fabric() -> None:
    snapshot = build_snapshot("kimi-k3", "fabric-64x32", bits_per_weight=4.0)
    assert snapshot.result.resident
    assert snapshot.physical_placement.per_tile_resident
    assert 0.70 < snapshot.capacity_utilization < 0.73
    assert snapshot.remote_traffic_fraction < 0.001
    assert snapshot.traffic.remote_expert_fraction > 0.95
    assert snapshot.traffic.hottest_link is not None


def test_kimi_k3_4bit_does_not_fit_one_tb_fabric() -> None:
    snapshot = build_snapshot("kimi-k3", "fabric-32x32", bits_per_weight=4.0)
    assert not snapshot.result.resident
    assert snapshot.capacity_utilization > 1.0
    lines = interpretation(snapshot)
    assert any("STOP" in line for line in lines)


def test_glm_dashboard_explains_physical_routes() -> None:
    snapshot = build_snapshot("glm-5.2", "fabric-32x32", bits_per_weight=4.0)
    lines = interpretation(snapshot)
    assert snapshot.result.resident
    assert snapshot.remote_traffic_fraction < 0.001
    assert snapshot.traffic.network_payload_bytes_per_token > 0
    assert snapshot.traffic.hop_bytes_per_token > snapshot.traffic.network_payload_bytes_per_token
    assert any("memory-stationary thesis" in line for line in lines)
    assert any("hot link" in line.lower() for line in lines)
