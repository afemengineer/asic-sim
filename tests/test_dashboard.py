from asic_sim.dashboard import build_snapshot, interpretation


def test_kimi_k3_4bit_fits_two_tb_fabric() -> None:
    snapshot = build_snapshot("kimi-k3", "fabric-64x32", bits_per_weight=4.0)
    assert snapshot.result.resident
    assert snapshot.placement.resident_per_tile_balanced
    assert 0.70 < snapshot.capacity_utilization < 0.73
    assert snapshot.remote_traffic_fraction < 0.001
    assert snapshot.placement.expected_remote_expert_fraction > 0.98


def test_kimi_k3_4bit_does_not_fit_one_tb_fabric() -> None:
    snapshot = build_snapshot("kimi-k3", "fabric-32x32", bits_per_weight=4.0)
    assert not snapshot.result.resident
    assert snapshot.capacity_utilization > 1.0
    lines = interpretation(snapshot)
    assert any("STOP" in line for line in lines)


def test_glm_dashboard_explains_remote_calls_vs_bytes() -> None:
    snapshot = build_snapshot("glm-5.2", "fabric-32x32", bits_per_weight=4.0)
    lines = interpretation(snapshot)
    assert snapshot.result.resident
    assert snapshot.result.local_data_fraction > 0.999
    assert any("remote activations" in line for line in lines)
    assert any("memory-stationary thesis" in line for line in lines)
