import math

from asic_sim.hardware import get_hardware
from asic_sim.m2 import simulate_m2
from asic_sim.m2_network_models import simulate_m2_cut_through
from asic_sim.models import get_model


def test_cut_through_is_not_slower_than_store_forward_for_single_token() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    common = dict(
        cluster_size=16,
        link_bandwidth_gb_s=64.0,
        concurrency=1,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
    )
    sf = simulate_m2(model, hardware, **common)
    ct = simulate_m2_cut_through(model, hardware, **common)
    assert sf.resident and ct.resident
    assert ct.throughput_tps >= sf.throughput_tps
    assert ct.p95_latency_s <= sf.p95_latency_s
    assert math.isclose(ct.network_payload_bytes_per_token, sf.network_payload_bytes_per_token, rel_tol=1e-12)
    assert math.isclose(ct.hop_bytes_per_token, sf.hop_bytes_per_token, rel_tol=1e-12)


def test_cut_through_converges_to_store_forward_when_serialization_is_negligible() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    common = dict(
        cluster_size=16,
        link_bandwidth_gb_s=1e9,
        concurrency=1,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
    )
    sf = simulate_m2(model, hardware, **common)
    ct = simulate_m2_cut_through(model, hardware, **common)
    assert sf.resident and ct.resident
    assert math.isclose(ct.throughput_tps, sf.throughput_tps, rel_tol=1e-4)


def test_cut_through_still_models_finite_link_queueing_under_burst() -> None:
    report = simulate_m2_cut_through(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_size=16,
        link_bandwidth_gb_s=16.0,
        concurrency=8,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
    )
    assert report.resident
    assert report.hottest_link_utilization > 0
    assert report.max_link_queue_wait_s >= 0
    assert report.p95_packet_latency_s > 0
