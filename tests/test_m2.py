from asic_sim.hardware import get_hardware
from asic_sim.m2 import compare_m2_candidates, simulate_m2
from asic_sim.models import get_model


def test_m2_cluster4_single_token_is_causal_and_resident() -> None:
    report = simulate_m2(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_size=4,
        link_bandwidth_gb_s=32.0,
        concurrency=1,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
    )
    assert report.resident
    assert report.throughput_tps > 0
    assert report.p50_latency_s == report.max_latency_s
    assert report.network_payload_bytes_per_token > 0
    assert report.hop_bytes_per_token >= report.network_payload_bytes_per_token
    assert report.packet_count_per_token > 0
    assert 0 <= report.hottest_link_utilization <= 1
    assert 0 <= report.hottest_memory_utilization <= 1
    assert report.event_count > 0


def test_more_link_bandwidth_cannot_reduce_single_token_throughput() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    slow = simulate_m2(
        model,
        hardware,
        cluster_size=4,
        link_bandwidth_gb_s=8.0,
        concurrency=1,
        shared_expert_bits=16.0,
    )
    fast = simulate_m2(
        model,
        hardware,
        cluster_size=4,
        link_bandwidth_gb_s=128.0,
        concurrency=1,
        shared_expert_bits=16.0,
    )
    assert slow.resident and fast.resident
    assert fast.throughput_tps >= slow.throughput_tps
    assert fast.p50_latency_s <= slow.p50_latency_s


def test_finite_candidate_is_compared_against_infinite_link_baseline() -> None:
    rows = compare_m2_candidates(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        candidates=((4, 32.0),),
        concurrency_levels=(2,),
        shared_expert_bits=16.0,
    )
    row = rows[0]
    assert row.report.resident
    assert row.baseline_throughput_tps >= row.report.throughput_tps
    assert 0 < row.retained_baseline_throughput <= 1.0
    assert 0 <= row.throughput_loss_fraction < 1.0


def test_layer_stationary_fp16_shared_case_remains_infeasible() -> None:
    report = simulate_m2(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_size=1,
        link_bandwidth_gb_s=32.0,
        concurrency=1,
        shared_expert_bits=16.0,
    )
    assert not report.resident
    assert report.throughput_tps == 0
