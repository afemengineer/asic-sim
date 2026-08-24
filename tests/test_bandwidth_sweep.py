import math

from asic_sim.bandwidth_sweep import (
    _cluster_shape,
    analyze_cluster_topology,
    sweep_cluster_bandwidths,
)
from asic_sim.hardware import get_hardware
from asic_sim.models import get_model


def test_cluster_shapes_tile_eight_by_eight_mesh_compactly() -> None:
    hardware = get_hardware("fabric-64x32")
    assert _cluster_shape(hardware, 1) == (1, 1)
    assert _cluster_shape(hardware, 2) in {(1, 2), (2, 1)}
    assert _cluster_shape(hardware, 4) == (2, 2)
    assert _cluster_shape(hardware, 16) == (4, 4)
    assert _cluster_shape(hardware, 64) == (8, 8)


def test_kimi_cluster_sweep_pressure_falls_with_bandwidth() -> None:
    summaries = sweep_cluster_bandwidths(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_sizes=(4,),
        link_bandwidths_gb_s=(8, 16, 32, 64, 128),
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
        tokens=8,
    )
    summary = summaries[0]
    pressures = [point.hot_link_pressure for point in summary.points]
    assert summary.topology.resident
    assert all(left > right for left, right in zip(pressures, pressures[1:]))
    assert summary.raw_minimum_bandwidth_gb_s is not None
    expected = (
        summary.topology.max_link_bytes_per_token
        * summary.topology.ideal_memory_roof_tps
        / 0.05
        / 1e9
    )
    assert math.isclose(summary.raw_minimum_bandwidth_gb_s, expected, rel_tol=1e-12)


def test_global_cluster_moves_more_than_layer_stationary_cluster() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    local = analyze_cluster_topology(model, hardware, cluster_size=1, tokens=8)
    global_mesh = analyze_cluster_topology(model, hardware, cluster_size=64, tokens=8)
    assert global_mesh.network_payload_bytes_per_token > local.network_payload_bytes_per_token
    assert global_mesh.hop_bytes_per_token > local.hop_bytes_per_token
    assert global_mesh.ideal_memory_roof_tps > local.ideal_memory_roof_tps


def test_infeasible_cluster_has_no_bandwidth_recommendation() -> None:
    summaries = sweep_cluster_bandwidths(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_sizes=(1,),
        link_bandwidths_gb_s=(1, 8, 64, 512),
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
        tokens=4,
    )
    summary = summaries[0]
    assert not summary.topology.resident
    assert summary.raw_minimum_bandwidth_gb_s is None
    assert summary.recommended_candidate_bandwidth_gb_s is None
