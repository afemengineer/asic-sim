import math

from asic_sim.hardware import get_hardware
from asic_sim.m3 import (
    KIMI_K3_STATE,
    _kda_state_bytes,
    _mla_entry_bytes,
    max_sequences_that_fit,
    simulate_m3,
)
from asic_sim.models import get_model


def test_kimi_cache_geometry_matches_configured_dimensions() -> None:
    spec = KIMI_K3_STATE
    assert len(spec.full_attention_layers) == 24
    assert spec.kda_layers == 69
    assert _mla_entry_bytes(spec, 16.0, "latent") == (512 + 64) * 2
    assert _mla_entry_bytes(spec, 16.0, "expanded") == 96 * (192 + 128) * 2
    assert _kda_state_bytes(spec, 16.0) > 3e6


def test_m3_uses_latent_moe_message_width() -> None:
    report = simulate_m3(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        cluster_size=4,
        link_bandwidth_gb_s=64.0,
        shared_shards=2,
        concurrency=1,
        context_length=4096,
    )
    assert report.resident
    assert report.routed_message_bytes == 3584 * 2
    assert report.shared_message_bytes == 7168 * 2
    assert report.throughput_tps > 0
    assert report.hottest_compute_utilization > 0


def test_latent_kv_uses_less_capacity_than_expanded_reference() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    latent = simulate_m3(
        model,
        hardware,
        cluster_size=16,
        link_bandwidth_gb_s=128.0,
        shared_shards=2,
        concurrency=1,
        context_length=16_384,
        kv_mode="latent",
    )
    expanded = simulate_m3(
        model,
        hardware,
        cluster_size=16,
        link_bandwidth_gb_s=128.0,
        shared_shards=2,
        concurrency=1,
        context_length=16_384,
        kv_mode="expanded",
    )
    assert latent.cache_bytes_per_sequence < expanded.cache_bytes_per_sequence


def test_max_sequences_falls_with_context_length() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    short, _, short_cache = max_sequences_that_fit(
        model,
        hardware,
        cluster_size=16,
        shared_shards=2,
        context_length=4096,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
        other_weight_bits=4.0,
        overhead_fraction=0.05,
        kv_mode="latent",
        kv_bits=16.0,
        kda_state_bits=16.0,
    )
    long, _, long_cache = max_sequences_that_fit(
        model,
        hardware,
        cluster_size=16,
        shared_shards=2,
        context_length=262_144,
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
        other_weight_bits=4.0,
        overhead_fraction=0.05,
        kv_mode="latent",
        kv_bits=16.0,
        kda_state_bits=16.0,
    )
    assert long_cache > short_cache
    assert long <= short


def test_shared_sharding_changes_physical_weight_balance() -> None:
    model = get_model("kimi-k3")
    hardware = get_hardware("fabric-64x32")
    one = simulate_m3(
        model,
        hardware,
        cluster_size=4,
        link_bandwidth_gb_s=64.0,
        shared_shards=1,
        concurrency=1,
        context_length=0,
    )
    two = simulate_m3(
        model,
        hardware,
        cluster_size=4,
        link_bandwidth_gb_s=64.0,
        shared_shards=2,
        concurrency=1,
        context_length=0,
    )
    assert one.resident and two.resident
    assert two.max_tile_storage_bytes <= one.max_tile_storage_bytes
    assert math.isclose(one.weight_storage_bytes, two.weight_storage_bytes, rel_tol=1e-12)
