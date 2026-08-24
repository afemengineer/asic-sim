from asic_sim.hardware import get_hardware
from asic_sim.m3_compute_sweep import attention_balance_points, sweep_compute_balance
from asic_sim.models import get_model


def test_latent_attention_trades_cache_bandwidth_for_arithmetic() -> None:
    points = {
        point.mode: point
        for point in attention_balance_points(
            get_model("kimi-k3"),
            get_hardware("fabric-64x32"),
            context_length=16_384,
            other_weight_bits=16.0,
        )
    }
    latent = points["latent"]
    expanded = points["expanded"]
    assert latent.entry_bytes_per_cached_token_layer < expanded.entry_bytes_per_cached_token_layer
    assert latent.state_traffic_bytes_per_layer < expanded.state_traffic_bytes_per_layer
    assert latent.state_flops_per_layer > expanded.state_flops_per_layer
    assert latent.state_arithmetic_intensity_flops_per_byte > expanded.state_arithmetic_intensity_flops_per_byte
    assert latent.state_balance_tops > expanded.state_balance_tops


def test_more_compute_does_not_reduce_single_token_m3_throughput() -> None:
    rows = sweep_compute_balance(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        candidates=((4, 64.0),),
        fp16_tops_values=(100.0, 400.0),
        routed_factor=4.0,
        concurrency=1,
        context_length=4096,
        shared_shards=2,
        other_weight_bits=16.0,
    )
    assert len(rows) == 2
    low, high = rows
    assert low.fp16_tops == 100.0
    assert high.fp16_tops == 400.0
    assert low.latent_resident and high.latent_resident
    assert high.latent_tps >= low.latent_tps
    assert high.expanded_tps >= low.expanded_tps


def test_compute_sweep_preserves_latent_capacity_advantage() -> None:
    row = sweep_compute_balance(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        candidates=((16, 128.0),),
        fp16_tops_values=(200.0,),
        concurrency=1,
        context_length=16_384,
        shared_shards=2,
        other_weight_bits=16.0,
    )[0]
    assert row.latent_resident and row.expanded_resident
    assert row.latent_cache_bytes_per_sequence < row.expanded_cache_bytes_per_sequence
    assert row.latent_tps > 0
    assert row.expanded_tps > 0
