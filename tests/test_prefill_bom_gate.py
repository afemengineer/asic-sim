from asic_sim.models import get_model
from asic_sim.prefill import PrefillWorkload
from asic_sim.prefill_bom_gate import (
    BomAssumptions,
    GDDR7Device,
    physical_memory_floor,
    run_bom_gate,
)


def test_kimi_k3_gddr_capacity_becomes_hundreds_of_packages() -> None:
    model = get_model("kimi-k3")
    floor = physical_memory_floor(
        model,
        GDDR7Device(),
        BomAssumptions(),
        weight_bits=4.0,
        overhead_fraction=0.05,
    )
    assert floor.model_weight_bytes == 1.47e12
    assert floor.gddr_chips == 735
    assert floor.modules == 46
    assert floor.capacity_per_module_gb == 32.0
    assert floor.raw_aggregate_bandwidth_tb_s > 80.0
    assert floor.target_to_raw_bandwidth_fraction < 0.10


def test_default_volume_gate_only_reaches_3x_at_very_high_volume() -> None:
    result = run_bom_gate(
        get_model("kimi-k3"),
        workload=PrefillWorkload(prompt_tokens=32_768, weight_bits=4.0),
    )
    assert result.slowdown_vs_reference < 2.0
    assert 100_000 < result.capex_3x_eur < 115_000
    assert 60_000 < result.capex_5x_eur < 70_000

    by_volume = {row.units: row for row in result.volumes}
    assert not by_volume[100].passes_3x
    assert not by_volume[1_000].passes_3x
    assert by_volume[10_000].passes_3x
    assert not by_volume[10_000].passes_5x
    assert by_volume[10_000].required_sell_price_eur < result.capex_3x_eur
    assert by_volume[10_000].required_sell_price_eur > result.capex_5x_eur


def test_zero_margin_still_does_not_rescue_low_volume_nre() -> None:
    result = run_bom_gate(
        get_model("kimi-k3"),
        workload=PrefillWorkload(prompt_tokens=32_768, weight_bits=4.0),
        assumptions=BomAssumptions(gross_margin=0.0),
    )
    by_volume = {row.units: row for row in result.volumes}
    assert by_volume[100].required_sell_price_eur > result.capex_3x_eur
    assert by_volume[1_000].required_sell_price_eur > result.capex_5x_eur
