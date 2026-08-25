from asic_sim.models import get_model
from asic_sim.prefill import (
    PrefillWorkload,
    REFERENCE_HARDWARE,
    candidate_hardware,
    compare_economics,
    simulate_prefill,
)


def test_kimi_prefill_candidate_is_compute_bound_at_4_pops():
    model = get_model("kimi-k3")
    workload = PrefillWorkload(prompt_tokens=32_768)
    hw = candidate_hardware(
        compute_pops=4.0,
        bandwidth_tb_s=2.0,
        capacity_gb=2048.0,
    )
    result = simulate_prefill(model, hw, workload)
    assert result.resident
    assert result.bottleneck == "compute"
    assert result.compute_time_s > result.memory_time_s
    assert result.input_tokens_per_s > 0


def test_more_compute_eventually_reaches_memory_wall():
    model = get_model("kimi-k3")
    workload = PrefillWorkload(prompt_tokens=32_768)
    slow = simulate_prefill(
        model,
        candidate_hardware(compute_pops=4.0, bandwidth_tb_s=2.0, capacity_gb=2048.0),
        workload,
    )
    fast = simulate_prefill(
        model,
        candidate_hardware(compute_pops=64.0, bandwidth_tb_s=2.0, capacity_gb=2048.0),
        workload,
    )
    assert fast.input_tokens_per_s > slow.input_tokens_per_s
    assert fast.bottleneck == "memory"


def test_capex_ceiling_scales_from_reference_throughput():
    model = get_model("kimi-k3")
    workload = PrefillWorkload(prompt_tokens=32_768)
    candidate_hw = candidate_hardware(
        compute_pops=8.0,
        bandwidth_tb_s=2.0,
        capacity_gb=2048.0,
        power_w=800.0,
    )
    reference_hw = REFERENCE_HARDWARE["b300-eu"]
    candidate = simulate_prefill(model, candidate_hw, workload)
    reference = simulate_prefill(model, reference_hw, workload)
    econ = compare_economics(
        candidate,
        reference,
        reference_price_eur=reference_hw.purchase_price_eur,
        candidate_power_w=candidate_hw.power_w,
        reference_power_w=reference_hw.power_w,
    )
    expected = (
        reference_hw.purchase_price_eur
        * candidate.input_tokens_per_s
        / reference.input_tokens_per_s
    )
    assert econ.capex_ceiling_1x_eur == expected
    assert econ.capex_ceiling_3x_eur == expected / 3.0
    assert econ.capex_ceiling_5x_eur == expected / 5.0
    assert econ.break_even_candidate_capex_tco_eur >= 0


def test_reference_prices_are_auditable():
    for ref in REFERENCE_HARDWARE.values():
        assert ref.purchase_price_eur and ref.purchase_price_eur > 0
        assert ref.price_source.startswith("https://")
        assert ref.price_date == "2026-08-25"
