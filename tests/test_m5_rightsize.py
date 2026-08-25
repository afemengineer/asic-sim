from asic_sim.m4_heterogeneous import HeterogeneousPhysicalSpec
from asic_sim.m5_commercial import Economics, Workload
from asic_sim.m5_rightsize import (
    GDDR7Part,
    RightSizeResult,
    bank_geometry,
    dedicated_one_node_budget,
    prefill_supply_request_rate,
)
from asic_sim.models import get_model


def test_production_gddr7_anchor_geometry() -> None:
    part = GDDR7Part()
    assert part.capacity_gb == 3.0
    assert part.bandwidth_gb_s() == 112.0


def test_kimi_capacity_sets_nearly_constant_gddr_package_count() -> None:
    model = get_model("kimi-k3")
    physical = HeterogeneousPhysicalSpec()
    g8 = bank_geometry(model, physical, GDDR7Part(), 8)
    g12 = bank_geometry(model, physical, GDDR7Part(), 12)
    g16 = bank_geometry(model, physical, GDDR7Part(), 16)

    assert g8.bank_count == 61
    assert g12.bank_count == 41
    assert g16.bank_count == 31
    assert 480 <= g8.total_devices <= 500
    assert 480 <= g12.total_devices <= 500
    assert 480 <= g16.total_devices <= 500


def test_one_mi355x_agent_prefill_supply_is_about_1p59_requests_per_second() -> None:
    workload = Workload("agent", 8192, 1024)
    rate = prefill_supply_request_rate("mi355x", workload)
    assert 1.58 < rate < 1.59


def test_dedicated_budget_charges_a_whole_prefill_node() -> None:
    model = get_model("kimi-k3")
    physical = HeterogeneousPhysicalSpec()
    workload = Workload("agent", 8192, 1024)
    economics = Economics(3.0, 0.70, 0.10, 1.20)
    decoder = RightSizeResult(
        placements=12,
        bank_count=41,
        bank_capacity_gb=36.0,
        total_devices=492,
        data_rate_gt_s=28.0,
        effective_bank_bandwidth_gb_s=1075.2,
        effective_aggregate_bandwidth_tb_s=44.0832,
        q4_tops_per_bank=8.0,
        resident=True,
        worst_profile="balanced",
        worst_decode_peak_tps=1800.0,
        worst_request_rate_s=1800.0 / 1024.0,
        feed_ratio=1.1,
        single_decode_tps=400.0,
        worst_collision_fraction=0.0,
    )
    budget = dedicated_one_node_budget(
        "mi355x",
        workload,
        decoder,
        model,
        physical,
        economics,
        handoff_gb_s=100.0,
        target_advantage=2.0,
        power_sensitivities_kw=(3.0, 5.0),
    )
    assert 0.99 <= budget.prefill_utilization <= 1.0
    assert budget.decoder_utilization < 1.0
    assert budget.max_decoder_cloud_hour_usd > 0
    assert len(budget.max_decoder_capex_by_power_kw) == 2
    assert budget.max_decoder_capex_by_power_kw[0][1] > budget.max_decoder_capex_by_power_kw[1][1]
