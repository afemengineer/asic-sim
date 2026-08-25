from asic_sim.m4_heterogeneous import ExpertBankCandidate, HeterogeneousPhysicalSpec, _cache_bytes_per_sequence
from asic_sim.m5_commercial import (
    COMMERCIAL_NODES,
    CustomMetrics,
    Economics,
    Workload,
    _custom_prefill_tps,
    commercial_metrics,
    hybrid_budget,
)
from asic_sim.models import get_model


def test_commercial_baselines_are_full_eight_gpu_nodes() -> None:
    b300 = COMMERCIAL_NODES["b300"]
    mi = COMMERCIAL_NODES["mi355x"]
    assert b300.gpu_count == mi.gpu_count == 8
    assert b300.memory_gb == mi.memory_gb == 2304.0
    assert b300.memory_bandwidth_tb_s == mi.memory_bandwidth_tb_s == 64.0
    assert b300.decode_peak_tps > mi.decode_peak_tps
    assert mi.cloud_node_hour_usd < b300.cloud_node_hour_usd


def test_commercial_cost_metrics_are_positive() -> None:
    economics = Economics(3.0, 0.70, 0.10, 1.20)
    workload = Workload("chat", 1024, 400)
    row = commercial_metrics(COMMERCIAL_NODES["mi355x"], workload, economics)
    assert row.request_rate_s > 0
    assert row.output_rate_tps > 0
    assert row.ttft_s > 0
    assert row.cloud_usd_per_m_output > 0
    assert row.owned_usd_per_m_output > 0


def test_custom_prefill_roof_is_finite_and_positive() -> None:
    model = get_model("kimi-k3")
    physical = HeterogeneousPhysicalSpec()
    candidate = ExpertBankCandidate(64, 24.0, 1344.0, 16.0, 64.0)
    rate = _custom_prefill_tps(
        model,
        physical,
        candidate,
        spine_modules=6,
        prompt_tokens=16_384,
        expert_efficiency=0.60,
        spine_efficiency=0.60,
    )
    assert 0 < rate < 20_000


def test_handoff_state_grows_with_prompt_length() -> None:
    model = get_model("kimi-k3")
    physical = HeterogeneousPhysicalSpec()
    short = _cache_bytes_per_sequence(model, 4096, physical)
    long = _cache_bytes_per_sequence(model, 131_072, physical)
    assert long > short > 0


def test_hybrid_budget_charges_prefill_and_handoff() -> None:
    economics = Economics(3.0, 0.70, 0.10, 1.20)
    workload = Workload("chat", 1024, 400)
    custom = CustomMetrics(
        spine_modules=6,
        cache_handoff_bytes=400e6,
        prefill_tps=5000.0,
        decode_single_tps=400.0,
        decode_peak_tps=3000.0,
        request_rate_s=1.0,
        ttft_s=0.2,
        e2e_single_s=1.2,
    )
    budget = hybrid_budget(
        COMMERCIAL_NODES["mi355x"],
        workload,
        custom,
        economics,
        handoff_gb_s=100.0,
        target_advantage=2.0,
        assumed_custom_power_kw=8.0,
    )
    assert budget.request_rate_s > 0
    assert budget.prefill_node_equivalents > 0
    assert budget.ttft_s > 0
