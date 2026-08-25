from asic_sim.m4_heterogeneous import (
    ExpertBankCandidate,
    HeterogeneousPhysicalSpec,
    build_capacity_report,
    economic_thresholds,
    simulate_all_fast_reference,
    simulate_candidate,
)
from asic_sim.models import get_model


def _setup(max_concurrency: int = 1):
    model = get_model("kimi-k3")
    spec = HeterogeneousPhysicalSpec()
    capacity = build_capacity_report(
        model,
        spec,
        context=16_384,
        max_concurrency=max_concurrency,
    )
    return model, spec, capacity


def test_m4_capacity_separates_routed_pool_from_fast_spine() -> None:
    model, spec, capacity = _setup()
    assert capacity.routed_storage_bytes > 1e12
    assert capacity.always_on_storage_bytes < capacity.routed_storage_bytes
    assert capacity.spine_modules >= 4
    assert capacity.reference_modules > capacity.spine_modules
    assert capacity.max_spine_storage_bytes <= spec.fast_capacity_bytes


def test_24x64gb_expert_banks_fit_but_16x64gb_do_not() -> None:
    model, spec, capacity = _setup()
    fit = ExpertBankCandidate(24, 64.0, 256.0, 16.0, 64.0)
    nofit = ExpertBankCandidate(16, 64.0, 256.0, 16.0, 64.0)
    fit_report = simulate_candidate(
        model, spec, capacity, fit, context=16_384, concurrency=1
    )
    nofit_report = simulate_candidate(
        model, spec, capacity, nofit, context=16_384, concurrency=1
    )
    assert fit_report.resident
    assert not nofit_report.resident


def test_more_expert_bank_bandwidth_cannot_hurt_single_token() -> None:
    model, spec, capacity = _setup()
    slow = ExpertBankCandidate(24, 64.0, 256.0, 16.0, 64.0)
    fast = ExpertBankCandidate(24, 64.0, 512.0, 16.0, 64.0)
    slow_report = simulate_candidate(
        model, spec, capacity, slow, context=16_384, concurrency=1
    )
    fast_report = simulate_candidate(
        model, spec, capacity, fast, context=16_384, concurrency=1
    )
    assert slow_report.resident and fast_report.resident
    assert fast_report.throughput_tps >= slow_report.throughput_tps
    assert fast_report.p95_latency_s <= slow_report.p95_latency_s


def test_fixed_expert_placement_exposes_collisions() -> None:
    model, spec, capacity = _setup()
    candidate = ExpertBankCandidate(24, 64.0, 512.0, 16.0, 64.0)
    report = simulate_candidate(
        model, spec, capacity, candidate, context=16_384, concurrency=1
    )
    assert 0 < report.average_distinct_banks <= model.experts_per_token
    assert 0 <= report.collision_fraction < 1
    assert report.p95_expert_barrier_s > 0


def test_all_fast_reference_and_economic_threshold_are_defined() -> None:
    model, spec, capacity = _setup(max_concurrency=8)
    candidate = ExpertBankCandidate(32, 64.0, 512.0, 16.0, 64.0)
    hetero = simulate_candidate(
        model, spec, capacity, candidate, context=16_384, concurrency=8
    )
    reference = simulate_all_fast_reference(
        model, spec, capacity, context=16_384, concurrency=8
    )
    thresholds = economic_thresholds(candidate, capacity, hetero, reference)
    assert hetero.resident and reference.resident
    assert hetero.throughput_tps > 0
    assert reference.throughput_tps > 0
    assert thresholds.bank_cost_ratio_for_lower_system_capex > 0
