from __future__ import annotations

import pytest

from asic_sim.models import get_model
from asic_sim.prefill import PrefillWorkload, REFERENCE_HARDWARE
from asic_sim.prefill_design_space import sweep_design_space


def _space(*, target_advantage: float = 3.0, max_slowdown: float | None = 2.0):
    return sweep_design_space(
        get_model("kimi-k3"),
        PrefillWorkload(prompt_tokens=32_768),
        reference_hw=REFERENCE_HARDWARE["b300-eu"],
        compute_pops_values=(4.0, 16.0),
        bandwidth_tb_s_values=(1.0, 2.0, 4.0),
        target_advantage=target_advantage,
        max_slowdown_vs_reference=max_slowdown,
    )


def test_grid_shape_and_positive_results() -> None:
    space = _space()
    assert len(space.points) == 6
    assert len(space.memory_knees) == 2
    assert len(space.compute_knees) == 3
    assert all(point.input_tokens_per_s > 0 for point in space.points)
    assert all(point.capex_ceiling_eur > 0 for point in space.points)


def test_bandwidth_never_reduces_throughput_at_fixed_compute() -> None:
    space = _space()
    for compute in (4.0, 16.0):
        rows = sorted(
            (point for point in space.points if point.compute_pops == compute),
            key=lambda point: point.bandwidth_tb_s,
        )
        throughputs = [point.input_tokens_per_s for point in rows]
        assert throughputs == sorted(throughputs)


def test_advantage_scales_capex_ceiling_inverse_linearly() -> None:
    parity = _space(target_advantage=1.0)
    triple = _space(target_advantage=3.0)
    parity_lookup = {
        (point.compute_pops, point.bandwidth_tb_s): point
        for point in parity.points
    }
    for point in triple.points:
        peer = parity_lookup[(point.compute_pops, point.bandwidth_tb_s)]
        assert point.capex_ceiling_eur == pytest.approx(peer.capex_ceiling_eur / 3.0)


def test_latency_guardrail_can_reject_all_points() -> None:
    space = _space(max_slowdown=0.01)
    assert not any(point.meets_latency_guardrail for point in space.points)


def test_memory_knees_use_requested_grid_values() -> None:
    space = _space()
    assert {knee.knee_value for knee in space.memory_knees}.issubset({1.0, 2.0, 4.0})
    assert {knee.knee_value for knee in space.compute_knees}.issubset({4.0, 16.0})
