from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from .models import ModelSpec, get_model
from .prefill import (
    EconomicComparison,
    PrefillHardware,
    PrefillWorkload,
    REFERENCE_HARDWARE,
    candidate_hardware,
    compare_economics,
    simulate_prefill,
)


DEFAULT_COMPUTE_POPS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DEFAULT_BANDWIDTH_TB_S = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True, slots=True)
class DesignPoint:
    compute_pops: float
    bandwidth_tb_s: float
    input_tokens_per_s: float
    prefill_time_s: float
    bottleneck: str
    resident: bool
    capex_ceiling_eur: float
    capex_parity_eur: float
    tco_break_even_eur: float
    throughput_fraction_of_reference: float
    slowdown_vs_reference: float
    meets_latency_guardrail: bool


@dataclass(frozen=True, slots=True)
class KneePoint:
    fixed_value: float
    knee_value: float
    max_input_tokens_per_s: float
    capture_fraction: float


@dataclass(frozen=True, slots=True)
class DesignSpace:
    points: tuple[DesignPoint, ...]
    reference_input_tokens_per_s: float
    reference_prefill_time_s: float
    target_advantage: float
    max_slowdown_vs_reference: float | None
    memory_knees: tuple[KneePoint, ...]
    compute_knees: tuple[KneePoint, ...]


def _positive_values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result or result[0] <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def _capex_for_advantage(comparison: EconomicComparison, advantage: float) -> float:
    if advantage <= 0:
        raise ValueError("target_advantage must be positive")
    return comparison.capex_ceiling_1x_eur / advantage


def sweep_design_space(
    model: ModelSpec,
    workload: PrefillWorkload,
    *,
    reference_hw: PrefillHardware,
    compute_pops_values: Iterable[float] = DEFAULT_COMPUTE_POPS,
    bandwidth_tb_s_values: Iterable[float] = DEFAULT_BANDWIDTH_TB_S,
    capacity_gb: float = 2048.0,
    compute_efficiency: float = 0.45,
    memory_efficiency: float = 0.75,
    attention_efficiency: float = 0.30,
    candidate_power_w: float = 800.0,
    overhead_fraction: float = 0.05,
    target_advantage: float = 3.0,
    max_slowdown_vs_reference: float | None = 2.0,
    utilization: float = 0.60,
    lifetime_years: float = 3.0,
    electricity_eur_per_kwh: float = 0.15,
    knee_capture_fraction: float = 0.95,
) -> DesignSpace:
    compute_values = _positive_values(compute_pops_values, "compute")
    bandwidth_values = _positive_values(bandwidth_tb_s_values, "bandwidth")
    if capacity_gb <= 0:
        raise ValueError("capacity_gb must be positive")
    if target_advantage <= 0:
        raise ValueError("target_advantage must be positive")
    if max_slowdown_vs_reference is not None and max_slowdown_vs_reference <= 0:
        raise ValueError("max_slowdown_vs_reference must be positive or None")
    if not 0 < knee_capture_fraction <= 1:
        raise ValueError("knee_capture_fraction must be in (0, 1]")
    if reference_hw.purchase_price_eur is None or reference_hw.purchase_price_eur <= 0:
        raise ValueError("reference hardware requires a positive purchase price")

    reference_result = simulate_prefill(
        model,
        reference_hw,
        workload,
        overhead_fraction=overhead_fraction,
    )
    if reference_result.input_tokens_per_s <= 0:
        raise ValueError("reference throughput must be positive")

    points: list[DesignPoint] = []
    for compute_pops in compute_values:
        for bandwidth_tb_s in bandwidth_values:
            candidate_hw = candidate_hardware(
                compute_pops=compute_pops,
                bandwidth_tb_s=bandwidth_tb_s,
                capacity_gb=capacity_gb,
                compute_efficiency=compute_efficiency,
                memory_efficiency=memory_efficiency,
                attention_efficiency=attention_efficiency,
                power_w=candidate_power_w,
            )
            candidate_result = simulate_prefill(
                model,
                candidate_hw,
                workload,
                overhead_fraction=overhead_fraction,
            )
            comparison = compare_economics(
                candidate_result,
                reference_result,
                reference_price_eur=reference_hw.purchase_price_eur,
                candidate_power_w=candidate_power_w,
                reference_power_w=reference_hw.power_w,
                utilization=utilization,
                lifetime_years=lifetime_years,
                electricity_eur_per_kwh=electricity_eur_per_kwh,
            )
            slowdown = candidate_result.prefill_time_s / reference_result.prefill_time_s
            meets_guardrail = (
                max_slowdown_vs_reference is None
                or slowdown <= max_slowdown_vs_reference
            )
            points.append(
                DesignPoint(
                    compute_pops=compute_pops,
                    bandwidth_tb_s=bandwidth_tb_s,
                    input_tokens_per_s=candidate_result.input_tokens_per_s,
                    prefill_time_s=candidate_result.prefill_time_s,
                    bottleneck=candidate_result.bottleneck,
                    resident=candidate_result.resident,
                    capex_ceiling_eur=_capex_for_advantage(comparison, target_advantage),
                    capex_parity_eur=comparison.capex_ceiling_1x_eur,
                    tco_break_even_eur=comparison.break_even_candidate_capex_tco_eur,
                    throughput_fraction_of_reference=(
                        candidate_result.input_tokens_per_s
                        / reference_result.input_tokens_per_s
                    ),
                    slowdown_vs_reference=slowdown,
                    meets_latency_guardrail=meets_guardrail,
                )
            )

    point_tuple = tuple(points)
    memory_knees = _knees_by_compute(
        point_tuple,
        compute_values,
        bandwidth_values,
        knee_capture_fraction,
    )
    compute_knees = _knees_by_bandwidth(
        point_tuple,
        compute_values,
        bandwidth_values,
        knee_capture_fraction,
    )
    return DesignSpace(
        points=point_tuple,
        reference_input_tokens_per_s=reference_result.input_tokens_per_s,
        reference_prefill_time_s=reference_result.prefill_time_s,
        target_advantage=target_advantage,
        max_slowdown_vs_reference=max_slowdown_vs_reference,
        memory_knees=memory_knees,
        compute_knees=compute_knees,
    )


def _knees_by_compute(
    points: tuple[DesignPoint, ...],
    compute_values: tuple[float, ...],
    bandwidth_values: tuple[float, ...],
    capture_fraction: float,
) -> tuple[KneePoint, ...]:
    knees: list[KneePoint] = []
    for compute in compute_values:
        rows = [point for point in points if point.compute_pops == compute]
        max_tps = max(point.input_tokens_per_s for point in rows)
        threshold = max_tps * capture_fraction
        eligible = [
            point.bandwidth_tb_s
            for point in rows
            if point.input_tokens_per_s >= threshold
        ]
        knees.append(
            KneePoint(
                fixed_value=compute,
                knee_value=min(eligible),
                max_input_tokens_per_s=max_tps,
                capture_fraction=capture_fraction,
            )
        )
    return tuple(knees)


def _knees_by_bandwidth(
    points: tuple[DesignPoint, ...],
    compute_values: tuple[float, ...],
    bandwidth_values: tuple[float, ...],
    capture_fraction: float,
) -> tuple[KneePoint, ...]:
    knees: list[KneePoint] = []
    for bandwidth in bandwidth_values:
        rows = [point for point in points if point.bandwidth_tb_s == bandwidth]
        max_tps = max(point.input_tokens_per_s for point in rows)
        threshold = max_tps * capture_fraction
        eligible = [
            point.compute_pops
            for point in rows
            if point.input_tokens_per_s >= threshold
        ]
        knees.append(
            KneePoint(
                fixed_value=bandwidth,
                knee_value=min(eligible),
                max_input_tokens_per_s=max_tps,
                capture_fraction=capture_fraction,
            )
        )
    return tuple(knees)


def _fmt_eur_compact(value: float) -> str:
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"€{value / 1_000:.1f}k"
    return f"€{value:.0f}"


def _fmt_point(point: DesignPoint) -> str:
    marker = "" if point.meets_latency_guardrail else "!"
    return f"{_fmt_eur_compact(point.capex_ceiling_eur)}{marker}"


def _parse_positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-prefill-space",
        description=(
            "Sweep prefill compute x external-memory bandwidth and report the maximum "
            "all-in candidate CAPEX that preserves a chosen economic advantage."
        ),
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--tokens", type=int, default=32_768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--weight-bits", type=float, default=4.0)
    parser.add_argument("--activation-bits", type=float, default=8.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--weight-reload-factor", type=float, default=1.15)
    parser.add_argument("--activation-traffic-factor", type=float, default=8.0)
    parser.add_argument("--kv-bytes-token", type=float, default=0.0)
    parser.add_argument("--fixed-overhead-ms", type=float, default=0.0)

    parser.add_argument(
        "--compute-pops",
        type=_parse_positive_float,
        nargs="+",
        default=list(DEFAULT_COMPUTE_POPS),
    )
    parser.add_argument(
        "--bandwidth-tb-s",
        type=_parse_positive_float,
        nargs="+",
        default=list(DEFAULT_BANDWIDTH_TB_S),
    )
    parser.add_argument("--capacity-gb", type=_parse_positive_float, default=2048.0)
    parser.add_argument("--compute-eff", type=float, default=0.45)
    parser.add_argument("--memory-eff", type=float, default=0.75)
    parser.add_argument("--attention-eff", type=float, default=0.30)
    parser.add_argument("--candidate-power-w", type=float, default=800.0)

    parser.add_argument(
        "--reference",
        choices=tuple(sorted(REFERENCE_HARDWARE)),
        default="b300-eu",
    )
    parser.add_argument(
        "--target-advantage",
        type=_parse_positive_float,
        default=3.0,
        help="required input tok/s per euro advantage; default 3x",
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=2.0,
        help=(
            "single-prompt prefill may be at most this many times slower than the "
            "reference; use 0 to disable the latency guardrail"
        ),
    )
    parser.add_argument("--knee-capture", type=float, default=0.95)
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument("--lifetime-years", type=float, default=3.0)
    parser.add_argument("--electricity-eur-kwh", type=float, default=0.15)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    workload = PrefillWorkload(
        prompt_tokens=args.tokens,
        concurrent_prompts=args.batch,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bytes_per_token=args.kv_bytes_token,
        weight_reload_factor=args.weight_reload_factor,
        activation_traffic_factor=args.activation_traffic_factor,
        fixed_overhead_ms=args.fixed_overhead_ms,
    )
    max_slowdown = None if args.max_slowdown == 0 else args.max_slowdown
    reference_hw = REFERENCE_HARDWARE[args.reference]
    space = sweep_design_space(
        model,
        workload,
        reference_hw=reference_hw,
        compute_pops_values=args.compute_pops,
        bandwidth_tb_s_values=args.bandwidth_tb_s,
        capacity_gb=args.capacity_gb,
        compute_efficiency=args.compute_eff,
        memory_efficiency=args.memory_eff,
        attention_efficiency=args.attention_eff,
        candidate_power_w=args.candidate_power_w,
        overhead_fraction=args.overhead,
        target_advantage=args.target_advantage,
        max_slowdown_vs_reference=max_slowdown,
        utilization=args.utilization,
        lifetime_years=args.lifetime_years,
        electricity_eur_per_kwh=args.electricity_eur_kwh,
        knee_capture_fraction=args.knee_capture,
    )

    compute_values = _positive_values(args.compute_pops, "compute")
    bandwidth_values = _positive_values(args.bandwidth_tb_s, "bandwidth")
    lookup = {
        (point.compute_pops, point.bandwidth_tb_s): point
        for point in space.points
    }

    print(
        f"{model.name} | {args.tokens:,} token prefill x batch {args.batch} | "
        f"target={args.target_advantage:g}x input-tok/s/€ vs {args.reference}"
    )
    print(
        f"Reference roofline: {space.reference_input_tokens_per_s:,.0f} input tok/s | "
        f"{space.reference_prefill_time_s:.3f}s prefill"
    )
    if max_slowdown is not None:
        print(
            f"Latency guardrail: <= {max_slowdown:g}x reference prefill time. "
            "'!' means the economic ceiling exists but the point fails latency."
        )
    else:
        print("Latency guardrail: disabled")
    print()
    print("MAXIMUM ALL-IN CANDIDATE CAPEX")
    header = f"{'POPS':>7}" + "".join(
        f"{bandwidth:>13g} TB/s" for bandwidth in bandwidth_values
    )
    print(header)
    for compute in compute_values:
        row = f"{compute:>7g}"
        for bandwidth in bandwidth_values:
            row += f"{_fmt_point(lookup[(compute, bandwidth)]):>18}"
        print(row)

    print()
    print(f"MEMORY KNEE ({args.knee_capture:.0%} of best throughput at each compute point)")
    print(f"{'POPS':>7} {'minimum TB/s':>14} {'best tok/s':>14}")
    for knee in space.memory_knees:
        print(
            f"{knee.fixed_value:>7g} {knee.knee_value:>14g} "
            f"{knee.max_input_tokens_per_s:>14,.0f}"
        )

    print()
    print(f"COMPUTE KNEE ({args.knee_capture:.0%} of best throughput at each bandwidth point)")
    print(f"{'TB/s':>7} {'minimum POPS':>14} {'best tok/s':>14}")
    for knee in space.compute_knees:
        print(
            f"{knee.fixed_value:>7g} {knee.knee_value:>14g} "
            f"{knee.max_input_tokens_per_s:>14,.0f}"
        )

    passing = [point for point in space.points if point.meets_latency_guardrail]
    if passing:
        richest = max(passing, key=lambda point: point.capex_ceiling_eur)
        leanest = min(
            passing,
            key=lambda point: (
                point.compute_pops * point.bandwidth_tb_s,
                point.compute_pops,
                point.bandwidth_tb_s,
            ),
        )
        print()
        print("GUARDRAIL SUMMARY")
        print(
            f"  {len(passing)}/{len(space.points)} grid points meet the latency guardrail."
        )
        print(
            f"  Largest {args.target_advantage:g}x CAPEX envelope: "
            f"{_fmt_eur_compact(richest.capex_ceiling_eur)} at "
            f"{richest.compute_pops:g} POPS / {richest.bandwidth_tb_s:g} TB/s."
        )
        print(
            f"  Smallest resource-product point that passes latency: "
            f"{leanest.compute_pops:g} POPS / {leanest.bandwidth_tb_s:g} TB/s, "
            f"CAPEX ceiling {_fmt_eur_compact(leanest.capex_ceiling_eur)}."
        )
    else:
        print()
        print("GUARDRAIL SUMMARY")
        print("  No design point in this grid meets the latency guardrail.")

    print()
    print(
        "Interpretation: a cell is not a BOM estimate. It is the maximum total purchase "
        "price the complete candidate appliance may have at that roofline point while "
        f"retaining the requested {args.target_advantage:g}x throughput-per-euro advantage."
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
