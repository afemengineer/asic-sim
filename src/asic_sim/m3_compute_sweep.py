from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from .architectures import _bytes, estimate_layers
from .formatting import fmt_bytes, fmt_rate
from .hardware import HardwareSpec, get_hardware
from .m3 import KIMI_K3_STATE, _attention_context_flops, _mla_entry_bytes, simulate_m3
from .models import ModelSpec, get_model


DEFAULT_CANDIDATES = ((4, 64.0), (8, 64.0), (16, 128.0))
DEFAULT_FP16_TOPS = (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)


@dataclass(frozen=True, slots=True)
class AttentionBalancePoint:
    mode: str
    entry_bytes_per_cached_token_layer: float
    state_traffic_bytes_per_layer: float
    state_flops_per_layer: float
    state_arithmetic_intensity_flops_per_byte: float
    state_balance_tops: float
    full_anchor_memory_bytes_per_layer: float
    full_anchor_flops_per_layer: float
    full_anchor_arithmetic_intensity_flops_per_byte: float
    full_anchor_balance_tops: float


@dataclass(frozen=True, slots=True)
class ComputeSweepRow:
    cluster_size: int
    link_bandwidth_gb_s: float
    fp16_tops: float
    routed_tops: float
    concurrency: int
    latent_resident: bool
    expanded_resident: bool
    latent_tps: float
    expanded_tps: float
    latent_over_expanded: float | None
    latent_compute_utilization: float
    latent_memory_utilization: float
    expanded_compute_utilization: float
    expanded_memory_utilization: float
    latent_cache_bytes_per_sequence: float
    expanded_cache_bytes_per_sequence: float


def attention_balance_points(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    context_length: int,
    other_weight_bits: float = 16.0,
    kv_bits: float = 16.0,
    overhead_fraction: float = 0.05,
) -> tuple[AttentionBalancePoint, ...]:
    if model.key != "kimi-k3":
        raise ValueError("compute balance is currently calibrated only for kimi-k3")
    if context_length < 0:
        raise ValueError("context_length cannot be negative")
    if min(other_weight_bits, kv_bits) <= 0:
        raise ValueError("bit widths must be positive")

    # M3's per-layer decomposition gives every layer the same 'other' parameter
    # bucket. Use a representative full-attention MoE layer so the analytical
    # balance matches the event model's anchor operation.
    layers = estimate_layers(model)
    representative = layers[KIMI_K3_STATE.full_attention_layers[0]]
    other_bytes = _bytes(representative.other_always_on_parameters, other_weight_bits, overhead_fraction)
    other_flops = 2.0 * representative.other_always_on_parameters
    memory_bw = hardware.local_memory_bandwidth_bytes_s

    points: list[AttentionBalancePoint] = []
    for mode in ("latent", "expanded"):
        entry = _mla_entry_bytes(KIMI_K3_STATE, kv_bits, mode)
        traffic = entry * (context_length + 1)
        state_flops = _attention_context_flops(KIMI_K3_STATE, context_length, mode)
        state_intensity = state_flops / traffic if traffic else 0.0
        state_balance = state_intensity * memory_bw / 1e12

        full_bytes = other_bytes + traffic
        full_flops = other_flops + state_flops
        full_intensity = full_flops / full_bytes if full_bytes else 0.0
        full_balance = full_intensity * memory_bw / 1e12
        points.append(
            AttentionBalancePoint(
                mode=mode,
                entry_bytes_per_cached_token_layer=entry,
                state_traffic_bytes_per_layer=traffic,
                state_flops_per_layer=state_flops,
                state_arithmetic_intensity_flops_per_byte=state_intensity,
                state_balance_tops=state_balance,
                full_anchor_memory_bytes_per_layer=full_bytes,
                full_anchor_flops_per_layer=full_flops,
                full_anchor_arithmetic_intensity_flops_per_byte=full_intensity,
                full_anchor_balance_tops=full_balance,
            )
        )
    return tuple(points)


def sweep_compute_balance(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    candidates: tuple[tuple[int, float], ...] = DEFAULT_CANDIDATES,
    fp16_tops_values: tuple[float, ...] = DEFAULT_FP16_TOPS,
    routed_factor: float = 4.0,
    concurrency: int = 1,
    context_length: int = 16_384,
    shared_shards: int = 2,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float = 16.0,
    other_weight_bits: float = 16.0,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    kv_bits: float = 16.0,
    kda_state_bits: float = 16.0,
    profile: str = "balanced",
    seed: int = 42,
) -> tuple[ComputeSweepRow, ...]:
    if routed_factor <= 0:
        raise ValueError("routed_factor must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if context_length < 0:
        raise ValueError("context_length cannot be negative")
    tops_values = tuple(sorted(set(float(value) for value in fp16_tops_values)))
    if not tops_values or tops_values[0] <= 0:
        raise ValueError("FP16 TOPS values must be positive")

    rows: list[ComputeSweepRow] = []
    for cluster_size, bandwidth in candidates:
        if shared_shards > cluster_size:
            continue
        for fp16_tops in tops_values:
            routed_tops = fp16_tops * routed_factor
            common = dict(
                cluster_size=cluster_size,
                link_bandwidth_gb_s=bandwidth,
                shared_shards=shared_shards,
                concurrency=concurrency,
                context_length=context_length,
                bits_per_weight=bits_per_weight,
                shared_expert_bits=shared_expert_bits,
                other_weight_bits=other_weight_bits,
                overhead_fraction=overhead_fraction,
                activation_bits=activation_bits,
                kv_bits=kv_bits,
                kda_state_bits=kda_state_bits,
                routed_tops=routed_tops,
                shared_tops=fp16_tops,
                other_tops=fp16_tops,
                profile=profile,
                seed=seed,
            )
            latent = simulate_m3(model, hardware, kv_mode="latent", **common)
            expanded = simulate_m3(model, hardware, kv_mode="expanded", **common)
            ratio = None
            if latent.resident and expanded.resident and expanded.throughput_tps > 0:
                ratio = latent.throughput_tps / expanded.throughput_tps
            rows.append(
                ComputeSweepRow(
                    cluster_size=cluster_size,
                    link_bandwidth_gb_s=bandwidth,
                    fp16_tops=fp16_tops,
                    routed_tops=routed_tops,
                    concurrency=concurrency,
                    latent_resident=latent.resident,
                    expanded_resident=expanded.resident,
                    latent_tps=latent.throughput_tps,
                    expanded_tps=expanded.throughput_tps,
                    latent_over_expanded=ratio,
                    latent_compute_utilization=latent.hottest_compute_utilization,
                    latent_memory_utilization=latent.hottest_memory_utilization,
                    expanded_compute_utilization=expanded.hottest_compute_utilization,
                    expanded_memory_utilization=expanded.hottest_memory_utilization,
                    latent_cache_bytes_per_sequence=latent.cache_bytes_per_sequence,
                    expanded_cache_bytes_per_sequence=expanded.cache_bytes_per_sequence,
                )
            )
    return tuple(rows)


def crossover_tops(rows: tuple[ComputeSweepRow, ...], cluster_size: int) -> float | None:
    candidates = [
        row for row in rows
        if row.cluster_size == cluster_size
        and row.latent_resident
        and row.expanded_resident
        and row.latent_over_expanded is not None
        and row.latent_over_expanded >= 1.0
    ]
    if not candidates:
        return None
    return min(row.fp16_tops for row in candidates)


def _parse_candidate(value: str) -> tuple[int, float]:
    try:
        cluster_text, bandwidth_text = value.split(":", 1)
        cluster = int(cluster_text)
        bandwidth = float(bandwidth_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("candidate must look like CLUSTER:GB/s, e.g. 4:64") from exc
    if cluster <= 0 or bandwidth <= 0:
        raise argparse.ArgumentTypeError("cluster and bandwidth must be positive")
    return cluster, bandwidth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m3-compute",
        description="Sweep Kimi K3 compute-to-memory balance and latent-vs-expanded MLA crossover.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--hardware", default="fabric-64x32")
    parser.add_argument("--candidate", type=_parse_candidate, action="append", dest="candidates")
    parser.add_argument("--tops", type=float, nargs="+", default=list(DEFAULT_FP16_TOPS), help="effective FP16 TOPS/tile")
    parser.add_argument("--routed-factor", type=float, default=4.0, help="routed-Q4 TOPS = factor * FP16 TOPS")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--shared-shards", type=int, default=2)
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=16.0)
    parser.add_argument("--other-bits", type=float, default=16.0)
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--kda-state-bits", type=float, default=16.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--activation-bits", type=float, default=16.0)
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _fmt_tops(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f}P"
    return f"{value:.0f}T"


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    candidates = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES

    balance = attention_balance_points(
        model,
        hardware,
        context_length=args.context,
        other_weight_bits=args.other_bits,
        kv_bits=args.kv_bits,
        overhead_fraction=args.overhead,
    )
    rows = sweep_compute_balance(
        model,
        hardware,
        candidates=candidates,
        fp16_tops_values=tuple(args.tops),
        routed_factor=args.routed_factor,
        concurrency=args.concurrency,
        context_length=args.context,
        shared_shards=args.shared_shards,
        bits_per_weight=args.bits,
        shared_expert_bits=args.shared_bits,
        other_weight_bits=args.other_bits,
        overhead_fraction=args.overhead,
        activation_bits=args.activation_bits,
        kv_bits=args.kv_bits,
        kda_state_bits=args.kda_state_bits,
        profile=args.profile,
        seed=args.seed,
    )

    print(f"{model.name} -> {hardware.name}")
    print(f"  context / concurrency:  {args.context:,} / {args.concurrency}")
    print(f"  local DRAM bandwidth:   {hardware.local_memory_bandwidth_bytes_s / 1e12:g} TB/s/tile")
    print(f"  precision:              routed {args.bits:g}b, shared {args.shared_bits:g}b, other {args.other_bits:g}b, KV {args.kv_bits:g}b")
    print(f"  compute coupling:       routed TOPS = {args.routed_factor:g} x FP16 TOPS")
    print(f"  shared MLP shards:      {args.shared_shards}")

    print("\nFULL-ATTENTION ANCHOR BALANCE")
    print("MODE      KV ENTRY   STATE/LAYER   STATE GFLOP   STATE AI   STATE BAL    FULL AI   FULL BAL")
    for point in balance:
        print(
            f"{point.mode:<9} {fmt_bytes(point.entry_bytes_per_cached_token_layer):>9}  "
            f"{fmt_bytes(point.state_traffic_bytes_per_layer):>11}  "
            f"{point.state_flops_per_layer / 1e9:>10.3f}  "
            f"{point.state_arithmetic_intensity_flops_per_byte:>8.2f}  "
            f"{_fmt_tops(point.state_balance_tops):>9}  "
            f"{point.full_anchor_arithmetic_intensity_flops_per_byte:>8.2f}  "
            f"{_fmt_tops(point.full_anchor_balance_tops):>9}"
        )

    print("\nM3 COMPUTE SWEEP")
    print("CLUSTER LINK  FP16   ROUTED  LATENT TPS   EXPAND TPS   L/E     LAT COMP LAT MEM  EXP COMP EXP MEM")
    for row in rows:
        latent = fmt_rate(row.latent_tps) if row.latent_resident else "NOFIT"
        expanded = fmt_rate(row.expanded_tps) if row.expanded_resident else "NOFIT"
        ratio = "—" if row.latent_over_expanded is None else f"{row.latent_over_expanded:5.2f}x"
        print(
            f"C{row.cluster_size:<6} {row.link_bandwidth_gb_s:>4g}G  {_fmt_tops(row.fp16_tops):>6}  "
            f"{_fmt_tops(row.routed_tops):>6}  {latent:>11}  {expanded:>11}  {ratio:>6}  "
            f"{row.latent_compute_utilization:>7.1%} {row.latent_memory_utilization:>7.1%}  "
            f"{row.expanded_compute_utilization:>7.1%} {row.expanded_memory_utilization:>7.1%}"
        )

    print("\nLATENT/EXPANDED CROSSOVER")
    for cluster_size, bandwidth in candidates:
        crossover = crossover_tops(rows, cluster_size)
        if crossover is None:
            print(f"  C{cluster_size}@{bandwidth:g}GB/s: no crossover inside supplied TOPS sweep (or expanded mode does not fit).")
        else:
            print(
                f"  C{cluster_size}@{bandwidth:g}GB/s: latent first matches/beats expanded at "
                f"~{crossover:g} FP16 TOPS/tile (~{crossover * args.routed_factor:g} routed TOPS/tile)."
            )

    if rows:
        first = rows[0]
        ratio = (
            first.expanded_cache_bytes_per_sequence / first.latent_cache_bytes_per_sequence
            if first.latent_cache_bytes_per_sequence > 0
            else math.inf
        )
        print("\nCAPACITY CONTEXT")
        print(f"  latent cache/sequence:  {fmt_bytes(first.latent_cache_bytes_per_sequence)}")
        print(f"  expanded cache/seq:     {fmt_bytes(first.expanded_cache_bytes_per_sequence)}")
        print(f"  expanded / latent:      {ratio:.1f}x")

    print("\nINTERPRETATION")
    print("  - FULL BAL is the effective FP16 TOPS/tile needed to match 105 TB/s for a representative full-attention anchor operation at this context.")
    print("  - The event sweep couples routed-Q4 compute to FP16 compute through --routed-factor; these are design assumptions, not measured silicon figures.")
    print("  - Expanded is retained only as a reference/cache-materialization baseline; latent MLA remains the intended serving architecture.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
