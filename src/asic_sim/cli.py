from __future__ import annotations

import argparse
import sys

from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .hardware import HARDWARE_PRESETS, get_hardware
from .models import MODEL_SPECS, get_model
from .placement import balanced_placement, decompose_model
from .simulator import SimulationResult, simulate_decode


def _percent(value: float) -> str:
    return f"{100.0 * value:.5f}%"


def _print_model_table() -> None:
    print("MODEL      TOTAL       ACTIVE      LAYERS  EXPERTS  TOP-K  ARCH")
    for spec in MODEL_SPECS.values():
        print(
            f"{spec.key:<10} {spec.total_parameters / 1e9:>7.0f}B "
            f"{spec.active_parameters / 1e9:>9.0f}B {spec.num_layers:>7} "
            f"{spec.num_experts:>8} {spec.experts_per_token:>6}  {spec.architecture}"
        )


def _print_hardware_table() -> None:
    print("HARDWARE            CAPACITY   AGG BW       TILES  LOCAL BW    NOC LINK")
    for hw in HARDWARE_PRESETS.values():
        local_bw = hw.tile_memory_bandwidth_tb_s or hw.memory_bandwidth_tb_s
        noc = f"{hw.noc_link_bandwidth_tb_s:.1f} TB/s" if hw.noc_link_bandwidth_tb_s else "-"
        print(
            f"{hw.key:<19} {hw.capacity_gb:>7.0f} GB  {hw.memory_bandwidth_tb_s:>7.0f} TB/s "
            f"{hw.tiles:>5}  {local_bw:>7.1f} TB/s  {noc:>8}"
        )


def _print_result(result: SimulationResult) -> None:
    print(f"\n{result.model} on {result.hardware}")
    print(f"  quantization:          {result.bits_per_weight:g} bit + {result.quant_overhead_fraction:.1%} overhead")
    print(f"  model storage:         {fmt_bytes(result.storage_bytes)}")
    print(f"  accelerator capacity:  {fmt_bytes(result.capacity_bytes)}")
    print(f"  resident:              {'YES' if result.resident else 'NO'}")
    print(f"  max bits that fit:     {result.max_bits_that_fit:.3f} bit/weight at this overhead")
    print(f"  active weight bytes:   {fmt_bytes(result.active_weight_bytes_per_token)} / token")
    print(f"  remote activation:     {fmt_bytes(result.remote_activation_bytes_per_token)} / token")
    print(f"  local data fraction:   {_percent(result.local_data_fraction)}")
    if result.resident:
        print(f"  local-memory time:     {fmt_time_s(result.memory_time_single_stream_s)}")
        print(f"  ideal NoC injection:   {fmt_time_s(result.noc_time_ideal_s)}")
        print(f"  ideal router latency:  {fmt_time_s(result.router_time_ideal_s)}")
        print(f"  single-stream roof:    {fmt_rate(result.single_stream_memory_noc_roofline_tps)}")
        print(f"  steady-state BW roof:  {fmt_rate(result.steady_state_memory_roofline_tps)}")
    if result.warning:
        print(f"  WARNING: {result.warning}")


def _run_compare(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware_keys = args.hardware or ["hbm4-illustrative", "raptor-like", "fabric-32x32", "fabric-64x32"]
    for key in hardware_keys:
        result = simulate_decode(
            model,
            get_hardware(key),
            bits_per_weight=args.bits,
            quant_overhead_fraction=args.overhead,
            activation_bits=args.activation_bits,
            remote_expert_fraction=args.remote_expert_fraction,
        )
        _print_result(result)
    return 0


def _run_capacity(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    print(f"{model.name}: {model.total_parameters / 1e12:.3f}T total parameters")
    print("BITS   OVERHEAD   STORAGE       FITS")
    for bits in args.bits:
        storage = model.storage_bytes(bits, args.overhead)
        fits = "yes" if storage <= args.capacity_gb * 1e9 else "no"
        print(f"{bits:>4.2f}   {args.overhead:>7.1%}   {fmt_bytes(storage):>10}   {fits}")
    max_bits = model.minimum_bits_for_capacity(args.capacity_gb * 1e9, args.overhead)
    print(f"\nMaximum average bit-width that fits in {args.capacity_gb:g} GB: {max_bits:.3f} bits/weight")
    return 0


def _run_placement(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    report = balanced_placement(
        model,
        hardware,
        bits_per_weight=args.bits,
        overhead_fraction=args.overhead,
        activation_bits=args.activation_bits,
    )
    decomp = decompose_model(model)
    print(f"{report.model} -> {report.hardware}")
    print(f"  total storage:          {fmt_bytes(report.total_storage_bytes)}")
    print(f"  average / tile:         {fmt_bytes(report.average_storage_per_tile_bytes)}")
    print(f"  max estimated / tile:   {fmt_bytes(report.max_estimated_storage_per_tile_bytes)}")
    print(f"  tile capacity:           {fmt_bytes(report.tile_capacity_bytes)}")
    print(f"  system resident:         {'YES' if report.resident_system else 'NO'}")
    print(f"  balanced tile fit:       {'YES' if report.resident_per_tile_balanced else 'NO'}")
    print(f"  routed parameter pool:   {report.routed_pool_parameters / 1e9:,.1f}B ({decomp.routed_fraction:.2%})")
    print(f"  always-on parameters:    {report.always_on_parameters / 1e9:,.1f}B")
    if report.expert_shard_parameters is not None:
        print(f"  expert shard:            {report.expert_shard_parameters / 1e6:,.2f}M params")
        print(f"  expert shards / tile:    {report.expert_shards_min_per_tile}-{report.expert_shards_max_per_tile}")
    print(f"  expected remote experts: {report.expected_remote_expert_fraction:.2%}")
    print(f"  MoE network bytes/token: {fmt_bytes(report.expected_moe_activation_bytes_per_token)}")
    print(f"  average mesh hops:       {report.average_hops:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim",
        description="Fast M0/M1 architecture simulator for memory-stationary LLM inference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="show built-in model specifications")
    sub.add_parser("list-hardware", help="show built-in hardware presets")

    compare = sub.add_parser("compare", help="compare one model across accelerator presets")
    compare.add_argument("--model", default="glm-5.2")
    compare.add_argument("--hardware", action="append", help="hardware preset; repeat to compare several")
    compare.add_argument("--bits", type=float, default=4.0, help="average weight bit-width")
    compare.add_argument("--overhead", type=float, default=0.05, help="fractional quantization/metadata overhead")
    compare.add_argument("--activation-bits", type=float, default=16.0)
    compare.add_argument(
        "--remote-expert-fraction",
        type=float,
        default=None,
        help="fraction of expert dispatches that cross tiles; default=1 for tiled fabrics",
    )

    capacity = sub.add_parser("capacity", help="sweep model storage against a capacity target")
    capacity.add_argument("--model", default="kimi-k3")
    capacity.add_argument("--capacity-gb", type=float, default=1024.0)
    capacity.add_argument("--bits", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 16.0])
    capacity.add_argument("--overhead", type=float, default=0.05)

    placement = sub.add_parser("placement", help="estimate balanced expert/layer placement across tiles")
    placement.add_argument("--model", default="glm-5.2")
    placement.add_argument("--hardware", default="fabric-32x32")
    placement.add_argument("--bits", type=float, default=4.0)
    placement.add_argument("--overhead", type=float, default=0.05)
    placement.add_argument("--activation-bits", type=float, default=16.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-models":
            _print_model_table()
            return 0
        if args.command == "list-hardware":
            _print_hardware_table()
            return 0
        if args.command == "compare":
            return _run_compare(args)
        if args.command == "capacity":
            return _run_capacity(args)
        if args.command == "placement":
            return _run_placement(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    sys.exit(main())
