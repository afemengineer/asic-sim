from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from .models import ModelSpec, get_model

GB = 1e9
TB = 1e12
PFLOP = 1e15
SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True, slots=True)
class PrefillHardware:
    key: str
    name: str
    compute_flops_s: float
    memory_bandwidth_bytes_s: float
    memory_capacity_bytes: float
    compute_efficiency: float
    memory_efficiency: float
    attention_efficiency: float
    power_w: float
    purchase_price_eur: float | None = None
    price_source: str = ""
    price_date: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class PrefillWorkload:
    prompt_tokens: int = 32_768
    concurrent_prompts: int = 1
    weight_bits: float = 4.0
    activation_bits: float = 8.0
    kv_bytes_per_token: float = 0.0
    weight_reload_factor: float = 1.15
    activation_traffic_factor: float = 8.0
    fixed_overhead_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class PrefillResult:
    model_name: str
    hardware_name: str
    prompt_tokens: int
    concurrent_prompts: int
    resident: bool
    weight_bytes: float
    linear_flops: float
    attention_flops: float
    total_flops: float
    weight_traffic_bytes: float
    activation_traffic_bytes: float
    kv_output_bytes: float
    total_memory_traffic_bytes: float
    compute_time_s: float
    memory_time_s: float
    fixed_time_s: float
    prefill_time_s: float
    input_tokens_per_s: float
    compute_utilization_at_roof: float
    memory_utilization_at_roof: float
    arithmetic_intensity_flops_per_byte: float
    balance_intensity_flops_per_byte: float
    bottleneck: str
    energy_j_per_input_token: float


@dataclass(frozen=True, slots=True)
class EconomicComparison:
    candidate: PrefillResult
    reference: PrefillResult
    reference_price_eur: float
    candidate_price_eur: float | None
    capex_ceiling_1x_eur: float
    capex_ceiling_3x_eur: float
    capex_ceiling_5x_eur: float
    candidate_input_tps_per_eur: float | None
    reference_input_tps_per_eur: float
    relative_input_tps_per_eur: float | None
    candidate_cost_per_million_input_tokens_eur: float | None
    reference_cost_per_million_input_tokens_eur: float
    break_even_candidate_capex_tco_eur: float


# Current, externally purchasable node-level reference points. Prices are
# intentionally kept separate from silicon specifications and timestamped.
REFERENCE_HARDWARE: dict[str, PrefillHardware] = {
    "b300-eu": PrefillHardware(
        key="b300-eu",
        name="8x NVIDIA HGX B300 (EU purchase baseline)",
        compute_flops_s=108.0 * PFLOP,
        memory_bandwidth_bytes_s=64.0 * TB,
        memory_capacity_bytes=2_304 * GB,
        compute_efficiency=0.35,
        memory_efficiency=0.80,
        attention_efficiency=0.30,
        power_w=14_500.0,
        purchase_price_eur=495_859.52,
        price_source="https://store.supermicro.com/nl_es/servers/gpu.html?system_gpu_family=1556&system_gpu_model=1735",
        price_date="2026-08-25",
        notes=(
            "Supermicro Europe starting price for a preconfigured 8U HGX B300 8-GPU system. "
            "Peak compute uses NVIDIA's dense FP4 HGX B300 figure; power uses NVIDIA DGX B300's "
            "14.5 kW published system consumption. Efficiency is a simulator assumption."
        ),
    ),
    "mi355x-eu": PrefillHardware(
        key="mi355x-eu",
        name="8x AMD Instinct MI355X (EU asking-price baseline)",
        compute_flops_s=80.5 * PFLOP,
        memory_bandwidth_bytes_s=64.0 * TB,
        memory_capacity_bytes=2_304 * GB,
        compute_efficiency=0.35,
        memory_efficiency=0.80,
        attention_efficiency=0.30,
        power_w=11_200.0,
        purchase_price_eur=648_587.0,
        price_source="https://servermall.com/sets/file-servers/?PAGEN_3=6&PAGEN_7=2",
        price_date="2026-08-25",
        notes=(
            "ServerMall EU ex-VAT asking price for a Gigabyte G893-ZX1-AAX4 with 8x MI355X. "
            "Treat as a market asking-price point, not an AMD MSRP. Peak compute uses AMD MXFP4. "
            "Power is accelerator TBP only (8 x 1.4 kW), so host/cooling overhead is excluded."
        ),
    ),
}


def _attention_layer_count(model: ModelSpec) -> int:
    if model.key == "kimi-k3":
        return 24
    if model.key == "qwen3.8-27b":
        return 16
    if model.num_experts == 0:
        return model.num_layers
    # Sparse-attention MoEs such as GLM are not accurately represented by dense
    # quadratic attention. Use zero rather than invent a kernel model.
    return 0


def _validate_hardware(hw: PrefillHardware) -> None:
    if hw.compute_flops_s <= 0 or hw.memory_bandwidth_bytes_s <= 0 or hw.memory_capacity_bytes <= 0:
        raise ValueError("hardware compute, bandwidth and capacity must be positive")
    for name, value in (
        ("compute_efficiency", hw.compute_efficiency),
        ("memory_efficiency", hw.memory_efficiency),
        ("attention_efficiency", hw.attention_efficiency),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if hw.power_w < 0:
        raise ValueError("power_w cannot be negative")


def _validate_workload(workload: PrefillWorkload) -> None:
    if workload.prompt_tokens <= 0:
        raise ValueError("prompt_tokens must be positive")
    if workload.concurrent_prompts <= 0:
        raise ValueError("concurrent_prompts must be positive")
    if workload.weight_bits <= 0 or workload.activation_bits <= 0:
        raise ValueError("bit widths must be positive")
    if workload.kv_bytes_per_token < 0:
        raise ValueError("kv_bytes_per_token cannot be negative")
    if workload.weight_reload_factor < 1.0:
        raise ValueError("weight_reload_factor must be >= 1")
    if workload.activation_traffic_factor < 0:
        raise ValueError("activation_traffic_factor cannot be negative")
    if workload.fixed_overhead_ms < 0:
        raise ValueError("fixed_overhead_ms cannot be negative")


def simulate_prefill(
    model: ModelSpec,
    hardware: PrefillHardware,
    workload: PrefillWorkload,
    *,
    overhead_fraction: float = 0.05,
) -> PrefillResult:
    _validate_hardware(hardware)
    _validate_workload(workload)
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")

    n = workload.prompt_tokens
    batch = workload.concurrent_prompts
    total_tokens = n * batch

    weight_bytes = model.storage_bytes(workload.weight_bits, overhead_fraction)
    resident = weight_bytes <= hardware.memory_capacity_bytes

    # For MoE prefill, active_parameters captures token-level routed compute.
    # The prompt is assumed large enough that the resident expert set is touched;
    # external memory traffic therefore starts from one full-model weight pass
    # (plus a configurable reload/tile penalty), rather than active bytes/token.
    linear_flops = 2.0 * model.active_parameters * total_tokens

    attention_layers = _attention_layer_count(model)
    attention_flops = (
        4.0 * attention_layers * batch * (n ** 2) * model.hidden_size
        if attention_layers
        else 0.0
    )
    total_flops = linear_flops + attention_flops

    weight_traffic = weight_bytes * workload.weight_reload_factor * batch
    activation_bytes = model.hidden_bytes(workload.activation_bits)
    activation_traffic = (
        activation_bytes
        * total_tokens
        * model.num_layers
        * workload.activation_traffic_factor
    )
    kv_output = workload.kv_bytes_per_token * total_tokens
    total_memory_traffic = weight_traffic + activation_traffic + kv_output

    linear_compute_time = linear_flops / (
        hardware.compute_flops_s * hardware.compute_efficiency
    )
    attention_compute_time = (
        attention_flops / (hardware.compute_flops_s * hardware.attention_efficiency)
        if attention_flops
        else 0.0
    )
    compute_time = linear_compute_time + attention_compute_time
    memory_time = total_memory_traffic / (
        hardware.memory_bandwidth_bytes_s * hardware.memory_efficiency
    )
    fixed_time = workload.fixed_overhead_ms / 1000.0
    roof_time = max(compute_time, memory_time)
    prefill_time = roof_time + fixed_time

    input_tps = total_tokens / prefill_time
    compute_util = min(1.0, compute_time / roof_time) if roof_time else 0.0
    memory_util = min(1.0, memory_time / roof_time) if roof_time else 0.0
    arithmetic_intensity = total_flops / total_memory_traffic if total_memory_traffic else math.inf
    effective_compute = hardware.compute_flops_s * hardware.compute_efficiency
    effective_bw = hardware.memory_bandwidth_bytes_s * hardware.memory_efficiency
    balance_intensity = effective_compute / effective_bw
    if compute_time > memory_time * 1.05:
        bottleneck = "compute"
    elif memory_time > compute_time * 1.05:
        bottleneck = "memory"
    else:
        bottleneck = "balanced"

    energy_j_per_token = hardware.power_w * prefill_time / total_tokens

    return PrefillResult(
        model_name=model.name,
        hardware_name=hardware.name,
        prompt_tokens=n,
        concurrent_prompts=batch,
        resident=resident,
        weight_bytes=weight_bytes,
        linear_flops=linear_flops,
        attention_flops=attention_flops,
        total_flops=total_flops,
        weight_traffic_bytes=weight_traffic,
        activation_traffic_bytes=activation_traffic,
        kv_output_bytes=kv_output,
        total_memory_traffic_bytes=total_memory_traffic,
        compute_time_s=compute_time,
        memory_time_s=memory_time,
        fixed_time_s=fixed_time,
        prefill_time_s=prefill_time,
        input_tokens_per_s=input_tps,
        compute_utilization_at_roof=compute_util,
        memory_utilization_at_roof=memory_util,
        arithmetic_intensity_flops_per_byte=arithmetic_intensity,
        balance_intensity_flops_per_byte=balance_intensity,
        bottleneck=bottleneck,
        energy_j_per_input_token=energy_j_per_token,
    )


def compare_economics(
    candidate: PrefillResult,
    reference: PrefillResult,
    *,
    reference_price_eur: float,
    candidate_price_eur: float | None = None,
    candidate_power_w: float,
    reference_power_w: float,
    utilization: float = 0.60,
    lifetime_years: float = 3.0,
    electricity_eur_per_kwh: float = 0.15,
) -> EconomicComparison:
    if reference_price_eur <= 0:
        raise ValueError("reference_price_eur must be positive")
    if candidate_price_eur is not None and candidate_price_eur <= 0:
        raise ValueError("candidate_price_eur must be positive when supplied")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    if electricity_eur_per_kwh < 0:
        raise ValueError("electricity_eur_per_kwh cannot be negative")

    ref_tps = reference.input_tokens_per_s
    cand_tps = candidate.input_tokens_per_s
    capex_1x = reference_price_eur * cand_tps / ref_tps
    capex_3x = capex_1x / 3.0
    capex_5x = capex_1x / 5.0

    ref_tps_per_eur = ref_tps / reference_price_eur
    cand_tps_per_eur = cand_tps / candidate_price_eur if candidate_price_eur else None
    relative = cand_tps_per_eur / ref_tps_per_eur if cand_tps_per_eur else None

    life_seconds = SECONDS_PER_YEAR * lifetime_years * utilization
    ref_life_tokens = ref_tps * life_seconds
    cand_life_tokens = cand_tps * life_seconds

    ref_energy_eur_per_token = (
        (reference_power_w / 1000.0)
        * electricity_eur_per_kwh
        / 3600.0
        / ref_tps
    )
    cand_energy_eur_per_token = (
        (candidate_power_w / 1000.0)
        * electricity_eur_per_kwh
        / 3600.0
        / cand_tps
    )
    ref_cost_per_token = reference_price_eur / ref_life_tokens + ref_energy_eur_per_token
    ref_cost_per_million = ref_cost_per_token * 1e6

    cand_cost_per_million = None
    if candidate_price_eur:
        cand_cost_per_token = candidate_price_eur / cand_life_tokens + cand_energy_eur_per_token
        cand_cost_per_million = cand_cost_per_token * 1e6

    break_even_capex = max(
        0.0,
        (ref_cost_per_token - cand_energy_eur_per_token) * cand_life_tokens,
    )

    return EconomicComparison(
        candidate=candidate,
        reference=reference,
        reference_price_eur=reference_price_eur,
        candidate_price_eur=candidate_price_eur,
        capex_ceiling_1x_eur=capex_1x,
        capex_ceiling_3x_eur=capex_3x,
        capex_ceiling_5x_eur=capex_5x,
        candidate_input_tps_per_eur=cand_tps_per_eur,
        reference_input_tps_per_eur=ref_tps_per_eur,
        relative_input_tps_per_eur=relative,
        candidate_cost_per_million_input_tokens_eur=cand_cost_per_million,
        reference_cost_per_million_input_tokens_eur=ref_cost_per_million,
        break_even_candidate_capex_tco_eur=break_even_capex,
    )


def candidate_hardware(
    *,
    compute_pops: float,
    bandwidth_tb_s: float,
    capacity_gb: float,
    compute_efficiency: float = 0.45,
    memory_efficiency: float = 0.75,
    attention_efficiency: float = 0.30,
    power_w: float = 800.0,
    purchase_price_eur: float | None = None,
) -> PrefillHardware:
    return PrefillHardware(
        key="candidate",
        name=f"Candidate {compute_pops:g} POPS / {bandwidth_tb_s:g} TB/s",
        compute_flops_s=compute_pops * PFLOP,
        memory_bandwidth_bytes_s=bandwidth_tb_s * TB,
        memory_capacity_bytes=capacity_gb * GB,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
        attention_efficiency=attention_efficiency,
        power_w=power_w,
        purchase_price_eur=purchase_price_eur,
        notes="User-defined prefill-only research point.",
    )


def _fmt_si(value: float, suffix: str = "") -> str:
    for scale, prefix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:,.2f}{prefix}{suffix}"
    return f"{value:,.2f}{suffix}"


def _fmt_bytes(value: float) -> str:
    if value >= TB:
        return f"{value / TB:,.3f} TB"
    if value >= GB:
        return f"{value / GB:,.2f} GB"
    return f"{value / 1e6:,.2f} MB"


def _fmt_eur(value: float) -> str:
    return f"€{value:,.0f}"


def _parse_positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-prefill",
        description=(
            "Analytical prefill economics simulator: compute-vs-memory roofline plus "
            "CAPEX ceilings against purchasable 8-GPU reference systems."
        ),
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--tokens", type=int, default=32_768)
    parser.add_argument("--batch", type=int, default=1, help="concurrent prompts")
    parser.add_argument("--weight-bits", type=float, default=4.0)
    parser.add_argument("--activation-bits", type=float, default=8.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--weight-reload-factor", type=float, default=1.15)
    parser.add_argument("--activation-traffic-factor", type=float, default=8.0)
    parser.add_argument("--kv-bytes-token", type=float, default=0.0)
    parser.add_argument("--fixed-overhead-ms", type=float, default=0.0)

    parser.add_argument("--candidate-compute-pops", type=_parse_positive_float, default=4.0)
    parser.add_argument("--candidate-bandwidth-tb-s", type=_parse_positive_float, default=2.0)
    parser.add_argument("--candidate-capacity-gb", type=_parse_positive_float, default=2048.0)
    parser.add_argument("--candidate-compute-eff", type=float, default=0.45)
    parser.add_argument("--candidate-memory-eff", type=float, default=0.75)
    parser.add_argument("--candidate-attention-eff", type=float, default=0.30)
    parser.add_argument("--candidate-power-w", type=float, default=800.0)
    parser.add_argument("--candidate-price-eur", type=float)

    parser.add_argument("--reference", choices=tuple(sorted(REFERENCE_HARDWARE)), default="b300-eu")
    parser.add_argument("--utilization", type=float, default=0.60)
    parser.add_argument("--lifetime-years", type=float, default=3.0)
    parser.add_argument("--electricity-eur-kwh", type=float, default=0.15)
    parser.add_argument(
        "--sweep-compute",
        type=float,
        nargs="+",
        help="optional POPS values; prints the CAPEX ceiling curve instead of one candidate",
    )
    return parser


def _workload_from_args(args: argparse.Namespace) -> PrefillWorkload:
    return PrefillWorkload(
        prompt_tokens=args.tokens,
        concurrent_prompts=args.batch,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        kv_bytes_per_token=args.kv_bytes_token,
        weight_reload_factor=args.weight_reload_factor,
        activation_traffic_factor=args.activation_traffic_factor,
        fixed_overhead_ms=args.fixed_overhead_ms,
    )


def _comparison_for(args: argparse.Namespace, compute_pops: float) -> EconomicComparison:
    model = get_model(args.model)
    workload = _workload_from_args(args)
    candidate = candidate_hardware(
        compute_pops=compute_pops,
        bandwidth_tb_s=args.candidate_bandwidth_tb_s,
        capacity_gb=args.candidate_capacity_gb,
        compute_efficiency=args.candidate_compute_eff,
        memory_efficiency=args.candidate_memory_eff,
        attention_efficiency=args.candidate_attention_eff,
        power_w=args.candidate_power_w,
        purchase_price_eur=args.candidate_price_eur,
    )
    reference_hw = REFERENCE_HARDWARE[args.reference]
    candidate_result = simulate_prefill(model, candidate, workload, overhead_fraction=args.overhead)
    reference_result = simulate_prefill(model, reference_hw, workload, overhead_fraction=args.overhead)
    return compare_economics(
        candidate_result,
        reference_result,
        reference_price_eur=reference_hw.purchase_price_eur or 0.0,
        candidate_price_eur=args.candidate_price_eur,
        candidate_power_w=candidate.power_w,
        reference_power_w=reference_hw.power_w,
        utilization=args.utilization,
        lifetime_years=args.lifetime_years,
        electricity_eur_per_kwh=args.electricity_eur_kwh,
    )


def _print_single(args: argparse.Namespace, comparison: EconomicComparison) -> None:
    cand = comparison.candidate
    ref = comparison.reference
    ref_hw = REFERENCE_HARDWARE[args.reference]

    print(f"{cand.model_name} prefill economics | {args.tokens:,} tokens x batch {args.batch}")
    print(
        f"Candidate: {cand.hardware_name} | capacity={args.candidate_capacity_gb:g} GB | "
        f"power={args.candidate_power_w:g} W"
    )
    print(
        f"Reference: {ref_hw.name} | purchase={_fmt_eur(comparison.reference_price_eur)} "
        f"| price date={ref_hw.price_date}"
    )
    print()
    print("ROOFLINE")
    print(
        f"{'':12} {'resident':>9} {'prefill':>10} {'input tok/s':>13} "
        f"{'compute':>10} {'memory':>10} {'bottleneck':>11}"
    )
    for label, result in (("candidate", cand), ("reference", ref)):
        print(
            f"{label:12} {str(result.resident):>9} {result.prefill_time_s:>9.3f}s "
            f"{result.input_tokens_per_s:>13,.0f} {result.compute_time_s:>9.3f}s "
            f"{result.memory_time_s:>9.3f}s {result.bottleneck:>11}"
        )
    print()
    print(
        f"work: {_fmt_si(cand.total_flops, 'FLOP')} total "
        f"({_fmt_si(cand.attention_flops, 'FLOP')} explicit attention)"
    )
    print(
        f"candidate memory traffic: {_fmt_bytes(cand.total_memory_traffic_bytes)} "
        f"(weights {_fmt_bytes(cand.weight_traffic_bytes)}, "
        f"activations {_fmt_bytes(cand.activation_traffic_bytes)}, "
        f"KV {_fmt_bytes(cand.kv_output_bytes)})"
    )
    print(
        f"candidate arithmetic intensity: {cand.arithmetic_intensity_flops_per_byte:,.1f} FLOP/B "
        f"| machine balance: {cand.balance_intensity_flops_per_byte:,.1f} FLOP/B"
    )
    print()
    print("CAPEX CEILING VS REFERENCE (input tok/s per euro)")
    print(f"  parity (1x): {_fmt_eur(comparison.capex_ceiling_1x_eur)}")
    print(f"  3x better:   {_fmt_eur(comparison.capex_ceiling_3x_eur)}")
    print(f"  5x better:   {_fmt_eur(comparison.capex_ceiling_5x_eur)}")
    print(
        f"  TCO break-even at {args.utilization:.0%} utilization / "
        f"{args.lifetime_years:g}y / €{args.electricity_eur_kwh:g}/kWh: "
        f"{_fmt_eur(comparison.break_even_candidate_capex_tco_eur)}"
    )
    if args.candidate_price_eur:
        print(
            f"  candidate at {_fmt_eur(args.candidate_price_eur)}: "
            f"{comparison.relative_input_tps_per_eur:.2f}x reference input-tok/s/€; "
            f"€{comparison.candidate_cost_per_million_input_tokens_eur:.4f}/M input tok"
        )
    print(
        f"  reference amortized+energy: "
        f"€{comparison.reference_cost_per_million_input_tokens_eur:.4f}/M input tok"
    )
    print()
    print("PRICE SOURCE")
    print(f"  {ref_hw.price_source}")
    print(
        "Caveat: this is an architectural roofline, not a kernel benchmark. "
        "Efficiency factors, attention model, weight reloads and KV bytes are explicit assumptions."
    )


def _print_sweep(args: argparse.Namespace) -> None:
    values = tuple(sorted(set(float(v) for v in args.sweep_compute or ())))
    if not values or values[0] <= 0:
        raise ValueError("--sweep-compute values must be positive")
    print(
        f"{get_model(args.model).name} | {args.tokens:,} token prefill | "
        f"{args.candidate_bandwidth_tb_s:g} TB/s candidate memory | "
        f"reference={args.reference}"
    )
    print(
        f"{'POPS':>7} {'cand tok/s':>12} {'bneck':>9} {'1x capex':>12} "
        f"{'3x capex':>12} {'5x capex':>12}"
    )
    for pops in values:
        comparison = _comparison_for(args, pops)
        cand = comparison.candidate
        print(
            f"{pops:>7.2f} {cand.input_tokens_per_s:>12,.0f} {cand.bottleneck:>9} "
            f"{_fmt_eur(comparison.capex_ceiling_1x_eur):>12} "
            f"{_fmt_eur(comparison.capex_ceiling_3x_eur):>12} "
            f"{_fmt_eur(comparison.capex_ceiling_5x_eur):>12}"
        )


def main() -> int:
    args = build_parser().parse_args()
    if args.sweep_compute:
        _print_sweep(args)
    else:
        _print_single(args, _comparison_for(args, args.candidate_compute_pops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
