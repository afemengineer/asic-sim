from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from .architectures import _bytes
from .m3 import KIMI_K3_STATE, _attention_context_flops, _kda_recurrent_flops
from .m4_heterogeneous import (
    ExpertBankCandidate,
    HeterogeneousPhysicalSpec,
    _cache_bytes_per_sequence,
    build_capacity_report,
    simulate_candidate,
)
from .models import ModelSpec, get_model
from .placement import decompose_model


@dataclass(frozen=True, slots=True)
class CommercialNode:
    key: str
    name: str
    gpu_count: int
    memory_gb: float
    memory_bandwidth_tb_s: float
    low_precision_pflops: float
    purchase_usd: float
    cloud_gpu_hour_usd: float
    node_power_kw: float
    decode_single_tps: float
    decode_peak_tps: float
    prefill_tps: float
    prefill_label: str

    @property
    def cloud_node_hour_usd(self) -> float:
        return self.gpu_count * self.cloud_gpu_hour_usd


# 2026-08 screening anchors. Hardware specs are vendor figures; prices/performance
# are public market/benchmark observations and should be refreshed over time.
COMMERCIAL_NODES = {
    "b300": CommercialNode(
        key="b300",
        name="8x NVIDIA B300",
        gpu_count=8,
        memory_gb=2304.0,
        memory_bandwidth_tb_s=64.0,
        low_precision_pflops=144.0,
        purchase_usd=580_475.0,
        cloud_gpu_hour_usd=7.40,
        node_power_kw=15.559,
        decode_single_tps=172.0,
        decode_peak_tps=1568.0,
        prefill_tps=172_000.0 / 23.0,
        prefill_label="~172k cold prefill / 23 s",
    ),
    "mi355x": CommercialNode(
        key="mi355x",
        name="8x AMD MI355X",
        gpu_count=8,
        memory_gb=2304.0,
        memory_bandwidth_tb_s=64.0,
        low_precision_pflops=80.5,
        purchase_usd=323_879.0,
        cloud_gpu_hour_usd=2.95,
        # 8x 1.4 kW accelerator TBP + a screening host/network allowance.
        node_power_kw=13.0,
        decode_single_tps=118.0,
        decode_peak_tps=952.0,
        prefill_tps=13_000.0,
        prefill_label="~13k tok/s tuned AITER MLA steady-state",
    ),
}


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    input_tokens: int
    output_tokens: int


DEFAULT_WORKLOADS = (
    Workload("chat", 1_024, 400),
    Workload("agent", 8_192, 1_024),
    Workload("long32k", 32_768, 1_024),
    Workload("long128k", 131_072, 1_024),
    Workload("wafer172k", 172_000, 400),
)


@dataclass(frozen=True, slots=True)
class Economics:
    amort_years: float
    utilization: float
    electricity_usd_kwh: float
    pue: float

    @property
    def active_lifetime_hours(self) -> float:
        return self.amort_years * 365.0 * 24.0 * self.utilization

    def owned_hourly_cost(self, purchase_usd: float, power_kw: float) -> float:
        capex = purchase_usd / self.active_lifetime_hours
        energy = power_kw * self.pue * self.electricity_usd_kwh
        return capex + energy


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    request_rate_s: float
    output_rate_tps: float
    ttft_s: float
    e2e_single_s: float
    cloud_usd_per_m_output: float
    owned_usd_per_m_output: float


@dataclass(frozen=True, slots=True)
class CustomMetrics:
    spine_modules: int
    cache_handoff_bytes: float
    prefill_tps: float
    decode_single_tps: float
    decode_peak_tps: float
    request_rate_s: float
    ttft_s: float
    e2e_single_s: float


@dataclass(frozen=True, slots=True)
class HybridBudget:
    prefill_node: str
    request_rate_s: float
    ttft_s: float
    e2e_single_s: float
    prefill_node_equivalents: float
    max_decoder_cloud_hour_usd: float
    max_decoder_capex_usd: float


def _fmt_money(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    if value < 0:
        return "impossible"
    if value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value/1_000:.0f}k"
    return f"${value:.2f}"


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds*1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds*1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _usd_per_m(hourly_usd: float, token_rate_s: float) -> float:
    if hourly_usd < 0 or token_rate_s <= 0:
        return math.inf
    return hourly_usd / (token_rate_s * 3600.0) * 1e6


def commercial_metrics(node: CommercialNode, workload: Workload, economics: Economics) -> RequestMetrics:
    prefill_s = workload.input_tokens / node.prefill_tps
    decode_single_s = workload.output_tokens / node.decode_single_tps
    decode_peak_s = workload.output_tokens / node.decode_peak_tps
    # Screening approximation: one node alternates prefill and decode work.
    request_rate = 1.0 / (prefill_s + decode_peak_s)
    output_rate = request_rate * workload.output_tokens
    ttft = prefill_s + 1.0 / node.decode_single_tps
    e2e = prefill_s + decode_single_s
    owned_hour = economics.owned_hourly_cost(node.purchase_usd, node.node_power_kw)
    return RequestMetrics(
        request_rate_s=request_rate,
        output_rate_tps=output_rate,
        ttft_s=ttft,
        e2e_single_s=e2e,
        cloud_usd_per_m_output=_usd_per_m(node.cloud_node_hour_usd, output_rate),
        owned_usd_per_m_output=_usd_per_m(owned_hour, output_rate),
    )


def _custom_prefill_tps(
    model: ModelSpec,
    physical: HeterogeneousPhysicalSpec,
    candidate: ExpertBankCandidate,
    spine_modules: int,
    prompt_tokens: int,
    expert_efficiency: float,
    spine_efficiency: float,
) -> float:
    d = decompose_model(model)
    routed_flops_per_token = 2.0 * d.active_routed_parameters
    always_flops_per_token = 2.0 * d.active_always_on_parameters

    # Average causal context during prefill is about L/2. Add Kimi's recurrent
    # KDA work and absorbed-MLA attention arithmetic to the always-on spine.
    mean_context = max(0, prompt_tokens // 2)
    attention_flops = (
        len(KIMI_K3_STATE.full_attention_layers)
        * _attention_context_flops(KIMI_K3_STATE, mean_context, "latent")
    )
    kda_flops = KIMI_K3_STATE.kda_layers * _kda_recurrent_flops(KIMI_K3_STATE)
    spine_flops_per_token = always_flops_per_token + attention_flops + kda_flops

    expert_compute = candidate.bank_count * candidate.q4_tops * 1e12 * expert_efficiency
    spine_compute = spine_modules * physical.fast_fp16_tops * 1e12 * spine_efficiency
    expert_roof = expert_compute / routed_flops_per_token if routed_flops_per_token else math.inf
    spine_roof = spine_compute / spine_flops_per_token if spine_flops_per_token else math.inf
    return min(expert_roof, spine_roof)


def custom_metrics(
    model: ModelSpec,
    workload: Workload,
    physical: HeterogeneousPhysicalSpec,
    candidate: ExpertBankCandidate,
    *,
    concurrency: int,
    profile: str,
    seed: int,
    expert_efficiency: float,
    spine_efficiency: float,
) -> CustomMetrics:
    # Size state at the end of the request, simulate decode around the midpoint.
    final_context = workload.input_tokens + workload.output_tokens
    mid_context = workload.input_tokens + workload.output_tokens // 2
    capacity = build_capacity_report(
        model,
        physical,
        context=final_context,
        max_concurrency=concurrency,
    )
    single = simulate_candidate(
        model,
        physical,
        capacity,
        candidate,
        context=mid_context,
        concurrency=1,
        profile=profile,
        seed=seed,
    )
    peak = simulate_candidate(
        model,
        physical,
        capacity,
        candidate,
        context=mid_context,
        concurrency=concurrency,
        profile=profile,
        seed=seed,
    )
    if not single.resident or not peak.resident:
        return CustomMetrics(capacity.spine_modules, 0.0, 0.0, 0.0, 0.0, 0.0, math.inf, math.inf)

    prefill_tps = _custom_prefill_tps(
        model,
        physical,
        candidate,
        capacity.spine_modules,
        workload.input_tokens,
        expert_efficiency,
        spine_efficiency,
    )
    prefill_s = workload.input_tokens / prefill_tps
    peak_decode_s = workload.output_tokens / peak.throughput_tps
    request_rate = 1.0 / (prefill_s + peak_decode_s)
    ttft = prefill_s + 1.0 / single.throughput_tps
    e2e = prefill_s + workload.output_tokens / single.throughput_tps
    handoff = _cache_bytes_per_sequence(model, workload.input_tokens, physical)
    return CustomMetrics(
        spine_modules=capacity.spine_modules,
        cache_handoff_bytes=handoff,
        prefill_tps=prefill_tps,
        decode_single_tps=single.throughput_tps,
        decode_peak_tps=peak.throughput_tps,
        request_rate_s=request_rate,
        ttft_s=ttft,
        e2e_single_s=e2e,
    )


def _best_baseline_costs(
    workload: Workload,
    economics: Economics,
) -> tuple[float, float]:
    rows = [commercial_metrics(node, workload, economics) for node in COMMERCIAL_NODES.values()]
    return (
        min(row.cloud_usd_per_m_output for row in rows),
        min(row.owned_usd_per_m_output for row in rows),
    )


def hybrid_budget(
    node: CommercialNode,
    workload: Workload,
    custom: CustomMetrics,
    economics: Economics,
    *,
    handoff_gb_s: float,
    target_advantage: float,
    assumed_custom_power_kw: float,
) -> HybridBudget:
    if custom.decode_peak_tps <= 0 or custom.cache_handoff_bytes <= 0:
        return HybridBudget(node.key, 0.0, math.inf, math.inf, 0.0, -1.0, -1.0)

    decode_req_s = custom.decode_peak_tps / workload.output_tokens
    handoff_req_s = handoff_gb_s * 1e9 / custom.cache_handoff_bytes
    # Prefill capacity can scale fractionally at fleet level; its cost is charged
    # in node-equivalents rather than pretending a dedicated full node is free.
    request_rate = min(decode_req_s, handoff_req_s)
    prefill_req_s_per_node = node.prefill_tps / workload.input_tokens
    prefill_nodes = request_rate / prefill_req_s_per_node
    output_rate = request_rate * workload.output_tokens

    handoff_s = custom.cache_handoff_bytes / (handoff_gb_s * 1e9)
    prefill_s = workload.input_tokens / node.prefill_tps
    ttft = prefill_s + handoff_s + 1.0 / custom.decode_single_tps
    e2e = prefill_s + handoff_s + workload.output_tokens / custom.decode_single_tps

    best_cloud_m, best_owned_m = _best_baseline_costs(workload, economics)
    allowed_total_cloud_hour = (best_cloud_m / 1e6) * output_rate * 3600.0 / target_advantage
    prefill_cloud_hour = prefill_nodes * node.cloud_node_hour_usd
    max_decoder_cloud = allowed_total_cloud_hour - prefill_cloud_hour

    allowed_total_owned_hour = (best_owned_m / 1e6) * output_rate * 3600.0 / target_advantage
    prefill_owned_hour = prefill_nodes * economics.owned_hourly_cost(node.purchase_usd, node.node_power_kw)
    decoder_owned_hour_budget = allowed_total_owned_hour - prefill_owned_hour
    decoder_energy_hour = assumed_custom_power_kw * economics.pue * economics.electricity_usd_kwh
    decoder_capex_hour_budget = decoder_owned_hour_budget - decoder_energy_hour
    max_capex = decoder_capex_hour_budget * economics.active_lifetime_hours

    return HybridBudget(
        prefill_node=node.key,
        request_rate_s=request_rate,
        ttft_s=ttft,
        e2e_single_s=e2e,
        prefill_node_equivalents=prefill_nodes,
        max_decoder_cloud_hour_usd=max_decoder_cloud,
        max_decoder_capex_usd=max_capex,
    )


def _parse_workload(value: str) -> Workload:
    try:
        name, inp, out = value.split(":", 2)
        workload = Workload(name, int(inp), int(out))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("workload must be NAME:INPUT_TOKENS:OUTPUT_TOKENS") from exc
    if not workload.name or workload.input_tokens <= 0 or workload.output_tokens <= 0:
        raise argparse.ArgumentTypeError("workload name and token counts must be positive")
    return workload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m5",
        description="Commercial Kimi K3 gate: B300/MI355X vs custom and disaggregated prefill/decode.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--workload", action="append", type=_parse_workload, dest="workloads")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--handoff-gb-s", type=float, default=100.0)
    parser.add_argument("--target-advantage", type=float, default=2.0)
    parser.add_argument("--expert-efficiency", type=float, default=0.60)
    parser.add_argument("--spine-efficiency", type=float, default=0.60)
    parser.add_argument("--assumed-custom-power-kw", type=float, default=8.0)
    parser.add_argument("--amort-years", type=float, default=3.0)
    parser.add_argument("--utilization", type=float, default=0.70)
    parser.add_argument("--electricity-usd-kwh", type=float, default=0.10)
    parser.add_argument("--pue", type=float, default=1.20)
    parser.add_argument("--bank-count", type=int, default=64)
    parser.add_argument("--bank-capacity-gb", type=float, default=24.0)
    parser.add_argument("--bank-bandwidth-gb-s", type=float, default=1344.0)
    parser.add_argument("--bank-q4-tops", type=float, default=16.0)
    parser.add_argument("--bank-link-gb-s", type=float, default=64.0)
    parser.add_argument("--bank-link-latency-ns", type=float, default=100.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    if model.key != "kimi-k3":
        raise ValueError("M5 currently supports kimi-k3 only")
    if args.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if args.handoff_gb_s <= 0 or args.target_advantage <= 1.0:
        raise ValueError("handoff bandwidth must be positive and target advantage must exceed 1x")
    if not (0 < args.expert_efficiency <= 1 and 0 < args.spine_efficiency <= 1):
        raise ValueError("efficiencies must be in (0, 1]")
    if min(args.amort_years, args.utilization, args.electricity_usd_kwh, args.pue) <= 0:
        raise ValueError("economic inputs must be positive")

    workloads = tuple(args.workloads) if args.workloads else DEFAULT_WORKLOADS
    economics = Economics(args.amort_years, args.utilization, args.electricity_usd_kwh, args.pue)
    physical = HeterogeneousPhysicalSpec()
    candidate = ExpertBankCandidate(
        args.bank_count,
        args.bank_capacity_gb,
        args.bank_bandwidth_gb_s,
        args.bank_q4_tops,
        args.bank_link_gb_s,
        args.bank_link_latency_ns,
    )

    print(f"{model.name} — M5 commercial gate")
    print(f"  custom candidate:       {candidate.label}, {candidate.q4_tops:g} Q4 TOPS/bank")
    print(f"  handoff fabric:         {args.handoff_gb_s:g} GB/s usable")
    print(f"  custom prefill eff.:    expert {args.expert_efficiency:.0%}, spine {args.spine_efficiency:.0%}")
    print(f"  economic gate:          >= {args.target_advantage:g}x cheaper output tokens")
    print(
        f"  owned-TCO assumptions:  {args.amort_years:g} y, {args.utilization:.0%} utilization, "
        f"${args.electricity_usd_kwh:g}/kWh, PUE {args.pue:g}"
    )
    print(f"  custom power sensitivity: {args.assumed_custom_power_kw:g} kW (not a BOM claim)")

    print("\nCOMMERCIAL BASELINES (2026-08 anchors)")
    print("NODE              MEM      HBM BW   LP PEAK   BUY PRICE   CLOUD/NODE-H  POWER   DEC1   DECPEAK  PREFILL")
    for node in COMMERCIAL_NODES.values():
        print(
            f"{node.name:<17} {node.memory_gb/1024:>4.2f} TB  {node.memory_bandwidth_tb_s:>5.0f} TB/s  "
            f"{node.low_precision_pflops:>6.1f}P  {_fmt_money(node.purchase_usd):>9}  "
            f"${node.cloud_node_hour_usd:>8.2f}      {node.node_power_kw:>4.1f}kW  "
            f"{node.decode_single_tps:>5.0f}  {node.decode_peak_tps:>7.0f}  {node.prefill_tps:>7.0f}"
        )
    print("  B300 prefill anchor is cold 172k/23s; MI355X anchor is tuned steady-state. Do not read their prefill ratio as an apples-to-apples benchmark.")

    custom_rows: dict[str, CustomMetrics] = {}
    commercial_rows: dict[tuple[str, str], RequestMetrics] = {}

    print("\nEND-TO-END PERFORMANCE SCREEN")
    print("WORKLOAD      I/O TOKENS      SPINE  CACHE-XFER  CUSTOM PREF  CUSTOM DEC1  CUSTOM PEAK  CUSTOM TTFT")
    for workload in workloads:
        custom = custom_metrics(
            model,
            workload,
            physical,
            candidate,
            concurrency=args.concurrency,
            profile=args.profile,
            seed=args.seed,
            expert_efficiency=args.expert_efficiency,
            spine_efficiency=args.spine_efficiency,
        )
        custom_rows[workload.name] = custom
        print(
            f"{workload.name:<12} {workload.input_tokens:>7,}/{workload.output_tokens:<5,}  "
            f"{custom.spine_modules:>5}  {custom.cache_handoff_bytes/1e9:>8.3f} GB  "
            f"{custom.prefill_tps:>9,.0f}  {custom.decode_single_tps:>10,.1f}  "
            f"{custom.decode_peak_tps:>10,.1f}  {_fmt_time(custom.ttft_s):>11}"
        )

    print("\nGPU-ONLY COMMERCIAL ECONOMICS")
    print("WORKLOAD      NODE              TTFT       E2E SINGLE   OUT TPS   CLOUD $/M OUT   OWNED $/M OUT")
    for workload in workloads:
        for node in COMMERCIAL_NODES.values():
            row = commercial_metrics(node, workload, economics)
            commercial_rows[(workload.name, node.key)] = row
            print(
                f"{workload.name:<12} {node.name:<17} {_fmt_time(row.ttft_s):>10}  {_fmt_time(row.e2e_single_s):>11}  "
                f"{row.output_rate_tps:>7.0f}      ${row.cloud_usd_per_m_output:>8.2f}       ${row.owned_usd_per_m_output:>8.2f}"
            )

    print("\nCUSTOM-ONLY 2x COST GATE")
    print("  This does not guess custom silicon price. It prints the maximum whole-system budget allowed at the simulated throughput.")
    print("WORKLOAD      REQ/S   MAX CLOUD-EQ $/H   MAX CAPEX @ POWER ASSUMPTION")
    for workload in workloads:
        custom = custom_rows[workload.name]
        output_rate = custom.request_rate_s * workload.output_tokens
        best_cloud_m, best_owned_m = _best_baseline_costs(workload, economics)
        allowed_cloud_hour = (best_cloud_m / 1e6) * output_rate * 3600.0 / args.target_advantage
        allowed_owned_hour = (best_owned_m / 1e6) * output_rate * 3600.0 / args.target_advantage
        energy_hour = args.assumed_custom_power_kw * economics.pue * economics.electricity_usd_kwh
        max_capex = (allowed_owned_hour - energy_hour) * economics.active_lifetime_hours
        print(
            f"{workload.name:<12} {custom.request_rate_s:>6.3f}       {_fmt_money(allowed_cloud_hour):>10}/h       {_fmt_money(max_capex):>12}"
        )

    print("\nDISAGGREGATED GPU-PREFILL -> CUSTOM-DECODE 2x COST GATE")
    print("WORKLOAD      PREFILL NODE       REQ/S  PREFILL NODE-EQ  TTFT       MAX DECODER $/H   MAX DECODER CAPEX")
    for workload in workloads:
        custom = custom_rows[workload.name]
        for node in COMMERCIAL_NODES.values():
            budget = hybrid_budget(
                node,
                workload,
                custom,
                economics,
                handoff_gb_s=args.handoff_gb_s,
                target_advantage=args.target_advantage,
                assumed_custom_power_kw=args.assumed_custom_power_kw,
            )
            print(
                f"{workload.name:<12} {node.name:<17} {budget.request_rate_s:>6.3f}      "
                f"{budget.prefill_node_equivalents:>7.3f}      {_fmt_time(budget.ttft_s):>9}  "
                f"{_fmt_money(budget.max_decoder_cloud_hour_usd):>11}/h    {_fmt_money(budget.max_decoder_capex_usd):>12}"
            )

    print("\nKILL CRITERIA")
    print(f"  - Target is not parity: custom must be >= {args.target_advantage:g}x cheaper per output token than the cheaper B300/MI355X baseline.")
    print("  - If MAX CAPEX is impossible or implausibly low after realistic power/NRE/support, the architecture fails the commercial gate.")
    print("  - Hybrid TTFT includes transfer of the full latent MLA + fixed KDA state; throughput is capped by decode and handoff bandwidth.")
    print("  - Commercial decode numbers are measured 1k-input/400-output anchors and are held constant vs context here; long-context GPU decode needs a trace-driven follow-up.")
    print("  - Custom prefill is an analytical compute roof with explicit efficiency factors, not a measured kernel benchmark.")
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
