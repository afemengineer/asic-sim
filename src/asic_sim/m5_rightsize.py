from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from .architectures import _bytes
from .m4_heterogeneous import (
    ExpertBankCandidate,
    HeterogeneousPhysicalSpec,
    _cache_bytes_per_sequence,
    build_capacity_report,
    simulate_candidate,
)
from .m5_commercial import (
    COMMERCIAL_NODES,
    Economics,
    Workload,
    _best_baseline_costs,
    _fmt_money,
)
from .models import ModelSpec, get_model
from .placement import decompose_model


# Micron 2026 production anchor used for the physical grouping sweep:
# 24-Gb x32 GDDR7 at 28 GT/s.  Each device therefore contributes 3 GB and
# 112 GB/s of peak interface bandwidth.  Bandwidth efficiency is kept as an
# explicit sensitivity rather than treating peak signaling rate as usable BW.
@dataclass(frozen=True, slots=True)
class GDDR7Part:
    density_gbit: float = 24.0
    bus_width_bits: int = 32
    max_rate_gt_s: float = 28.0

    @property
    def capacity_gb(self) -> float:
        return self.density_gbit / 8.0

    def bandwidth_gb_s(self, rate_gt_s: float | None = None) -> float:
        rate = self.max_rate_gt_s if rate_gt_s is None else rate_gt_s
        return rate * self.bus_width_bits / 8.0


@dataclass(frozen=True, slots=True)
class BankGeometry:
    placements: int
    capacity_gb: float
    peak_bandwidth_gb_s: float
    bank_count: int
    total_devices: int


@dataclass(frozen=True, slots=True)
class RightSizeResult:
    placements: int
    bank_count: int
    bank_capacity_gb: float
    total_devices: int
    data_rate_gt_s: float
    effective_bank_bandwidth_gb_s: float
    effective_aggregate_bandwidth_tb_s: float
    q4_tops_per_bank: float
    resident: bool
    worst_profile: str
    worst_decode_peak_tps: float
    worst_request_rate_s: float
    feed_ratio: float
    single_decode_tps: float
    worst_collision_fraction: float


@dataclass(frozen=True, slots=True)
class DedicatedBudget:
    request_rate_s: float
    output_rate_tps: float
    prefill_utilization: float
    decoder_utilization: float
    max_decoder_cloud_hour_usd: float
    max_decoder_capex_by_power_kw: tuple[tuple[float, float], ...]


def _minimum_resident_banks(
    model: ModelSpec,
    physical: HeterogeneousPhysicalSpec,
    bank_capacity_gb: float,
) -> int:
    d = decompose_model(model)
    expert_params = d.expert_shard_parameters or 0.0
    expert_bytes = _bytes(expert_params, physical.routed_bits, physical.overhead_fraction)
    total_bytes = _bytes(d.routed_pool_parameters, physical.routed_bits, physical.overhead_fraction)
    shards = model.moe_layers * (model.num_experts or 0)
    capacity = bank_capacity_gb * 1e9
    if shards <= 0 or expert_bytes <= 0:
        return 1

    count = max(1, math.ceil(total_bytes / capacity))
    while True:
        max_shards = math.ceil(shards / count)
        if max_shards * expert_bytes <= capacity and total_bytes <= count * capacity:
            return count
        count += 1
        if count > shards:
            raise RuntimeError("could not place routed experts in bank geometry")


def bank_geometry(
    model: ModelSpec,
    physical: HeterogeneousPhysicalSpec,
    part: GDDR7Part,
    placements: int,
    *,
    rate_gt_s: float | None = None,
) -> BankGeometry:
    if placements <= 0:
        raise ValueError("placements must be positive")
    rate = part.max_rate_gt_s if rate_gt_s is None else rate_gt_s
    if rate <= 0 or rate > part.max_rate_gt_s:
        raise ValueError("data rate must be in (0, part.max_rate_gt_s]")
    capacity = placements * part.capacity_gb
    bandwidth = placements * part.bandwidth_gb_s(rate)
    banks = _minimum_resident_banks(model, physical, capacity)
    return BankGeometry(
        placements=placements,
        capacity_gb=capacity,
        peak_bandwidth_gb_s=bandwidth,
        bank_count=banks,
        total_devices=banks * placements,
    )


def prefill_supply_request_rate(node_key: str, workload: Workload) -> float:
    node = COMMERCIAL_NODES[node_key]
    return node.prefill_tps / workload.input_tokens


def evaluate_geometry(
    model: ModelSpec,
    workload: Workload,
    physical: HeterogeneousPhysicalSpec,
    part: GDDR7Part,
    *,
    placements: int,
    rate_gt_s: float,
    bandwidth_efficiency: float,
    q4_tops_per_bank: float,
    concurrency: int,
    spine_modules: int,
    profiles: tuple[str, ...],
    target_request_rate_s: float,
    seed: int,
    bank_link_gb_s: float,
    bank_link_latency_ns: float,
) -> RightSizeResult:
    if not (0 < bandwidth_efficiency <= 1.0):
        raise ValueError("bandwidth efficiency must be in (0, 1]")
    geometry = bank_geometry(model, physical, part, placements, rate_gt_s=rate_gt_s)
    effective_bank_bw = geometry.peak_bandwidth_gb_s * bandwidth_efficiency
    candidate = ExpertBankCandidate(
        geometry.bank_count,
        geometry.capacity_gb,
        effective_bank_bw,
        q4_tops_per_bank,
        bank_link_gb_s,
        bank_link_latency_ns,
    )
    final_context = workload.input_tokens + workload.output_tokens
    mid_context = workload.input_tokens + workload.output_tokens // 2
    try:
        capacity = build_capacity_report(
            model,
            physical,
            context=final_context,
            max_concurrency=concurrency,
            forced_spine_modules=spine_modules,
        )
    except ValueError:
        return RightSizeResult(
            placements,
            geometry.bank_count,
            geometry.capacity_gb,
            geometry.total_devices,
            rate_gt_s,
            effective_bank_bw,
            geometry.bank_count * effective_bank_bw / 1e3,
            q4_tops_per_bank,
            False,
            "spine-nofit",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    reports = {}
    for profile in profiles:
        reports[profile] = simulate_candidate(
            model,
            physical,
            capacity,
            candidate,
            context=mid_context,
            concurrency=concurrency,
            profile=profile,
            seed=seed,
        )
    resident = all(report.resident for report in reports.values())
    if not resident:
        return RightSizeResult(
            placements,
            geometry.bank_count,
            geometry.capacity_gb,
            geometry.total_devices,
            rate_gt_s,
            effective_bank_bw,
            geometry.bank_count * effective_bank_bw / 1e3,
            q4_tops_per_bank,
            False,
            "bank-nofit",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    worst_profile, worst = min(reports.items(), key=lambda item: item[1].throughput_tps)
    single = simulate_candidate(
        model,
        physical,
        capacity,
        candidate,
        context=mid_context,
        concurrency=1,
        profile=worst_profile,
        seed=seed,
    )
    req_s = worst.throughput_tps / workload.output_tokens
    return RightSizeResult(
        placements=placements,
        bank_count=geometry.bank_count,
        bank_capacity_gb=geometry.capacity_gb,
        total_devices=geometry.total_devices,
        data_rate_gt_s=rate_gt_s,
        effective_bank_bandwidth_gb_s=effective_bank_bw,
        effective_aggregate_bandwidth_tb_s=geometry.bank_count * effective_bank_bw / 1e3,
        q4_tops_per_bank=q4_tops_per_bank,
        resident=True,
        worst_profile=worst_profile,
        worst_decode_peak_tps=worst.throughput_tps,
        worst_request_rate_s=req_s,
        feed_ratio=req_s / target_request_rate_s if target_request_rate_s else 0.0,
        single_decode_tps=single.throughput_tps,
        worst_collision_fraction=worst.collision_fraction,
    )


def search_minimum_rate(
    model: ModelSpec,
    workload: Workload,
    physical: HeterogeneousPhysicalSpec,
    part: GDDR7Part,
    *,
    placements: int,
    rates_gt_s: tuple[float, ...],
    bandwidth_efficiency: float,
    q4_tops_per_bank: float,
    concurrency: int,
    spine_modules: int,
    profiles: tuple[str, ...],
    target_request_rate_s: float,
    headroom_fraction: float,
    seed: int,
    bank_link_gb_s: float,
    bank_link_latency_ns: float,
) -> RightSizeResult:
    required = target_request_rate_s * (1.0 + headroom_fraction)
    last: RightSizeResult | None = None
    for rate in sorted(set(rates_gt_s)):
        if rate <= 0 or rate > part.max_rate_gt_s:
            continue
        last = evaluate_geometry(
            model,
            workload,
            physical,
            part,
            placements=placements,
            rate_gt_s=rate,
            bandwidth_efficiency=bandwidth_efficiency,
            q4_tops_per_bank=q4_tops_per_bank,
            concurrency=concurrency,
            spine_modules=spine_modules,
            profiles=profiles,
            target_request_rate_s=target_request_rate_s,
            seed=seed,
            bank_link_gb_s=bank_link_gb_s,
            bank_link_latency_ns=bank_link_latency_ns,
        )
        if last.resident and last.worst_request_rate_s >= required:
            return last
    if last is None:
        raise ValueError("rate sweep contains no valid GDDR7 rate")
    return last


def dedicated_one_node_budget(
    node_key: str,
    workload: Workload,
    decoder: RightSizeResult,
    model: ModelSpec,
    physical: HeterogeneousPhysicalSpec,
    economics: Economics,
    *,
    handoff_gb_s: float,
    target_advantage: float,
    power_sensitivities_kw: tuple[float, ...],
) -> DedicatedBudget:
    node = COMMERCIAL_NODES[node_key]
    prefill_req_s = node.prefill_tps / workload.input_tokens
    decoder_req_s = decoder.worst_request_rate_s
    state_bytes = _cache_bytes_per_sequence(model, workload.input_tokens, physical)
    handoff_req_s = handoff_gb_s * 1e9 / state_bytes
    request_rate = min(prefill_req_s, decoder_req_s, handoff_req_s)
    output_rate = request_rate * workload.output_tokens

    best_cloud_m, best_owned_m = _best_baseline_costs(workload, economics)
    allowed_total_cloud_hour = best_cloud_m / 1e6 * output_rate * 3600.0 / target_advantage
    max_decoder_cloud = allowed_total_cloud_hour - node.cloud_node_hour_usd

    allowed_total_owned_hour = best_owned_m / 1e6 * output_rate * 3600.0 / target_advantage
    prefill_owned_hour = economics.owned_hourly_cost(node.purchase_usd, node.node_power_kw)
    capex_rows = []
    for power_kw in power_sensitivities_kw:
        energy_hour = power_kw * economics.pue * economics.electricity_usd_kwh
        capex_hour = allowed_total_owned_hour - prefill_owned_hour - energy_hour
        capex_rows.append((power_kw, capex_hour * economics.active_lifetime_hours))

    return DedicatedBudget(
        request_rate_s=request_rate,
        output_rate_tps=output_rate,
        prefill_utilization=request_rate / prefill_req_s if prefill_req_s else 0.0,
        decoder_utilization=request_rate / decoder_req_s if decoder_req_s else 0.0,
        max_decoder_cloud_hour_usd=max_decoder_cloud,
        max_decoder_capex_by_power_kw=tuple(capex_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m5-rightsize",
        description="Right-size a Kimi K3 decode appliance to exactly one commercial GPU prefill node.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--prefill-node", choices=tuple(COMMERCIAL_NODES), default="mi355x")
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--spine-modules", type=int, default=5)
    parser.add_argument("--placements", type=int, nargs="+", default=[8, 10, 12, 16])
    parser.add_argument("--rates", type=float, nargs="+", default=[16, 18, 20, 22, 24, 26, 28])
    parser.add_argument("--bandwidth-efficiency", type=float, default=0.80)
    parser.add_argument("--q4-tops-per-bank", type=float, default=8.0)
    parser.add_argument("--headroom", type=float, default=0.05)
    parser.add_argument("--profiles", nargs="+", choices=("balanced", "hot", "zipf"), default=["balanced", "hot", "zipf"])
    parser.add_argument("--handoff-gb-s", type=float, default=100.0)
    parser.add_argument("--bank-link-gb-s", type=float, default=64.0)
    parser.add_argument("--bank-link-latency-ns", type=float, default=100.0)
    parser.add_argument("--target-advantage", type=float, default=2.0)
    parser.add_argument("--power-kw", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    parser.add_argument("--amort-years", type=float, default=3.0)
    parser.add_argument("--utilization", type=float, default=0.70)
    parser.add_argument("--electricity-usd-kwh", type=float, default=0.10)
    parser.add_argument("--pue", type=float, default=1.20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    if model.key != "kimi-k3":
        raise ValueError("M5.1 currently supports kimi-k3 only")
    if min(args.input_tokens, args.output_tokens, args.concurrency, args.spine_modules) <= 0:
        raise ValueError("token counts, concurrency and spine modules must be positive")
    if not (0 < args.bandwidth_efficiency <= 1):
        raise ValueError("bandwidth-efficiency must be in (0, 1]")
    if args.q4_tops_per_bank <= 0 or args.handoff_gb_s <= 0:
        raise ValueError("bank compute and handoff bandwidth must be positive")
    if args.headroom < 0 or args.target_advantage <= 1:
        raise ValueError("headroom cannot be negative and target advantage must exceed 1x")
    if any(x <= 0 for x in args.power_kw):
        raise ValueError("power sensitivities must be positive")

    workload = Workload("agent-rightsize", args.input_tokens, args.output_tokens)
    node = COMMERCIAL_NODES[args.prefill_node]
    physical = HeterogeneousPhysicalSpec()
    part = GDDR7Part()
    economics = Economics(args.amort_years, args.utilization, args.electricity_usd_kwh, args.pue)
    target_req_s = prefill_supply_request_rate(args.prefill_node, workload)
    target_out_tps = target_req_s * workload.output_tokens
    state_bytes = _cache_bytes_per_sequence(model, workload.input_tokens, physical)
    min_handoff = target_req_s * state_bytes / 1e9

    d = decompose_model(model)
    routed_bytes = _bytes(d.routed_pool_parameters, physical.routed_bits, physical.overhead_fraction)

    print(f"{model.name} — M5.1 one-prefill-node decode rightsizing")
    print(f"  workload:                {workload.input_tokens:,} input / {workload.output_tokens:,} output tokens")
    print(f"  prefill source:          exactly 1 x {node.name} (no fractional node accounting)")
    print(f"  prefill supply:          {target_req_s:.3f} req/s -> {target_out_tps:,.1f} output tok/s")
    print(f"  fixed decoder spine:     {args.spine_modules} x 32 GB / 100 TB/s screening modules")
    print(f"  routed expert storage:   {routed_bytes/1e12:.3f} TB")
    print(f"  handoff state/request:   {state_bytes/1e9:.3f} GB; minimum fabric for full feed {min_handoff:.3f} GB/s")
    print(f"  routing stress:          {', '.join(args.profiles)}; selection uses worst result")
    print(f"  decode target:           {args.headroom:.0%} headroom over one-node prefill supply")
    print(
        f"  GDDR7 anchor:            {part.density_gbit:g} Gb x{part.bus_width_bits}, {part.max_rate_gt_s:g} GT/s production; "
        f"{part.capacity_gb:g} GB and {part.bandwidth_gb_s():g} GB/s per package"
    )
    print(f"  usable-memory BW factor: {args.bandwidth_efficiency:.0%} of signaling peak")

    print("\nRIGHT-SIZED GDDR7 EXPERT BANKS")
    print("PKG/BANK  BANK CAP  BANKS  GDDR PKGS  MIN RATE  EFF BW/BANK  AGG EFF BW  WORST PROF  DEC PEAK  REQ/S  FEED  DEC1  COLLIDE")
    rows: list[RightSizeResult] = []
    for placements in sorted(set(args.placements)):
        result = search_minimum_rate(
            model,
            workload,
            physical,
            part,
            placements=placements,
            rates_gt_s=tuple(args.rates),
            bandwidth_efficiency=args.bandwidth_efficiency,
            q4_tops_per_bank=args.q4_tops_per_bank,
            concurrency=args.concurrency,
            spine_modules=args.spine_modules,
            profiles=tuple(args.profiles),
            target_request_rate_s=target_req_s,
            headroom_fraction=args.headroom,
            seed=args.seed,
            bank_link_gb_s=args.bank_link_gb_s,
            bank_link_latency_ns=args.bank_link_latency_ns,
        )
        rows.append(result)
        fit = result.worst_request_rate_s >= target_req_s * (1 + args.headroom)
        status = "PASS" if fit else "FAIL"
        print(
            f"{result.placements:>8}  {result.bank_capacity_gb:>7.0f}GB  {result.bank_count:>5}  {result.total_devices:>9}  "
            f"{result.data_rate_gt_s:>7.0f}G  {result.effective_bank_bandwidth_gb_s:>10.0f}G  "
            f"{result.effective_aggregate_bandwidth_tb_s:>9.1f}T  {result.worst_profile:>10}  "
            f"{result.worst_decode_peak_tps:>8.0f}  {result.worst_request_rate_s:>5.3f}  "
            f"{result.feed_ratio:>4.2f}x  {result.single_decode_tps:>5.0f}  {result.worst_collision_fraction:>6.1%}  {status}"
        )

    passing = [row for row in rows if row.worst_request_rate_s >= target_req_s * (1 + args.headroom)]
    if passing:
        # Prefer fewer custom bank ASICs, then fewer GDDR packages, then lower line rate.
        chosen = min(passing, key=lambda row: (row.bank_count, row.total_devices, row.data_rate_gt_s))
    else:
        chosen = max(rows, key=lambda row: row.worst_request_rate_s)

    budget = dedicated_one_node_budget(
        args.prefill_node,
        workload,
        chosen,
        model,
        physical,
        economics,
        handoff_gb_s=args.handoff_gb_s,
        target_advantage=args.target_advantage,
        power_sensitivities_kw=tuple(args.power_kw),
    )

    print("\nONE-NODE PAIRING")
    print(
        f"  selected geometry:       {chosen.bank_count} banks x {chosen.bank_capacity_gb:g} GB "
        f"({chosen.placements} packages/bank, {chosen.total_devices} GDDR7 packages)"
    )
    print(
        f"  selected memory rate:    {chosen.data_rate_gt_s:g} GT/s @ {args.bandwidth_efficiency:.0%} usable -> "
        f"{chosen.effective_aggregate_bandwidth_tb_s:.1f} TB/s effective"
    )
    print(f"  worst routing profile:   {chosen.worst_profile}; decoder {chosen.worst_request_rate_s:.3f} req/s")
    print(f"  paired system rate:      {budget.request_rate_s:.3f} req/s / {budget.output_rate_tps:,.0f} output tok/s")
    print(f"  prefill utilization:     {budget.prefill_utilization:.1%}")
    print(f"  decoder utilization:     {budget.decoder_utilization:.1%}")
    print(f"  maximum decoder cloud-equivalent budget for {args.target_advantage:g}x target: {_fmt_money(budget.max_decoder_cloud_hour_usd)}/h")

    print("\nDEDICATED OWNED-SYSTEM 2x CAPEX GATE")
    print("  One complete MI355X node is charged to this decoder. Decoder power remains a sensitivity, not a BOM claim.")
    print("DECODER POWER   MAX DECODER CAPEX")
    for power_kw, capex in budget.max_decoder_capex_by_power_kw:
        print(f"{power_kw:>8.1f} kW      {_fmt_money(capex):>12}")

    print("\nINTERPRETATION")
    print("  - Capacity is non-negotiable: grouping GDDR packages into fewer banks reduces ASIC count, not routed-weight bytes.")
    print("  - A one-to-one prefill/decode pair should not provision 2x excess decode bandwidth; excess capacity only raises cost and power.")
    print("  - PASS means the worst of balanced/hot/Zipf synthetic routing can absorb one full prefill node plus the requested headroom.")
    print("  - The five 100-TB/s spine modules are still an unpriced screening abstraction. If their physical cost/power is not credible, this branch still fails.")
    print("  - No adversarial real routing trace is modeled yet; this is the last synthetic-routing screen before traces are required.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
