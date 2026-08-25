from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import math
import random
from statistics import mean
from typing import Any

from .architectures import _bytes, estimate_layers
from .fabric import _profile_cumulative, _selected_experts
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .m3 import (
    KIMI_K3_STATE,
    _attention_context_flops,
    _kda_recurrent_flops,
    _kda_state_bytes,
    _mla_entry_bytes,
)
from .models import ModelSpec, get_model
from .placement import decompose_model


@dataclass(frozen=True, slots=True)
class ExpertBankCandidate:
    bank_count: int
    capacity_gb: float
    bandwidth_gb_s: float
    q4_tops: float
    link_gb_s: float
    link_latency_ns: float = 100.0

    @property
    def label(self) -> str:
        return f"E{self.bank_count}x{self.capacity_gb:g}@{self.bandwidth_gb_s:g}G"


DEFAULT_CANDIDATES = (
    ExpertBankCandidate(24, 64.0, 256.0, 16.0, 64.0),
    ExpertBankCandidate(32, 64.0, 256.0, 16.0, 64.0),
    ExpertBankCandidate(24, 64.0, 512.0, 16.0, 64.0),
    ExpertBankCandidate(32, 64.0, 512.0, 16.0, 64.0),
    ExpertBankCandidate(16, 128.0, 512.0, 16.0, 64.0),
    ExpertBankCandidate(32, 64.0, 1024.0, 16.0, 64.0),
)
DEFAULT_CONCURRENCY = (1, 64)


@dataclass(frozen=True, slots=True)
class HeterogeneousPhysicalSpec:
    fast_capacity_gb: float = 32.0
    fast_bandwidth_tb_s: float = 100.0
    fast_fp16_tops: float = 256.0
    fast_q4_tops: float = 1024.0
    shared_shards: int = 2
    routed_bits: float = 4.0
    shared_bits: float = 16.0
    other_bits: float = 16.0
    kv_bits: float = 16.0
    overhead_fraction: float = 0.05
    activation_bits: float = 16.0

    @property
    def fast_capacity_bytes(self) -> float:
        return self.fast_capacity_gb * 1e9

    @property
    def fast_bandwidth_bytes_s(self) -> float:
        return self.fast_bandwidth_tb_s * 1e12


@dataclass(frozen=True, slots=True)
class CapacityReport:
    routed_storage_bytes: float
    always_on_storage_bytes: float
    cache_bytes_per_sequence: float
    spine_modules: int
    reference_modules: int
    max_spine_storage_bytes: float


@dataclass(frozen=True, slots=True)
class M4RunReport:
    label: str
    resident: bool
    concurrency: int
    throughput_tps: float
    p50_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    average_distinct_banks: float
    collision_fraction: float
    p95_expert_barrier_s: float
    hottest_bank_utilization: float
    hottest_spine_utilization: float
    max_bank_queue_wait_s: float
    max_spine_queue_wait_s: float
    expert_service_s: float
    events: int


@dataclass(frozen=True, slots=True)
class EconomicThresholds:
    bank_cost_ratio_for_lower_system_capex: float
    bank_cost_ratio_for_lower_capex_per_token: float
    bank_power_ratio_for_lower_energy_per_token: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    frac = position - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _coprime_stride(size: int) -> int:
    if size <= 1:
        return 1
    preferred = max(1, int(round(math.sqrt(size))))
    for delta in range(size):
        for candidate in (preferred + delta, preferred - delta):
            if 1 <= candidate < size and math.gcd(candidate, size) == 1:
                return candidate
    return 1


def _cache_bytes_per_sequence(model: ModelSpec, context: int, spec: HeterogeneousPhysicalSpec) -> float:
    if model.key != "kimi-k3":
        raise ValueError("M4 currently has calibrated inference-state geometry only for kimi-k3")
    full = KIMI_K3_STATE.full_attention_layer_set
    mla = _mla_entry_bytes(KIMI_K3_STATE, spec.kv_bits, "latent")
    kda = _kda_state_bytes(KIMI_K3_STATE, spec.kv_bits)
    total = 0.0
    for layer in range(model.num_layers):
        total += mla * context if layer in full else kda
    return total


def _mixed_storage(model: ModelSpec, spec: HeterogeneousPhysicalSpec) -> tuple[float, float]:
    d = decompose_model(model)
    routed = _bytes(d.routed_pool_parameters, spec.routed_bits, spec.overhead_fraction)
    always_on = _bytes(d.shared_expert_parameters, spec.shared_bits, spec.overhead_fraction) + _bytes(
        d.other_always_on_parameters, spec.other_bits, spec.overhead_fraction
    )
    return routed, always_on


def _spine_storage_map(
    model: ModelSpec,
    spec: HeterogeneousPhysicalSpec,
    *,
    spine_modules: int,
    context: int,
    max_concurrency: int,
) -> tuple[list[float], float]:
    layers = estimate_layers(model)
    capacity = [0.0] * spine_modules
    cache_per_sequence = [0.0] * spine_modules
    full = KIMI_K3_STATE.full_attention_layer_set
    mla = _mla_entry_bytes(KIMI_K3_STATE, spec.kv_bits, "latent")
    kda = _kda_state_bytes(KIMI_K3_STATE, spec.kv_bits)

    for i, layer in enumerate(layers):
        anchor = i % spine_modules
        capacity[anchor] += _bytes(layer.other_always_on_parameters, spec.other_bits, spec.overhead_fraction)
        if layer.shared_parameters:
            shards = min(spec.shared_shards, spine_modules)
            shard_bytes = _bytes(layer.shared_parameters, spec.shared_bits, spec.overhead_fraction) / shards
            for offset in range(shards):
                capacity[(anchor + offset) % spine_modules] += shard_bytes
        if i in full:
            cache_per_sequence[anchor] += mla * context
        else:
            cache_per_sequence[anchor] += kda

    total = [w + max_concurrency * c for w, c in zip(capacity, cache_per_sequence)]
    return total, sum(cache_per_sequence)


def build_capacity_report(
    model: ModelSpec,
    spec: HeterogeneousPhysicalSpec,
    *,
    context: int,
    max_concurrency: int,
    forced_spine_modules: int | None = None,
) -> CapacityReport:
    routed, always_on = _mixed_storage(model, spec)
    cache_seq = _cache_bytes_per_sequence(model, context, spec)

    if forced_spine_modules is not None:
        if forced_spine_modules <= 0:
            raise ValueError("spine_modules must be positive")
        spine_modules = forced_spine_modules
        mapping, _ = _spine_storage_map(
            model,
            spec,
            spine_modules=spine_modules,
            context=context,
            max_concurrency=max_concurrency,
        )
        if max(mapping, default=0.0) > spec.fast_capacity_bytes:
            raise ValueError("forced spine module count does not fit always-on weights + KV state")
    else:
        spine_modules = 1
        while True:
            mapping, _ = _spine_storage_map(
                model,
                spec,
                spine_modules=spine_modules,
                context=context,
                max_concurrency=max_concurrency,
            )
            if max(mapping, default=0.0) <= spec.fast_capacity_bytes:
                break
            spine_modules += 1
            if spine_modules > 256:
                raise RuntimeError("could not place spine state within 256 fast modules")

    max_spine = max(mapping, default=0.0)
    total_reference_storage = routed + always_on + cache_seq * max_concurrency
    reference_modules = max(1, math.ceil(total_reference_storage / spec.fast_capacity_bytes))
    return CapacityReport(
        routed_storage_bytes=routed,
        always_on_storage_bytes=always_on,
        cache_bytes_per_sequence=cache_seq,
        spine_modules=spine_modules,
        reference_modules=reference_modules,
        max_spine_storage_bytes=max_spine,
    )


class _M4Simulator:
    def __init__(
        self,
        model: ModelSpec,
        spec: HeterogeneousPhysicalSpec,
        capacity: CapacityReport,
        *,
        context: int,
        concurrency: int,
        profile: str,
        seed: int,
        candidate: ExpertBankCandidate | None,
        ideal_reference: bool,
    ) -> None:
        self.model = model
        self.spec = spec
        self.capacity = capacity
        self.context = context
        self.concurrency = concurrency
        self.profile = profile
        self.seed = seed
        self.candidate = candidate
        self.ideal_reference = ideal_reference
        self.layers = estimate_layers(model)
        self.decomp = decompose_model(model)
        self.profile_cumulative = _profile_cumulative(model, profile)
        self.events: list[tuple[float, int, str, tuple[Any, ...]]] = []
        self.seq = 0
        self.events_processed = 0
        self.token_done: list[float | None] = [None] * concurrency
        self.spine_available = [0.0] * capacity.spine_modules
        self.spine_busy = [0.0] * capacity.spine_modules
        self.spine_waits: list[float] = []
        self.expert_barriers: list[float] = []
        self.distinct_samples: list[int] = []
        self.collisions = 0
        self.expert_calls = 0

        if ideal_reference:
            self.bank_count = capacity.reference_modules
            self.bank_bandwidth = spec.fast_bandwidth_bytes_s
            self.bank_q4_flops_s = spec.fast_q4_tops * 1e12
            self.link_bytes_s = math.inf
            self.link_latency_s = 0.0
            self.bank_capacity_bytes = spec.fast_capacity_bytes
            self.label = f"ALL-FAST-{capacity.reference_modules}"
        else:
            if candidate is None:
                raise ValueError("heterogeneous simulation requires candidate")
            self.bank_count = candidate.bank_count
            self.bank_bandwidth = candidate.bandwidth_gb_s * 1e9
            self.bank_q4_flops_s = candidate.q4_tops * 1e12
            self.link_bytes_s = candidate.link_gb_s * 1e9
            self.link_latency_s = candidate.link_latency_ns * 1e-9
            self.bank_capacity_bytes = candidate.capacity_gb * 1e9
            self.label = candidate.label

        self.bank_available = [0.0] * self.bank_count
        self.bank_busy = [0.0] * self.bank_count
        self.bank_waits: list[float] = []
        self.stride = _coprime_stride(self.bank_count)
        self.expert_weight_bytes = _bytes(
            self.decomp.expert_shard_parameters or 0.0,
            spec.routed_bits,
            spec.overhead_fraction,
        )
        self.expert_flops = 2.0 * (self.decomp.expert_shard_parameters or 0.0)
        memory_s = self.expert_weight_bytes / self.bank_bandwidth if self.bank_bandwidth else math.inf
        compute_s = self.expert_flops / self.bank_q4_flops_s if self.bank_q4_flops_s else math.inf
        routed_message = KIMI_K3_STATE.routed_expert_hidden_size * spec.activation_bits / 8.0
        network_s = 0.0
        if math.isfinite(self.link_bytes_s):
            network_s = 2.0 * routed_message / self.link_bytes_s + 2.0 * self.link_latency_s
        self.expert_service_s = max(memory_s, compute_s) + network_s

        shard_count = model.moe_layers * model.num_experts
        max_shards = math.ceil(shard_count / self.bank_count)
        self.max_bank_storage = max_shards * self.expert_weight_bytes
        self.resident = ideal_reference or (
            self.max_bank_storage <= self.bank_capacity_bytes
            and self.capacity.routed_storage_bytes <= self.bank_count * self.bank_capacity_bytes
        )

    def push(self, time_s: float, kind: str, *data: Any) -> None:
        self.seq += 1
        heapq.heappush(self.events, (time_s, self.seq, kind, tuple(data)))

    def _selected(self, token: int, moe_layer: int) -> tuple[int, ...]:
        if self.profile == "balanced":
            rng = random.Random(0)
        else:
            rng = random.Random(self.seed + token * 1_000_003 + moe_layer * 9_176)
        return _selected_experts(
            self.model,
            token,
            moe_layer,
            self.profile,
            self.profile_cumulative,
            rng,
        )

    def _state_cost(self, layer_index: int) -> tuple[float, float]:
        if layer_index in KIMI_K3_STATE.full_attention_layer_set:
            entry = _mla_entry_bytes(KIMI_K3_STATE, self.spec.kv_bits, "latent")
            memory = entry * (self.context + 1)
            flops = _attention_context_flops(KIMI_K3_STATE, self.context, "latent")
            return memory, flops
        state = _kda_state_bytes(KIMI_K3_STATE, self.spec.kv_bits)
        return 2.0 * state, _kda_recurrent_flops(KIMI_K3_STATE)

    def _reserve_spine(self, ready_s: float, module: int, service_s: float) -> float:
        start = max(ready_s, self.spine_available[module])
        wait = start - ready_s
        finish = start + service_s
        self.spine_available[module] = finish
        self.spine_busy[module] += service_s
        self.spine_waits.append(wait)
        return finish

    def _reserve_bank(self, ready_s: float, bank: int, jobs: int) -> float:
        start = max(ready_s, self.bank_available[bank])
        wait = start - ready_s
        service = jobs * self.expert_service_s
        finish = start + service
        self.bank_available[bank] = finish
        self.bank_busy[bank] += service
        self.bank_waits.append(wait)
        return finish

    def _begin_layer(self, time_s: float, token: int, layer_index: int) -> None:
        layer = self.layers[layer_index]
        anchor = layer_index % self.capacity.spine_modules
        state_bytes, state_flops = self._state_cost(layer_index)
        other_bytes = _bytes(layer.other_always_on_parameters, self.spec.other_bits, self.spec.overhead_fraction)
        other_flops = 2.0 * layer.other_always_on_parameters + state_flops
        memory_s = (other_bytes + state_bytes) / self.spec.fast_bandwidth_bytes_s
        compute_s = other_flops / (self.spec.fast_fp16_tops * 1e12)
        finish = self._reserve_spine(time_s, anchor, max(memory_s, compute_s))
        if layer.is_moe:
            self.push(finish, "moe_start", token, layer_index)
        else:
            self.push(finish, "layer_done", token, layer_index)

    def _schedule_shared(self, time_s: float, layer_index: int) -> list[float]:
        layer = self.layers[layer_index]
        if layer.shared_parameters <= 0:
            return []
        anchor = layer_index % self.capacity.spine_modules
        shards = min(self.spec.shared_shards, self.capacity.spine_modules)
        params = layer.shared_parameters / shards
        memory_s = _bytes(params, self.spec.shared_bits, self.spec.overhead_fraction) / self.spec.fast_bandwidth_bytes_s
        compute_s = (2.0 * params) / (self.spec.fast_fp16_tops * 1e12)
        service = max(memory_s, compute_s)
        return [
            self._reserve_spine(time_s, (anchor + offset) % self.capacity.spine_modules, service)
            for offset in range(shards)
        ]

    def _schedule_experts(self, time_s: float, token: int, layer_index: int) -> list[float]:
        moe_layer = layer_index - self.model.dense_layers
        selected = self._selected(token, moe_layer)
        self.expert_calls += len(selected)

        if self.ideal_reference:
            # Optimistic lower bound: expert weights are perfectly placed and each
            # selected expert can use the currently least-loaded fast module.
            order = sorted(range(self.bank_count), key=lambda bank: self.bank_available[bank])
            finishes = []
            distinct = min(len(selected), self.bank_count)
            for i, _expert in enumerate(selected):
                bank = order[i % len(order)]
                finishes.append(self._reserve_bank(time_s, bank, 1))
            self.distinct_samples.append(distinct)
            self.collisions += max(0, len(selected) - distinct)
            return finishes

        grouped: dict[int, int] = {}
        for expert in selected:
            bank = (expert + moe_layer * self.stride) % self.bank_count
            grouped[bank] = grouped.get(bank, 0) + 1
        distinct = len(grouped)
        self.distinct_samples.append(distinct)
        self.collisions += len(selected) - distinct
        return [self._reserve_bank(time_s, bank, jobs) for bank, jobs in grouped.items()]

    def _moe_start(self, time_s: float, token: int, layer_index: int) -> None:
        shared = self._schedule_shared(time_s, layer_index)
        experts = self._schedule_experts(time_s, token, layer_index)
        finish = max(shared + experts, default=time_s)
        self.expert_barriers.append(finish - time_s)
        self.push(finish, "layer_done", token, layer_index)

    def _layer_done(self, time_s: float, token: int, layer_index: int) -> None:
        if layer_index + 1 >= len(self.layers):
            self.token_done[token] = time_s
        else:
            self.push(time_s, "begin_layer", token, layer_index + 1)

    def run(self) -> M4RunReport:
        if not self.resident:
            return M4RunReport(
                label=self.label,
                resident=False,
                concurrency=self.concurrency,
                throughput_tps=0.0,
                p50_latency_s=0.0,
                p95_latency_s=0.0,
                max_latency_s=0.0,
                average_distinct_banks=0.0,
                collision_fraction=0.0,
                p95_expert_barrier_s=0.0,
                hottest_bank_utilization=0.0,
                hottest_spine_utilization=0.0,
                max_bank_queue_wait_s=0.0,
                max_spine_queue_wait_s=0.0,
                expert_service_s=self.expert_service_s,
                events=0,
            )

        for token in range(self.concurrency):
            self.push(0.0, "begin_layer", token, 0)

        while self.events:
            time_s, _seq, kind, data = heapq.heappop(self.events)
            self.events_processed += 1
            if kind == "begin_layer":
                self._begin_layer(time_s, *data)
            elif kind == "moe_start":
                self._moe_start(time_s, *data)
            elif kind == "layer_done":
                self._layer_done(time_s, *data)
            else:
                raise RuntimeError(f"unknown M4 event {kind!r}")

        if any(value is None for value in self.token_done):
            raise RuntimeError("M4 ended before all tokens completed")
        done = [float(value) for value in self.token_done if value is not None]
        makespan = max(done, default=0.0)
        throughput = self.concurrency / makespan if makespan else 0.0
        bank_util = max(self.bank_busy, default=0.0) / makespan if makespan else 0.0
        spine_util = max(self.spine_busy, default=0.0) / makespan if makespan else 0.0
        return M4RunReport(
            label=self.label,
            resident=True,
            concurrency=self.concurrency,
            throughput_tps=throughput,
            p50_latency_s=_percentile(done, 0.50),
            p95_latency_s=_percentile(done, 0.95),
            max_latency_s=max(done, default=0.0),
            average_distinct_banks=mean(self.distinct_samples) if self.distinct_samples else 0.0,
            collision_fraction=self.collisions / self.expert_calls if self.expert_calls else 0.0,
            p95_expert_barrier_s=_percentile(self.expert_barriers, 0.95),
            hottest_bank_utilization=bank_util,
            hottest_spine_utilization=spine_util,
            max_bank_queue_wait_s=max(self.bank_waits, default=0.0),
            max_spine_queue_wait_s=max(self.spine_waits, default=0.0),
            expert_service_s=self.expert_service_s,
            events=self.events_processed,
        )


def simulate_candidate(
    model: ModelSpec,
    spec: HeterogeneousPhysicalSpec,
    capacity: CapacityReport,
    candidate: ExpertBankCandidate,
    *,
    context: int,
    concurrency: int,
    profile: str = "balanced",
    seed: int = 42,
) -> M4RunReport:
    return _M4Simulator(
        model,
        spec,
        capacity,
        context=context,
        concurrency=concurrency,
        profile=profile,
        seed=seed,
        candidate=candidate,
        ideal_reference=False,
    ).run()


def simulate_all_fast_reference(
    model: ModelSpec,
    spec: HeterogeneousPhysicalSpec,
    capacity: CapacityReport,
    *,
    context: int,
    concurrency: int,
    profile: str = "balanced",
    seed: int = 42,
) -> M4RunReport:
    return _M4Simulator(
        model,
        spec,
        capacity,
        context=context,
        concurrency=concurrency,
        profile=profile,
        seed=seed,
        candidate=None,
        ideal_reference=True,
    ).run()


def economic_thresholds(
    candidate: ExpertBankCandidate,
    capacity: CapacityReport,
    hetero: M4RunReport,
    reference: M4RunReport,
) -> EconomicThresholds:
    system_capex = max(0.0, (capacity.reference_modules - capacity.spine_modules) / candidate.bank_count)
    if hetero.throughput_tps <= 0 or reference.throughput_tps <= 0:
        return EconomicThresholds(system_capex, 0.0, 0.0)
    normalized_budget = capacity.reference_modules * hetero.throughput_tps / reference.throughput_tps
    per_token = (normalized_budget - capacity.spine_modules) / candidate.bank_count
    return EconomicThresholds(
        bank_cost_ratio_for_lower_system_capex=system_capex,
        bank_cost_ratio_for_lower_capex_per_token=per_token,
        bank_power_ratio_for_lower_energy_per_token=per_token,
    )


def _parse_candidate(value: str) -> ExpertBankCandidate:
    try:
        fields = value.split(":")
        if len(fields) not in {5, 6}:
            raise ValueError
        banks, cap, bw, tops, link = fields[:5]
        latency = fields[5] if len(fields) == 6 else "100"
        candidate = ExpertBankCandidate(
            bank_count=int(banks),
            capacity_gb=float(cap),
            bandwidth_gb_s=float(bw),
            q4_tops=float(tops),
            link_gb_s=float(link),
            link_latency_ns=float(latency),
        )
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be BANKS:CAP_GB:BW_GB/s:Q4_TOPS:LINK_GB/s[:LAT_NS]"
        ) from exc
    if min(candidate.bank_count, candidate.capacity_gb, candidate.bandwidth_gb_s, candidate.q4_tops, candidate.link_gb_s) <= 0:
        raise argparse.ArgumentTypeError("candidate values must be positive")
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m4",
        description="Heterogeneous Kimi K3 simulator: fast attention/KV spine + resident expert-memory banks.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY))
    parser.add_argument("--candidate", type=_parse_candidate, action="append", dest="candidates")
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spine-modules", type=int, default=None)
    parser.add_argument("--fast-capacity-gb", type=float, default=32.0)
    parser.add_argument("--fast-bandwidth-tb-s", type=float, default=100.0)
    parser.add_argument("--fast-fp16-tops", type=float, default=256.0)
    parser.add_argument("--fast-q4-tops", type=float, default=1024.0)
    parser.add_argument("--shared-shards", type=int, default=2)
    parser.add_argument("--routed-bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=16.0)
    parser.add_argument("--other-bits", type=float, default=16.0)
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    if model.key != "kimi-k3":
        raise ValueError("M4 currently supports kimi-k3 only")
    if args.context < 0:
        raise ValueError("context cannot be negative")
    concurrency = tuple(sorted(set(args.concurrency)))
    if not concurrency or any(x <= 0 for x in concurrency):
        raise ValueError("concurrency values must be positive")
    if min(args.fast_capacity_gb, args.fast_bandwidth_tb_s, args.fast_fp16_tops, args.fast_q4_tops) <= 0:
        raise ValueError("fast-module parameters must be positive")
    if args.shared_shards <= 0:
        raise ValueError("shared-shards must be positive")

    spec = HeterogeneousPhysicalSpec(
        fast_capacity_gb=args.fast_capacity_gb,
        fast_bandwidth_tb_s=args.fast_bandwidth_tb_s,
        fast_fp16_tops=args.fast_fp16_tops,
        fast_q4_tops=args.fast_q4_tops,
        shared_shards=args.shared_shards,
        routed_bits=args.routed_bits,
        shared_bits=args.shared_bits,
        other_bits=args.other_bits,
        kv_bits=args.kv_bits,
        overhead_fraction=args.overhead,
    )
    capacity = build_capacity_report(
        model,
        spec,
        context=args.context,
        max_concurrency=max(concurrency),
        forced_spine_modules=args.spine_modules,
    )
    candidates = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES

    d = decompose_model(model)
    expert_params = d.expert_shard_parameters or 0.0
    expert_bytes = _bytes(expert_params, spec.routed_bits, spec.overhead_fraction)
    print(f"{model.name} — heterogeneous resident-expert architecture")
    print(f"  context / max concurrency: {args.context:,} / {max(concurrency)}")
    print(f"  routed pool:              {fmt_bytes(capacity.routed_storage_bytes)} ({d.routed_pool_parameters/1e12:.3f}T params)")
    print(f"  always-on spine weights:  {fmt_bytes(capacity.always_on_storage_bytes)}")
    print(f"  latent KV/state per seq:  {fmt_bytes(capacity.cache_bytes_per_sequence)}")
    print(f"  routed expert block:      {expert_params/1e6:.3f}M params / {fmt_bytes(expert_bytes)}")
    print(
        f"  fast-module abstraction:  {spec.fast_capacity_gb:g} GB, {spec.fast_bandwidth_tb_s:g} TB/s, "
        f"{spec.fast_fp16_tops:g} FP16 TOPS, {spec.fast_q4_tops:g} Q4 TOPS"
    )
    print(f"  minimum fast spine:       {capacity.spine_modules} modules (max used {fmt_bytes(capacity.max_spine_storage_bytes)})")
    print(f"  optimistic all-fast ref:  {capacity.reference_modules} modules for weights + max-concurrency KV")
    print(f"  routing profile:          {args.profile}")

    references: dict[int, M4RunReport] = {}
    print("\nOPTIMISTIC ALL-FAST REFERENCE")
    print("CONC   THROUGHPUT    P50 LAT     P95 LAT     BANK UTIL  SPINE UTIL  P95 EXPERT")
    for conc in concurrency:
        report = simulate_all_fast_reference(
            model,
            spec,
            capacity,
            context=args.context,
            concurrency=conc,
            profile=args.profile,
            seed=args.seed,
        )
        references[conc] = report
        print(
            f"{conc:>4}  {fmt_rate(report.throughput_tps):>12}  {fmt_time_s(report.p50_latency_s):>10}  "
            f"{fmt_time_s(report.p95_latency_s):>10}  {report.hottest_bank_utilization:>8.1%}  "
            f"{report.hottest_spine_utilization:>9.1%}  {fmt_time_s(report.p95_expert_barrier_s):>10}"
        )

    print("\nHETEROGENEOUS EXPERT-BANK SWEEP")
    print(
        "CANDIDATE         CONC FIT   THROUGHPUT   VS FAST  P95 LAT    DISTINCT  COLLIDE  BANK UTIL  "
        "P95 BARRIER  EXPERT SVC"
    )
    reports: dict[tuple[str, int], M4RunReport] = {}
    for candidate in candidates:
        for conc in concurrency:
            report = simulate_candidate(
                model,
                spec,
                capacity,
                candidate,
                context=args.context,
                concurrency=conc,
                profile=args.profile,
                seed=args.seed,
            )
            reports[(candidate.label, conc)] = report
            if not report.resident:
                print(f"{candidate.label:<17} {conc:>4}  NO   —")
                continue
            ref = references[conc]
            ratio = report.throughput_tps / ref.throughput_tps if ref.throughput_tps else 0.0
            print(
                f"{candidate.label:<17} {conc:>4} yes  {fmt_rate(report.throughput_tps):>11}  {ratio:>7.1%}  "
                f"{fmt_time_s(report.p95_latency_s):>9}  {report.average_distinct_banks:>8.2f}  "
                f"{report.collision_fraction:>7.1%}  {report.hottest_bank_utilization:>8.1%}  "
                f"{fmt_time_s(report.p95_expert_barrier_s):>11}  {fmt_time_s(report.expert_service_s):>10}"
            )

    highest = max(concurrency)
    ref = references[highest]
    print(f"\nECONOMIC BREAK-EVEN AT CONCURRENCY {highest}")
    print("  Fast-module cost and power are normalized to 1.0; no vendor price/power is assumed.")
    print("CANDIDATE          BANK$/FAST$<  BANK$/FAST$ FOR <=$/TOK  BANKW/FASTW FOR <=J/TOK")
    for candidate in candidates:
        report = reports[(candidate.label, highest)]
        if not report.resident:
            print(f"{candidate.label:<18} NO FIT")
            continue
        threshold = economic_thresholds(candidate, capacity, report, ref)
        capex_tok = threshold.bank_cost_ratio_for_lower_capex_per_token
        power_tok = threshold.bank_power_ratio_for_lower_energy_per_token
        capex_text = "impossible" if capex_tok <= 0 else f"{capex_tok:.3f}"
        power_text = "impossible" if power_tok <= 0 else f"{power_tok:.3f}"
        print(
            f"{candidate.label:<18} {threshold.bank_cost_ratio_for_lower_system_capex:>10.3f}  "
            f"{capex_text:>22}  {power_text:>22}"
        )

    print("\nHOW TO READ THIS")
    print("  - ALL-FAST is deliberately optimistic: routed experts get ideal load-balanced placement on fast modules.")
    print("  - Expert-bank weights never move. A selected expert pays local weight streaming + local Q4 compute + one latent activation round trip.")
    print("  - DISTINCT is the average number of physical expert banks hit by Kimi top-16; COLLIDE is the fraction of calls serialized behind another selected expert on the same bank.")
    print("  - BANK$/FAST$< is the maximum bank/fast-module price ratio for lower raw system capex; the per-token columns additionally penalize lower throughput.")
    print("  - Negative per-token break-even is printed as impossible: even zero-cost/zero-power expert banks would not compensate for the throughput loss under this model.")
    print("  - This remains a screening model: real expert traces, DRAM timing, packet backpressure, thermals, chip/package cost and measured kernel efficiency are not yet included.")
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
