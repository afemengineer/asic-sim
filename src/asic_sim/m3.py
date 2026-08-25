from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import math
from statistics import mean
from typing import Any

from .architectures import _bytes, estimate_layers
from .bandwidth_sweep import _cluster_tiles, _coprime_stride
from .fabric import _profile_cumulative, _selected_experts
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .hardware import HardwareSpec, get_hardware
from .m2_network_models import _CutThroughSimulator
from .models import ModelSpec, get_model


DEFAULT_CANDIDATES = ((4, 64.0), (8, 64.0), (16, 128.0))
DEFAULT_CONCURRENCY = (1, 64)
DEFAULT_CONTEXT_SWEEP = (4_096, 16_384, 65_536, 262_144, 1_048_576)


@dataclass(frozen=True, slots=True)
class KimiInferenceStateSpec:
    full_attention_layers: tuple[int, ...]
    num_heads: int = 96
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    kv_lora_rank: int = 512
    kda_heads: int = 96
    kda_head_dim: int = 128
    kda_conv_kernel: int = 4
    routed_expert_hidden_size: int = 3584

    @property
    def full_attention_layer_set(self) -> frozenset[int]:
        return frozenset(self.full_attention_layers)

    @property
    def kda_layers(self) -> int:
        return 93 - len(self.full_attention_layers)


# Official Kimi K3 config: full attention is layer numbers 4, 8, ..., 92, 93.
# Internally the simulator uses zero-based layer indices.
KIMI_K3_STATE = KimiInferenceStateSpec(
    full_attention_layers=tuple([value - 1 for value in range(4, 93, 4)] + [92]),
)


@dataclass(frozen=True, slots=True)
class M3RunReport:
    cluster_size: int
    link_bandwidth_gb_s: float
    shared_shards: int
    concurrency: int
    context_length: int
    kv_mode: str
    resident: bool
    weight_storage_bytes: float
    cache_bytes_per_sequence: float
    total_cache_bytes: float
    max_tile_storage_bytes: float
    tile_capacity_bytes: float
    throughput_tps: float
    p50_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    hottest_link_utilization: float
    hottest_memory_utilization: float
    hottest_compute_utilization: float
    max_link_queue_wait_s: float
    max_memory_queue_wait_s: float
    max_compute_queue_wait_s: float
    p95_barrier_tail_s: float
    network_payload_bytes_per_token: float
    hop_bytes_per_token: float
    routed_message_bytes: float
    shared_message_bytes: float
    event_count: int


@dataclass(slots=True)
class _WorkBarrier:
    remaining: int
    callback: str
    callback_data: tuple[Any, ...]
    last_done_s: float = 0.0


@dataclass(slots=True)
class _MoeBarrier:
    remaining: int
    first_done_s: float | None = None
    last_done_s: float | None = None


def _state_spec(model: ModelSpec) -> KimiInferenceStateSpec:
    if model.key != "kimi-k3":
        raise ValueError("M3 currently has a calibrated inference-state model only for kimi-k3")
    return KIMI_K3_STATE


def _mla_entry_bytes(spec: KimiInferenceStateSpec, bits: float, mode: str) -> float:
    bytes_per_element = bits / 8.0
    if mode == "latent":
        # Optimized MLA cache: compressed KV latent plus decoupled RoPE key.
        return (spec.kv_lora_rank + spec.qk_rope_head_dim) * bytes_per_element
    if mode == "expanded":
        # Hugging Face reference path stores expanded K and V tensors.
        key = spec.num_heads * (spec.qk_nope_head_dim + spec.qk_rope_head_dim)
        value = spec.num_heads * spec.v_head_dim
        return (key + value) * bytes_per_element
    raise ValueError("kv_mode must be 'latent' or 'expanded'")


def _kda_state_bytes(spec: KimiInferenceStateSpec, bits: float) -> float:
    bytes_per_element = bits / 8.0
    recurrent = spec.kda_heads * spec.kda_head_dim * spec.kda_head_dim
    # q/k/v short-convolution state; use the configured four-value window as
    # a conservative capacity estimate.
    conv = 3 * spec.kda_heads * spec.kda_head_dim * spec.kda_conv_kernel
    return (recurrent + conv) * bytes_per_element


def _attention_context_flops(spec: KimiInferenceStateSpec, context: int, mode: str) -> float:
    if context <= 0:
        return 0.0
    if mode == "expanded":
        # QK and AV matvecs over expanded head dimensions.
        dims = spec.num_heads * (
            spec.qk_nope_head_dim + spec.qk_rope_head_dim + spec.v_head_dim
        )
        return 2.0 * dims * context
    if mode == "latent":
        # Absorbed-MLA screening model: per-head content+RoPE score against the
        # compressed latent, then weighted latent accumulation. This trades much
        # lower cache bandwidth for more arithmetic than the expanded reference.
        dims = spec.num_heads * (
            (spec.kv_lora_rank + spec.qk_rope_head_dim) + spec.kv_lora_rank
        )
        return 2.0 * dims * context
    raise ValueError("kv_mode must be 'latent' or 'expanded'")


def _kda_recurrent_flops(spec: KimiInferenceStateSpec) -> float:
    # Screening approximation for recurrent read/update/output work.
    return 4.0 * spec.kda_heads * spec.kda_head_dim * spec.kda_head_dim


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
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


class _M3Simulator(_CutThroughSimulator):
    """M3 event model: cut-through NoC + memory + compute + inference state.

    Memory and tensor work for one operation are scheduled concurrently and the
    operation completes when both resources finish, approximating a streamed,
    double-buffered datapath. This is intentionally an optimistic overlap model.
    """

    def __init__(
        self,
        model: ModelSpec,
        hardware: HardwareSpec,
        *,
        cluster_size: int,
        link_bandwidth_gb_s: float,
        bits_per_weight: float,
        shared_expert_bits: float,
        other_weight_bits: float,
        overhead_fraction: float,
        activation_bits: float,
        concurrency: int,
        context_length: int,
        kv_mode: str,
        kv_bits: float,
        kda_state_bits: float,
        shared_shards: int,
        routed_tops: float,
        shared_tops: float,
        other_tops: float,
        profile: str,
        seed: int,
        issue_spacing_s: float,
    ) -> None:
        super().__init__(
            model,
            hardware,
            cluster_size=cluster_size,
            link_bandwidth_gb_s=link_bandwidth_gb_s,
            bits_per_weight=bits_per_weight,
            shared_expert_bits=shared_expert_bits,
            overhead_fraction=overhead_fraction,
            activation_bits=activation_bits,
            concurrency=concurrency,
            profile=profile,
            seed=seed,
            issue_spacing_s=issue_spacing_s,
        )
        if context_length < 0:
            raise ValueError("context_length cannot be negative")
        if shared_shards <= 0 or shared_shards > cluster_size:
            raise ValueError("shared_shards must be in [1, cluster_size]")
        if min(routed_tops, shared_tops, other_tops) <= 0:
            raise ValueError("effective TOPS values must be positive")
        if min(other_weight_bits, kv_bits, kda_state_bits) <= 0:
            raise ValueError("bit widths must be positive")
        if kv_mode not in {"latent", "expanded"}:
            raise ValueError("kv_mode must be 'latent' or 'expanded'")

        self.state_spec = _state_spec(model)
        self.context_length = context_length
        self.kv_mode = kv_mode
        self.kv_bits = kv_bits
        self.kda_state_bits = kda_state_bits
        self.shared_shards = shared_shards
        self.other_bits = other_weight_bits
        self.routed_tops = routed_tops * 1e12
        self.shared_tops = shared_tops * 1e12
        self.other_tops = other_tops * 1e12

        self.compute_available = [0.0] * hardware.tiles
        self.compute_busy = [0.0] * hardware.tiles
        self.compute_wait_total = 0.0
        self.compute_wait_samples: list[float] = []
        self.memory_wait_samples: list[float] = []
        self.work_barriers: dict[int, _WorkBarrier] = {}
        self.work_seq = 0
        self.moe_barriers: dict[tuple[int, int], _MoeBarrier] = {}
        self.moe_barrier_tails: list[float] = []

        # Kimi Latent-MoE routes the 3584-wide expert representation, while
        # shared experts consume the full 7168 hidden representation.
        self.routed_message_bytes = self.state_spec.routed_expert_hidden_size * activation_bits / 8.0
        self.shared_message_bytes = model.hidden_bytes(activation_bits)

        self.weight_storage_by_tile, self.cache_per_sequence_by_tile = self._build_storage()
        self.weight_storage_bytes = sum(self.weight_storage_by_tile)
        self.cache_bytes_per_sequence = sum(self.cache_per_sequence_by_tile)
        self.total_storage_by_tile = [
            w + concurrency * c
            for w, c in zip(self.weight_storage_by_tile, self.cache_per_sequence_by_tile)
        ]
        self.tile_capacity_bytes = (
            hardware.tile_capacity_gb * 1e9
            if hardware.tile_capacity_gb is not None
            else hardware.capacity_bytes
        )
        self.resident_m3 = (
            sum(self.total_storage_by_tile) <= hardware.capacity_bytes
            and max(self.total_storage_by_tile, default=0.0) <= self.tile_capacity_bytes
        )

    def _build_storage(self) -> tuple[list[float], list[float]]:
        storage = [0.0] * self.hardware.tiles
        cache = [0.0] * self.hardware.tiles
        full_layers = self.state_spec.full_attention_layer_set
        mla_entry = _mla_entry_bytes(self.state_spec, self.kv_bits, self.kv_mode)
        kda_state = _kda_state_bytes(self.state_spec, self.kda_state_bits)

        for layer_index, layer in enumerate(self.layers):
            cluster = self.clusters[self.layer_cluster[layer_index]]
            anchor = self.layer_anchor[layer_index]
            storage[anchor] += _bytes(layer.other_always_on_parameters, self.other_bits, self.overhead)

            if layer.shared_parameters:
                owners = self._shared_owners(layer_index)
                shard_bytes = _bytes(layer.shared_parameters, self.shared_bits, self.overhead) / len(owners)
                for owner in owners:
                    storage[owner] += shard_bytes

            if layer.routed_pool_parameters:
                routed_each = _bytes(layer.routed_pool_parameters, self.bits, self.overhead) / len(cluster)
                for tile in cluster:
                    storage[tile] += routed_each

            if layer_index in full_layers:
                cache[anchor] += mla_entry * self.context_length
            else:
                cache[anchor] += kda_state

        return storage, cache

    def _shared_owners(self, layer_index: int) -> tuple[int, ...]:
        cluster = self.clusters[self.layer_cluster[layer_index]]
        anchor = self.layer_anchor[layer_index]
        anchor_pos = cluster.index(anchor)
        count = min(self.shared_shards, len(cluster))
        return tuple(cluster[(anchor_pos + offset) % len(cluster)] for offset in range(count))

    def _schedule_memory(self, ready_s: float, tile: int, payload_bytes: float, callback: str, *cb_data: Any) -> None:
        if payload_bytes <= 0:
            self.push(ready_s, callback, *cb_data)
            return
        start = max(ready_s, self.memory_available[tile])
        wait = start - ready_s
        service = payload_bytes / self.memory_bandwidth_bytes_s
        finish = start + service
        self.memory_available[tile] = finish
        self.memory_busy[tile] += service
        self.memory_wait_total += wait
        self.memory_wait_samples.append(wait)
        self.push(finish, callback, *cb_data)

    def _schedule_compute(self, ready_s: float, tile: int, flops: float, effective_flops_s: float, callback: str, *cb_data: Any) -> None:
        if flops <= 0:
            self.push(ready_s, callback, *cb_data)
            return
        start = max(ready_s, self.compute_available[tile])
        wait = start - ready_s
        service = flops / effective_flops_s
        finish = start + service
        self.compute_available[tile] = finish
        self.compute_busy[tile] += service
        self.compute_wait_total += wait
        self.compute_wait_samples.append(wait)
        self.push(finish, callback, *cb_data)

    def _schedule_work(
        self,
        ready_s: float,
        tile: int,
        memory_bytes: float,
        flops: float,
        effective_flops_s: float,
        callback: str,
        *cb_data: Any,
    ) -> None:
        self.work_seq += 1
        work_id = self.work_seq
        self.work_barriers[work_id] = _WorkBarrier(2, callback, tuple(cb_data))
        self._schedule_memory(ready_s, tile, memory_bytes, "work_part_done", work_id)
        self._schedule_compute(ready_s, tile, flops, effective_flops_s, "work_part_done", work_id)

    def _work_part_done(self, time_s: float, work_id: int) -> None:
        barrier = self.work_barriers[work_id]
        barrier.remaining -= 1
        barrier.last_done_s = max(barrier.last_done_s, time_s)
        if barrier.remaining == 0:
            del self.work_barriers[work_id]
            self.push(barrier.last_done_s, barrier.callback, *barrier.callback_data)

    def _state_traffic_and_flops(self, layer_index: int) -> tuple[float, float]:
        if layer_index in self.state_spec.full_attention_layer_set:
            entry = _mla_entry_bytes(self.state_spec, self.kv_bits, self.kv_mode)
            # Read all previous cache entries and write the new one.
            traffic = entry * (self.context_length + 1)
            flops = _attention_context_flops(self.state_spec, self.context_length, self.kv_mode)
            return traffic, flops
        state = _kda_state_bytes(self.state_spec, self.kda_state_bits)
        # Recurrent state is read and written every decode step.
        return 2.0 * state, _kda_recurrent_flops(self.state_spec)

    def _begin_layer(self, time_s: float, token: int, layer_index: int) -> None:
        layer = self.layers[layer_index]
        anchor = self.layer_anchor[layer_index]
        state_bytes, state_flops = self._state_traffic_and_flops(layer_index)
        other_bytes = _bytes(layer.other_always_on_parameters, self.other_bits, self.overhead)
        other_flops = 2.0 * layer.other_always_on_parameters + state_flops
        self._schedule_work(
            time_s,
            anchor,
            other_bytes + state_bytes,
            other_flops,
            self.other_tops,
            "other_done",
            token,
            layer_index,
        )

    def _other_done(self, time_s: float, token: int, layer_index: int) -> None:
        layer = self.layers[layer_index]
        if not layer.is_moe or not self.model.num_experts:
            self.push(time_s, "layer_done", token, layer_index)
            return

        selected = self._selected(token, layer_index - self.model.dense_layers)
        shared_owners = self._shared_owners(layer_index) if layer.shared_parameters else ()
        key = (token, layer_index)
        self.moe_barriers[key] = _MoeBarrier(len(selected) + len(shared_owners))

        if shared_owners:
            shared_bytes = _bytes(layer.shared_parameters, self.shared_bits, self.overhead) / len(shared_owners)
            shared_flops = 2.0 * layer.shared_parameters / len(shared_owners)
            anchor = self.layer_anchor[layer_index]
            for owner in shared_owners:
                if owner == anchor:
                    self._schedule_work(
                        time_s, owner, shared_bytes, shared_flops, self.shared_tops,
                        "branch_done", token, layer_index,
                    )
                else:
                    self._send_packet(
                        time_s, anchor, owner, self.shared_message_bytes,
                        "shared_arrive", token, layer_index, owner, anchor,
                        shared_bytes, shared_flops,
                    )

        expert_params = layer.active_routed_parameters / max(1, len(selected))
        expert_bytes = _bytes(expert_params, self.bits, self.overhead)
        expert_flops = 2.0 * expert_params
        cluster = self.clusters[self.layer_cluster[layer_index]]
        moe_layer = layer_index - self.model.dense_layers
        anchor = self.layer_anchor[layer_index]
        for expert in selected:
            slot = (expert + moe_layer * self.expert_stride) % len(cluster)
            expert_tile = cluster[slot]
            if expert_tile == anchor:
                self._schedule_work(
                    time_s, expert_tile, expert_bytes, expert_flops, self.routed_tops,
                    "branch_done", token, layer_index,
                )
            else:
                self._send_packet(
                    time_s, anchor, expert_tile, self.routed_message_bytes,
                    "expert_arrive_m3", token, layer_index, expert_tile, anchor,
                    expert_bytes, expert_flops,
                )

    def _shared_arrive(
        self,
        time_s: float,
        token: int,
        layer_index: int,
        tile: int,
        anchor: int,
        memory_bytes: float,
        flops: float,
    ) -> None:
        self._schedule_work(
            time_s, tile, memory_bytes, flops, self.shared_tops,
            "shared_work_done", token, layer_index, tile, anchor,
        )

    def _shared_work_done(self, time_s: float, token: int, layer_index: int, tile: int, anchor: int) -> None:
        self._send_packet(
            time_s, tile, anchor, self.shared_message_bytes,
            "branch_done", token, layer_index,
        )

    def _expert_arrive_m3(
        self,
        time_s: float,
        token: int,
        layer_index: int,
        tile: int,
        anchor: int,
        memory_bytes: float,
        flops: float,
    ) -> None:
        self._schedule_work(
            time_s, tile, memory_bytes, flops, self.routed_tops,
            "expert_work_done_m3", token, layer_index, tile, anchor,
        )

    def _expert_work_done_m3(self, time_s: float, token: int, layer_index: int, tile: int, anchor: int) -> None:
        self._send_packet(
            time_s, tile, anchor, self.routed_message_bytes,
            "branch_done", token, layer_index,
        )

    def _branch_done(self, time_s: float, token: int, layer_index: int) -> None:
        key = (token, layer_index)
        barrier = self.moe_barriers[key]
        if barrier.first_done_s is None:
            barrier.first_done_s = time_s
        barrier.last_done_s = time_s
        barrier.remaining -= 1
        if barrier.remaining == 0:
            if barrier.first_done_s is not None and barrier.last_done_s is not None:
                self.moe_barrier_tails.append(barrier.last_done_s - barrier.first_done_s)
            del self.moe_barriers[key]
            self.push(time_s, "layer_done", token, layer_index)

    def run_m3(self) -> M3RunReport:
        if not self.resident_m3:
            return M3RunReport(
                self.cluster_size, self.link_bandwidth_gb_s, self.shared_shards,
                self.concurrency, self.context_length, self.kv_mode, False,
                self.weight_storage_bytes, self.cache_bytes_per_sequence,
                self.cache_bytes_per_sequence * self.concurrency,
                max(self.total_storage_by_tile, default=0.0), self.tile_capacity_bytes,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                self.routed_message_bytes, self.shared_message_bytes, 0,
            )

        for token, issue_time in enumerate(self.token_issue):
            self.push(issue_time, "begin_layer", token, 0)

        while self.events:
            time_s, _seq, kind, data = heapq.heappop(self.events)
            self.events_processed += 1
            if kind == "begin_layer":
                self._begin_layer(time_s, *data)
            elif kind == "other_done":
                self._other_done(time_s, *data)
            elif kind == "work_part_done":
                self._work_part_done(time_s, *data)
            elif kind == "shared_arrive":
                self._shared_arrive(time_s, *data)
            elif kind == "shared_work_done":
                self._shared_work_done(time_s, *data)
            elif kind == "expert_arrive_m3":
                self._expert_arrive_m3(time_s, *data)
            elif kind == "expert_work_done_m3":
                self._expert_work_done_m3(time_s, *data)
            elif kind == "branch_done":
                self._branch_done(time_s, *data)
            elif kind == "layer_done":
                self._layer_done(time_s, *data)
            elif kind == "packet_hop":
                self._handle_packet_hop(time_s, *data)
            else:
                raise RuntimeError(f"unknown M3 event kind {kind!r}")

        if any(value is None for value in self.token_done):
            raise RuntimeError("M3 ended before every token completed")
        completions = [float(value) for value in self.token_done if value is not None]
        latencies = [done - issue for done, issue in zip(completions, self.token_issue)]
        start = min(self.token_issue, default=0.0)
        makespan = max(completions, default=0.0) - start
        throughput = self.concurrency / makespan if makespan > 0 else 0.0

        link_util = max(self.link_busy.values(), default=0.0) / makespan if makespan else 0.0
        mem_util = max(self.memory_busy, default=0.0) / makespan if makespan else 0.0
        compute_util = max(self.compute_busy, default=0.0) / makespan if makespan else 0.0

        return M3RunReport(
            cluster_size=self.cluster_size,
            link_bandwidth_gb_s=self.link_bandwidth_gb_s,
            shared_shards=self.shared_shards,
            concurrency=self.concurrency,
            context_length=self.context_length,
            kv_mode=self.kv_mode,
            resident=True,
            weight_storage_bytes=self.weight_storage_bytes,
            cache_bytes_per_sequence=self.cache_bytes_per_sequence,
            total_cache_bytes=self.cache_bytes_per_sequence * self.concurrency,
            max_tile_storage_bytes=max(self.total_storage_by_tile, default=0.0),
            tile_capacity_bytes=self.tile_capacity_bytes,
            throughput_tps=throughput,
            p50_latency_s=_percentile(latencies, 0.50),
            p95_latency_s=_percentile(latencies, 0.95),
            max_latency_s=max(latencies, default=0.0),
            hottest_link_utilization=link_util,
            hottest_memory_utilization=mem_util,
            hottest_compute_utilization=compute_util,
            max_link_queue_wait_s=max(self.link_wait_samples, default=0.0),
            max_memory_queue_wait_s=max(self.memory_wait_samples, default=0.0),
            max_compute_queue_wait_s=max(self.compute_wait_samples, default=0.0),
            p95_barrier_tail_s=_percentile(self.moe_barrier_tails, 0.95),
            network_payload_bytes_per_token=self.network_payload_bytes / self.concurrency,
            hop_bytes_per_token=self.hop_bytes / self.concurrency,
            routed_message_bytes=self.routed_message_bytes,
            shared_message_bytes=self.shared_message_bytes,
            event_count=self.events_processed,
        )


def simulate_m3(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_size: int,
    link_bandwidth_gb_s: float,
    shared_shards: int,
    concurrency: int,
    context_length: int,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float = 16.0,
    other_weight_bits: float = 4.0,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    kv_mode: str = "latent",
    kv_bits: float = 16.0,
    kda_state_bits: float = 16.0,
    routed_tops: float = 400.0,
    shared_tops: float = 100.0,
    other_tops: float = 100.0,
    profile: str = "balanced",
    seed: int = 42,
    issue_spacing_s: float = 0.0,
) -> M3RunReport:
    sim = _M3Simulator(
        model,
        hardware,
        cluster_size=cluster_size,
        link_bandwidth_gb_s=link_bandwidth_gb_s,
        bits_per_weight=bits_per_weight,
        shared_expert_bits=shared_expert_bits,
        other_weight_bits=other_weight_bits,
        overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
        concurrency=concurrency,
        context_length=context_length,
        kv_mode=kv_mode,
        kv_bits=kv_bits,
        kda_state_bits=kda_state_bits,
        shared_shards=shared_shards,
        routed_tops=routed_tops,
        shared_tops=shared_tops,
        other_tops=other_tops,
        profile=profile,
        seed=seed,
        issue_spacing_s=issue_spacing_s,
    )
    return sim.run_m3()


def max_sequences_that_fit(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_size: int,
    shared_shards: int,
    context_length: int,
    bits_per_weight: float,
    shared_expert_bits: float,
    other_weight_bits: float,
    overhead_fraction: float,
    kv_mode: str,
    kv_bits: float,
    kda_state_bits: float,
) -> tuple[int, float, float]:
    # Build a one-sequence simulator solely to obtain the physical storage map.
    sim = _M3Simulator(
        model,
        hardware,
        cluster_size=cluster_size,
        link_bandwidth_gb_s=1e9,
        bits_per_weight=bits_per_weight,
        shared_expert_bits=shared_expert_bits,
        other_weight_bits=other_weight_bits,
        overhead_fraction=overhead_fraction,
        activation_bits=16.0,
        concurrency=1,
        context_length=context_length,
        kv_mode=kv_mode,
        kv_bits=kv_bits,
        kda_state_bits=kda_state_bits,
        shared_shards=shared_shards,
        routed_tops=1.0,
        shared_tops=1.0,
        other_tops=1.0,
        profile="balanced",
        seed=42,
        issue_spacing_s=0.0,
    )
    max_sequences = math.inf
    for weight, cache in zip(sim.weight_storage_by_tile, sim.cache_per_sequence_by_tile):
        if weight > sim.tile_capacity_bytes:
            return 0, sim.weight_storage_bytes, sim.cache_bytes_per_sequence
        if cache > 0:
            max_sequences = min(max_sequences, math.floor((sim.tile_capacity_bytes - weight) / cache))
    aggregate_cache = sim.cache_bytes_per_sequence
    if aggregate_cache > 0:
        max_sequences = min(
            max_sequences,
            math.floor((hardware.capacity_bytes - sim.weight_storage_bytes) / aggregate_cache),
        )
    if max_sequences is math.inf:
        max_sequences = 0
    return max(0, int(max_sequences)), sim.weight_storage_bytes, sim.cache_bytes_per_sequence


def _parse_candidate(value: str) -> tuple[int, float]:
    try:
        cluster_text, bandwidth_text = value.split(":", 1)
        cluster = int(cluster_text)
        bandwidth = float(bandwidth_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("candidate must look like CLUSTER:GB/s, e.g. 16:128") from exc
    if cluster <= 0 or bandwidth <= 0:
        raise argparse.ArgumentTypeError("cluster and bandwidth must be positive")
    return cluster, bandwidth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m3",
        description="M3 compute/KV/shared-expert event simulator for Kimi K3.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--hardware", default="fabric-64x32")
    parser.add_argument("--candidate", type=_parse_candidate, action="append", dest="candidates")
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY))
    parser.add_argument("--context", type=int, default=16_384)
    parser.add_argument("--context-sweep", type=int, nargs="+", default=list(DEFAULT_CONTEXT_SWEEP))
    parser.add_argument("--shared-shards", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=16.0)
    parser.add_argument("--other-bits", type=float, default=4.0)
    parser.add_argument("--kv-mode", choices=("latent", "expanded"), default="latent")
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--kda-state-bits", type=float, default=16.0)
    parser.add_argument("--routed-tops", type=float, default=400.0)
    parser.add_argument("--shared-tops", type=float, default=100.0)
    parser.add_argument("--other-tops", type=float, default=100.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--activation-bits", type=float, default=16.0)
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-spacing-us", type=float, default=0.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    _state_spec(model)
    candidates = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES
    concurrency = tuple(sorted(set(args.concurrency)))
    shards = tuple(sorted(set(args.shared_shards)))
    if not concurrency or any(value <= 0 for value in concurrency):
        raise ValueError("concurrency values must be positive")
    if args.context < 0 or any(value < 0 for value in args.context_sweep):
        raise ValueError("context lengths cannot be negative")

    print(f"{model.name} -> {hardware.name}")
    print("  M3 model:               cut-through NoC + FIFO DRAM/compute + streamed memory/compute overlap")
    print(f"  context / KV mode:      {args.context:,} / {args.kv_mode}")
    print(f"  precision:              routed {args.bits:g}b, shared {args.shared_bits:g}b, other {args.other_bits:g}b, KV {args.kv_bits:g}b")
    print(f"  effective compute:      routed {args.routed_tops:g}, shared {args.shared_tops:g}, other {args.other_tops:g} TOPS/tile")
    print(f"  routing profile:        {args.profile}")
    print("  candidates:             " + ", ".join(f"C{c}@{bw:g}GB/s" for c, bw in candidates))
    print("  shared split:           sharding the single wide shared MLP along its intermediate dimension")

    print("\nM3 EXECUTION COMPARISON")
    print(
        "CLUSTER LINK SHARED CONC FIT   THROUGHPUT   P50 LAT    P95 LAT    LINK   MEM    COMP   "
        "MAX TILE    CACHE/SEQ   P95 BARRIER"
    )
    reports: list[M3RunReport] = []
    for cluster_size, bandwidth in candidates:
        for shared_shards in shards:
            if shared_shards > cluster_size:
                continue
            for conc in concurrency:
                report = simulate_m3(
                    model,
                    hardware,
                    cluster_size=cluster_size,
                    link_bandwidth_gb_s=bandwidth,
                    shared_shards=shared_shards,
                    concurrency=conc,
                    context_length=args.context,
                    bits_per_weight=args.bits,
                    shared_expert_bits=args.shared_bits,
                    other_weight_bits=args.other_bits,
                    overhead_fraction=args.overhead,
                    activation_bits=args.activation_bits,
                    kv_mode=args.kv_mode,
                    kv_bits=args.kv_bits,
                    kda_state_bits=args.kda_state_bits,
                    routed_tops=args.routed_tops,
                    shared_tops=args.shared_tops,
                    other_tops=args.other_tops,
                    profile=args.profile,
                    seed=args.seed,
                    issue_spacing_s=args.issue_spacing_us * 1e-6,
                )
                reports.append(report)
                if not report.resident:
                    print(
                        f"C{cluster_size:<6} {bandwidth:>4g}G {shared_shards:>6} {conc:>4}  NO   "
                        f"—            —          —          —      —      —      {fmt_bytes(report.max_tile_storage_bytes):>10}  "
                        f"{fmt_bytes(report.cache_bytes_per_sequence):>10}  —"
                    )
                    continue
                print(
                    f"C{cluster_size:<6} {bandwidth:>4g}G {shared_shards:>6} {conc:>4}  yes  "
                    f"{fmt_rate(report.throughput_tps):>11}  "
                    f"{fmt_time_s(report.p50_latency_s):>9}  {fmt_time_s(report.p95_latency_s):>9}  "
                    f"{report.hottest_link_utilization:>5.1%}  {report.hottest_memory_utilization:>5.1%}  "
                    f"{report.hottest_compute_utilization:>5.1%}  "
                    f"{fmt_bytes(report.max_tile_storage_bytes):>10}  {fmt_bytes(report.cache_bytes_per_sequence):>10}  "
                    f"{fmt_time_s(report.p95_barrier_tail_s):>11}"
                )

    if reports:
        first = reports[0]
        print("\nMESSAGE SIZE CORRECTION")
        print(f"  routed Latent-MoE activation: {fmt_bytes(first.routed_message_bytes)} / message")
        print(f"  shared full-hidden activation: {fmt_bytes(first.shared_message_bytes)} / message")

    print("\nKV CAPACITY SWEEP (MAX SAME-CONTEXT SEQUENCES THAT FIT)")
    header = "CONTEXT" + "".join(f"  C{c}/S{s}".rjust(12) for c, _ in candidates for s in shards if s <= c)
    print(header)
    for context in args.context_sweep:
        values: list[str] = []
        for cluster_size, _bandwidth in candidates:
            for shared_shards in shards:
                if shared_shards > cluster_size:
                    continue
                max_seq, _weights, cache_seq = max_sequences_that_fit(
                    model,
                    hardware,
                    cluster_size=cluster_size,
                    shared_shards=shared_shards,
                    context_length=context,
                    bits_per_weight=args.bits,
                    shared_expert_bits=args.shared_bits,
                    other_weight_bits=args.other_bits,
                    overhead_fraction=args.overhead,
                    kv_mode=args.kv_mode,
                    kv_bits=args.kv_bits,
                    kda_state_bits=args.kda_state_bits,
                )
                values.append(f"{max_seq:>6}({fmt_bytes(cache_seq)})")
        print(f"{context:>7,}" + "".join(value.rjust(12) for value in values))

    print("\nASSUMPTIONS / BOUNDS")
    print("  - M3 latent KV mode assumes an optimized absorbed-MLA cache (512 latent + 64 RoPE values per full-attention layer/token).")
    print("  - KDA capacity uses a fixed recurrent matrix plus q/k/v short-convolution state; state traffic is modeled as one read + one write per decode step.")
    print("  - Memory and tensor work overlap ideally within an operation; compute TOPS are user-supplied effective rates, not a Raptor/d-Matrix claim.")
    print("  - Shared sharding tensor-parallelizes Kimi's single wide shared MLP; helper shards return a full-hidden partial output for reduction.")
    print("  - KV placement stays with each layer anchor. Compute, DRAM and NoC queues are causal, but flit credits/backpressure, bank conflicts and thermals remain excluded.")
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
