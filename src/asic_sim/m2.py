from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import math
import random
from statistics import mean
from typing import Any

from .architectures import _bytes, _layer_active_components, estimate_layers
from .bandwidth_sweep import _cluster_tiles, _coprime_stride, analyze_cluster_topology
from .fabric import _profile_cumulative, _selected_experts, xy_route
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .hardware import HardwareSpec, get_hardware
from .models import ModelSpec, get_model


DEFAULT_CANDIDATES = ((4, 32.0), (16, 64.0))
DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True, slots=True)
class M2RunReport:
    cluster_size: int
    link_bandwidth_gb_s: float
    concurrency: int
    resident: bool
    makespan_s: float
    throughput_tps: float
    p50_latency_s: float
    p95_latency_s: float
    max_latency_s: float
    network_payload_bytes_per_token: float
    hop_bytes_per_token: float
    packet_count_per_token: float
    mean_packet_latency_s: float
    p95_packet_latency_s: float
    mean_link_queue_wait_s: float
    max_link_queue_wait_s: float
    hottest_link: tuple[int, int] | None
    hottest_link_utilization: float
    hottest_memory_tile: int | None
    hottest_memory_utilization: float
    mean_barrier_tail_s: float
    p95_barrier_tail_s: float
    link_queue_wait_s_total: float
    memory_queue_wait_s_total: float
    event_count: int


@dataclass(frozen=True, slots=True)
class M2ComparisonRow:
    report: M2RunReport
    baseline_throughput_tps: float
    retained_baseline_throughput: float
    throughput_loss_fraction: float


@dataclass(slots=True)
class _Barrier:
    remaining: int
    first_done_s: float | None = None
    last_done_s: float | None = None


class _M2Simulator:
    """Small deterministic event simulator for M1-selected cluster topologies.

    Network packets use a conservative store-and-forward model. Each directed
    link is a FIFO serialization resource. Each tile has one local-memory
    service resource at the configured 3D-DRAM bandwidth. The model includes
    causal layer dependencies, top-k fan-out/fan-in barriers, shared experts,
    memory contention and link contention. Tensor compute, KV traffic, packet
    headers/flits, adaptive routing, virtual channels and backpressure are not
    yet modeled.
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
        overhead_fraction: float,
        activation_bits: float,
        concurrency: int,
        profile: str,
        seed: int,
        issue_spacing_s: float,
    ) -> None:
        if cluster_size <= 0 or cluster_size > hardware.tiles or hardware.tiles % cluster_size:
            raise ValueError("cluster_size must divide the hardware tile count")
        if link_bandwidth_gb_s <= 0:
            raise ValueError("link_bandwidth_gb_s must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if issue_spacing_s < 0:
            raise ValueError("issue_spacing_s cannot be negative")

        self.model = model
        self.hardware = hardware
        self.cluster_size = cluster_size
        self.link_bandwidth_bytes_s = link_bandwidth_gb_s * 1e9
        self.link_bandwidth_gb_s = link_bandwidth_gb_s
        self.bits = bits_per_weight
        self.shared_bits = shared_expert_bits
        self.overhead = overhead_fraction
        self.activation_bits = activation_bits
        self.concurrency = concurrency
        self.profile = profile
        self.seed = seed
        self.issue_spacing_s = issue_spacing_s

        self.layers = estimate_layers(model)
        self.clusters = _cluster_tiles(hardware, cluster_size)
        self.cluster_count = len(self.clusters)
        self.layer_cluster = tuple(
            min(self.cluster_count - 1, (layer * self.cluster_count) // len(self.layers))
            for layer in range(len(self.layers))
        )
        local_layer_index = [0] * self.cluster_count
        anchors: list[int] = []
        for cluster_index in self.layer_cluster:
            cluster = self.clusters[cluster_index]
            local_index = local_layer_index[cluster_index]
            anchors.append(cluster[local_index % len(cluster)])
            local_layer_index[cluster_index] += 1
        self.layer_anchor = tuple(anchors)
        self.expert_stride = _coprime_stride(cluster_size)
        self.hidden_bytes = model.hidden_bytes(activation_bits)
        self.router_latency_s = hardware.router_latency_ns * 1e-9
        self.memory_bandwidth_bytes_s = hardware.local_memory_bandwidth_bytes_s
        self.profile_cumulative = _profile_cumulative(model, profile) if model.num_experts else None

        self.events: list[tuple[float, int, str, tuple[Any, ...]]] = []
        self.event_seq = 0
        self.events_processed = 0
        self.memory_available = [0.0] * hardware.tiles
        self.memory_busy = [0.0] * hardware.tiles
        self.memory_wait_total = 0.0
        self.link_available: dict[tuple[int, int], float] = {}
        self.link_busy: dict[tuple[int, int], float] = {}
        self.link_wait_total = 0.0
        self.link_wait_samples: list[float] = []
        self.packet_latencies: list[float] = []
        self.packet_count = 0
        self.network_payload_bytes = 0.0
        self.hop_bytes = 0.0
        self.token_issue = [token * issue_spacing_s for token in range(concurrency)]
        self.token_done: list[float | None] = [None] * concurrency
        self.barriers: dict[tuple[int, int], _Barrier] = {}
        self.barrier_tails: list[float] = []

    def push(self, time_s: float, kind: str, *data: Any) -> None:
        self.event_seq += 1
        heapq.heappush(self.events, (time_s, self.event_seq, kind, data))

    def _selected(self, token: int, moe_layer: int) -> tuple[int, ...]:
        if self.profile == "balanced":
            return _selected_experts(
                self.model,
                token,
                moe_layer,
                self.profile,
                self.profile_cumulative,
                random.Random(0),
            )
        # Keep routing choices independent of event ordering/concurrency.
        rng = random.Random(self.seed + token * 1_000_003 + moe_layer * 9_176)
        return _selected_experts(
            self.model,
            token,
            moe_layer,
            self.profile,
            self.profile_cumulative,
            rng,
        )

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
        self.push(finish, callback, *cb_data)

    def _send_packet(self, ready_s: float, source: int, destination: int, payload_bytes: float, callback: str, *cb_data: Any) -> None:
        if source == destination or payload_bytes <= 0:
            self.push(ready_s, callback, *cb_data)
            return
        path = xy_route(self.hardware, source, destination)
        packet_id = self.packet_count
        self.packet_count += 1
        self.network_payload_bytes += payload_bytes
        self.hop_bytes += payload_bytes * (len(path) - 1)
        self.push(ready_s, "packet_hop", packet_id, ready_s, path, 0, payload_bytes, callback, cb_data)

    def _handle_packet_hop(
        self,
        time_s: float,
        packet_id: int,
        packet_start_s: float,
        path: tuple[int, ...],
        hop_index: int,
        payload_bytes: float,
        callback: str,
        cb_data: tuple[Any, ...],
    ) -> None:
        del packet_id
        if hop_index >= len(path) - 1:
            self.packet_latencies.append(time_s - packet_start_s)
            self.push(time_s, callback, *cb_data)
            return

        link = (path[hop_index], path[hop_index + 1])
        start = max(time_s, self.link_available.get(link, 0.0))
        wait = start - time_s
        serialization = payload_bytes / self.link_bandwidth_bytes_s
        link_finish = start + serialization
        arrival = link_finish + self.router_latency_s
        self.link_available[link] = link_finish
        self.link_busy[link] = self.link_busy.get(link, 0.0) + serialization
        self.link_wait_total += wait
        self.link_wait_samples.append(wait)
        self.push(
            arrival,
            "packet_hop",
            -1,
            packet_start_s,
            path,
            hop_index + 1,
            payload_bytes,
            callback,
            cb_data,
        )

    def _begin_layer(self, time_s: float, token: int, layer_index: int) -> None:
        layer = self.layers[layer_index]
        anchor = self.layer_anchor[layer_index]
        _, _, other_bytes = _layer_active_components(layer, self.bits, self.shared_bits, self.overhead)
        self._schedule_memory(time_s, anchor, other_bytes, "other_done", token, layer_index)

    def _other_done(self, time_s: float, token: int, layer_index: int) -> None:
        layer = self.layers[layer_index]
        anchor = self.layer_anchor[layer_index]
        routed_bytes, shared_bytes, _ = _layer_active_components(layer, self.bits, self.shared_bits, self.overhead)
        if not layer.is_moe or not self.model.num_experts:
            self.push(time_s, "layer_done", token, layer_index)
            return

        selected = self._selected(token, layer_index - self.model.dense_layers)
        barrier_key = (token, layer_index)
        self.barriers[barrier_key] = _Barrier(remaining=len(selected) + (1 if shared_bytes > 0 else 0))

        if shared_bytes > 0:
            self._schedule_memory(time_s, anchor, shared_bytes, "branch_done", token, layer_index)

        expert_bytes = routed_bytes / max(1, len(selected))
        cluster = self.clusters[self.layer_cluster[layer_index]]
        moe_layer = layer_index - self.model.dense_layers
        for expert in selected:
            slot = (expert + moe_layer * self.expert_stride) % len(cluster)
            expert_tile = cluster[slot]
            if expert_tile == anchor:
                self._schedule_memory(time_s, expert_tile, expert_bytes, "branch_done", token, layer_index)
            else:
                self._send_packet(
                    time_s,
                    anchor,
                    expert_tile,
                    self.hidden_bytes,
                    "expert_arrive",
                    token,
                    layer_index,
                    expert_tile,
                    anchor,
                    expert_bytes,
                )

    def _expert_arrive(
        self,
        time_s: float,
        token: int,
        layer_index: int,
        expert_tile: int,
        anchor: int,
        expert_bytes: float,
    ) -> None:
        self._schedule_memory(
            time_s,
            expert_tile,
            expert_bytes,
            "expert_memory_done",
            token,
            layer_index,
            expert_tile,
            anchor,
        )

    def _expert_memory_done(
        self,
        time_s: float,
        token: int,
        layer_index: int,
        expert_tile: int,
        anchor: int,
    ) -> None:
        self._send_packet(
            time_s,
            expert_tile,
            anchor,
            self.hidden_bytes,
            "branch_done",
            token,
            layer_index,
        )

    def _branch_done(self, time_s: float, token: int, layer_index: int) -> None:
        key = (token, layer_index)
        barrier = self.barriers[key]
        if barrier.first_done_s is None:
            barrier.first_done_s = time_s
        barrier.last_done_s = time_s
        barrier.remaining -= 1
        if barrier.remaining == 0:
            if barrier.first_done_s is not None and barrier.last_done_s is not None:
                self.barrier_tails.append(barrier.last_done_s - barrier.first_done_s)
            del self.barriers[key]
            self.push(time_s, "layer_done", token, layer_index)

    def _layer_done(self, time_s: float, token: int, layer_index: int) -> None:
        if layer_index + 1 >= len(self.layers):
            self.token_done[token] = time_s
            return
        current_anchor = self.layer_anchor[layer_index]
        next_anchor = self.layer_anchor[layer_index + 1]
        if current_anchor == next_anchor:
            self.push(time_s, "begin_layer", token, layer_index + 1)
        else:
            self._send_packet(
                time_s,
                current_anchor,
                next_anchor,
                self.hidden_bytes,
                "begin_layer",
                token,
                layer_index + 1,
            )

    def run(self) -> M2RunReport:
        topology = analyze_cluster_topology(
            self.model,
            self.hardware,
            cluster_size=self.cluster_size,
            bits_per_weight=self.bits,
            shared_expert_bits=self.shared_bits,
            overhead_fraction=self.overhead,
            activation_bits=self.activation_bits,
            tokens=min(16, self.concurrency),
            profile=self.profile,
            seed=self.seed,
        )
        if not topology.resident:
            return M2RunReport(
                cluster_size=self.cluster_size,
                link_bandwidth_gb_s=self.link_bandwidth_gb_s,
                concurrency=self.concurrency,
                resident=False,
                makespan_s=0.0,
                throughput_tps=0.0,
                p50_latency_s=0.0,
                p95_latency_s=0.0,
                max_latency_s=0.0,
                network_payload_bytes_per_token=0.0,
                hop_bytes_per_token=0.0,
                packet_count_per_token=0.0,
                mean_packet_latency_s=0.0,
                p95_packet_latency_s=0.0,
                mean_link_queue_wait_s=0.0,
                max_link_queue_wait_s=0.0,
                hottest_link=None,
                hottest_link_utilization=0.0,
                hottest_memory_tile=None,
                hottest_memory_utilization=0.0,
                mean_barrier_tail_s=0.0,
                p95_barrier_tail_s=0.0,
                link_queue_wait_s_total=0.0,
                memory_queue_wait_s_total=0.0,
                event_count=0,
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
            elif kind == "expert_arrive":
                self._expert_arrive(time_s, *data)
            elif kind == "expert_memory_done":
                self._expert_memory_done(time_s, *data)
            elif kind == "branch_done":
                self._branch_done(time_s, *data)
            elif kind == "layer_done":
                self._layer_done(time_s, *data)
            elif kind == "packet_hop":
                self._handle_packet_hop(time_s, *data)
            else:
                raise RuntimeError(f"unknown event kind {kind!r}")

        if any(value is None for value in self.token_done):
            raise RuntimeError("simulation ended before every token completed")
        completions = [float(value) for value in self.token_done if value is not None]
        latencies = [done - issue for done, issue in zip(completions, self.token_issue)]
        start = min(self.token_issue, default=0.0)
        makespan = max(completions, default=0.0) - start
        throughput = self.concurrency / makespan if makespan > 0 else 0.0

        hottest_link = None
        hottest_link_util = 0.0
        if makespan > 0 and self.link_busy:
            hottest_link, busiest = max(self.link_busy.items(), key=lambda item: item[1])
            hottest_link_util = busiest / makespan

        hottest_memory_tile = None
        hottest_memory_util = 0.0
        if makespan > 0 and self.memory_busy:
            hottest_memory_tile = max(range(len(self.memory_busy)), key=lambda tile: self.memory_busy[tile])
            hottest_memory_util = self.memory_busy[hottest_memory_tile] / makespan

        return M2RunReport(
            cluster_size=self.cluster_size,
            link_bandwidth_gb_s=self.link_bandwidth_gb_s,
            concurrency=self.concurrency,
            resident=True,
            makespan_s=makespan,
            throughput_tps=throughput,
            p50_latency_s=_percentile(latencies, 0.50),
            p95_latency_s=_percentile(latencies, 0.95),
            max_latency_s=max(latencies, default=0.0),
            network_payload_bytes_per_token=self.network_payload_bytes / self.concurrency,
            hop_bytes_per_token=self.hop_bytes / self.concurrency,
            packet_count_per_token=self.packet_count / self.concurrency,
            mean_packet_latency_s=mean(self.packet_latencies) if self.packet_latencies else 0.0,
            p95_packet_latency_s=_percentile(self.packet_latencies, 0.95),
            mean_link_queue_wait_s=mean(self.link_wait_samples) if self.link_wait_samples else 0.0,
            max_link_queue_wait_s=max(self.link_wait_samples, default=0.0),
            hottest_link=hottest_link,
            hottest_link_utilization=hottest_link_util,
            hottest_memory_tile=hottest_memory_tile,
            hottest_memory_utilization=hottest_memory_util,
            mean_barrier_tail_s=mean(self.barrier_tails) if self.barrier_tails else 0.0,
            p95_barrier_tail_s=_percentile(self.barrier_tails, 0.95),
            link_queue_wait_s_total=self.link_wait_total,
            memory_queue_wait_s_total=self.memory_wait_total,
            event_count=self.events_processed,
        )


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


def simulate_m2(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_size: int,
    link_bandwidth_gb_s: float,
    concurrency: int,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    profile: str = "balanced",
    seed: int = 42,
    issue_spacing_s: float = 0.0,
) -> M2RunReport:
    shared_bits = bits_per_weight if shared_expert_bits is None else shared_expert_bits
    simulator = _M2Simulator(
        model,
        hardware,
        cluster_size=cluster_size,
        link_bandwidth_gb_s=link_bandwidth_gb_s,
        bits_per_weight=bits_per_weight,
        shared_expert_bits=shared_bits,
        overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
        concurrency=concurrency,
        profile=profile,
        seed=seed,
        issue_spacing_s=issue_spacing_s,
    )
    return simulator.run()


def compare_m2_candidates(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    candidates: tuple[tuple[int, float], ...] = DEFAULT_CANDIDATES,
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    profile: str = "balanced",
    seed: int = 42,
    issue_spacing_s: float = 0.0,
) -> tuple[M2ComparisonRow, ...]:
    rows: list[M2ComparisonRow] = []
    for cluster_size, bandwidth in candidates:
        for concurrency in concurrency_levels:
            actual = simulate_m2(
                model,
                hardware,
                cluster_size=cluster_size,
                link_bandwidth_gb_s=bandwidth,
                concurrency=concurrency,
                bits_per_weight=bits_per_weight,
                shared_expert_bits=shared_expert_bits,
                overhead_fraction=overhead_fraction,
                activation_bits=activation_bits,
                profile=profile,
                seed=seed,
                issue_spacing_s=issue_spacing_s,
            )
            if not actual.resident:
                rows.append(M2ComparisonRow(actual, 0.0, 0.0, 1.0))
                continue
            baseline = simulate_m2(
                model,
                hardware,
                cluster_size=cluster_size,
                link_bandwidth_gb_s=1e9,
                concurrency=concurrency,
                bits_per_weight=bits_per_weight,
                shared_expert_bits=shared_expert_bits,
                overhead_fraction=overhead_fraction,
                activation_bits=activation_bits,
                profile=profile,
                seed=seed,
                issue_spacing_s=issue_spacing_s,
            )
            retained = actual.throughput_tps / baseline.throughput_tps if baseline.throughput_tps else 0.0
            rows.append(
                M2ComparisonRow(
                    report=actual,
                    baseline_throughput_tps=baseline.throughput_tps,
                    retained_baseline_throughput=retained,
                    throughput_loss_fraction=max(0.0, 1.0 - retained),
                )
            )
    return tuple(rows)


def _parse_candidate(value: str) -> tuple[int, float]:
    try:
        cluster_text, bandwidth_text = value.split(":", 1)
        cluster = int(cluster_text)
        bandwidth = float(bandwidth_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("candidate must look like CLUSTER:GB/s, e.g. 4:32") from exc
    if cluster <= 0 or bandwidth <= 0:
        raise argparse.ArgumentTypeError("cluster and bandwidth must be positive")
    return cluster, bandwidth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-m2",
        description="Event-driven M2 memory/NoC contention simulator for clustered MoE mappings.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--hardware", default="fabric-64x32")
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=16.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--activation-bits", type=float, default=16.0)
    parser.add_argument("--candidate", type=_parse_candidate, action="append", dest="candidates")
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_CONCURRENCY))
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--issue-spacing-us", type=float, default=0.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    candidates = tuple(args.candidates) if args.candidates else DEFAULT_CANDIDATES
    concurrency = tuple(sorted(set(args.concurrency)))
    if any(value <= 0 for value in concurrency):
        raise ValueError("concurrency values must be positive")
    if args.issue_spacing_us < 0:
        raise ValueError("issue spacing cannot be negative")

    rows = compare_m2_candidates(
        model,
        hardware,
        candidates=candidates,
        concurrency_levels=concurrency,
        bits_per_weight=args.bits,
        shared_expert_bits=args.shared_bits,
        overhead_fraction=args.overhead,
        activation_bits=args.activation_bits,
        profile=args.profile,
        seed=args.seed,
        issue_spacing_s=args.issue_spacing_us * 1e-6,
    )

    print(f"{model.name} -> {hardware.name}")
    print(f"  M2 model:               causal layer dependencies + FIFO tile-memory/link queues")
    print(f"  packet model:           conservative store-and-forward XY routing")
    print(f"  base/shared precision:  {args.bits:g} / {args.shared_bits:g} bit")
    print(f"  routing profile:        {args.profile}")
    print(f"  issue spacing:          {args.issue_spacing_us:g} us ({'burst' if args.issue_spacing_us == 0 else 'staggered'})")
    print("  candidates:             " + ", ".join(f"C{cluster}@{bandwidth:g}GB/s" for cluster, bandwidth in candidates))
    print("  baseline:               same causal simulation with effectively infinite link bandwidth")

    print("\nM2 CONTENTION SWEEP")
    print("CLUSTER LINK     CONC  FIT   THROUGHPUT   RETAIN  P50 LAT    P95 LAT    LINK UTIL  MEM UTIL  P95 PKT    P95 BARRIER")
    for row in rows:
        r = row.report
        if not r.resident:
            print(f"C{r.cluster_size:<6} {r.link_bandwidth_gb_s:>5g}G {r.concurrency:>6}   NO   —")
            continue
        print(
            f"C{r.cluster_size:<6} {r.link_bandwidth_gb_s:>5g}G {r.concurrency:>6}  yes  "
            f"{fmt_rate(r.throughput_tps):>11}  "
            f"{row.retained_baseline_throughput:>6.1%}  "
            f"{fmt_time_s(r.p50_latency_s):>9}  "
            f"{fmt_time_s(r.p95_latency_s):>9}  "
            f"{r.hottest_link_utilization:>8.1%}  "
            f"{r.hottest_memory_utilization:>7.1%}  "
            f"{fmt_time_s(r.p95_packet_latency_s):>9}  "
            f"{fmt_time_s(r.p95_barrier_tail_s):>11}"
        )

    print("\nRESOURCE DETAIL AT HIGHEST CONCURRENCY")
    highest = max(concurrency)
    for row in rows:
        r = row.report
        if r.concurrency != highest or not r.resident:
            continue
        hot_link = "none" if r.hottest_link is None else f"T{r.hottest_link[0]:02d}->T{r.hottest_link[1]:02d}"
        hot_mem = "none" if r.hottest_memory_tile is None else f"T{r.hottest_memory_tile:02d}"
        print(
            f"  C{r.cluster_size}@{r.link_bandwidth_gb_s:g}GB/s: "
            f"hot-link={hot_link} {r.hottest_link_utilization:.1%}; "
            f"hot-memory={hot_mem} {r.hottest_memory_utilization:.1%}; "
            f"NoC={fmt_bytes(r.network_payload_bytes_per_token)}/tok; "
            f"hop-bytes={fmt_bytes(r.hop_bytes_per_token)}/tok; "
            f"max link queue={fmt_time_s(r.max_link_queue_wait_s)}; "
            f"events={r.event_count:,}"
        )

    print("\nINTERPRETATION RULE")
    print("  RETAIN compares each finite-bandwidth run with the same event simulation using effectively infinite links.")
    print("  If RETAIN stays near 100% while memory utilization rises, the proposed link is not the limiting resource in this M2 model.")
    print("  Compute, KV traffic, flit-level backpressure, virtual channels, bank conflicts and thermals remain outside this stage.")
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
