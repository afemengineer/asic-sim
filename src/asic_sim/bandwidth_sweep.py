from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random

from .architectures import (
    _TrafficAccumulator,
    _bytes,
    _layer_active_components,
    _memory_floor,
    estimate_layers,
)
from .fabric import _profile_cumulative, _selected_experts
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .hardware import HardwareSpec, get_hardware
from .models import ModelSpec, get_model


DEFAULT_CLUSTER_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_LINK_BANDWIDTHS_GB_S = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2000)


@dataclass(frozen=True, slots=True)
class ClusterTopologyReport:
    cluster_size: int
    cluster_rows: int
    cluster_cols: int
    resident: bool
    total_storage_bytes: float
    max_tile_storage_bytes: float
    min_tile_storage_bytes: float
    tile_capacity_bytes: float
    network_payload_bytes_per_token: float
    hop_bytes_per_token: float
    max_link_bytes_per_token: float
    hotspot_ratio: float
    active_directed_links: int
    hottest_link: tuple[int, int] | None
    ideal_memory_floor_s: float
    ideal_memory_roof_tps: float

    @property
    def shape_label(self) -> str:
        return f"{self.cluster_rows}x{self.cluster_cols}"


@dataclass(frozen=True, slots=True)
class BandwidthSweepPoint:
    cluster_size: int
    link_bandwidth_gb_s: float
    resident: bool
    hot_link_pressure: float
    hot_link_serialization_s: float
    conservative_total_s: float
    conservative_tps: float
    retained_memory_roof_fraction: float
    adequate: bool


@dataclass(frozen=True, slots=True)
class ClusterBandwidthSummary:
    topology: ClusterTopologyReport
    raw_minimum_bandwidth_gb_s: float | None
    recommended_candidate_bandwidth_gb_s: float | None
    points: tuple[BandwidthSweepPoint, ...]


def _mesh_shape(hardware: HardwareSpec) -> tuple[int, int]:
    if hardware.mesh_rows and hardware.mesh_cols:
        return hardware.mesh_rows, hardware.mesh_cols
    rows = max(1, int(math.sqrt(hardware.tiles)))
    cols = math.ceil(hardware.tiles / rows)
    return rows, cols


def _cluster_shape(hardware: HardwareSpec, cluster_size: int) -> tuple[int, int]:
    """Choose a compact rectangular cluster that tiles the physical mesh."""
    if cluster_size <= 0 or cluster_size > hardware.tiles:
        raise ValueError("cluster size must be between 1 and hardware.tiles")
    if hardware.tiles % cluster_size:
        raise ValueError("cluster size must divide the hardware tile count")

    rows, cols = _mesh_shape(hardware)
    fabric_aspect = rows / cols
    candidates: list[tuple[float, int, int]] = []
    for cluster_rows in range(1, rows + 1):
        if cluster_size % cluster_rows:
            continue
        cluster_cols = cluster_size // cluster_rows
        if cluster_cols > cols:
            continue
        if rows % cluster_rows or cols % cluster_cols:
            continue
        aspect = cluster_rows / cluster_cols
        score = abs(math.log(aspect / fabric_aspect))
        candidates.append((score, cluster_rows, cluster_cols))
    if not candidates:
        raise ValueError(f"cluster size {cluster_size} cannot tile the {_mesh_shape(hardware)} mesh")
    _, cluster_rows, cluster_cols = min(candidates)
    return cluster_rows, cluster_cols


def _cluster_tiles(
    hardware: HardwareSpec,
    cluster_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition the mesh into compact rectangles ordered as a serpentine pipeline."""
    rows, cols = _mesh_shape(hardware)
    cluster_rows, cluster_cols = _cluster_shape(hardware, cluster_size)
    block_rows = rows // cluster_rows
    block_cols = cols // cluster_cols
    clusters: list[tuple[int, ...]] = []

    for block_row in range(block_rows):
        block_col_iter = range(block_cols) if block_row % 2 == 0 else range(block_cols - 1, -1, -1)
        for block_col in block_col_iter:
            top = block_row * cluster_rows
            left = block_col * cluster_cols
            local: list[int] = []
            for local_row in range(cluster_rows):
                cols_iter = range(cluster_cols) if local_row % 2 == 0 else range(cluster_cols - 1, -1, -1)
                for local_col in cols_iter:
                    tile = (top + local_row) * cols + left + local_col
                    if tile >= hardware.tiles:
                        raise ValueError("cluster crosses an unpopulated mesh coordinate")
                    local.append(tile)
            clusters.append(tuple(local))

    if sum(len(cluster) for cluster in clusters) != hardware.tiles:
        raise AssertionError("cluster partition does not cover every physical tile")
    return tuple(clusters)


def _coprime_stride(size: int) -> int:
    if size <= 1:
        return 1
    candidate = max(3, int(math.sqrt(size)) | 1)
    while math.gcd(candidate, size) != 1:
        candidate += 2
    return candidate


def analyze_cluster_topology(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_size: int,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    tokens: int = 64,
    profile: str = "balanced",
    seed: int = 42,
) -> ClusterTopologyReport:
    """Map consecutive layer groups to fixed-size local clusters and trace traffic.

    Always-on/shared weights stay at a layer anchor. Anchors rotate inside the
    cluster across consecutive layers so shared/attention storage is not pinned
    to one physical tile. Routed expert weights are striped across the cluster.
    """
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    shared_bits = bits_per_weight if shared_expert_bits is None else shared_expert_bits
    if shared_bits <= 0:
        raise ValueError("shared_expert_bits must be positive")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")
    if activation_bits <= 0:
        raise ValueError("activation_bits must be positive")
    if tokens <= 0:
        raise ValueError("tokens must be positive")

    clusters = _cluster_tiles(hardware, cluster_size)
    cluster_rows, cluster_cols = _cluster_shape(hardware, cluster_size)
    layers = estimate_layers(model)
    cluster_count = len(clusters)
    tile_capacity = hardware.tile_capacity_gb * 1e9 if hardware.tile_capacity_gb is not None else hardware.capacity_bytes

    # Consecutive model layers are assigned to consecutive physical clusters.
    layer_cluster = tuple(min(cluster_count - 1, (layer * cluster_count) // len(layers)) for layer in range(len(layers)))
    layer_anchor: list[int] = []
    local_layer_index = [0] * cluster_count
    storage = [0.0] * hardware.tiles

    for layer, cluster_idx in zip(layers, layer_cluster):
        cluster = clusters[cluster_idx]
        local_idx = local_layer_index[cluster_idx]
        anchor = cluster[local_idx % len(cluster)]
        local_layer_index[cluster_idx] += 1
        layer_anchor.append(anchor)

        storage[anchor] += _bytes(layer.other_always_on_parameters, bits_per_weight, overhead_fraction)
        storage[anchor] += _bytes(layer.shared_parameters, shared_bits, overhead_fraction)
        if layer.routed_pool_parameters:
            routed_each = _bytes(layer.routed_pool_parameters, bits_per_weight, overhead_fraction) / len(cluster)
            for tile in cluster:
                storage[tile] += routed_each

    total_storage = model.mixed_storage_bytes(
        bits_per_weight,
        overhead_fraction,
        shared_expert_bits=shared_bits,
    )

    acc = _TrafficAccumulator(hardware, {})
    hidden_bytes = model.hidden_bytes(activation_bits)
    rng = random.Random(seed)
    cumulative = _profile_cumulative(model, profile) if model.num_experts else None
    stride = _coprime_stride(cluster_size)

    for token in range(tokens):
        previous_anchor = layer_anchor[0]
        for layer_index, layer in enumerate(layers):
            anchor = layer_anchor[layer_index]
            if layer_index > 0:
                acc.record(previous_anchor, anchor, hidden_bytes)

            if model.num_experts and layer.is_moe:
                moe_layer = layer_index - model.dense_layers
                selected = _selected_experts(model, token, moe_layer, profile, cumulative, rng)
                cluster = clusters[layer_cluster[layer_index]]
                for expert in selected:
                    slot = (expert + moe_layer * stride) % len(cluster)
                    expert_tile = cluster[slot]
                    if expert_tile != anchor:
                        acc.record(anchor, expert_tile, hidden_bytes)
                        acc.record(expert_tile, anchor, hidden_bytes)
            previous_anchor = anchor

    traffic = acc.summarize(tokens)
    routed_parallel = 1
    if model.num_experts:
        routed_parallel = min(cluster_size, model.experts_per_token)
    memory_floor = _memory_floor(
        layers,
        hardware,
        bits=bits_per_weight,
        shared_bits=shared_bits,
        overhead=overhead_fraction,
        routed_parallel_tiles=routed_parallel,
    )

    return ClusterTopologyReport(
        cluster_size=cluster_size,
        cluster_rows=cluster_rows,
        cluster_cols=cluster_cols,
        resident=total_storage <= hardware.capacity_bytes and max(storage, default=0.0) <= tile_capacity,
        total_storage_bytes=total_storage,
        max_tile_storage_bytes=max(storage, default=0.0),
        min_tile_storage_bytes=min(storage, default=0.0),
        tile_capacity_bytes=tile_capacity,
        network_payload_bytes_per_token=traffic[0],
        hop_bytes_per_token=traffic[1],
        active_directed_links=traffic[2],
        max_link_bytes_per_token=traffic[3],
        hotspot_ratio=traffic[4],
        hottest_link=traffic[5],
        ideal_memory_floor_s=memory_floor,
        ideal_memory_roof_tps=(1.0 / memory_floor if memory_floor else 0.0),
    )


def sweep_cluster_bandwidths(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    cluster_sizes: tuple[int, ...] = DEFAULT_CLUSTER_SIZES,
    link_bandwidths_gb_s: tuple[float, ...] = DEFAULT_LINK_BANDWIDTHS_GB_S,
    max_hot_link_pressure: float = 0.05,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    tokens: int = 64,
    profile: str = "balanced",
    seed: int = 42,
) -> tuple[ClusterBandwidthSummary, ...]:
    """Sweep cluster size and link rate using hottest-link load at the memory roof.

    hot_link_pressure is the fraction of one directed link's line rate consumed
    if the architecture sustained its ideal local-memory-only token roof. A 5%
    threshold therefore leaves ~20x steady-state bandwidth headroom for bursts,
    overlap imperfections and future M2 queueing effects.
    """
    if not 0 < max_hot_link_pressure <= 1:
        raise ValueError("max_hot_link_pressure must be in (0, 1]")
    bandwidths = tuple(sorted(set(float(value) for value in link_bandwidths_gb_s)))
    if not bandwidths or bandwidths[0] <= 0:
        raise ValueError("link bandwidths must be positive")

    valid_sizes = tuple(
        size for size in cluster_sizes if 0 < size <= hardware.tiles and hardware.tiles % size == 0
    )
    if not valid_sizes:
        raise ValueError("no valid cluster sizes for this hardware")

    summaries: list[ClusterBandwidthSummary] = []
    for cluster_size in valid_sizes:
        topology = analyze_cluster_topology(
            model,
            hardware,
            cluster_size=cluster_size,
            bits_per_weight=bits_per_weight,
            shared_expert_bits=shared_expert_bits,
            overhead_fraction=overhead_fraction,
            activation_bits=activation_bits,
            tokens=tokens,
            profile=profile,
            seed=seed,
        )

        if topology.max_link_bytes_per_token > 0 and topology.ideal_memory_roof_tps > 0:
            raw_minimum = (
                topology.max_link_bytes_per_token
                * topology.ideal_memory_roof_tps
                / max_hot_link_pressure
                / 1e9
            )
        else:
            raw_minimum = 0.0

        points: list[BandwidthSweepPoint] = []
        recommendation: float | None = None
        for bandwidth_gb_s in bandwidths:
            bandwidth_bytes_s = bandwidth_gb_s * 1e9
            serialization = topology.max_link_bytes_per_token / bandwidth_bytes_s
            pressure = (
                topology.max_link_bytes_per_token * topology.ideal_memory_roof_tps / bandwidth_bytes_s
                if topology.ideal_memory_roof_tps
                else 0.0
            )
            total_s = topology.ideal_memory_floor_s + serialization
            conservative_tps = 1.0 / total_s if total_s > 0 else 0.0
            retained = (
                conservative_tps / topology.ideal_memory_roof_tps
                if topology.ideal_memory_roof_tps > 0
                else 0.0
            )
            adequate = topology.resident and pressure <= max_hot_link_pressure
            if adequate and recommendation is None:
                recommendation = bandwidth_gb_s
            points.append(
                BandwidthSweepPoint(
                    cluster_size=cluster_size,
                    link_bandwidth_gb_s=bandwidth_gb_s,
                    resident=topology.resident,
                    hot_link_pressure=pressure,
                    hot_link_serialization_s=serialization,
                    conservative_total_s=total_s,
                    conservative_tps=conservative_tps,
                    retained_memory_roof_fraction=retained,
                    adequate=adequate,
                )
            )

        summaries.append(
            ClusterBandwidthSummary(
                topology=topology,
                raw_minimum_bandwidth_gb_s=(raw_minimum if topology.resident else None),
                recommended_candidate_bandwidth_gb_s=recommendation,
                points=tuple(points),
            )
        )
    return tuple(summaries)


def _format_bandwidth(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:g} TB/s"
    return f"{value:g} GB/s"


def _run(args: argparse.Namespace) -> int:
    model = get_model(args.model)
    hardware = get_hardware(args.hardware)
    cluster_sizes = tuple(args.clusters)
    bandwidths = tuple(args.bandwidths)
    summaries = sweep_cluster_bandwidths(
        model,
        hardware,
        cluster_sizes=cluster_sizes,
        link_bandwidths_gb_s=bandwidths,
        max_hot_link_pressure=args.max_hot_pressure,
        bits_per_weight=args.bits,
        shared_expert_bits=args.shared_bits,
        overhead_fraction=args.overhead,
        activation_bits=args.activation_bits,
        tokens=args.tokens,
        profile=args.profile,
        seed=args.seed,
    )
    shared_bits = args.bits if args.shared_bits is None else args.shared_bits

    print(f"{model.name} -> {hardware.name}")
    print(f"  base/shared precision:  {args.bits:g} / {shared_bits:g} bit")
    print(f"  routing profile:        {args.profile}, {args.tokens} synthetic token(s)")
    print(f"  adequacy threshold:     <= {args.max_hot_pressure:.1%} hot-link pressure at the memory-only roof")
    print("  candidate link rates:   " + ", ".join(_format_bandwidth(value) for value in bandwidths))

    print("\nMINIMUM BANDWIDTH BY CLUSTER SIZE")
    print("CLUSTER  SHAPE  FIT   MAX TILE    MEM ROOF      NOC/TOK   HOT/TOK   RAW MIN      CANDIDATE")
    for summary in summaries:
        t = summary.topology
        print(
            f"{t.cluster_size:>7}  {t.shape_label:>5}  "
            f"{'yes' if t.resident else 'NO ':>3}  "
            f"{fmt_bytes(t.max_tile_storage_bytes):>10}  "
            f"{fmt_rate(t.ideal_memory_roof_tps):>10}  "
            f"{fmt_bytes(t.network_payload_bytes_per_token):>10}  "
            f"{fmt_bytes(t.max_link_bytes_per_token):>9}  "
            f"{_format_bandwidth(summary.raw_minimum_bandwidth_gb_s):>11}  "
            f"{_format_bandwidth(summary.recommended_candidate_bandwidth_gb_s):>11}"
        )

    print("\nHOT-LINK PRESSURE AT EACH LINK RATE")
    header = "CLUSTER" + "".join(f" {value:g}G".rjust(9) for value in bandwidths)
    print(header)
    for summary in summaries:
        cells = []
        for point in summary.points:
            if not summary.topology.resident:
                cell = "NOFIT"
            else:
                marker = "*" if point.adequate else ""
                cell = f"{point.hot_link_pressure * 100:.1f}%{marker}"
            cells.append(cell.rjust(9))
        print(f"{summary.topology.cluster_size:>7}" + "".join(cells))

    print("\n* = at or below the selected hot-link-pressure threshold.")
    print(
        "RAW MIN is an M1.5 screening value, not a SerDes/NoC specification: it uses aggregate per-token link load "
        "at the ideal memory roof and still excludes packet timing, queueing, arbitration, compute, KV traffic and thermals."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-sweep",
        description="Sweep cluster size and per-neighbor fabric bandwidth for memory-stationary MoE inference.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--hardware", default="fabric-64x32")
    parser.add_argument("--bits", type=float, default=4.0)
    parser.add_argument("--shared-bits", type=float, default=None)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--activation-bits", type=float, default=16.0)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--profile", choices=("balanced", "hot", "zipf"), default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, nargs="+", default=list(DEFAULT_CLUSTER_SIZES))
    parser.add_argument(
        "--bandwidths",
        type=float,
        nargs="+",
        default=list(DEFAULT_LINK_BANDWIDTHS_GB_S),
        help="per-directed-neighbor link bandwidths in GB/s",
    )
    parser.add_argument(
        "--max-hot-pressure",
        type=float,
        default=0.05,
        help="maximum fraction of link line-rate consumed at the ideal memory roof; default 0.05",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
