from __future__ import annotations

from dataclasses import dataclass
import bisect
import math
import random
from typing import Iterable

from .hardware import HardwareSpec
from .models import ModelSpec
from .placement import decompose_model


DirectedLink = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TileState:
    tile_id: int
    row: int
    col: int
    storage_bytes: float
    expert_shards: int
    layer_anchors: int


@dataclass(frozen=True, slots=True)
class PhysicalPlacement:
    """Deterministic physical model placement on a 2D tile mesh.

    M1 remains metadata-only: an expert shard is represented by its layer/expert
    coordinates and a deterministic tile mapping, not by materialized weights.
    """

    model: ModelSpec
    hardware: HardwareSpec
    bits_per_weight: float
    overhead_fraction: float
    tiles: tuple[TileState, ...]
    layer_anchor_tiles: tuple[int, ...]
    expert_layer_stride: int
    total_storage_bytes: float
    tile_capacity_bytes: float

    @property
    def max_storage_bytes(self) -> float:
        return max((tile.storage_bytes for tile in self.tiles), default=0.0)

    @property
    def min_storage_bytes(self) -> float:
        return min((tile.storage_bytes for tile in self.tiles), default=0.0)

    @property
    def system_resident(self) -> bool:
        return self.total_storage_bytes <= self.hardware.capacity_bytes

    @property
    def per_tile_resident(self) -> bool:
        return all(tile.storage_bytes <= self.tile_capacity_bytes for tile in self.tiles)

    def anchor_tile(self, layer: int) -> int:
        return self.layer_anchor_tiles[layer]

    def expert_tile(self, moe_layer: int, expert: int) -> int:
        if not self.model.num_experts:
            raise ValueError("dense models do not have routed experts")
        if not 0 <= moe_layer < self.model.moe_layers:
            raise IndexError("moe_layer out of range")
        if not 0 <= expert < self.model.num_experts:
            raise IndexError("expert out of range")
        # Layer skew prevents expert ID e from landing on the same tile at every
        # layer when expert count is divisible by tile count (true for Kimi/GLM).
        return (expert + moe_layer * self.expert_layer_stride) % self.hardware.tiles


@dataclass(frozen=True, slots=True)
class RouteSample:
    token: int
    layer: int
    kind: str
    source: int
    destination: int
    path: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrafficReport:
    model: str
    hardware: str
    profile: str
    tokens: int
    activation_bits: float
    network_payload_bytes: float
    hop_bytes: float
    network_payload_bytes_per_token: float
    hop_bytes_per_token: float
    remote_expert_dispatches: int
    total_expert_dispatches: int
    remote_expert_fraction: float
    average_hops: float
    active_directed_links: int
    max_link_bytes_per_token: float
    mean_active_link_bytes_per_token: float
    hotspot_ratio: float
    hottest_link: DirectedLink | None
    ideal_hottest_link_time_s: float | None
    link_bytes_per_token: tuple[tuple[DirectedLink, float], ...]
    tile_tx_bytes_per_token: tuple[float, ...]
    tile_rx_bytes_per_token: tuple[float, ...]
    sample_routes: tuple[RouteSample, ...]

    @property
    def hottest_link_share(self) -> float:
        total = sum(value for _, value in self.link_bytes_per_token)
        return self.max_link_bytes_per_token / total if total else 0.0


def _mesh_shape(hardware: HardwareSpec) -> tuple[int, int]:
    if hardware.mesh_rows and hardware.mesh_cols:
        return hardware.mesh_rows, hardware.mesh_cols
    rows = max(1, int(math.sqrt(hardware.tiles)))
    cols = math.ceil(hardware.tiles / rows)
    return rows, cols


def tile_coord(hardware: HardwareSpec, tile: int) -> tuple[int, int]:
    if not 0 <= tile < hardware.tiles:
        raise IndexError("tile out of range")
    _, cols = _mesh_shape(hardware)
    return divmod(tile, cols)


def xy_route(hardware: HardwareSpec, source: int, destination: int) -> tuple[int, ...]:
    """Return an X-then-Y route as tile IDs, including endpoints."""
    if source == destination:
        return (source,)
    rows, cols = _mesh_shape(hardware)
    sr, sc = tile_coord(hardware, source)
    dr, dc = tile_coord(hardware, destination)
    path = [source]
    row, col = sr, sc

    step_col = 1 if dc > col else -1
    while col != dc:
        col += step_col
        tile = row * cols + col
        if tile >= hardware.tiles:
            raise ValueError("XY route crosses an unpopulated mesh coordinate")
        path.append(tile)

    step_row = 1 if dr > row else -1
    while row != dr:
        row += step_row
        tile = row * cols + col
        if tile >= hardware.tiles:
            raise ValueError("XY route crosses an unpopulated mesh coordinate")
        path.append(tile)

    if not (0 <= dr < rows and 0 <= dc < cols):
        raise ValueError("destination outside mesh")
    return tuple(path)


def _coprime_stride(tiles: int) -> int:
    if tiles <= 1:
        return 1
    candidate = max(3, int(math.sqrt(tiles)) | 1)
    while math.gcd(candidate, tiles) != 1:
        candidate += 2
    return candidate


def build_physical_placement(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    bits_per_weight: float = 4.0,
    overhead_fraction: float = 0.05,
) -> PhysicalPlacement:
    """Place layer anchors and expert shards on concrete mesh tiles."""
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")

    tile_count = hardware.tiles
    bytes_per_parameter = bits_per_weight / 8.0 * (1.0 + overhead_fraction)
    total_storage = model.storage_bytes(bits_per_weight, overhead_fraction)
    tile_capacity = (
        hardware.tile_capacity_gb * 1e9
        if hardware.tile_capacity_gb is not None
        else hardware.capacity_bytes
    )
    storage = [0.0] * tile_count
    expert_shards = [0] * tile_count
    anchors = [0] * tile_count
    layer_anchor_tiles = tuple(layer % tile_count for layer in range(model.num_layers))
    for tile in layer_anchor_tiles:
        anchors[tile] += 1

    stride = _coprime_stride(tile_count)
    decomposition = decompose_model(model)

    if not model.num_experts:
        # Dense M1 baseline: layer-partition equal slices. On one tile this is
        # exact for capacity; on a mesh it is a deliberately simple first map.
        per_layer_params = model.total_parameters / model.num_layers
        for layer, tile in enumerate(layer_anchor_tiles):
            del layer
            storage[tile] += per_layer_params * bytes_per_parameter
    else:
        always_on_per_layer = decomposition.always_on_parameters / model.num_layers
        for tile in layer_anchor_tiles:
            storage[tile] += always_on_per_layer * bytes_per_parameter

        shard_params = decomposition.expert_shard_parameters or 0.0
        shard_bytes = shard_params * bytes_per_parameter
        for moe_layer in range(model.moe_layers):
            for expert in range(model.num_experts):
                tile = (expert + moe_layer * stride) % tile_count
                storage[tile] += shard_bytes
                expert_shards[tile] += 1

    rows, cols = _mesh_shape(hardware)
    tile_states = []
    for tile in range(tile_count):
        row, col = divmod(tile, cols)
        if row >= rows:
            raise ValueError("tile index exceeds mesh geometry")
        tile_states.append(
            TileState(
                tile_id=tile,
                row=row,
                col=col,
                storage_bytes=storage[tile],
                expert_shards=expert_shards[tile],
                layer_anchors=anchors[tile],
            )
        )

    return PhysicalPlacement(
        model=model,
        hardware=hardware,
        bits_per_weight=bits_per_weight,
        overhead_fraction=overhead_fraction,
        tiles=tuple(tile_states),
        layer_anchor_tiles=layer_anchor_tiles,
        expert_layer_stride=stride,
        total_storage_bytes=total_storage,
        tile_capacity_bytes=tile_capacity,
    )


def _balanced_experts(model: ModelSpec, token: int, moe_layer: int) -> tuple[int, ...]:
    n = model.num_experts
    k = model.experts_per_token
    base = (token * 131 + moe_layer * 17) % n
    stride = 37
    while math.gcd(stride, n) != 1:
        stride += 2
    return tuple((base + rank * stride) % n for rank in range(k))


def _weighted_unique(
    rng: random.Random,
    cumulative: tuple[float, ...],
    k: int,
) -> tuple[int, ...]:
    chosen: set[int] = set()
    total = cumulative[-1]
    attempts = 0
    max_attempts = max(100, k * 100)
    while len(chosen) < k and attempts < max_attempts:
        target = rng.random() * total
        chosen.add(bisect.bisect_left(cumulative, target))
        attempts += 1
    if len(chosen) < k:
        for idx in range(len(cumulative)):
            chosen.add(idx)
            if len(chosen) == k:
                break
    return tuple(chosen)


def _profile_cumulative(model: ModelSpec, profile: str) -> tuple[float, ...] | None:
    if profile == "balanced":
        return None
    n = model.num_experts
    if profile == "hot":
        hot_count = max(model.experts_per_token, max(1, n // 10))
        weights = [8.0 if idx < hot_count else 1.0 for idx in range(n)]
    elif profile == "zipf":
        weights = [1.0 / ((idx + 1) ** 1.1) for idx in range(n)]
    else:
        raise ValueError("profile must be one of: balanced, hot, zipf")
    cumulative: list[float] = []
    total = 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    return tuple(cumulative)


def _selected_experts(
    model: ModelSpec,
    token: int,
    moe_layer: int,
    profile: str,
    cumulative: tuple[float, ...] | None,
    rng: random.Random,
) -> tuple[int, ...]:
    if profile == "balanced":
        return _balanced_experts(model, token, moe_layer)
    assert cumulative is not None
    return _weighted_unique(rng, cumulative, model.experts_per_token)


def trace_traffic(
    placement: PhysicalPlacement,
    *,
    tokens: int = 32,
    profile: str = "balanced",
    activation_bits: float = 16.0,
    seed: int = 42,
    sample_limit: int = 12,
) -> TrafficReport:
    """Replay metadata-only token routes and aggregate exact per-link M1 traffic.

    This is not a cycle-accurate contention simulator. It gives the exact link
    loads induced by the chosen placement/routing trace; timing/queues arrive in M2.
    """
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if activation_bits <= 0:
        raise ValueError("activation_bits must be positive")

    model = placement.model
    hardware = placement.hardware
    hidden_bytes = model.hidden_bytes(activation_bits)
    rng = random.Random(seed)
    cumulative = _profile_cumulative(model, profile) if model.num_experts else None

    link_bytes: dict[DirectedLink, float] = {}
    tile_tx = [0.0] * hardware.tiles
    tile_rx = [0.0] * hardware.tiles
    injected = 0.0
    hop_bytes = 0.0
    remote_dispatches = 0
    total_dispatches = 0
    samples: list[RouteSample] = []

    def record_route(token: int, layer: int, kind: str, source: int, destination: int, payload: float) -> None:
        nonlocal injected, hop_bytes
        if source == destination:
            return
        path = xy_route(hardware, source, destination)
        injected += payload
        tile_tx[source] += payload
        tile_rx[destination] += payload
        hops = len(path) - 1
        hop_bytes += payload * hops
        for left, right in zip(path, path[1:]):
            link = (left, right)
            link_bytes[link] = link_bytes.get(link, 0.0) + payload
        if len(samples) < sample_limit:
            samples.append(RouteSample(token, layer, kind, source, destination, path))

    for token in range(tokens):
        previous_anchor = placement.anchor_tile(0)
        for layer in range(model.num_layers):
            anchor = placement.anchor_tile(layer)
            if layer > 0:
                record_route(token, layer, "layer", previous_anchor, anchor, hidden_bytes)

            if model.num_experts and layer >= model.dense_layers:
                moe_layer = layer - model.dense_layers
                selected = _selected_experts(model, token, moe_layer, profile, cumulative, rng)
                for expert in selected:
                    total_dispatches += 1
                    expert_tile = placement.expert_tile(moe_layer, expert)
                    if expert_tile != anchor:
                        remote_dispatches += 1
                        record_route(token, layer, "dispatch", anchor, expert_tile, hidden_bytes)
                        record_route(token, layer, "return", expert_tile, anchor, hidden_bytes)
            previous_anchor = anchor

    per_token_links = tuple(
        sorted(((link, value / tokens) for link, value in link_bytes.items()), key=lambda item: item[1], reverse=True)
    )
    if per_token_links:
        hottest_link, max_link = per_token_links[0]
        mean_link = sum(value for _, value in per_token_links) / len(per_token_links)
        hotspot_ratio = max_link / mean_link if mean_link else 0.0
    else:
        hottest_link = None
        max_link = mean_link = hotspot_ratio = 0.0

    ideal_hot_time = None
    if hottest_link is not None and hardware.noc_link_bandwidth_bytes_s:
        ideal_hot_time = max_link / hardware.noc_link_bandwidth_bytes_s

    return TrafficReport(
        model=model.name,
        hardware=hardware.name,
        profile=profile,
        tokens=tokens,
        activation_bits=activation_bits,
        network_payload_bytes=injected,
        hop_bytes=hop_bytes,
        network_payload_bytes_per_token=injected / tokens,
        hop_bytes_per_token=hop_bytes / tokens,
        remote_expert_dispatches=remote_dispatches,
        total_expert_dispatches=total_dispatches,
        remote_expert_fraction=(remote_dispatches / total_dispatches if total_dispatches else 0.0),
        average_hops=(hop_bytes / injected if injected else 0.0),
        active_directed_links=len(link_bytes),
        max_link_bytes_per_token=max_link,
        mean_active_link_bytes_per_token=mean_link,
        hotspot_ratio=hotspot_ratio,
        hottest_link=hottest_link,
        ideal_hottest_link_time_s=ideal_hot_time,
        link_bytes_per_token=per_token_links,
        tile_tx_bytes_per_token=tuple(value / tokens for value in tile_tx),
        tile_rx_bytes_per_token=tuple(value / tokens for value in tile_rx),
        sample_routes=tuple(samples),
    )
