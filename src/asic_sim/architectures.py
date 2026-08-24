from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .fabric import (
    DirectedLink,
    _profile_cumulative,
    _selected_experts,
    build_physical_placement,
    trace_traffic,
    xy_route,
)
from .hardware import HardwareSpec
from .models import ModelSpec
from .placement import decompose_model


@dataclass(frozen=True, slots=True)
class LayerEstimate:
    layer: int
    is_moe: bool
    total_parameters: float
    active_parameters: float
    routed_pool_parameters: float
    active_routed_parameters: float
    shared_parameters: float
    other_always_on_parameters: float


@dataclass(frozen=True, slots=True)
class ArchitectureReport:
    mode: str
    label: str
    model: str
    hardware: str
    bits_per_weight: float
    shared_expert_bits: float
    cluster_size: int
    resident: bool
    total_storage_bytes: float
    max_tile_storage_bytes: float
    min_tile_storage_bytes: float
    tile_capacity_bytes: float
    network_payload_bytes_per_token: float
    hop_bytes_per_token: float
    active_directed_links: int
    max_link_bytes_per_token: float
    hotspot_ratio: float
    hottest_link: DirectedLink | None
    ideal_memory_floor_s: float
    ideal_memory_roof_tps: float
    shared_active_bytes_per_token: float
    routed_active_bytes_per_token: float
    other_active_bytes_per_token: float
    notes: str


@dataclass
class _TrafficAccumulator:
    hardware: HardwareSpec
    link_bytes: dict[DirectedLink, float]
    payload_bytes: float = 0.0
    hop_bytes: float = 0.0

    def record(self, source: int, destination: int, payload: float) -> None:
        if source == destination or payload <= 0:
            return
        path = xy_route(self.hardware, source, destination)
        self.payload_bytes += payload
        hops = len(path) - 1
        self.hop_bytes += payload * hops
        for left, right in zip(path, path[1:]):
            link = (left, right)
            self.link_bytes[link] = self.link_bytes.get(link, 0.0) + payload

    def summarize(self, tokens: int) -> tuple[float, float, int, float, float, DirectedLink | None]:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        per_token = [(link, value / tokens) for link, value in self.link_bytes.items()]
        per_token.sort(key=lambda item: item[1], reverse=True)
        if per_token:
            hottest_link, max_link = per_token[0]
            mean = sum(value for _, value in per_token) / len(per_token)
            hotspot = max_link / mean if mean else 0.0
        else:
            hottest_link = None
            max_link = hotspot = 0.0
        return (
            self.payload_bytes / tokens,
            self.hop_bytes / tokens,
            len(per_token),
            max_link,
            hotspot,
            hottest_link,
        )


def estimate_layers(model: ModelSpec) -> tuple[LayerEstimate, ...]:
    """Build a count-consistent per-layer decomposition.

    The routed pool is divided over MoE layers. Explicit shared experts are
    placed only in MoE layers. Remaining always-on parameters (attention,
    embeddings/projections/norms/dense-path residual in this abstraction) are
    spread evenly over layers. This is intentionally an M1 estimate, but sums
    exactly to the vendor total and active parameter counts.
    """
    if model.num_layers <= 0:
        raise ValueError("model must have at least one layer")

    d = decompose_model(model)
    if not model.moe_layers:
        per = model.total_parameters / model.num_layers
        return tuple(
            LayerEstimate(
                layer=layer,
                is_moe=False,
                total_parameters=per,
                active_parameters=per,
                routed_pool_parameters=0.0,
                active_routed_parameters=0.0,
                shared_parameters=0.0,
                other_always_on_parameters=per,
            )
            for layer in range(model.num_layers)
        )

    routed_pool_per_moe = d.routed_pool_parameters / model.moe_layers
    active_routed_per_moe = d.active_routed_parameters / model.moe_layers
    shared_per_moe = d.shared_expert_parameters_per_moe_layer
    other_per_layer = d.other_always_on_parameters / model.num_layers

    layers: list[LayerEstimate] = []
    for layer in range(model.num_layers):
        is_moe = layer >= model.dense_layers
        if is_moe:
            total = routed_pool_per_moe + shared_per_moe + other_per_layer
            active = active_routed_per_moe + shared_per_moe + other_per_layer
            routed_pool = routed_pool_per_moe
            active_routed = active_routed_per_moe
            shared = shared_per_moe
        else:
            total = active = other_per_layer
            routed_pool = active_routed = shared = 0.0
        layers.append(
            LayerEstimate(
                layer=layer,
                is_moe=is_moe,
                total_parameters=total,
                active_parameters=active,
                routed_pool_parameters=routed_pool,
                active_routed_parameters=active_routed,
                shared_parameters=shared,
                other_always_on_parameters=other_per_layer,
            )
        )

    # Guard against accidental drift when changing the decomposition later.
    if not math.isclose(sum(x.total_parameters for x in layers), model.total_parameters, rel_tol=1e-10):
        raise AssertionError("layer estimates do not sum to total parameters")
    if not math.isclose(sum(x.active_parameters for x in layers), model.active_parameters, rel_tol=1e-10):
        raise AssertionError("layer estimates do not sum to active parameters")
    return tuple(layers)


def _bytes(parameters: float, bits: float, overhead: float) -> float:
    return parameters * bits / 8.0 * (1.0 + overhead)


def _layer_storage_bytes(layer: LayerEstimate, bits: float, shared_bits: float, overhead: float) -> float:
    return _bytes(layer.routed_pool_parameters + layer.other_always_on_parameters, bits, overhead) + _bytes(
        layer.shared_parameters, shared_bits, overhead
    )


def _layer_active_components(
    layer: LayerEstimate,
    bits: float,
    shared_bits: float,
    overhead: float,
) -> tuple[float, float, float]:
    routed = _bytes(layer.active_routed_parameters, bits, overhead)
    shared = _bytes(layer.shared_parameters, shared_bits, overhead)
    other = _bytes(layer.other_always_on_parameters, bits, overhead)
    return routed, shared, other


def _mesh_shape(hardware: HardwareSpec) -> tuple[int, int]:
    if hardware.mesh_rows and hardware.mesh_cols:
        return hardware.mesh_rows, hardware.mesh_cols
    rows = max(1, int(math.sqrt(hardware.tiles)))
    cols = math.ceil(hardware.tiles / rows)
    return rows, cols


def _serpentine_tiles(hardware: HardwareSpec) -> tuple[int, ...]:
    rows, cols = _mesh_shape(hardware)
    order: list[int] = []
    for row in range(rows):
        row_tiles = [row * cols + col for col in range(cols) if row * cols + col < hardware.tiles]
        if row % 2:
            row_tiles.reverse()
        order.extend(row_tiles)
    return tuple(order)


def _balanced_owner(index: int, count: int, owners: tuple[int, ...]) -> int:
    if not owners:
        raise ValueError("owners cannot be empty")
    group = min(len(owners) - 1, (index * len(owners)) // count)
    return owners[group]


def _clusters_2x2(hardware: HardwareSpec) -> tuple[tuple[int, ...], ...]:
    if hardware.tiles == 1:
        return ((0,),)
    rows, cols = _mesh_shape(hardware)
    if rows % 2 or cols % 2:
        raise ValueError("cluster-4 currently requires even mesh rows and columns")
    cluster_rows = rows // 2
    cluster_cols = cols // 2
    result: list[tuple[int, ...]] = []
    for cr in range(cluster_rows):
        cols_iter = range(cluster_cols) if cr % 2 == 0 else range(cluster_cols - 1, -1, -1)
        for cc in cols_iter:
            r = cr * 2
            c = cc * 2
            result.append(
                (
                    r * cols + c,
                    r * cols + c + 1,
                    (r + 1) * cols + c + 1,
                    (r + 1) * cols + c,
                )
            )
    return tuple(result)


def _memory_floor(
    layers: tuple[LayerEstimate, ...],
    hardware: HardwareSpec,
    *,
    bits: float,
    shared_bits: float,
    overhead: float,
    routed_parallel_tiles: int,
) -> float:
    """Ideal per-token memory floor with bounded routed-expert parallelism.

    Shared experts and the remaining always-on path stay on the layer anchor.
    Routed expert bytes may be spread over routed_parallel_tiles. This is a
    bandwidth roof only: compute, synchronization and NoC timing are excluded.
    """
    bw = hardware.local_memory_bandwidth_bytes_s
    total = 0.0
    for layer in layers:
        routed, shared, other = _layer_active_components(layer, bits, shared_bits, overhead)
        if layer.is_moe and routed > 0:
            parallel = max(1, min(routed_parallel_tiles, hardware.tiles))
            routed_per_tile = routed / parallel
            # The anchor also owns a routed share in the ideal balanced mapping.
            critical_bytes = shared + other + routed_per_tile
        else:
            critical_bytes = shared + other + routed
        total += critical_bytes / bw
    return total


def _storage_pipeline(
    layers: tuple[LayerEstimate, ...],
    hardware: HardwareSpec,
    bits: float,
    shared_bits: float,
    overhead: float,
) -> tuple[list[float], tuple[int, ...]]:
    order = _serpentine_tiles(hardware)
    owners = tuple(_balanced_owner(i, len(layers), order) for i in range(len(layers)))
    storage = [0.0] * hardware.tiles
    for layer, tile in zip(layers, owners):
        storage[tile] += _layer_storage_bytes(layer, bits, shared_bits, overhead)
    return storage, owners


def _traffic_pipeline(
    model: ModelSpec,
    hardware: HardwareSpec,
    owners: tuple[int, ...],
    *,
    tokens: int,
    activation_bits: float,
) -> tuple[float, float, int, float, float, DirectedLink | None]:
    acc = _TrafficAccumulator(hardware, {})
    hidden = model.hidden_bytes(activation_bits)
    for _token in range(tokens):
        for layer in range(1, len(owners)):
            acc.record(owners[layer - 1], owners[layer], hidden)
    return acc.summarize(tokens)


def _storage_cluster4(
    layers: tuple[LayerEstimate, ...],
    hardware: HardwareSpec,
    bits: float,
    shared_bits: float,
    overhead: float,
) -> tuple[list[float], tuple[int, ...], tuple[tuple[int, ...], ...]]:
    clusters = _clusters_2x2(hardware)
    layer_cluster = tuple(min(len(clusters) - 1, (i * len(clusters)) // len(layers)) for i in range(len(layers)))
    storage = [0.0] * hardware.tiles
    for layer, cluster_idx in zip(layers, layer_cluster):
        cluster = clusters[cluster_idx]
        anchor = cluster[0]
        storage[anchor] += _bytes(layer.other_always_on_parameters, bits, overhead)
        storage[anchor] += _bytes(layer.shared_parameters, shared_bits, overhead)
        if layer.routed_pool_parameters:
            routed_each = _bytes(layer.routed_pool_parameters, bits, overhead) / len(cluster)
            for tile in cluster:
                storage[tile] += routed_each
    return storage, layer_cluster, clusters


def _traffic_cluster4(
    model: ModelSpec,
    hardware: HardwareSpec,
    layer_cluster: tuple[int, ...],
    clusters: tuple[tuple[int, ...], ...],
    *,
    tokens: int,
    profile: str,
    activation_bits: float,
    seed: int,
) -> tuple[float, float, int, float, float, DirectedLink | None]:
    acc = _TrafficAccumulator(hardware, {})
    hidden = model.hidden_bytes(activation_bits)
    rng = random.Random(seed)
    cumulative = _profile_cumulative(model, profile) if model.num_experts else None

    previous_anchor = clusters[layer_cluster[0]][0]
    for token in range(tokens):
        previous_anchor = clusters[layer_cluster[0]][0]
        for layer in range(model.num_layers):
            cluster = clusters[layer_cluster[layer]]
            anchor = cluster[0]
            if layer > 0:
                acc.record(previous_anchor, anchor, hidden)
            if model.num_experts and layer >= model.dense_layers:
                moe_layer = layer - model.dense_layers
                selected = _selected_experts(model, token, moe_layer, profile, cumulative, rng)
                for expert in selected:
                    # Rotate logical expert IDs inside the local cluster across layers.
                    slot = (expert + 3 * moe_layer) % len(cluster)
                    expert_tile = cluster[slot]
                    if expert_tile != anchor:
                        acc.record(anchor, expert_tile, hidden)
                        acc.record(expert_tile, anchor, hidden)
            previous_anchor = anchor
    return acc.summarize(tokens)


def _storage_expert_mesh(
    model: ModelSpec,
    hardware: HardwareSpec,
    layers: tuple[LayerEstimate, ...],
    bits: float,
    shared_bits: float,
    overhead: float,
) -> list[float]:
    storage = [0.0] * hardware.tiles
    # Keep the existing expert-mesh convention: layer anchors advance by tile ID.
    stride = 1
    if hardware.tiles > 1:
        candidate = max(3, int(math.sqrt(hardware.tiles)) | 1)
        while math.gcd(candidate, hardware.tiles) != 1:
            candidate += 2
        stride = candidate

    for layer in layers:
        anchor = layer.layer % hardware.tiles
        storage[anchor] += _bytes(layer.other_always_on_parameters, bits, overhead)
        storage[anchor] += _bytes(layer.shared_parameters, shared_bits, overhead)
        if layer.routed_pool_parameters:
            moe_layer = layer.layer - model.dense_layers
            shard = layer.routed_pool_parameters / model.num_experts
            shard_bytes = _bytes(shard, bits, overhead)
            for expert in range(model.num_experts):
                tile = (expert + moe_layer * stride) % hardware.tiles
                storage[tile] += shard_bytes
    return storage


def compare_architectures(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    bits_per_weight: float = 4.0,
    shared_expert_bits: float | None = None,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    tokens: int = 64,
    profile: str = "balanced",
    seed: int = 42,
) -> tuple[ArchitectureReport, ...]:
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    shared_bits = bits_per_weight if shared_expert_bits is None else shared_expert_bits
    if shared_bits <= 0:
        raise ValueError("shared_expert_bits must be positive")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")
    if tokens <= 0:
        raise ValueError("tokens must be positive")

    layers = estimate_layers(model)
    tile_capacity = hardware.tile_capacity_gb * 1e9 if hardware.tile_capacity_gb is not None else hardware.capacity_bytes

    d = decompose_model(model)
    shared_active = _bytes(d.shared_expert_parameters, shared_bits, overhead_fraction)
    routed_active = _bytes(d.active_routed_parameters, bits_per_weight, overhead_fraction)
    other_active = _bytes(d.other_always_on_parameters, bits_per_weight, overhead_fraction)
    total_storage = model.mixed_storage_bytes(
        bits_per_weight,
        overhead_fraction,
        shared_expert_bits=shared_bits,
    )

    reports: list[ArchitectureReport] = []

    # 1) Whole-layer stationary pipeline.
    p_storage, p_owners = _storage_pipeline(layers, hardware, bits_per_weight, shared_bits, overhead_fraction)
    p_traffic = _traffic_pipeline(model, hardware, p_owners, tokens=tokens, activation_bits=activation_bits)
    p_floor = _memory_floor(
        layers,
        hardware,
        bits=bits_per_weight,
        shared_bits=shared_bits,
        overhead=overhead_fraction,
        routed_parallel_tiles=1,
    )
    reports.append(
        ArchitectureReport(
            mode="layer-pipeline",
            label="Layer pipeline",
            model=model.name,
            hardware=hardware.name,
            bits_per_weight=bits_per_weight,
            shared_expert_bits=shared_bits,
            cluster_size=1,
            resident=total_storage <= hardware.capacity_bytes and max(p_storage, default=0.0) <= tile_capacity,
            total_storage_bytes=total_storage,
            max_tile_storage_bytes=max(p_storage, default=0.0),
            min_tile_storage_bytes=min(p_storage, default=0.0),
            tile_capacity_bytes=tile_capacity,
            network_payload_bytes_per_token=p_traffic[0],
            hop_bytes_per_token=p_traffic[1],
            active_directed_links=p_traffic[2],
            max_link_bytes_per_token=p_traffic[3],
            hotspot_ratio=p_traffic[4],
            hottest_link=p_traffic[5],
            ideal_memory_floor_s=p_floor,
            ideal_memory_roof_tps=(1.0 / p_floor if p_floor else 0.0),
            shared_active_bytes_per_token=shared_active,
            routed_active_bytes_per_token=routed_active,
            other_active_bytes_per_token=other_active,
            notes="Whole layers are stationary; a serpentine physical pipeline keeps successive tile groups adjacent.",
        )
    )

    # Dense models do not have a meaningful expert-sharding comparison.
    if not model.num_experts or hardware.tiles == 1:
        return tuple(reports)

    # 2) Four-tile local clusters per layer group.
    c_storage, layer_cluster, clusters = _storage_cluster4(layers, hardware, bits_per_weight, shared_bits, overhead_fraction)
    c_traffic = _traffic_cluster4(
        model,
        hardware,
        layer_cluster,
        clusters,
        tokens=tokens,
        profile=profile,
        activation_bits=activation_bits,
        seed=seed,
    )
    c_parallel = min(4, model.experts_per_token)
    c_floor = _memory_floor(
        layers,
        hardware,
        bits=bits_per_weight,
        shared_bits=shared_bits,
        overhead=overhead_fraction,
        routed_parallel_tiles=c_parallel,
    )
    reports.append(
        ArchitectureReport(
            mode="cluster-4",
            label="Cluster-4",
            model=model.name,
            hardware=hardware.name,
            bits_per_weight=bits_per_weight,
            shared_expert_bits=shared_bits,
            cluster_size=4,
            resident=total_storage <= hardware.capacity_bytes and max(c_storage, default=0.0) <= tile_capacity,
            total_storage_bytes=total_storage,
            max_tile_storage_bytes=max(c_storage, default=0.0),
            min_tile_storage_bytes=min(c_storage, default=0.0),
            tile_capacity_bytes=tile_capacity,
            network_payload_bytes_per_token=c_traffic[0],
            hop_bytes_per_token=c_traffic[1],
            active_directed_links=c_traffic[2],
            max_link_bytes_per_token=c_traffic[3],
            hotspot_ratio=c_traffic[4],
            hottest_link=c_traffic[5],
            ideal_memory_floor_s=c_floor,
            ideal_memory_roof_tps=(1.0 / c_floor if c_floor else 0.0),
            shared_active_bytes_per_token=shared_active,
            routed_active_bytes_per_token=routed_active,
            other_active_bytes_per_token=other_active,
            notes="Shared/always-on path stays at a 2x2 cluster anchor; routed experts are striped only inside that cluster.",
        )
    )

    # 3) Global expert mesh (the previous M1 mapping), now with explicit shared storage.
    e_storage = _storage_expert_mesh(model, hardware, layers, bits_per_weight, shared_bits, overhead_fraction)
    placement = build_physical_placement(
        model,
        hardware,
        bits_per_weight=bits_per_weight,
        overhead_fraction=overhead_fraction,
    )
    e_trace = trace_traffic(
        placement,
        tokens=tokens,
        profile=profile,
        activation_bits=activation_bits,
        seed=seed,
    )
    e_parallel = min(model.experts_per_token, hardware.tiles)
    e_floor = _memory_floor(
        layers,
        hardware,
        bits=bits_per_weight,
        shared_bits=shared_bits,
        overhead=overhead_fraction,
        routed_parallel_tiles=e_parallel,
    )
    reports.append(
        ArchitectureReport(
            mode="expert-mesh",
            label="Expert mesh",
            model=model.name,
            hardware=hardware.name,
            bits_per_weight=bits_per_weight,
            shared_expert_bits=shared_bits,
            cluster_size=hardware.tiles,
            resident=total_storage <= hardware.capacity_bytes and max(e_storage, default=0.0) <= tile_capacity,
            total_storage_bytes=total_storage,
            max_tile_storage_bytes=max(e_storage, default=0.0),
            min_tile_storage_bytes=min(e_storage, default=0.0),
            tile_capacity_bytes=tile_capacity,
            network_payload_bytes_per_token=e_trace.network_payload_bytes_per_token,
            hop_bytes_per_token=e_trace.hop_bytes_per_token,
            active_directed_links=e_trace.active_directed_links,
            max_link_bytes_per_token=e_trace.max_link_bytes_per_token,
            hotspot_ratio=e_trace.hotspot_ratio,
            hottest_link=e_trace.hottest_link,
            ideal_memory_floor_s=e_floor,
            ideal_memory_roof_tps=(1.0 / e_floor if e_floor else 0.0),
            shared_active_bytes_per_token=shared_active,
            routed_active_bytes_per_token=routed_active,
            other_active_bytes_per_token=other_active,
            notes="Shared/always-on path stays at each layer anchor; routed experts may use the full mesh.",
        )
    )

    return tuple(reports)
