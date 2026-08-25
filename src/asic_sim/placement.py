from __future__ import annotations

from dataclasses import dataclass
import math

from .hardware import HardwareSpec
from .models import ModelSpec


@dataclass(frozen=True, slots=True)
class ModelDecomposition:
    """Parameter decomposition inferred from published total/active counts.

    Routed-vs-always-on is inferred from the published top-k activation ratio.
    Shared experts are then carved explicitly out of the always-on pool using
    the vendor architecture dimensions. This keeps the total/active counts
    exactly consistent while exposing the shared path as a first-class cost.
    """

    total_parameters: float
    active_parameters: float
    routed_pool_parameters: float
    always_on_parameters: float
    active_routed_parameters: float
    active_always_on_parameters: float
    routed_fraction: float
    expert_shard_parameters: float | None
    shared_expert_parameters: float
    shared_expert_parameters_per_moe_layer: float
    other_always_on_parameters: float


def decompose_model(model: ModelSpec) -> ModelDecomposition:
    if not model.num_experts or not model.experts_per_token or not model.moe_layers:
        return ModelDecomposition(
            total_parameters=model.total_parameters,
            active_parameters=model.active_parameters,
            routed_pool_parameters=0.0,
            always_on_parameters=model.total_parameters,
            active_routed_parameters=0.0,
            active_always_on_parameters=model.active_parameters,
            routed_fraction=0.0,
            expert_shard_parameters=None,
            shared_expert_parameters=0.0,
            shared_expert_parameters_per_moe_layer=0.0,
            other_always_on_parameters=model.total_parameters,
        )

    activation_fraction = model.experts_per_token / model.num_experts
    if activation_fraction >= 1.0:
        raise ValueError("MoE activation fraction must be < 1 for decomposition")

    routed_pool = (model.total_parameters - model.active_parameters) / (1.0 - activation_fraction)
    always_on = model.total_parameters - routed_pool
    active_routed = routed_pool * activation_fraction
    active_always_on = model.active_parameters - active_routed

    tolerance = max(1.0, model.total_parameters * 1e-10)
    if routed_pool < -tolerance or always_on < -tolerance or active_always_on < -tolerance:
        raise ValueError("published model counts are inconsistent with the simple equal-expert top-k model")

    routed_pool = max(0.0, routed_pool)
    always_on = max(0.0, always_on)
    active_always_on = max(0.0, active_always_on)
    shard = routed_pool / (model.moe_layers * model.num_experts)

    explicit_shared = model.shared_expert_parameters_total
    # Architecture dimensions should fit inside the inferred always-on pool. If
    # a future model's vendor counts disagree slightly, preserve count
    # consistency and cap the explicit carve-out rather than going negative.
    shared = min(always_on, explicit_shared)
    shared_per_layer = shared / model.moe_layers if model.moe_layers else 0.0
    other_always_on = max(0.0, always_on - shared)

    return ModelDecomposition(
        total_parameters=model.total_parameters,
        active_parameters=model.active_parameters,
        routed_pool_parameters=routed_pool,
        always_on_parameters=always_on,
        active_routed_parameters=active_routed,
        active_always_on_parameters=active_always_on,
        routed_fraction=routed_pool / model.total_parameters,
        expert_shard_parameters=shard,
        shared_expert_parameters=shared,
        shared_expert_parameters_per_moe_layer=shared_per_layer,
        other_always_on_parameters=other_always_on,
    )


@dataclass(frozen=True, slots=True)
class PlacementReport:
    model: str
    hardware: str
    bits_per_weight: float
    overhead_fraction: float
    total_storage_bytes: float
    average_storage_per_tile_bytes: float
    max_estimated_storage_per_tile_bytes: float
    tile_capacity_bytes: float
    resident_system: bool
    resident_per_tile_balanced: bool
    routed_pool_parameters: float
    always_on_parameters: float
    expert_shard_parameters: float | None
    expert_shards_total: int
    expert_shards_min_per_tile: int
    expert_shards_max_per_tile: int
    expected_remote_expert_fraction: float
    expected_moe_activation_bytes_per_token: float
    average_hops: float


def balanced_placement(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    bits_per_weight: float = 4.0,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
) -> PlacementReport:
    """Estimate a balanced round-robin placement without materializing experts."""
    if hardware.tiles < 1:
        raise ValueError("hardware must contain at least one tile")
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")

    decomposition = decompose_model(model)
    bytes_per_parameter = bits_per_weight / 8.0 * (1.0 + overhead_fraction)
    total_storage = model.total_parameters * bytes_per_parameter
    average = total_storage / hardware.tiles

    shard_count = model.moe_layers * model.num_experts if model.num_experts else 0
    min_shards = shard_count // hardware.tiles if shard_count else 0
    max_shards = math.ceil(shard_count / hardware.tiles) if shard_count else 0

    always_on_per_tile = decomposition.always_on_parameters / hardware.tiles
    max_routed_params = max_shards * (decomposition.expert_shard_parameters or 0.0)
    max_storage = (always_on_per_tile + max_routed_params) * bytes_per_parameter

    if hardware.tile_capacity_gb is not None:
        tile_capacity = hardware.tile_capacity_gb * 1e9
    else:
        tile_capacity = hardware.capacity_bytes

    expected_remote_fraction = 0.0 if hardware.tiles == 1 else 1.0 - (1.0 / hardware.tiles)
    moe_bytes = model.approximate_moe_network_bytes(
        activation_bits=activation_bits,
        remote_expert_fraction=expected_remote_fraction,
    )

    return PlacementReport(
        model=model.name,
        hardware=hardware.name,
        bits_per_weight=bits_per_weight,
        overhead_fraction=overhead_fraction,
        total_storage_bytes=total_storage,
        average_storage_per_tile_bytes=average,
        max_estimated_storage_per_tile_bytes=max_storage,
        tile_capacity_bytes=tile_capacity,
        resident_system=total_storage <= hardware.capacity_bytes,
        resident_per_tile_balanced=max_storage <= tile_capacity,
        routed_pool_parameters=decomposition.routed_pool_parameters,
        always_on_parameters=decomposition.always_on_parameters,
        expert_shard_parameters=decomposition.expert_shard_parameters,
        expert_shards_total=shard_count,
        expert_shards_min_per_tile=min_shards,
        expert_shards_max_per_tile=max_shards,
        expected_remote_expert_fraction=expected_remote_fraction,
        expected_moe_activation_bytes_per_token=moe_bytes,
        average_hops=hardware.average_manhattan_hops,
    )
