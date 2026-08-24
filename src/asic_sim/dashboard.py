from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareSpec, get_hardware
from .models import ModelSpec, get_model
from .placement import PlacementReport, balanced_placement
from .simulator import SimulationResult, simulate_decode


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """One coherent, human-facing view of a model/hardware experiment."""

    model: ModelSpec
    hardware: HardwareSpec
    result: SimulationResult
    placement: PlacementReport
    capacity_utilization: float
    tile_utilization: float
    remote_traffic_fraction: float
    total_ideal_latency_s: float | None
    memory_latency_fraction: float | None
    noc_latency_fraction: float | None
    router_latency_fraction: float | None


def build_snapshot(
    model_key: str,
    hardware_key: str,
    *,
    bits_per_weight: float = 4.0,
    overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
) -> DashboardSnapshot:
    if bits_per_weight <= 0:
        raise ValueError("bits_per_weight must be positive")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction cannot be negative")

    model = get_model(model_key)
    hardware = get_hardware(hardware_key)
    placement = balanced_placement(
        model,
        hardware,
        bits_per_weight=bits_per_weight,
        overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
    )

    # The dashboard uses the same balanced-placement remote-expert estimate that
    # it displays, avoiding the confusing 100%-remote conservative CLI default.
    result = simulate_decode(
        model,
        hardware,
        bits_per_weight=bits_per_weight,
        quant_overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
        remote_expert_fraction=placement.expected_remote_expert_fraction,
    )

    capacity_utilization = result.storage_bytes / result.capacity_bytes
    tile_utilization = (
        placement.max_estimated_storage_per_tile_bytes / placement.tile_capacity_bytes
        if placement.tile_capacity_bytes
        else 0.0
    )
    relevant_bytes = result.active_weight_bytes_per_token + result.remote_activation_bytes_per_token
    remote_traffic_fraction = (
        result.remote_activation_bytes_per_token / relevant_bytes if relevant_bytes else 0.0
    )

    if result.resident:
        memory = result.memory_time_single_stream_s or 0.0
        noc = result.noc_time_ideal_s or 0.0
        router = result.router_time_ideal_s or 0.0
        total = memory + noc + router
        if total > 0:
            memory_fraction = memory / total
            noc_fraction = noc / total
            router_fraction = router / total
        else:
            memory_fraction = noc_fraction = router_fraction = 0.0
    else:
        total = None
        memory_fraction = noc_fraction = router_fraction = None

    return DashboardSnapshot(
        model=model,
        hardware=hardware,
        result=result,
        placement=placement,
        capacity_utilization=capacity_utilization,
        tile_utilization=tile_utilization,
        remote_traffic_fraction=remote_traffic_fraction,
        total_ideal_latency_s=total,
        memory_latency_fraction=memory_fraction,
        noc_latency_fraction=noc_fraction,
        router_latency_fraction=router_fraction,
    )


def interpretation(snapshot: DashboardSnapshot) -> tuple[str, ...]:
    """Return terse interpretations intended for the TUI and future GUIs."""
    r = snapshot.result
    p = snapshot.placement
    lines: list[str] = []

    if not r.resident:
        lines.append(
            f"STOP: the model needs {snapshot.capacity_utilization:.1%} of system capacity, so it is not resident."
        )
        lines.append(
            f"At the current overhead, this hardware can fit at most {r.max_bits_that_fit:.2f} average bits/weight."
        )
        return tuple(lines)

    headroom = max(0.0, 1.0 - snapshot.capacity_utilization)
    if headroom >= 0.20:
        lines.append(f"Capacity looks comfortable at this abstraction: {headroom:.1%} system headroom remains.")
    else:
        lines.append(f"Capacity is tight: only {headroom:.1%} system headroom remains before KV/cache/runtime overheads.")

    if snapshot.hardware.tiles > 1:
        lines.append(
            f"{p.expected_remote_expert_fraction:.1%} of selected experts are expected to be remote under naive balanced placement, "
            f"but remote activations are only {snapshot.remote_traffic_fraction:.4%} of modeled bytes."
        )
        lines.append(
            "That is the core memory-stationary thesis: expert calls move across the mesh; expert weights do not."
        )

    if snapshot.total_ideal_latency_s and snapshot.memory_latency_fraction is not None:
        fractions = {
            "local-memory streaming": snapshot.memory_latency_fraction,
            "NoC injection": snapshot.noc_latency_fraction or 0.0,
            "router traversal": snapshot.router_latency_fraction or 0.0,
        }
        dominant = max(fractions, key=fractions.get)
        lines.append(
            f"In the current ideal single-stream model, {dominant} dominates modeled latency ({fractions[dominant]:.1%})."
        )

    single = r.single_stream_memory_noc_roofline_tps
    steady = r.steady_state_memory_roofline_tps
    if single and steady:
        lines.append(
            f"The {steady / single:.1f}x gap between single-stream and steady-state roofs is pipeline opportunity, not free latency speedup."
        )

    lines.append(
        "Do not treat either TPS number as product performance yet: compute, KV traffic, NoC contention, bank conflicts and thermals are not modeled."
    )
    return tuple(lines)
