from __future__ import annotations

from dataclasses import dataclass

from .fabric import PhysicalPlacement, TrafficReport, build_physical_placement, trace_traffic
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
    physical_placement: PhysicalPlacement
    traffic: TrafficReport
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
    routing_profile: str = "balanced",
    trace_tokens: int = 16,
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
    physical = build_physical_placement(
        model,
        hardware,
        bits_per_weight=bits_per_weight,
        overhead_fraction=overhead_fraction,
    )
    traffic = trace_traffic(
        physical,
        tokens=trace_tokens,
        profile=routing_profile,
        activation_bits=activation_bits,
    )

    # Feed the physically observed remote-dispatch fraction back into the M0
    # roofline. M0 still uses its simple latency model; M1 now supplies a real
    # deterministic placement/routing trace rather than 1 - 1/N by fiat.
    result = simulate_decode(
        model,
        hardware,
        bits_per_weight=bits_per_weight,
        quant_overhead_fraction=overhead_fraction,
        activation_bits=activation_bits,
        remote_expert_fraction=traffic.remote_expert_fraction,
    )

    capacity_utilization = result.storage_bytes / result.capacity_bytes
    tile_utilization = (
        physical.max_storage_bytes / physical.tile_capacity_bytes
        if physical.tile_capacity_bytes
        else 0.0
    )
    relevant_bytes = result.active_weight_bytes_per_token + traffic.network_payload_bytes_per_token
    remote_traffic_fraction = (
        traffic.network_payload_bytes_per_token / relevant_bytes if relevant_bytes else 0.0
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
        physical_placement=physical,
        traffic=traffic,
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
    physical = snapshot.physical_placement
    traffic = snapshot.traffic
    lines: list[str] = []

    if not r.resident:
        lines.append(
            f"STOP: the model needs {snapshot.capacity_utilization:.1%} of system capacity, so it is not resident."
        )
        lines.append(
            f"At the current overhead, this hardware can fit at most {r.max_bits_that_fit:.2f} average bits/weight."
        )
        return tuple(lines)

    if not physical.per_tile_resident:
        lines.append(
            "STOP: total capacity fits, but the explicit M1 placement overflows at least one local tile."
        )

    headroom = max(0.0, 1.0 - snapshot.capacity_utilization)
    if headroom >= 0.20:
        lines.append(f"Capacity looks comfortable at this abstraction: {headroom:.1%} system headroom remains.")
    else:
        lines.append(f"Capacity is tight: only {headroom:.1%} system headroom remains before KV/cache/runtime overheads.")

    if snapshot.hardware.tiles > 1:
        if snapshot.model.num_experts:
            lines.append(
                f"The physical balanced trace sends {traffic.remote_expert_fraction:.1%} of expert dispatches off-tile, "
                f"yet all NoC payload is only {snapshot.remote_traffic_fraction:.4%} of modeled bytes."
            )
            lines.append(
                "That is the memory-stationary thesis in concrete routes: activations move; expert weights remain on their owning tiles."
            )
        else:
            lines.append(
                f"Dense layer partitioning produces {traffic.network_payload_bytes_per_token / 1e6:.3f} MB/token of inter-tile activation traffic."
            )

        if traffic.hottest_link is not None:
            left, right = traffic.hottest_link
            lines.append(
                f"M1 hot link is T{left:02d}->T{right:02d}: {traffic.max_link_bytes_per_token / 1e6:.3f} MB/token, "
                f"{traffic.hotspot_ratio:.2f}x the mean active-link load."
            )

    if snapshot.total_ideal_latency_s and snapshot.memory_latency_fraction is not None:
        fractions = {
            "local-memory streaming": snapshot.memory_latency_fraction,
            "NoC injection": snapshot.noc_latency_fraction or 0.0,
            "router traversal": snapshot.router_latency_fraction or 0.0,
        }
        dominant = max(fractions, key=fractions.get)
        lines.append(
            f"In the current ideal single-stream M0 model, {dominant} dominates modeled latency ({fractions[dominant]:.1%})."
        )

    single = r.single_stream_memory_noc_roofline_tps
    steady = r.steady_state_memory_roofline_tps
    if single and steady:
        lines.append(
            f"The {steady / single:.1f}x gap between single-stream and steady-state roofs is pipeline opportunity, not free latency speedup."
        )

    lines.append(
        "M1 now measures exact route/link loads for a synthetic token trace, but it is not cycle-accurate: queueing, overlap, compute, KV traffic, bank conflicts and thermals remain unmodeled."
    )
    return tuple(lines)
