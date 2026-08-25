from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareSpec
from .models import ModelSpec


@dataclass(frozen=True, slots=True)
class SimulationResult:
    model: str
    hardware: str
    bits_per_weight: float
    quant_overhead_fraction: float
    activation_bits: float
    storage_bytes: float
    capacity_bytes: float
    resident: bool
    max_bits_that_fit: float
    active_weight_bytes_per_token: float
    remote_activation_bytes_per_token: float
    local_data_fraction: float
    memory_time_single_stream_s: float | None
    noc_time_ideal_s: float | None
    router_time_ideal_s: float | None
    single_stream_memory_noc_roofline_tps: float | None
    steady_state_memory_roofline_tps: float | None
    warning: str = ""


def simulate_decode(
    model: ModelSpec,
    hardware: HardwareSpec,
    *,
    bits_per_weight: float = 4.0,
    quant_overhead_fraction: float = 0.05,
    activation_bits: float = 16.0,
    remote_expert_fraction: float | None = None,
) -> SimulationResult:
    """Run the M0 decode data-movement roofline."""
    storage = model.storage_bytes(bits_per_weight, quant_overhead_fraction)
    resident = storage <= hardware.capacity_bytes
    max_bits = model.minimum_bits_for_capacity(hardware.capacity_bytes, quant_overhead_fraction)
    active_bytes = model.active_weight_bytes(bits_per_weight, quant_overhead_fraction)

    if remote_expert_fraction is None:
        remote_expert_fraction = 1.0 if hardware.tiles > 1 else 0.0

    remote_bytes = model.approximate_moe_network_bytes(
        activation_bits=activation_bits,
        remote_expert_fraction=remote_expert_fraction,
    )
    total_relevant_bytes = active_bytes + remote_bytes
    local_fraction = active_bytes / total_relevant_bytes if total_relevant_bytes else 1.0

    if not resident:
        warning = "model weights do not fit in accelerator capacity; off-package paging is not modeled"
        return SimulationResult(
            model=model.name,
            hardware=hardware.name,
            bits_per_weight=bits_per_weight,
            quant_overhead_fraction=quant_overhead_fraction,
            activation_bits=activation_bits,
            storage_bytes=storage,
            capacity_bytes=hardware.capacity_bytes,
            resident=False,
            max_bits_that_fit=max_bits,
            active_weight_bytes_per_token=active_bytes,
            remote_activation_bytes_per_token=remote_bytes,
            local_data_fraction=local_fraction,
            memory_time_single_stream_s=None,
            noc_time_ideal_s=None,
            router_time_ideal_s=None,
            single_stream_memory_noc_roofline_tps=None,
            steady_state_memory_roofline_tps=None,
            warning=warning,
        )

    memory_time_single = active_bytes / hardware.local_memory_bandwidth_bytes_s

    noc_time = 0.0
    router_time = 0.0
    if remote_bytes and hardware.noc_link_bandwidth_bytes_s:
        noc_time = remote_bytes / hardware.noc_link_bandwidth_bytes_s
        router_time = (
            model.moe_layers
            * model.experts_per_token
            * 2.0
            * hardware.average_manhattan_hops
            * hardware.router_latency_ns
            * 1e-9
            * remote_expert_fraction
        )

    latency = memory_time_single + noc_time + router_time
    single_tps = 1.0 / latency if latency > 0 else None

    throughput_time = active_bytes / hardware.aggregate_memory_bandwidth_bytes_s
    steady_tps = 1.0 / throughput_time if throughput_time > 0 else None

    warning_parts: list[str] = []
    if hardware.tiles > 1:
        warning_parts.append("NoC result is injection-only and excludes contention/bisection limits")
        warning_parts.append("steady-state TPS assumes ideal pipeline/load balance across tiles")

    return SimulationResult(
        model=model.name,
        hardware=hardware.name,
        bits_per_weight=bits_per_weight,
        quant_overhead_fraction=quant_overhead_fraction,
        activation_bits=activation_bits,
        storage_bytes=storage,
        capacity_bytes=hardware.capacity_bytes,
        resident=True,
        max_bits_that_fit=max_bits,
        active_weight_bytes_per_token=active_bytes,
        remote_activation_bytes_per_token=remote_bytes,
        local_data_fraction=local_fraction,
        memory_time_single_stream_s=memory_time_single,
        noc_time_ideal_s=noc_time,
        router_time_ideal_s=router_time,
        single_stream_memory_noc_roofline_tps=single_tps,
        steady_state_memory_roofline_tps=steady_tps,
        warning="; ".join(warning_parts),
    )
