from __future__ import annotations

from dataclasses import dataclass
import math


GB = 1e9
TB = 1e12


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """M0 hardware description."""

    key: str
    name: str
    capacity_gb: float
    memory_bandwidth_tb_s: float
    tiles: int = 1
    tile_capacity_gb: float | None = None
    tile_memory_bandwidth_tb_s: float | None = None
    noc_link_bandwidth_tb_s: float | None = None
    mesh_rows: int | None = None
    mesh_cols: int | None = None
    router_latency_ns: float = 0.0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.capacity_gb <= 0 or self.memory_bandwidth_tb_s <= 0:
            raise ValueError("capacity and memory bandwidth must be positive")
        if self.tiles <= 0:
            raise ValueError("tiles must be positive")
        if self.tiles > 1:
            if self.tile_capacity_gb is None or self.tile_memory_bandwidth_tb_s is None:
                raise ValueError("tiled hardware requires tile capacity and tile bandwidth")
            if self.mesh_rows is not None and self.mesh_cols is not None:
                if self.mesh_rows * self.mesh_cols < self.tiles:
                    raise ValueError("mesh dimensions cannot contain all tiles")

    @property
    def capacity_bytes(self) -> float:
        return self.capacity_gb * GB

    @property
    def aggregate_memory_bandwidth_bytes_s(self) -> float:
        return self.memory_bandwidth_tb_s * TB

    @property
    def local_memory_bandwidth_bytes_s(self) -> float:
        if self.tile_memory_bandwidth_tb_s is None:
            return self.aggregate_memory_bandwidth_bytes_s
        return self.tile_memory_bandwidth_tb_s * TB

    @property
    def noc_link_bandwidth_bytes_s(self) -> float | None:
        if self.noc_link_bandwidth_tb_s is None:
            return None
        return self.noc_link_bandwidth_tb_s * TB

    @property
    def average_manhattan_hops(self) -> float:
        if self.tiles <= 1:
            return 0.0
        rows, cols = self._mesh_shape()
        row_distance = (rows**2 - 1) / (3 * rows) if rows > 1 else 0.0
        col_distance = (cols**2 - 1) / (3 * cols) if cols > 1 else 0.0
        return row_distance + col_distance

    def _mesh_shape(self) -> tuple[int, int]:
        if self.mesh_rows and self.mesh_cols:
            return self.mesh_rows, self.mesh_cols
        rows = max(1, int(math.sqrt(self.tiles)))
        cols = math.ceil(self.tiles / rows)
        return rows, cols


HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "hbm4-illustrative": HardwareSpec(
        key="hbm4-illustrative",
        name="Illustrative HBM4 accelerator",
        capacity_gb=192,
        memory_bandwidth_tb_s=18.0,
        notes=(
            "Deliberately generic comparison point, not a claim about a specific shipping GPU. "
            "Edit or override it when comparing against a concrete accelerator."
        ),
    ),
    "raptor-like": HardwareSpec(
        key="raptor-like",
        name="Raptor-like 3D-DRAM tile",
        capacity_gb=32,
        memory_bandwidth_tb_s=105.0,
        notes=(
            "Research reference using the public ~32 GB / ~100 TB/s-class 3D-DRAM point. "
            "The simulator does not claim to reproduce d-Matrix microarchitecture."
        ),
    ),
    "fabric-32x32": HardwareSpec(
        key="fabric-32x32",
        name="32 x 32 GB distributed 3D-DRAM fabric",
        capacity_gb=1024,
        memory_bandwidth_tb_s=3360.0,
        tiles=32,
        tile_capacity_gb=32,
        tile_memory_bandwidth_tb_s=105.0,
        noc_link_bandwidth_tb_s=2.0,
        mesh_rows=4,
        mesh_cols=8,
        router_latency_ns=5.0,
        notes=(
            "Hypothetical research target: 32 autonomous memory-compute tiles on a 4x8 mesh. "
            "Aggregate bandwidth is only used for steady-state throughput; a single token does not "
            "automatically receive 3.36 PB/s."
        ),
    ),
    "fabric-64x32": HardwareSpec(
        key="fabric-64x32",
        name="64 x 32 GB distributed 3D-DRAM fabric",
        capacity_gb=2048,
        memory_bandwidth_tb_s=6720.0,
        tiles=64,
        tile_capacity_gb=32,
        tile_memory_bandwidth_tb_s=105.0,
        noc_link_bandwidth_tb_s=2.0,
        mesh_rows=8,
        mesh_cols=8,
        router_latency_ns=5.0,
        notes="2 TB-class capacity option added primarily because Kimi K3 does not fit in 1 TB at 4-bit.",
    ),
}

ALIASES = {
    "hbm": "hbm4-illustrative",
    "raptor": "raptor-like",
    "fabric32": "fabric-32x32",
    "fabric64": "fabric-64x32",
}


def get_hardware(key: str) -> HardwareSpec:
    normalized = key.strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    try:
        return HARDWARE_PRESETS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(HARDWARE_PRESETS))
        raise KeyError(f"unknown hardware {key!r}; available: {available}") from exc
