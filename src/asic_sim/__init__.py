"""asic-sim: fast architecture exploration for memory-stationary LLM inference."""

from .dashboard import DashboardSnapshot, build_snapshot, interpretation
from .hardware import HARDWARE_PRESETS, HardwareSpec, get_hardware
from .models import MODEL_SPECS, ModelSpec, get_model
from .placement import ModelDecomposition, PlacementReport, balanced_placement, decompose_model
from .simulator import SimulationResult, simulate_decode

__all__ = [
    "DashboardSnapshot",
    "HARDWARE_PRESETS",
    "MODEL_SPECS",
    "HardwareSpec",
    "ModelSpec",
    "ModelDecomposition",
    "PlacementReport",
    "SimulationResult",
    "build_snapshot",
    "interpretation",
    "get_hardware",
    "get_model",
    "balanced_placement",
    "decompose_model",
    "simulate_decode",
]

__version__ = "0.2.0"
