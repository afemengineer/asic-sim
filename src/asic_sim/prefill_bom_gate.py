from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from .models import ModelSpec, get_model
from .prefill import (
    PrefillWorkload,
    REFERENCE_HARDWARE,
    candidate_hardware,
    compare_economics,
    simulate_prefill,
)

GB = 1e9
USD_TO_EUR_DEFAULT = 0.85673


@dataclass(frozen=True, slots=True)
class GDDR7Device:
    """A concrete GDDR7 package used to turn capacity into a physical chip count."""

    name: str = "Samsung K4VAF325ZC-SC28"
    capacity_gb: float = 2.0
    data_rate_gbps_per_pin: float = 28.0
    bus_width_bits: int = 32
    quoted_price_usd: float = 46.50
    price_source: str = "https://microworkskorea.tistory.com/2659"
    spec_source: str = "https://semiconductor.samsung.com/us/dram/gddr/gddr7/k4vaf325zc-sc28/"
    price_date: str = "2026-08-24"

    @property
    def bandwidth_gb_s(self) -> float:
        return self.data_rate_gbps_per_pin * self.bus_width_bits / 8.0


@dataclass(frozen=True, slots=True)
class VolumeCase:
    units: int
    memory_price_multiplier: float


DEFAULT_VOLUMES = (
    VolumeCase(100, 1.00),
    VolumeCase(1_000, 0.80),
    VolumeCase(10_000, 0.65),
)


@dataclass(frozen=True, slots=True)
class BomAssumptions:
    compute_pops: float = 64.0
    bandwidth_tb_s: float = 8.0
    capacity_gb: float = 2048.0
    candidate_power_w: float = 1200.0
    compute_efficiency: float = 0.45
    memory_efficiency: float = 0.75
    attention_efficiency: float = 0.30
    chips_per_module: int = 16

    # Epoch AI estimates B200 variable logic fabrication at about $900 for a
    # 10-PFLOP dense-FP4 package. We grant the custom inference-only design a
    # deliberately optimistic 2x compute-per-dollar specialization advantage.
    b200_logic_cost_usd: float = 900.0
    b200_dense_fp4_pops: float = 10.0
    specialization_density_gain: float = 2.0

    # Epoch's B200 auxiliary estimate is $480 for VRMs, PCB and module assembly.
    # We use it as a per-module proxy, despite our module being very different.
    module_auxiliary_usd: float = 480.0
    simple_package_usd: float = 100.0
    module_fabric_usd: float = 100.0

    # Optimistic appliance-level floors. They intentionally exclude sales,
    # support, software engineering, spares, inventory financing and warranty.
    chassis_power_cooling_eur: float = 7000.0
    external_network_eur: float = 1990.0
    variable_cost_contingency: float = 0.10

    # 5-nm full-mask NRE reference. A leading-edge program can be much higher;
    # the default is deliberately not a worst-case value.
    nre_usd: float = 47_000_000.0
    gross_margin: float = 0.30
    usd_to_eur: float = USD_TO_EUR_DEFAULT


@dataclass(frozen=True, slots=True)
class PhysicalMemoryFloor:
    model_weight_bytes: float
    gddr_chips: int
    modules: int
    capacity_per_module_gb: float
    raw_bandwidth_per_module_tb_s: float
    raw_aggregate_bandwidth_tb_s: float
    target_to_raw_bandwidth_fraction: float


@dataclass(frozen=True, slots=True)
class VolumeBomResult:
    units: int
    memory_price_multiplier: float
    memory_chip_price_usd: float
    memory_eur: float
    logic_silicon_eur: float
    module_auxiliary_eur: float
    module_package_eur: float
    module_fabric_eur: float
    chassis_power_cooling_eur: float
    external_network_eur: float
    variable_bom_before_contingency_eur: float
    variable_bom_eur: float
    nre_amortized_eur: float
    fully_loaded_cost_eur: float
    required_sell_price_eur: float
    passes_3x: bool
    passes_5x: bool
    headroom_3x_eur: float
    headroom_5x_eur: float


@dataclass(frozen=True, slots=True)
class BomGateResult:
    model_name: str
    candidate_prefill_s: float
    reference_prefill_s: float
    candidate_input_tps: float
    reference_input_tps: float
    slowdown_vs_reference: float
    capex_1x_eur: float
    capex_3x_eur: float
    capex_5x_eur: float
    memory: PhysicalMemoryFloor
    volumes: tuple[VolumeBomResult, ...]


def physical_memory_floor(
    model: ModelSpec,
    gddr: GDDR7Device,
    assumptions: BomAssumptions,
    *,
    weight_bits: float = 4.0,
    overhead_fraction: float = 0.05,
) -> PhysicalMemoryFloor:
    if gddr.capacity_gb <= 0 or gddr.bandwidth_gb_s <= 0:
        raise ValueError("GDDR capacity and bandwidth must be positive")
    if assumptions.chips_per_module <= 0:
        raise ValueError("chips_per_module must be positive")
    if assumptions.bandwidth_tb_s <= 0:
        raise ValueError("bandwidth_tb_s must be positive")

    weight_bytes = model.storage_bytes(weight_bits, overhead_fraction)
    chips = math.ceil(weight_bytes / (gddr.capacity_gb * GB))
    modules = math.ceil(chips / assumptions.chips_per_module)
    capacity_per_module_gb = assumptions.chips_per_module * gddr.capacity_gb
    raw_module_tb_s = assumptions.chips_per_module * gddr.bandwidth_gb_s / 1000.0
    raw_total_tb_s = chips * gddr.bandwidth_gb_s / 1000.0

    return PhysicalMemoryFloor(
        model_weight_bytes=weight_bytes,
        gddr_chips=chips,
        modules=modules,
        capacity_per_module_gb=capacity_per_module_gb,
        raw_bandwidth_per_module_tb_s=raw_module_tb_s,
        raw_aggregate_bandwidth_tb_s=raw_total_tb_s,
        target_to_raw_bandwidth_fraction=assumptions.bandwidth_tb_s / raw_total_tb_s,
    )


def _logic_silicon_cost_usd(assumptions: BomAssumptions) -> float:
    values = (
        assumptions.compute_pops,
        assumptions.b200_logic_cost_usd,
        assumptions.b200_dense_fp4_pops,
        assumptions.specialization_density_gain,
    )
    if min(values) <= 0:
        raise ValueError("compute and logic-cost assumptions must be positive")
    baseline_cost_per_pops = (
        assumptions.b200_logic_cost_usd / assumptions.b200_dense_fp4_pops
    )
    return (
        assumptions.compute_pops
        * baseline_cost_per_pops
        / assumptions.specialization_density_gain
    )


def _validate_assumptions(assumptions: BomAssumptions) -> None:
    if not 0 <= assumptions.variable_cost_contingency < 1:
        raise ValueError("variable_cost_contingency must be in [0, 1)")
    if not 0 <= assumptions.gross_margin < 1:
        raise ValueError("gross_margin must be in [0, 1)")
    if assumptions.nre_usd < 0:
        raise ValueError("nre_usd cannot be negative")
    if assumptions.usd_to_eur <= 0:
        raise ValueError("usd_to_eur must be positive")
    for value in (
        assumptions.module_auxiliary_usd,
        assumptions.simple_package_usd,
        assumptions.module_fabric_usd,
        assumptions.chassis_power_cooling_eur,
        assumptions.external_network_eur,
    ):
        if value < 0:
            raise ValueError("BOM cost assumptions cannot be negative")


def run_bom_gate(
    model: ModelSpec,
    *,
    workload: PrefillWorkload,
    assumptions: BomAssumptions = BomAssumptions(),
    gddr: GDDR7Device = GDDR7Device(),
    volume_cases: tuple[VolumeCase, ...] = DEFAULT_VOLUMES,
    reference_key: str = "b300-eu",
    overhead_fraction: float = 0.05,
) -> BomGateResult:
    _validate_assumptions(assumptions)
    if not volume_cases:
        raise ValueError("at least one volume case is required")
    for case in volume_cases:
        if case.units <= 0 or not 0 < case.memory_price_multiplier <= 1:
            raise ValueError("volume units must be positive and memory multiplier in (0, 1]")

    reference_hw = REFERENCE_HARDWARE[reference_key]
    if reference_hw.purchase_price_eur is None:
        raise ValueError("reference hardware requires a purchase price")

    candidate_hw = candidate_hardware(
        compute_pops=assumptions.compute_pops,
        bandwidth_tb_s=assumptions.bandwidth_tb_s,
        capacity_gb=assumptions.capacity_gb,
        compute_efficiency=assumptions.compute_efficiency,
        memory_efficiency=assumptions.memory_efficiency,
        attention_efficiency=assumptions.attention_efficiency,
        power_w=assumptions.candidate_power_w,
    )
    candidate = simulate_prefill(model, candidate_hw, workload, overhead_fraction=overhead_fraction)
    reference = simulate_prefill(model, reference_hw, workload, overhead_fraction=overhead_fraction)
    economics = compare_economics(
        candidate,
        reference,
        reference_price_eur=reference_hw.purchase_price_eur,
        candidate_power_w=candidate_hw.power_w,
        reference_power_w=reference_hw.power_w,
    )
    capex_1x = economics.capex_ceiling_1x_eur
    capex_3x = capex_1x / 3.0
    capex_5x = capex_1x / 5.0

    memory = physical_memory_floor(
        model,
        gddr,
        assumptions,
        weight_bits=workload.weight_bits,
        overhead_fraction=overhead_fraction,
    )

    fx = assumptions.usd_to_eur
    logic_eur = _logic_silicon_cost_usd(assumptions) * fx
    module_aux_eur = memory.modules * assumptions.module_auxiliary_usd * fx
    package_eur = memory.modules * assumptions.simple_package_usd * fx
    fabric_eur = memory.modules * assumptions.module_fabric_usd * fx

    rows: list[VolumeBomResult] = []
    for case in volume_cases:
        memory_price_usd = gddr.quoted_price_usd * case.memory_price_multiplier
        memory_eur = memory.gddr_chips * memory_price_usd * fx
        before_contingency = (
            memory_eur
            + logic_eur
            + module_aux_eur
            + package_eur
            + fabric_eur
            + assumptions.chassis_power_cooling_eur
            + assumptions.external_network_eur
        )
        variable_bom = before_contingency * (1.0 + assumptions.variable_cost_contingency)
        nre_eur = assumptions.nre_usd * fx / case.units
        loaded = variable_bom + nre_eur
        sell = loaded / (1.0 - assumptions.gross_margin)
        rows.append(
            VolumeBomResult(
                units=case.units,
                memory_price_multiplier=case.memory_price_multiplier,
                memory_chip_price_usd=memory_price_usd,
                memory_eur=memory_eur,
                logic_silicon_eur=logic_eur,
                module_auxiliary_eur=module_aux_eur,
                module_package_eur=package_eur,
                module_fabric_eur=fabric_eur,
                chassis_power_cooling_eur=assumptions.chassis_power_cooling_eur,
                external_network_eur=assumptions.external_network_eur,
                variable_bom_before_contingency_eur=before_contingency,
                variable_bom_eur=variable_bom,
                nre_amortized_eur=nre_eur,
                fully_loaded_cost_eur=loaded,
                required_sell_price_eur=sell,
                passes_3x=sell <= capex_3x,
                passes_5x=sell <= capex_5x,
                headroom_3x_eur=capex_3x - sell,
                headroom_5x_eur=capex_5x - sell,
            )
        )

    return BomGateResult(
        model_name=model.name,
        candidate_prefill_s=candidate.prefill_time_s,
        reference_prefill_s=reference.prefill_time_s,
        candidate_input_tps=candidate.input_tokens_per_s,
        reference_input_tps=reference.input_tokens_per_s,
        slowdown_vs_reference=candidate.prefill_time_s / reference.prefill_time_s,
        capex_1x_eur=capex_1x,
        capex_3x_eur=capex_3x,
        capex_5x_eur=capex_5x,
        memory=memory,
        volumes=tuple(rows),
    )


def _fmt_eur(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:.2f}M"
    return f"€{value / 1000:.1f}k"


def _parse_volume(value: str) -> VolumeCase:
    try:
        units_text, multiplier_text = value.split(":", 1)
        units = int(units_text)
        multiplier = float(multiplier_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("volume must look like UNITS:MEMORY_MULTIPLIER") from exc
    if units <= 0 or not 0 < multiplier <= 1:
        raise argparse.ArgumentTypeError("units must be positive and multiplier in (0, 1]")
    return VolumeCase(units, multiplier)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asic-sim-prefill-bom",
        description="Bottom-up physical/BOM gate for the prefill accelerator design point.",
    )
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--tokens", type=int, default=32_768)
    parser.add_argument("--weight-bits", type=float, default=4.0)
    parser.add_argument("--overhead", type=float, default=0.05)
    parser.add_argument("--compute-pops", type=float, default=64.0)
    parser.add_argument("--bandwidth-tb-s", type=float, default=8.0)
    parser.add_argument("--capacity-gb", type=float, default=2048.0)
    parser.add_argument("--candidate-power-w", type=float, default=1200.0)
    parser.add_argument("--chips-per-module", type=int, default=16)
    parser.add_argument("--nre-usd", type=float, default=47_000_000.0)
    parser.add_argument("--gross-margin", type=float, default=0.30)
    parser.add_argument("--usd-eur", type=float, default=USD_TO_EUR_DEFAULT)
    parser.add_argument("--specialization-gain", type=float, default=2.0)
    parser.add_argument("--module-aux-usd", type=float, default=480.0)
    parser.add_argument("--module-package-usd", type=float, default=100.0)
    parser.add_argument("--module-fabric-usd", type=float, default=100.0)
    parser.add_argument("--chassis-eur", type=float, default=7000.0)
    parser.add_argument("--network-eur", type=float, default=1990.0)
    parser.add_argument("--contingency", type=float, default=0.10)
    parser.add_argument("--gddr-price-usd", type=float, default=46.50)
    parser.add_argument("--gddr-capacity-gb", type=float, default=2.0)
    parser.add_argument("--volume", action="append", type=_parse_volume)
    parser.add_argument("--reference", choices=tuple(sorted(REFERENCE_HARDWARE)), default="b300-eu")
    return parser


def _print_result(result: BomGateResult, assumptions: BomAssumptions, gddr: GDDR7Device) -> None:
    print(f"{result.model_name} bottom-up prefill BOM gate")
    print(
        f"Target: {assumptions.compute_pops:g} POPS / {assumptions.bandwidth_tb_s:g} TB/s | "
        f"candidate {result.candidate_prefill_s:.3f}s vs reference {result.reference_prefill_s:.3f}s "
        f"({result.slowdown_vs_reference:.2f}x slower)"
    )
    print(
        f"Economic ceilings: parity {_fmt_eur(result.capex_1x_eur)} | "
        f"3x {_fmt_eur(result.capex_3x_eur)} | 5x {_fmt_eur(result.capex_5x_eur)}"
    )
    print()
    mem = result.memory
    print("PHYSICAL MEMORY FLOOR")
    print(f"  resident weights: {mem.model_weight_bytes / 1e12:.3f} TB")
    print(
        f"  {gddr.name}: {gddr.capacity_gb:g} GB/chip, {gddr.bandwidth_gb_s:.0f} GB/s/chip, "
        f"quoted ${gddr.quoted_price_usd:.2f}/chip"
    )
    print(f"  required GDDR packages: {mem.gddr_chips:,}")
    print(
        f"  at {assumptions.chips_per_module} chips/module: {mem.modules} modules, "
        f"{mem.capacity_per_module_gb:g} GB/module, {mem.raw_bandwidth_per_module_tb_s:.3f} TB/s raw/module"
    )
    print(
        f"  raw aggregate GDDR bandwidth: {mem.raw_aggregate_bandwidth_tb_s:.1f} TB/s; "
        f"target uses only {mem.target_to_raw_bandwidth_fraction:.1%} of raw pin bandwidth"
    )
    print()
    print("VOLUME GATE")
    print(
        f"{'units':>8} {'GDDR/chip':>11} {'memory':>10} {'variable':>10} {'NRE/u':>10} "
        f"{'sell floor':>11} {'3x':>6} {'5x':>6}"
    )
    for row in result.volumes:
        print(
            f"{row.units:>8,} ${row.memory_chip_price_usd:>9.2f} {_fmt_eur(row.memory_eur):>10} "
            f"{_fmt_eur(row.variable_bom_eur):>10} {_fmt_eur(row.nre_amortized_eur):>10} "
            f"{_fmt_eur(row.required_sell_price_eur):>11} "
            f"{('PASS' if row.passes_3x else 'FAIL'):>6} {('PASS' if row.passes_5x else 'FAIL'):>6}"
        )
    print()
    print("Default discounts (100/1k/10k) are optimistic assumptions, not supplier quotes.")
    print("Excluded: software, support, warranty reserve, inventory financing, sales/channel margin, respins.")
    print(f"FX: 1 USD = {assumptions.usd_to_eur:.5f} EUR (2026-08-25 default).")
    print("Sources:")
    print(f"  GDDR quote: {gddr.price_source}")
    print(f"  GDDR spec:  {gddr.spec_source}")
    print("  B200 BOM:   https://epoch.ai/data-insights/b200-cost-breakdown")
    print("  NRE ref:    https://siliconandsteel.co/tools/market-data/")
    print("  800G NIC:   https://www.snswitch.com/products/900-9x81e-00ex-st0")


def main() -> int:
    args = build_parser().parse_args()
    assumptions = BomAssumptions(
        compute_pops=args.compute_pops,
        bandwidth_tb_s=args.bandwidth_tb_s,
        capacity_gb=args.capacity_gb,
        candidate_power_w=args.candidate_power_w,
        chips_per_module=args.chips_per_module,
        specialization_density_gain=args.specialization_gain,
        module_auxiliary_usd=args.module_aux_usd,
        simple_package_usd=args.module_package_usd,
        module_fabric_usd=args.module_fabric_usd,
        chassis_power_cooling_eur=args.chassis_eur,
        external_network_eur=args.network_eur,
        variable_cost_contingency=args.contingency,
        nre_usd=args.nre_usd,
        gross_margin=args.gross_margin,
        usd_to_eur=args.usd_eur,
    )
    gddr = GDDR7Device(capacity_gb=args.gddr_capacity_gb, quoted_price_usd=args.gddr_price_usd)
    volumes = tuple(args.volume) if args.volume else DEFAULT_VOLUMES
    workload = PrefillWorkload(prompt_tokens=args.tokens, weight_bits=args.weight_bits)
    result = run_bom_gate(
        get_model(args.model),
        workload=workload,
        assumptions=assumptions,
        gddr=gddr,
        volume_cases=volumes,
        reference_key=args.reference,
        overhead_fraction=args.overhead,
    )
    _print_result(result, assumptions, gddr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
