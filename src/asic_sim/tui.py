from __future__ import annotations

import math

from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Select, Static

from .dashboard import DashboardSnapshot, build_snapshot, interpretation
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .hardware import HARDWARE_PRESETS
from .models import MODEL_SPECS
from .simulator import simulate_decode


ORANGE = "#ff9d24"
GREEN = "#6bd968"
YELLOW = "#f0c75e"
RED = "#ff6b6b"
MUTED = "#8d939b"


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _bar(value: float, width: int = 28) -> Text:
    value = max(0.0, min(1.0, value))
    filled = round(width * value)
    text = Text()
    if value < 0.75:
        color = GREEN
    elif value <= 1.0:
        color = YELLOW
    else:
        color = RED
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style="#34383e")
    return text


def _metric(title: str, value: str, detail: str = "") -> Text:
    text = Text()
    text.append(title.upper() + "\n", style=f"bold {MUTED}")
    text.append(value, style=f"bold {ORANGE}")
    if detail:
        text.append("\n" + detail, style=MUTED)
    return text


def _tile_map(snapshot: DashboardSnapshot) -> Text:
    hw = snapshot.hardware
    p = snapshot.placement
    if hw.tiles == 1:
        load = p.max_estimated_storage_per_tile_bytes / p.tile_capacity_bytes
        return Text.assemble(
            ("T00 ", f"bold {ORANGE}"),
            (f"{_pct(load, 0)} full  ", "bold"),
            (f"{fmt_bytes(p.max_estimated_storage_per_tile_bytes)} / {fmt_bytes(p.tile_capacity_bytes)}", MUTED),
        )

    rows = hw.mesh_rows or max(1, int(math.sqrt(hw.tiles)))
    cols = hw.mesh_cols or math.ceil(hw.tiles / rows)
    load = p.max_estimated_storage_per_tile_bytes / p.tile_capacity_bytes
    color = GREEN if load < 0.75 else YELLOW if load <= 1.0 else RED
    text = Text()
    tile = 0
    for _row in range(rows):
        for _col in range(cols):
            if tile >= hw.tiles:
                break
            text.append(f" T{tile:02d} ", style=f"bold black on {color}")
            text.append(f" {_pct(load, 0):>4} ", style="#d7dadd")
            tile += 1
        text.append("\n")
    text.append(
        f"Balanced M1 estimate: ~{fmt_bytes(p.max_estimated_storage_per_tile_bytes)} used of "
        f"{fmt_bytes(p.tile_capacity_bytes)} per tile. Exact per-tile placement comes next.",
        style=MUTED,
    )
    return text


def _traffic_table(snapshot: DashboardSnapshot) -> Table:
    r = snapshot.result
    p = snapshot.placement
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("FLOW", style=MUTED)
    table.add_column("PER TOKEN", justify="right")
    table.add_column("INTERPRETATION")
    table.add_row("Active weights", fmt_bytes(r.active_weight_bytes_per_token), "Read locally beside compute")
    table.add_row("Remote activations", fmt_bytes(r.remote_activation_bytes_per_token), "Cross the tile mesh")
    table.add_row("Remote traffic share", _pct(snapshot.remote_traffic_fraction, 4), "Share of modeled bytes")
    if snapshot.hardware.tiles > 1:
        table.add_row("Remote expert calls", _pct(p.expected_remote_expert_fraction, 2), "Calls, not weight bytes")
        table.add_row("Average mesh hops", f"{p.average_hops:.3f}", "Naive balanced placement")
    return table


def _latency_table(snapshot: DashboardSnapshot) -> Table:
    r = snapshot.result
    table = Table(box=None, expand=True, pad_edge=False)
    table.add_column("COMPONENT", style=MUTED)
    table.add_column("IDEAL TIME", justify="right")
    table.add_column("SHARE", justify="right")
    if not r.resident:
        table.add_row("Unavailable", "—", "model does not fit")
        return table

    rows = (
        ("Local memory", r.memory_time_single_stream_s or 0.0, snapshot.memory_latency_fraction or 0.0),
        ("NoC injection", r.noc_time_ideal_s or 0.0, snapshot.noc_latency_fraction or 0.0),
        ("Router traversal", r.router_time_ideal_s or 0.0, snapshot.router_latency_fraction or 0.0),
    )
    for name, seconds, fraction in rows:
        table.add_row(name, fmt_time_s(seconds), _pct(fraction, 1))
    table.add_row("TOTAL", fmt_time_s(snapshot.total_ideal_latency_s or 0.0), "100%")
    return table


def _comparison_table(snapshot: DashboardSnapshot) -> Table:
    table = Table(expand=True, pad_edge=False)
    table.add_column("HARDWARE")
    table.add_column("FIT", justify="center")
    table.add_column("CAPACITY", justify="right")
    table.add_column("SINGLE STREAM", justify="right")
    table.add_column("STEADY BW ROOF", justify="right")

    model = snapshot.model
    bits = snapshot.result.bits_per_weight
    overhead = snapshot.result.quant_overhead_fraction
    for key, hw in HARDWARE_PRESETS.items():
        remote_fraction = 0.0 if hw.tiles == 1 else 1.0 - (1.0 / hw.tiles)
        result = simulate_decode(
            model,
            hw,
            bits_per_weight=bits,
            quant_overhead_fraction=overhead,
            remote_expert_fraction=remote_fraction,
        )
        fit = Text("YES", style=f"bold {GREEN}") if result.resident else Text("NO", style=f"bold {RED}")
        single = fmt_rate(result.single_stream_memory_noc_roofline_tps) if result.resident else "—"
        steady = fmt_rate(result.steady_state_memory_roofline_tps) if result.resident else "—"
        table.add_row(hw.name, fit, fmt_bytes(result.capacity_bytes), single, steady)
    return table


class AsicSimTui(App[None]):
    """Interactive architecture explorer for M0/M1 experiments."""

    TITLE = "ASIC-SIM // MEMORY-STATIONARY ARCHITECTURE EXPLORER"
    SUB_TITLE = "M0 + early M1 — roofs, not silicon claims"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_snapshot", "Refresh"),
    ]

    CSS = f"""
    Screen {{
        background: #0b0d0f;
        color: #e8eaed;
    }}

    Header {{
        background: #14171a;
        color: {ORANGE};
    }}

    Footer {{
        background: #14171a;
    }}

    #controls {{
        height: auto;
        padding: 1 2;
        background: #101316;
        border-bottom: solid #33383e;
    }}

    .control {{
        width: 1fr;
        min-width: 18;
        margin-right: 1;
    }}

    .control-label {{
        height: 1;
        color: {MUTED};
        text-style: bold;
    }}

    Select, Input {{
        margin-top: 1;
        border: tall #3b4148;
        background: #0d1012;
    }}

    Select:focus, Input:focus {{
        border: tall {ORANGE};
    }}

    #verdict {{
        height: auto;
        margin: 1 2 0 2;
        padding: 1 2;
        border: round #3b4148;
        background: #111417;
    }}

    #metrics {{
        grid-size: 3;
        grid-gutter: 1 1;
        height: auto;
        margin: 1 2 0 2;
    }}

    .metric {{
        height: 5;
        padding: 1 2;
        border: round #30353b;
        background: #101316;
    }}

    #breakdowns {{
        height: auto;
        margin: 1 2 0 2;
    }}

    .panel {{
        width: 1fr;
        height: 12;
        padding: 1 2;
        border: round #30353b;
        background: #101316;
    }}

    #traffic {{
        margin-right: 1;
    }}

    #tile-map, #comparison, #insights {{
        height: auto;
        min-height: 8;
        margin: 1 2 0 2;
        padding: 1 2;
        border: round #30353b;
        background: #101316;
    }}

    #comparison {{
        min-height: 11;
    }}

    #insights {{
        margin-bottom: 1;
    }}

    VerticalScroll {{
        scrollbar-color: {ORANGE};
        scrollbar-background: #111417;
    }}
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll():
            with Horizontal(id="controls"):
                with Vertical(classes="control"):
                    yield Static("MODEL", classes="control-label")
                    yield Select(
                        [(spec.name, key) for key, spec in MODEL_SPECS.items()],
                        value="kimi-k3",
                        allow_blank=False,
                        id="model-select",
                    )
                with Vertical(classes="control"):
                    yield Static("HARDWARE", classes="control-label")
                    yield Select(
                        [(hw.name, key) for key, hw in HARDWARE_PRESETS.items()],
                        value="fabric-64x32",
                        allow_blank=False,
                        id="hardware-select",
                    )
                with Vertical(classes="control"):
                    yield Static("WEIGHT BITS", classes="control-label")
                    yield Select(
                        [(f"{bits:g} bit", bits) for bits in (2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 16.0)],
                        value=4.0,
                        allow_blank=False,
                        id="bits-select",
                    )
                with Vertical(classes="control"):
                    yield Static("QUANT OVERHEAD", classes="control-label")
                    yield Input(value="5", placeholder="percent", id="overhead-input")

            yield Static(id="verdict")
            with Grid(id="metrics"):
                yield Static(id="metric-capacity", classes="metric")
                yield Static(id="metric-tile", classes="metric")
                yield Static(id="metric-locality", classes="metric")
                yield Static(id="metric-remote", classes="metric")
                yield Static(id="metric-single", classes="metric")
                yield Static(id="metric-steady", classes="metric")

            with Horizontal(id="breakdowns"):
                yield Static(id="traffic", classes="panel")
                yield Static(id="latency", classes="panel")

            yield Static(id="tile-map")
            yield Static(id="comparison")
            yield Static(id="insights")
        yield Footer()

    def on_mount(self) -> None:
        self.refresh_snapshot()

    @on(Select.Changed)
    def _selection_changed(self, _event: Select.Changed) -> None:
        self.refresh_snapshot()

    @on(Input.Submitted)
    def _overhead_submitted(self, _event: Input.Submitted) -> None:
        self.refresh_snapshot()

    def action_refresh_snapshot(self) -> None:
        self.refresh_snapshot()

    def _selected(self) -> tuple[str, str, float, float] | None:
        model_value = self.query_one("#model-select", Select).value
        hardware_value = self.query_one("#hardware-select", Select).value
        bits_value = self.query_one("#bits-select", Select).value
        if model_value is Select.NULL or hardware_value is Select.NULL or bits_value is Select.NULL:
            return None
        try:
            overhead = float(self.query_one("#overhead-input", Input).value) / 100.0
        except ValueError:
            self.query_one("#verdict", Static).update(Text("Quant overhead must be a number such as 5.", style=f"bold {RED}"))
            return None
        if overhead < 0 or overhead > 1:
            self.query_one("#verdict", Static).update(Text("Quant overhead must be between 0 and 100 percent.", style=f"bold {RED}"))
            return None
        return str(model_value), str(hardware_value), float(bits_value), overhead

    def refresh_snapshot(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        model_key, hardware_key, bits, overhead = selected
        snapshot = build_snapshot(
            model_key,
            hardware_key,
            bits_per_weight=bits,
            overhead_fraction=overhead,
        )
        self._render_snapshot(snapshot)

    def _render_snapshot(self, snapshot: DashboardSnapshot) -> None:
        r = snapshot.result
        p = snapshot.placement

        if r.resident and p.resident_per_tile_balanced:
            verdict = Text("RESIDENT  ", style=f"bold black on {GREEN}")
            verdict.append("  Model fits the system and the current balanced per-tile estimate.", style="bold")
        elif r.resident:
            verdict = Text("TILE OVERFLOW  ", style=f"bold black on {YELLOW}")
            verdict.append("  Total capacity fits, but the balanced tile estimate exceeds local tile capacity.", style="bold")
        else:
            verdict = Text("DOES NOT FIT  ", style=f"bold white on {RED}")
            verdict.append(
                f"  Needs {fmt_bytes(r.storage_bytes)} but hardware has {fmt_bytes(r.capacity_bytes)}. "
                f"Max average weight width: {r.max_bits_that_fit:.2f} bits.",
                style="bold",
            )
        self.query_one("#verdict", Static).update(verdict)

        cap_detail = Text()
        cap_detail.append_text(_bar(snapshot.capacity_utilization))
        cap_detail.append(f"  {_pct(snapshot.capacity_utilization, 1)}")
        self.query_one("#metric-capacity", Static).update(
            _metric("System capacity", f"{fmt_bytes(r.storage_bytes)} / {fmt_bytes(r.capacity_bytes)}", cap_detail.plain)
        )
        self.query_one("#metric-tile", Static).update(
            _metric(
                "Worst tile estimate",
                f"{fmt_bytes(p.max_estimated_storage_per_tile_bytes)} / {fmt_bytes(p.tile_capacity_bytes)}",
                f"{_pct(snapshot.tile_utilization, 1)} full",
            )
        )
        self.query_one("#metric-locality", Static).update(
            _metric("Modeled bytes local", _pct(r.local_data_fraction, 4), "weights stay beside compute")
        )
        self.query_one("#metric-remote", Static).update(
            _metric(
                "Remote expert calls",
                _pct(p.expected_remote_expert_fraction, 2),
                f"but only {_pct(snapshot.remote_traffic_fraction, 4)} of bytes",
            )
        )

        if r.resident:
            self.query_one("#metric-single", Static).update(
                _metric("Single-stream roof", fmt_rate(r.single_stream_memory_noc_roofline_tps), "ideal memory + NoC only")
            )
            self.query_one("#metric-steady", Static).update(
                _metric("Steady BW roof", fmt_rate(r.steady_state_memory_roofline_tps), "ideal pipelined upper bound")
            )
        else:
            self.query_one("#metric-single", Static).update(_metric("Single-stream roof", "—", "model is not resident"))
            self.query_one("#metric-steady", Static).update(_metric("Steady BW roof", "—", "model is not resident"))

        traffic = Text("DATA MOVEMENT\n", style=f"bold {ORANGE}")
        self.query_one("#traffic", Static).update(traffic + Text.from_markup("") if False else _traffic_table(snapshot))
        self.query_one("#traffic", Static).border_title = "DATA MOVEMENT"
        self.query_one("#latency", Static).update(_latency_table(snapshot))
        self.query_one("#latency", Static).border_title = "SINGLE-STREAM LATENCY FLOOR"

        self.query_one("#tile-map", Static).update(_tile_map(snapshot))
        self.query_one("#tile-map", Static).border_title = f"TILE MAP — {snapshot.hardware.tiles} TILE(S)"

        self.query_one("#comparison", Static).update(_comparison_table(snapshot))
        self.query_one("#comparison", Static).border_title = f"{snapshot.model.name} — HARDWARE COMPARISON @ {r.bits_per_weight:g} BIT"

        insight_text = Text()
        for idx, line in enumerate(interpretation(snapshot), start=1):
            insight_text.append(f"{idx}. ", style=f"bold {ORANGE}")
            insight_text.append(line + "\n")
        self.query_one("#insights", Static).update(insight_text)
        self.query_one("#insights", Static).border_title = "WHAT THIS MEANS"


def run_tui() -> None:
    AsicSimTui().run()


if __name__ == "__main__":
    run_tui()
