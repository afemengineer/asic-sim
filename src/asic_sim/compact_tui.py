from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from .architectures import compare_architectures
from .dashboard import DashboardSnapshot
from .formatting import fmt_bytes, fmt_rate, fmt_time_s
from .tui import AsicSimTui, GREEN, MUTED, ORANGE, RED


def _mapping_table(snapshot: DashboardSnapshot) -> Table:
    bits = snapshot.result.bits_per_weight
    reports = compare_architectures(
        snapshot.model,
        snapshot.hardware,
        bits_per_weight=bits,
        shared_expert_bits=bits,
        overhead_fraction=snapshot.result.quant_overhead_fraction,
        tokens=32,
        profile="balanced",
    )
    table = Table(expand=True, pad_edge=False)
    table.add_column("MAPPING")
    table.add_column("FIT", justify="center")
    table.add_column("MAX TILE", justify="right")
    table.add_column("NOC/TOK", justify="right")
    table.add_column("HOP-BYTES", justify="right")
    table.add_column("MEM FLOOR", justify="right")
    table.add_column("MEM ROOF", justify="right")

    for report in reports:
        fit = Text("YES", style=f"bold {GREEN}") if report.resident else Text("NO", style=f"bold {RED}")
        table.add_row(
            report.label,
            fit,
            fmt_bytes(report.max_tile_storage_bytes),
            fmt_bytes(report.network_payload_bytes_per_token),
            fmt_bytes(report.hop_bytes_per_token),
            fmt_time_s(report.ideal_memory_floor_s),
            fmt_rate(report.ideal_memory_roof_tps),
        )

    if snapshot.model.shared_experts:
        shared = reports[0].shared_active_bytes_per_token if reports else 0.0
        table.caption = (
            f"Shared experts: {snapshot.model.shared_experts}/MoE layer, always active; "
            f"{fmt_bytes(shared)}/token at selected {bits:g}-bit assumption. "
            "Kimi stress case: CLI architectures --shared-bits 16."
        )
        table.caption_style = MUTED
    return table


class CompactAsicSimTui(AsicSimTui):
    """Denser terminal layout that keeps the useful dashboard above the fold."""

    CSS = f"""
    #controls {{
        height: 5;
        padding: 0 1;
        background: #101316;
        border-bottom: solid #33383e;
    }}

    .control {{
        width: 1fr;
        min-width: 16;
        height: 4;
        margin-right: 1;
    }}

    .control-label {{
        height: 1;
        color: {MUTED};
        text-style: bold;
    }}

    /* Select owns an internal SelectCurrent widget with its own border.
       Do not add another border to the outer Select or the selected text
       gets clipped in this compact three-row layout. */
    Select {{
        height: 3;
        margin-top: 0;
        background: #0d1012;
    }}

    Input {{
        height: 3;
        margin-top: 0;
        border: tall #3b4148;
        background: #0d1012;
    }}

    Input:focus {{
        border: tall {ORANGE};
    }}

    #verdict {{
        height: 3;
        margin: 1 1 0 1;
        padding: 0 1;
    }}

    #metrics {{
        grid-size: 3;
        grid-gutter: 1 1;
        height: auto;
        margin: 1 1 0 1;
    }}

    .metric {{
        height: 4;
        padding: 0 1;
    }}

    #breakdowns {{
        height: auto;
        margin: 1 1 0 1;
    }}

    .panel {{
        height: 9;
        padding: 0 1;
    }}

    #tile-map, #comparison, #insights {{
        margin: 1 1 0 1;
        padding: 0 1;
    }}

    #tile-map {{
        min-height: 7;
    }}

    #comparison {{
        min-height: 9;
    }}

    #insights {{
        min-height: 6;
        margin-bottom: 1;
    }}
    """

    def _render_snapshot(self, snapshot: DashboardSnapshot) -> None:
        super()._render_snapshot(snapshot)
        comparison = self.query_one("#comparison", Static)
        comparison.update(_mapping_table(snapshot))
        comparison.border_title = (
            f"M1 PHYSICAL MAPPING COMPARISON — {snapshot.model.name} / "
            f"{snapshot.hardware.tiles} TILE(S)"
        )


def run_tui() -> None:
    CompactAsicSimTui().run()


if __name__ == "__main__":
    run_tui()
