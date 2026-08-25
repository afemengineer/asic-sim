import asyncio

from textual.widgets import Static

from asic_sim.compact_tui import CompactAsicSimTui, _mapping_table
from asic_sim.dashboard import build_snapshot
from asic_sim.tui import AsicSimTui


def test_tui_mounts_and_renders_default_snapshot() -> None:
    async def run() -> None:
        app = AsicSimTui()
        async with app.run_test(size=(160, 60)) as pilot:
            await pilot.pause()
            verdict = app.query_one("#verdict", Static)
            assert "RESIDENT" in str(verdict.render())
            assert app.query_one("#tile-map", Static)
            assert app.query_one("#insights", Static)

    asyncio.run(run())


def test_compact_tui_renders_mapping_comparison() -> None:
    snapshot = build_snapshot("kimi-k3", "fabric-64x32", bits_per_weight=4.0)
    table = _mapping_table(snapshot)
    assert len(table.rows) == 3

    async def run() -> None:
        app = CompactAsicSimTui()
        async with app.run_test(size=(180, 65)) as pilot:
            await pilot.pause()
            assert app.query_one("#comparison", Static)

    asyncio.run(run())
