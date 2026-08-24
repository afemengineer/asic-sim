import asyncio

from textual.widgets import Static

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
