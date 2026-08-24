from __future__ import annotations

from .tui import AsicSimTui, ORANGE, MUTED


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


def run_tui() -> None:
    CompactAsicSimTui().run()


if __name__ == "__main__":
    run_tui()
