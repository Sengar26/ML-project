"""
Shared chart styling.

One palette, applied by role, so the figures read as one system rather than as five
scripts that each reached for matplotlib's defaults. Categorical hues are assigned in a
fixed order and never cycled; charts here never use more than three series, which is the
number that stays separable under colour-vision deficiency across all pairs.

The palette is validated: worst all-pairs CVD deltaE 9.2, worst normal-vision deltaE 24.0
against a #fcfcfb surface. Aqua sits below 3:1 contrast on that surface, so anything drawn
in aqua carries a visible direct label rather than relying on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29a"
GRID = "#e6e5e0"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua — fixed order
BLUE, ORANGE, AQUA = SERIES

# Sequential ramp for density: one hue, light to dark. Never a rainbow.
DENSITY = LinearSegmentedColormap.from_list("density", ["#eaf1fb", "#2a78d6", "#123457"])


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        # Room for the one-line subtitle that sits between the title and the axes.
        "axes.titlepad": 30,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "font.size": 10,
        "figure.dpi": 130,
    })


def subtitle(ax, text: str) -> None:
    """One line under the title carrying the interpretation, in secondary ink."""
    ax.annotate(text, xy=(0, 1.02), xycoords="axes fraction", fontsize=9,
                color=INK_2, va="bottom", ha="left")


def panel_is_simulated() -> bool:
    """Did the current panel come from src/simulate.py rather than real data?

    build_panel.py records the raw directory it read, and the simulator writes a truth.json
    beside its CSVs. A chart built from simulated data that isn't labelled as such is the
    single easiest way for this repo to end up asserting something it hasn't shown.
    """
    import json

    marker = Path(__file__).resolve().parent.parent / "data" / "interim" / "panel_source.json"
    try:
        return bool(json.loads(marker.read_text()).get("simulated"))
    except (OSError, ValueError):
        return False


def save(fig, path: Path, note: str | None = None) -> Path:
    """Write a figure, with a provenance note and, when warranted, a simulation stamp."""
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED, ha="left", va="bottom")
    if panel_is_simulated():
        fig.text(0.995, 0.995, "SIMULATED DATA — NOT A FINDING",
                 fontsize=8, color="#e34948", ha="right", va="top", fontweight="bold",
                 bbox={"facecolor": "#fdecec", "edgecolor": "#e34948",
                       "linewidth": 0.8, "boxstyle": "round,pad=0.35"})
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure: {path}")
    return path
