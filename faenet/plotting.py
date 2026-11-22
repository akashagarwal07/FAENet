"""Shared figure style for report plots (static PNG/PDF for the LaTeX paper)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots in fixed order (validated reference palette); never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e4e3df"; SURFACE = "#fcfcfb"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=200)
    fig.savefig(path_no_ext + ".pdf")
    plt.close(fig)
