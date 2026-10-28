"""One palette and one axis treatment for every figure in the paper."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
ACCENT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES = {"BCT": ACCENT[0], "NCT": ACCENT[1], "MCO2": ACCENT[2], "KLIMA": ACCENT[3]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "figure.dpi": 130,
})


def style(ax, title=None, sub=None, ylabel=None):
    """Left-aligned title with an optional subtitle, light grid, open frame."""
    if title:
        lines = sub.count("\n") + 1 if sub else 0
        ax.set_title(title, loc="left", pad=10 + 15 * lines)
        if sub:
            ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                    color=INK2, va="bottom", linespacing=1.4)
    ax.grid(True, alpha=0.18, linewidth=0.7, color=MUTED)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


# The paper's figures are drawn at the width they print at (the journal's text
# width), so that the sizes below are the sizes on the page.
PAGE_WIDTH = 5.47                                      # inches
PAPER = {"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5}
PAPER_DPI = 600


def save(fig, path: str, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
