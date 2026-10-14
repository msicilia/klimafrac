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


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
