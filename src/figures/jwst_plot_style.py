# jwst_plot_style.py  (safe, drop-in replacement)
from __future__ import annotations
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-friendly palette
CB_PALETTE = {
    "blue":    "#1f77b4",
    "orange":  "#ff7f0e",
    "green":   "#2ca02c",
    "red":     "#d62728",
    "purple":  "#9467bd",
    "brown":   "#8c564b",
    "pink":    "#e377c2",
    "gray":    "#7f7f7f",
    "olive":   "#bcbd22",
    "cyan":    "#17becf",
    "black":   "#000000",
}

def set_mpl_defaults(scale: str = "paper") -> None:
    """
    Minimal, journal-friendly defaults.
    scale: "paper" (slightly larger text) or "talk" (bigger)
    """
    if scale == "talk":
        fs = 12
    else:
        fs = 10  # good for MNRAS column width

    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": fs,
        "axes.titlesize": fs + 1,
        "axes.labelsize": fs,
        "legend.fontsize": fs - 1,
        "xtick.labelsize": fs - 1,
        "ytick.labelsize": fs - 1,
        "axes.linewidth": 0.8,
        "grid.linestyle": ":",
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "pdf.fonttype": 42,  # embed as TrueType
        "ps.fonttype": 42,
    })

def ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)

def fig_ax(nrows: int = 1, ncols: int = 1,
           width_in: float = 3.4, height_in: float = 2.6,
           wspace: float | None = None, hspace: float | None = None):
    """
    Create a figure+axes with optional subplot spacing control.
    Returns (fig, ax) when 1x1; (fig, axs_flat) otherwise.
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=(width_in, height_in))
    if (wspace is not None) or (hspace is not None):
        fig.subplots_adjust(wspace=(wspace or 0.0), hspace=(hspace or 0.0))

    if nrows == 1 and ncols == 1:
        return fig, axs
    else:
        return fig, np.array(axs).flatten()

def axis_labels(ax, xlab: str, ylab: str) -> None:
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

def ticks_and_grid(ax, which: str = "both") -> None:
    ax.grid(True, which="major")
    if which == "both":
        try:
            ax.minorticks_on()
            ax.grid(True, which="minor", alpha=0.15)
        except Exception:
            pass

def _ensure_ext(path_no_ext: str, ext: str) -> str:
    return f"{path_no_ext}.{ext.lstrip('.')}"

def save_figure(fig, base_path_no_ext: str,
                save_pdf: bool = True, save_png: bool = True,
                bbox_inches: str = "tight"):
    """
    Save as PDF (vector) and PNG. Returns list of written paths.
    """
    written = []
    if save_pdf:
        p = _ensure_ext(base_path_no_ext, "pdf")
        fig.savefig(p, bbox_inches=bbox_inches)
        written.append(p)
    if save_png:
        p = _ensure_ext(base_path_no_ext, "png")
        fig.savefig(p, bbox_inches=bbox_inches)
        written.append(p)
    return written
