# C:\JWST-Mature-Galaxies\src\figures\fig_variance_publication_v5b.py
# v5b (adjusted): inset lowered a bit more; everything else unchanged.

from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Run with:  python -m figures.fig_variance_publication_v5b   (from src>)
from .jwst_plot_style import (
    set_mpl_defaults, fig_ax, axis_labels, ticks_and_grid, save_figure, CB_PALETTE
)

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES_11C   = os.path.join(PROJ_ROOT, "results", "step11c")
OUT_DIR   = os.path.join(PROJ_ROOT, "results", "figures_v2", "variance")
os.makedirs(OUT_DIR, exist_ok=True)

def latest_run_dir(root: str) -> str:
    runs = [d for d in glob.glob(os.path.join(root, "run_*")) if os.path.isdir(d)]
    if not runs:
        raise FileNotFoundError(f"No run_* folder found under {root}.")
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def find_files(run_dir: str, tag: str) -> tuple[str, str]:
    obs = glob.glob(os.path.join(run_dir, f"{tag}_field_densities_raw_and_norm_qc.csv"))
    mvr = glob.glob(os.path.join(run_dir, f"{tag}_mock_variance_norm_qc.csv"))
    if not obs:
        raise FileNotFoundError(f"Missing observed table for {tag} in {run_dir}")
    if not mvr:
        raise FileNotFoundError(f"Missing mock normalized-variance CSV for {tag} in {run_dir}")
    return obs[0], mvr[0]

def load_obs_norm_variance(per_field_csv: str) -> float:
    df = pd.read_csv(per_field_csv)
    col = "rho_norm" if "rho_norm" in df.columns else next(c for c in df.columns if "rho_norm" in c)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.var(vals, ddof=1)) if vals.size >= 2 else float("nan")

def load_mock_norm_variances(mv_csv: str) -> np.ndarray:
    df = pd.read_csv(mv_csv)
    col = df.columns[0]
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]

def decimal_xticks(ax, decimals=3):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.{decimals}f}"))

def make_hist_z8_10(run_dir: str) -> list[str]:
    obs_csv, mv_csv = find_files(run_dir, "z8_10")
    obs_norm = load_obs_norm_variance(obs_csv)
    mock_norm = load_mock_norm_variances(mv_csv)

    mu  = float(np.mean(mock_norm)) if mock_norm.size else float("nan")
    sig = float(np.std(mock_norm, ddof=0)) if mock_norm.size else float("nan")

    fig, axs = fig_ax(1, 1, width_in=6.8, height_in=4.8)
    ax = axs[0]

    # Main histogram
    bins = min(70, max(30, int(np.sqrt(mock_norm.size)))) if mock_norm.size else 40
    ax.hist(mock_norm, bins=bins, color=CB_PALETTE["blue"], alpha=0.32, label="mocks (normalized)")

    # Mean and μ+1σ (dotted)
    if np.isfinite(mu):
        ax.axvline(mu, color=CB_PALETTE["blue"], lw=1.2, ls=":", label="mock mean", zorder=2, clip_on=True)
    if np.isfinite(mu) and np.isfinite(sig):
        ax.axvline(mu + sig, color="gray", lw=1.2, ls=":", label=r"mock $\mu{+}1\sigma$", zorder=2, clip_on=True)

    # Observed (dashed red)
    if np.isfinite(obs_norm):
        ax.axvline(obs_norm, color=CB_PALETTE["red"], lw=2.0, ls="--", label="observed norm var", zorder=3, clip_on=True)

    # Labels, grid
    decimal_xticks(ax, decimals=3)
    axis_labels(ax, "Inter-field variance (normalized)", "Count of mocks")
    ax.set_title("z8–10 (QC): normalized inter-field variance\nObserved vs. 5000 lognormal mocks")
    ticks_and_grid(ax)

    # Legend INSIDE, slightly left of the right border (clears the red line)
    ax.legend(frameon=True, loc="upper right", bbox_to_anchor=(0.86, 0.97))

    # Inset: lowered a bit more (y0 from 0.44 -> 0.36)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=(0.20, 0.36, 0.46, 0.36),  # (x0, y0, w, h) in axes coords
        bbox_transform=ax.transAxes, loc="upper left", borderpad=0.8
    )
    axins.set_facecolor("white")
    axins.set_zorder(10)

    bins_in = min(40, max(20, int(np.sqrt(mock_norm.size)))) if mock_norm.size else 20
    axins.hist(mock_norm, bins=bins_in, color=CB_PALETTE["blue"], alpha=0.35, linewidth=0.0)

    if mock_norm.size:
        lo, hi = np.percentile(mock_norm, [5, 95])
        axins.set_xlim(lo, hi)
    if np.isfinite(mu):
        axins.axvline(mu, color=CB_PALETTE["blue"], lw=1.1, ls=":", clip_on=True)
    if np.isfinite(mu) and np.isfinite(sig):
        axins.axvline(mu + sig, color="gray", lw=1.1, ls=":", clip_on=True)

    decimal_xticks(axins, decimals=3)
    axins.set_yticklabels([])

    out = os.path.join(OUT_DIR, "variance_hist_z8_10_qc_linear_inset_v5b")
    written = save_figure(fig, out)
    plt.close(fig)
    return written

def main():
    set_mpl_defaults(scale="paper")
    run_dir = latest_run_dir(RES_11C)
    print(f"Step 11c latest run: {run_dir}")
    w = make_hist_z8_10(run_dir)
    print("=== Saved v5b ===")
    for p in w:
        print("  -", p)

if __name__ == "__main__":
    main()
