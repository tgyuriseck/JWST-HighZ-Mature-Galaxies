# C:\JWST-Mature-Galaxies\src\figures\fig_perfield_rank_qc_v1.py
from __future__ import annotations
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run from src>:  python -m figures.fig_perfield_rank_qc_v1
from .jwst_plot_style import (
    set_mpl_defaults, fig_ax, axis_labels, ticks_and_grid, save_figure, CB_PALETTE
)

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STEP11C   = os.path.join(PROJ_ROOT, "results", "step11c")
OUT_DIR   = os.path.join(PROJ_ROOT, "results", "figures_v2", "variance")
os.makedirs(OUT_DIR, exist_ok=True)

def latest_run_dir() -> str:
    runs = [d for d in glob.glob(os.path.join(STEP11C, "run_*")) if os.path.isdir(d)]
    if not runs:
        raise FileNotFoundError(f"No run_* directories under {STEP11C}")
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def find_perfield_csv(run_dir: str, tag: str) -> str:
    """
    Find the per-field densities CSV produced by Step 11c.
    Tries several patterns to be robust to naming differences.
    """
    patterns = [
        f"{tag}*field*densit*raw*norm*qc*.csv",
        f"{tag}*per_field*raw*norm*qc*.csv",
        f"{tag}*field*raw*norm*qc*.csv",
        f"{tag}*field*qc*.csv",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(run_dir, pat))
        if matches:
            matches.sort(key=os.path.getmtime)
            return matches[-1]
    # If not found, show what exists to help debug
    all_csvs = sorted(glob.glob(os.path.join(run_dir, "*.csv")))
    preview = "\n  ".join(os.path.basename(p) for p in all_csvs[:12])
    raise FileNotFoundError(
        f"Per-field CSV for {tag} not found in {run_dir}.\n"
        f"Tried patterns: {patterns}\nFound CSVs (first 12):\n  {preview}"
    )

def load_perfield(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Find columns robustly
    field_col = next((c for c in df.columns if c.lower().startswith("field")), None)
    raw_col   = next((c for c in df.columns if "rho_raw" in c.lower() or "raw" in c.lower()), None)
    norm_col  = next((c for c in df.columns if "rho_norm" in c.lower() or "norm" in c.lower()), None)
    if not (field_col and raw_col and norm_col):
        raise ValueError(f"Needed columns not found in {csv_path}. Have: {list(df.columns)}")
    out = df[[field_col, raw_col, norm_col]].copy()
    out.columns = ["field", "raw", "norm"]
    out["field"] = out["field"].astype(str)
    return out

def to_ranks(s: pd.Series, descending=True) -> pd.Series:
    # Rank 1 = highest density
    return s.rank(method="min", ascending=not descending).astype(int)

def plot_rank_panel(ax, df: pd.DataFrame, title: str):
    df = df.copy()
    df["rank_raw"]  = to_ranks(df["raw"])
    df["rank_norm"] = to_ranks(df["norm"])
    # order by normalized rank for stable y ordering
    order = df.sort_values(["rank_norm", "rank_raw"])["field"].tolist()
    y = np.arange(len(order))
    idx = {f:i for i,f in enumerate(order)}

    # connectors + points
    for _, r in df.iterrows():
        yi = idx[r["field"]]
        ax.plot([r["rank_raw"], r["rank_norm"]], [yi, yi], color="0.82", lw=1, zorder=1)
        ax.scatter(r["rank_raw"],  yi, color=CB_PALETTE["blue"],   s=36, zorder=2,
                   label="raw" if _ == df.index[0] else None)
        ax.scatter(r["rank_norm"], yi, color=CB_PALETTE["orange"], s=36, zorder=3,
                   label="normalized" if _ == df.index[0] else None)

    # y-ticks are field names
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_xaxis()  # show rank 1 at the left
    ax.set_xlim(0.5, len(order) + 0.5)
    ticks_and_grid(ax)
    axis_labels(ax, "Rank (1 = highest density)", "Field")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)

    # emphasise PRIMER-COSMOS / NGDEEP if present
    for name, col in [("PRIMER-COSMOS", CB_PALETTE["blue"]), ("NGDEEP", CB_PALETTE["orange"])]:
        if name in idx:
            yi = idx[name]
            ax.text(0.02, yi, name, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", color="0.25", fontsize=9)

def main():
    set_mpl_defaults(scale="paper")
    run_dir = latest_run_dir()

    f8  = find_perfield_csv(run_dir, "z8_10")
    f10 = find_perfield_csv(run_dir, "z10_20")
    df8  = load_perfield(f8)
    df10 = load_perfield(f10)

    fig, axs = fig_ax(1, 2, width_in=7.0, height_in=3.8, wspace=0.30)
    plot_rank_panel(axs[0], df8,  "z=8–10")
    plot_rank_panel(axs[1], df10, "z=10–20")

    out = os.path.join(OUT_DIR, "perfield_rank_qc_v1")
    written = save_figure(fig, out)
    print("Saved:", *written, sep="\n  ")
    plt.close(fig)

if __name__ == "__main__":
    main()
