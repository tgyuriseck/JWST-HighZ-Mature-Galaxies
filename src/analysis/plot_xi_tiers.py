#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 9D — Plot xi(d) with error bars (and optional smoothed curve) for all tiers.

Inputs
------
Reads the CSVs written in Step 9C:
  results/step9c/<tag>/xi_<tier>.csv

Outputs (per run)
-----------------
figures/step9d/<timestamp>/xi_<tier>.png          # per-tier plot
figures/step9d/<timestamp>/xi_panel_all.png       # optional small-multiples panel
results/step9d/<timestamp>/step9d_run_summary.csv # one-line per tier

Run examples (from src/)
------------------------
python analysis\\plot_xi_tiers.py
python analysis\\plot_xi_tiers.py --xi-dir "../results/step9c/<your_tag>"
python analysis\\plot_xi_tiers.py --ymin -0.5 --ymax 2.0 --no-smooth --no-panel
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------- helpers -----------------

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def latest_subdir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    subs = sorted([p for p in root.iterdir() if p.is_dir()])
    return subs[-1] if subs else None

def tier_from_filename(path: Path) -> str:
    stem = path.stem
    return stem[3:] if stem.startswith("xi_") else stem

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_pos_errors(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Make error bars valid for Matplotlib: |arr| with a tiny floor, NaNs -> eps."""
    out = np.abs(arr.astype(np.float64))
    out[~np.isfinite(out)] = eps
    out[out < eps] = eps
    return out

@dataclass
class TierPlotData:
    tier: str
    centers: np.ndarray
    xi: np.ndarray
    sigma: np.ndarray
    xi_smooth: Optional[np.ndarray]


# ----------------- IO -----------------

def load_xi_dir(xi_dir: Path) -> List[TierPlotData]:
    files = sorted(xi_dir.glob("xi_*.csv"))
    out: List[TierPlotData] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        needed = {"bin_center_Mpc", "xi", "sigma_xi"}
        if not needed.issubset(df.columns):
            continue
        tier = tier_from_filename(f)
        centers = df["bin_center_Mpc"].to_numpy(np.float64)
        xi = df["xi"].to_numpy(np.float64)
        sig = df["sigma_xi"].to_numpy(np.float64)
        xi_s = df["xi_smooth"].to_numpy(np.float64) if "xi_smooth" in df.columns else None
        out.append(TierPlotData(tier, centers, xi, sig, xi_s))
    return out


# ----------------- plotting -----------------

def plot_single(t: TierPlotData, out_png: Path, ymin: Optional[float], ymax: Optional[float], show_smooth: bool):
    plt.figure(figsize=(7,5))
    plt.axhline(0.0, lw=1, color="k", alpha=0.5)
    plt.axvline(150.0, lw=1, ls="--")

    yerr = safe_pos_errors(t.sigma)
    plt.errorbar(t.centers, t.xi, yerr=yerr, fmt="o", ms=3, lw=1, alpha=0.9, capsize=2, label=r"$\xi(d)$")

    if show_smooth and t.xi_smooth is not None and np.isfinite(t.xi_smooth).any():
        plt.plot(t.centers, t.xi_smooth, lw=2, label=r"$\xi(d)$ (smoothed)")

    plt.xlabel("Separation d  [Mpc (comoving)]")
    plt.ylabel(r"Correlation $\xi(d)$")
    plt.title(f"Two-point correlation — {t.tier}")
    if (ymin is not None) or (ymax is not None):
        plt.ylim(ymin, ymax)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_panel_all(data: List[TierPlotData], out_png: Path, ymin: Optional[float], ymax: Optional[float], show_smooth: bool):
    if not data:
        return
    n = len(data)
    ncols = 3 if n >= 6 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.6*nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, data):
        ax.axhline(0.0, lw=1, color="k", alpha=0.4)
        ax.axvline(150.0, lw=1, ls="--")
        yerr = safe_pos_errors(t.sigma)
        ax.errorbar(t.centers, t.xi, yerr=yerr, fmt="o", ms=2.5, lw=0.8, alpha=0.9, capsize=2, label="xi")
        if show_smooth and t.xi_smooth is not None and np.isfinite(t.xi_smooth).any():
            ax.plot(t.centers, t.xi_smooth, lw=1.5, label="smoothed")
        ax.set_title(t.tier)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)

    for ax in axes[len(data):]:
        ax.axis("off")

    fig.text(0.5, 0.04, "Separation d  [Mpc (comoving)]", ha="center")
    fig.text(0.04, 0.5, r"Correlation $\xi(d)$", va="center", rotation="vertical")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0.04, 0.04, 0.98, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------- main -----------------

def main():
    here = Path(__file__).resolve()
    project_root = here.parents[2]

    parser = argparse.ArgumentParser(description="Step 9D: plot xi(d) for all tiers.")
    parser.add_argument("--xi-dir", default="", help="Directory with xi_*.csv from step9c. Default = latest under results/step9c/")
    parser.add_argument("--ymin", type=float, default=None, help="y-axis lower limit (optional)")
    parser.add_argument("--ymax", type=float, default=None, help="y-axis upper limit (optional)")
    parser.add_argument("--no-smooth", action="store_true", help="Hide smoothed curve if present.")
    parser.add_argument("--no-panel", action="store_true", help="Skip multi-panel summary figure.")
    args = parser.parse_args()

    step9c_root = project_root / "results" / "step9c"
    xi_dir = Path(args.xi_dir) if args.xi_dir else latest_subdir(step9c_root)
    if not xi_dir or not xi_dir.exists():
        print("Could not locate step9c outputs. Provide --xi-dir explicitly.")
        return

    run_tag = timestamp_tag()
    out_fig_dir = project_root / "figures" / "step9d" / run_tag
    out_res_dir = project_root / "results" / "step9d" / run_tag
    ensure_dir(out_fig_dir); ensure_dir(out_res_dir)

    tiers = load_xi_dir(xi_dir)
    if not tiers:
        print(f"No xi_*.csv files found in: {xi_dir}")
        return
    print(f"Found {len(tiers)} tier(s) in {xi_dir}")

    rows = []
    for t in tiers:
        out_png = out_fig_dir / f"xi_{t.tier}.png"
        plot_single(
            t,
            out_png,
            ymin=args.ymin,
            ymax=args.ymax,
            show_smooth=(not args.no_smooth)
        )
        rows.append({
            "tier": t.tier,
            "xi_png": out_png.name,
            "xi_source_csv": f"xi_{t.tier}.csv",
            "ymin": args.ymin,
            "ymax": args.ymax,
            "smoothed_plotted": (not args.no_smooth)
        })

    if not args.no_panel:
        out_panel = out_fig_dir / "xi_panel_all.png"
        plot_panel_all(tiers, out_panel, ymin=args.ymin, ymax=args.ymax, show_smooth=(not args.no_smooth))

    pd.DataFrame(rows).to_csv(out_res_dir / "step9d_run_summary.csv", index=False)
    print(f"Done. Figures: {out_fig_dir}")
    print(f"Summary CSV: {out_res_dir / 'step9d_run_summary.csv'}")

if __name__ == "__main__":
    main()
