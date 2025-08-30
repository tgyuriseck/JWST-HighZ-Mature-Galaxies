#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 9C — Landy–Szalay correlation function ξ(d) for JWST tiers.

What this script does
---------------------
• Finds your latest (or specified) Step 9 (DD) and Step 9B (RR/DR) outputs.
• For each tier present in both steps, it:
    - Loads histograms and meta, checks binning consistency.
    - Computes normalized counts:
         DD_norm = DD_count / [ N_D * (N_D - 1) / 2 ]
         DR_norm, RR_norm are read from Step 9B CSVs (already normalized there).
    - Computes Landy–Szalay per bin:
         xi = (DD_norm - 2*DR_norm + RR_norm) / max(RR_norm, eps)
    - Adds a quick error proxy:
         sigma_xi ≈ (1 + xi) / sqrt(max(RR_count, 1))
      (Poisson-like; good for QA bands — not a publication-grade covariance.)
    - Optionally applies a **moving-average smooth** to ξ(d): --smooth-window
• Writes results to new timestamped folders:
    results/step9c/<timestamp>/
    figures/step9c/<timestamp>/

Inputs
------
• DD from results/step9/<tag>/DD_hist_<tier>.csv (+ DD_meta_*.json)
• RR & DR from results/step9b/<tag>/RR_hist_<tier>.csv, DR_hist_<tier>.csv (+ meta)

Run (from src/)
---------------
# Auto-pick latest step9 and step9b
python analysis\\xi_astrodeep.py --smooth-window 5

# Or specify exact input folders (useful when you have many runs)
python analysis\\xi_astrodeep.py ^
  --dd-dir "../results/step9/20250816_162200" ^
  --rdir   "../results/step9b/20250816_165044" ^
  --smooth-window 5

Outputs per tier
----------------
results/step9c/<ts>/xi_<tier>.csv
results/step9c/<ts>/xi_meta_<tier>.json
figures/step9c/<ts>/xi_<tier>.png

CSV columns include: bin edges/centers, raw counts, normalized counts, xi, xi_smooth (if requested), sigma_xi
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------- helpers ---------------------

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_subdirs(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return sorted([d for d in p.iterdir() if d.is_dir()])

def latest_subdir(p: Path) -> Optional[Path]:
    subs = list_subdirs(p)
    return subs[-1] if subs else None

def load_json(p: Path) -> Dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; edges are left as NaN if not enough neighbors."""
    if window <= 1:
        return a.copy()
    if window % 2 == 0:
        raise ValueError("smooth-window must be odd (e.g., 3,5,7).")
    pad = window // 2
    out = np.full_like(a, np.nan, dtype=np.float64)
    csum = np.cumsum(np.insert(a, 0, 0.0))
    for i in range(pad, len(a) - pad):
        s = csum[i + pad + 1] - csum[i - pad]
        out[i] = s / window
    return out

@dataclass
class TierInputs:
    tier: str
    dd_csv: Path
    dd_meta: Path
    rr_csv: Path
    rr_meta: Path
    dr_csv: Path
    dr_meta: Path

# --------------------- IO scan ---------------------

def scan_inputs(dd_dir: Path, rdir: Path) -> List[TierInputs]:
    """Match tiers that exist in both step9 (DD) and step9b (RR/DR)."""
    dd_files = list(dd_dir.glob("DD_hist_*.csv"))
    tiers_dd = {f.stem.replace("DD_hist_", ""): f for f in dd_files}
    out: List[TierInputs] = []

    for tier, dd_csv in sorted(tiers_dd.items()):
        rr_csv = rdir / f"RR_hist_{tier}.csv"
        dr_csv = rdir / f"DR_hist_{tier}.csv"
        if rr_csv.is_file() and dr_csv.is_file():
            ti = TierInputs(
                tier=tier,
                dd_csv=dd_csv,
                dd_meta=dd_dir / f"DD_meta_{tier}.json",
                rr_csv=rr_csv,
                rr_meta=rdir / f"RR_meta_{tier}.json",
                dr_csv=dr_csv,
                dr_meta=rdir / f"DR_meta_{tier}.json",
            )
            out.append(ti)
    return out

# --------------------- core math ---------------------

def check_bins(dd: pd.DataFrame, rr: pd.DataFrame, dr: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Verify identical edges/centers across DD, RR, DR; return edges and centers."""
    for df in (dd, rr, dr):
        if not {"bin_left_Mpc", "bin_right_Mpc", "bin_center_Mpc"}.issubset(df.columns):
            raise ValueError("Histogram CSV missing required columns (bin_left_Mpc, bin_right_Mpc, bin_center_Mpc).")

    # Use DD as the reference:
    edges_ref = np.r_[dd["bin_left_Mpc"].values, dd["bin_right_Mpc"].values[-1]]
    centers_ref = dd["bin_center_Mpc"].values

    def edges_from(df):
        return np.r_[df["bin_left_Mpc"].values, df["bin_right_Mpc"].values[-1]]

    if not np.allclose(edges_ref, edges_from(rr), rtol=0, atol=1e-8) or \
       not np.allclose(edges_ref, edges_from(dr), rtol=0, atol=1e-8):
        raise ValueError("DD/RR/DR bin edges do not match. Re-run with consistent binning.")
    if not np.allclose(centers_ref, rr["bin_center_Mpc"].values, rtol=0, atol=1e-8) or \
       not np.allclose(centers_ref, dr["bin_center_Mpc"].values, rtol=0, atol=1e-8):
        raise ValueError("DD/RR/DR bin centers do not match. Re-run with consistent binning.")
    return edges_ref, centers_ref

def compute_xi(
    dd_df: pd.DataFrame, rr_df: pd.DataFrame, dr_df: pd.DataFrame,
    dd_meta: Dict, rr_meta: Dict
) -> pd.DataFrame:
    """Compute normalized counts and Landy–Szalay xi per bin."""
    # Pull normalization constants
    # Prefer N_D from meta; fall back to infer from pairs if needed.
    if "pairs_DD_all" in rr_meta:
        # rr_meta stores n_data and normalization terms for RR/DR
        n_data = int(rr_meta.get("n_data", 0))
        n_rand = int(rr_meta.get("n_random", 0))
        pairs_DD_all = int(rr_meta["pairs_DD_all"])
        pairs_RR_all = int(rr_meta["pairs_RR_all"])
        pairs_DR_all = int(rr_meta["pairs_DR_all"])
    else:
        # Old step9b meta (if any); here dd_meta has n_points; compute pairs
        n_data = int(dd_meta.get("n_points", 0))
        n_rand = int(rr_meta.get("n_random", 0))
        pairs_DD_all = n_data * (n_data - 1) // 2
        pairs_RR_all = int(rr_meta.get("pairs_RR_all", n_rand * (n_rand - 1) // 2))
        pairs_DR_all = n_data * n_rand

    if pairs_DD_all <= 0 or pairs_RR_all <= 0 or pairs_DR_all <= 0:
        raise ValueError("Invalid normalization constants (pairs_*_all).")

    # Normalize
    DD_norm = dd_df["DD_count"].to_numpy(dtype=np.float64) / float(pairs_DD_all)
    RR_norm = rr_df["RR_norm"].to_numpy(dtype=np.float64)  # already normalized in step9b
    DR_norm = dr_df["DR_norm"].to_numpy(dtype=np.float64)

    # Landy–Szalay
    eps = 1e-12
    xi = (DD_norm - 2.0 * DR_norm + RR_norm) / np.maximum(RR_norm, eps)

    # Quick error proxy using RR raw counts
    RR_count = rr_df["RR_count"].to_numpy(dtype=np.float64)
    sigma_xi = (1.0 + xi) / np.sqrt(np.maximum(RR_count, 1.0))

    # Assemble output frame
    out = pd.DataFrame({
        "bin_left_Mpc": dd_df["bin_left_Mpc"],
        "bin_right_Mpc": dd_df["bin_right_Mpc"],
        "bin_center_Mpc": dd_df["bin_center_Mpc"],
        "DD_count": dd_df["DD_count"],
        "DR_count": dr_df["DR_count"],
        "RR_count": rr_df["RR_count"],
        "DD_norm": DD_norm,
        "DR_norm": DR_norm,
        "RR_norm": RR_norm,
        "xi": xi,
        "sigma_xi": sigma_xi
    })
    return out

def plot_xi(centers: np.ndarray, xi: np.ndarray, xi_smooth: Optional[np.ndarray], out_png: Path, title: str):
    plt.figure(figsize=(7,5))
    plt.axhline(0.0, color="k", lw=1, alpha=0.5)
    plt.plot(centers, xi, lw=1.2, label=r"$\xi(d)$")
    if xi_smooth is not None:
        plt.plot(centers, xi_smooth, lw=2.0, label=r"$\xi(d)$ (smoothed)")
    plt.axvline(150.0, ls="--", lw=1)  # BAO-ish visual guide
    plt.xlabel("Separation d  [Mpc (comoving)]")
    plt.ylabel(r"Correlation $\xi(d)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# --------------------- main ---------------------

def main():
    here = Path(__file__).resolve()
    project_root = here.parents[2]

    parser = argparse.ArgumentParser(description="Step 9C: compute Landy–Szalay ξ(d) from DD, DR, RR.")
    # Input directory options
    parser.add_argument("--dd-dir", default="", help="Folder with Step 9 outputs (DD). Default: latest under results/step9/")
    parser.add_argument("--rdir",    default="", help="Folder with Step 9B outputs (RR/DR). Default: latest under results/step9b/")
    # Smoothing
    parser.add_argument("--smooth-window", type=int, default=0, help="Odd integer window size for centered moving-average smoothing (0 = off).")
    args = parser.parse_args()

    # Resolve input folders
    step9_root = project_root / "results" / "step9"
    step9b_root = project_root / "results" / "step9b"

    dd_dir = Path(args.dd_dir) if args.dd_dir else latest_subdir(step9_root)
    rdir   = Path(args.rdir)   if args.rdir   else latest_subdir(step9b_root)

    if not dd_dir or not rdir:
        print("Could not locate input folders. Provide --dd-dir and --rdir explicitly.")
        sys.exit(1)
    if not dd_dir.exists() or not rdir.exists():
        print(f"Input folder(s) not found:\n  dd-dir: {dd_dir}\n  rdir:   {rdir}")
        sys.exit(1)

    # Prepare outputs
    run_tag = timestamp_tag()
    out_res = (project_root / "results" / "step9c" / run_tag)
    out_fig = (project_root / "figures" / "step9c" / run_tag)
    ensure_dir(out_res); ensure_dir(out_fig)

    # Match tiers
    tiers = scan_inputs(dd_dir, rdir)
    if not tiers:
        print("No common tiers found between:\n  DD:  {}\n  RR/DR: {}".format(dd_dir, rdir))
        sys.exit(1)
    print(f"Found {len(tiers)} tier(s).")
    print(f"DD dir:  {dd_dir}\nRR/DR:   {rdir}\nOut:     {out_res}")

    # For run summary
    rows = []

    for ti in tiers:
        tier = ti.tier
        print(f"\n=== Tier {tier} ===")

        dd_df = pd.read_csv(ti.dd_csv)
        rr_df = pd.read_csv(ti.rr_csv)
        dr_df = pd.read_csv(ti.dr_csv)
        dd_meta = load_json(ti.dd_meta) if ti.dd_meta.is_file() else {}
        rr_meta = load_json(ti.rr_meta) if ti.rr_meta.is_file() else {}

        # Check bins
        edges, centers = check_bins(dd_df, rr_df, dr_df)

        # Compute xi
        xi_df = compute_xi(dd_df, rr_df, dr_df, dd_meta, rr_meta)

        # Optional smoothing
        xi_smooth = None
        if args.smooth_window and args.smooth_window > 1:
            xi_smooth = moving_average(xi_df["xi"].to_numpy(np.float64), args.smooth_window)
            xi_df["xi_smooth"] = xi_smooth

        # Save CSV
        out_csv = out_res / f"xi_{tier}.csv"
        xi_df.to_csv(out_csv, index=False)

        # Save META
        meta = {
            "tier": tier,
            "dd_dir": str(dd_dir.resolve()),
            "rdir": str(rdir.resolve()),
            "dd_csv": ti.dd_csv.name,
            "rr_csv": ti.rr_csv.name,
            "dr_csv": ti.dr_csv.name,
            "dd_meta": ti.dd_meta.name if ti.dd_meta.is_file() else None,
            "rr_meta": ti.rr_meta.name if ti.rr_meta.is_file() else None,
            "method": "Landy-Szalay",
            "formula": "(DD - 2*DR + RR)/RR, with normalized counts",
            "smoothing": {"type": "moving_average", "window": int(args.smooth_window) if args.smooth_window else 0},
            "timestamp": run_tag
        }
        with open(out_res / f"xi_meta_{tier}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Plot
        out_png = out_fig / f"xi_{tier}.png"
        plot_xi(centers, xi_df["xi"].to_numpy(np.float64), xi_smooth, out_png, f"Two-point correlation — {tier}")

        # Run summary row
        rows.append({
            "tier": tier,
            "bins": len(xi_df),
            "smooth_window": int(args.smooth_window),
            "xi_csv": out_csv.name,
            "xi_meta": f"xi_meta_{tier}.json",
            "xi_png": out_png.name
        })

    # Write summary
    if rows:
        summary_df = pd.DataFrame(rows)
        summary_csv = out_res / "step9c_run_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nRun summary: {summary_csv}")
    else:
        print("\nNo tiers processed (empty rows).")

if __name__ == "__main__":
    main()
