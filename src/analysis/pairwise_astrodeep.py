#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 9A — Pairwise separations (DD) for JWST ASTRODEEP tiers.

This script:
  - Finds tier CSVs (default: ANY *.csv under <project>/data_processed/tiers).
  - Accepts either:
      (A) Ready-made Cartesian columns X,Y,Z in Mpc (comoving), or
      (B) RA/Dec (deg) + redshift (zspec/zphot), and converts to Cartesian using Planck18.
  - Computes pairwise separation histogram DD(d) within [r_min, r_max] with fixed bin width.
  - Uses SciPy cKDTree if available; otherwise an exact grid-neighborhood fallback.
  - Writes per-tier CSV + JSON + PNG into timestamped results/step9/ and figures/step9/.
  - Optional brute-force self-test on a tiny subset.

Run examples:

  # Full run (searches all *.csv in tiers)
  python analysis\pairwise_astrodeep.py

  # Only z≥10 tiers by pattern
  python analysis\pairwise_astrodeep.py --tiers-glob "../data_processed/tiers/*z10*.csv" --bin-min 1 --bin-max 250 --bin-width 5

  # Add a small 600-point self-test for confidence
  python analysis\pairwise_astrodeep.py --self-test 600
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SciPy KDTree (optional)
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Astropy (for RA/Dec/z -> Cartesian)
ASTROPY_OK = True
try:
    from astropy.cosmology import Planck18
    import astropy.units as u
except Exception:
    ASTROPY_OK = False

import glob
import os


# ----------------------------- Utilities ------------------------------------ #

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_tier_name(path: Path) -> str:
    """
    Try to extract a concise tier label from filename.
    'astrodeep_<tier>[_coords].csv' -> <tier>, else stem.
    """
    stem = path.stem
    if "astrodeep_" in stem:
        after = stem.split("astrodeep_", 1)[1]
        if after.endswith("_coords"):
            after = after[:-7]
        if after:
            return after
    return stem


def ensure_dirs(base_results: Path, base_figures: Path) -> Tuple[Path, Path]:
    base_results.mkdir(parents=True, exist_ok=True)
    base_figures.mkdir(parents=True, exist_ok=True)
    return base_results, base_figures


def select_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """
    Decide how to get coordinates:
      - If X,Y,Z exist (case-insensitive), return ("xyz", Xcol, Ycol, Zcol).
      - Else, find RA/DEC and z columns:
          RA: RA | RA_optap | RA_photoz
          DEC: DEC | DEC_optap | DEC_photoz
          z:  zspec (preferred if finite), else zphot, else z, else z_best
        return ("radecz", RAcol, DECcol, zcol).
    Raises ValueError if neither option is available.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Option A: Cartesian present
    if all(k in cols_lower for k in ("x", "y", "z")):
        return ("xyz", cols_lower["x"], cols_lower["y"], cols_lower["z"])

    # Option B: RA/Dec/z
    ra_candidates  = ["ra", "ra_optap", "ra_photoz"]
    dec_candidates = ["dec", "dec_optap", "dec_photoz"]
    z_candidates   = ["zspec", "zphot", "z", "z_best"]

    ra_col  = next((cols_lower[k] for k in ra_candidates  if k in cols_lower), None)
    dec_col = next((cols_lower[k] for k in dec_candidates if k in cols_lower), None)
    z_col   = next((cols_lower[k] for k in z_candidates   if k in cols_lower), None)

    if ra_col and dec_col and z_col:
        return ("radecz", ra_col, dec_col, z_col)

    raise ValueError("No usable coordinate columns found. Need X,Y,Z or (RA,DEC and zspec/zphot).")


def load_points(csv_path: Path, mode_hint: str = "auto",
                max_rows: int = 0, subsample: Optional[int] = None,
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, Dict]:
    """
    Load positions as an (N,3) array in comoving Mpc.
    If mode is 'xyz', read X,Y,Z directly.
    If mode is 'radecz', compute Cartesian from RA/DEC (deg) and z via Planck18.

    Returns:
      points (N,3), meta dict describing selection and filtering.
    """
    df = pd.read_csv(csv_path)

    # Determine columns
    sel_mode, c1, c2, c3 = select_columns(df)

    used_mode = sel_mode
    used_cols = (c1, c2, c3)

    # Subset to columns for speed
    df = df.loc[:, list(set(used_cols))].copy()

    # Truncate first if requested (keeps determinism pre-subsample)
    if max_rows and max_rows > 0:
        df = df.iloc[:max_rows].copy()

    if sel_mode == "xyz":
        X = pd.to_numeric(df[c1], errors="coerce").to_numpy(np.float64, copy=False)
        Y = pd.to_numeric(df[c2], errors="coerce").to_numpy(np.float64, copy=False)
        Z = pd.to_numeric(df[c3], errors="coerce").to_numpy(np.float64, copy=False)
        pts = np.vstack([X, Y, Z]).T
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]

    else:
        # radecz path
        if not ASTROPY_OK:
            raise RuntimeError("Astropy not available. Install with: pip install astropy")
        ra_deg  = pd.to_numeric(df[c1], errors="coerce").to_numpy(np.float64, copy=False)
        dec_deg = pd.to_numeric(df[c2], errors="coerce").to_numpy(np.float64, copy=False)
        zz      = pd.to_numeric(df[c3], errors="coerce").to_numpy(np.float64, copy=False)

        # filter finite and z > 0
        mask = np.isfinite(ra_deg) & np.isfinite(dec_deg) & np.isfinite(zz) & (zz > 0)
        ra = np.deg2rad(ra_deg[mask])
        dec = np.deg2rad(dec_deg[mask])
        z = zz[mask]

        # Prefer zspec if present and finite; otherwise zphot.
        # If we only had one z column (from select_columns), we just use it.
        # (This can be enhanced by reading both columns if available, but we keep it simple.)
        # Cosmology: comoving distance in Mpc
        r = Planck18.comoving_distance(z).to(u.Mpc).value

        cosd = np.cos(dec)
        X = r * cosd * np.cos(ra)
        Y = r * cosd * np.sin(ra)
        Z = r * np.sin(dec)
        pts = np.vstack([X, Y, Z]).T

    if subsample is not None and subsample > 0 and subsample < len(pts):
        if rng is None:
            rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=subsample, replace=False)
        pts = pts[idx]

    meta = {"mode": used_mode, "cols": list(used_cols), "n_after": int(len(pts))}
    return pts, meta


def make_bins(r_min: float, r_max: float, dr: float) -> np.ndarray:
    if r_max <= r_min:
        raise ValueError("bin-max must be greater than bin-min")
    if dr <= 0:
        raise ValueError("bin-width must be positive")
    nbin = int(math.ceil((r_max - r_min) / dr))
    edges = r_min + np.arange(nbin + 1) * dr
    return edges


def hist_from_distances(dists: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(dists, bins=bin_edges)
    return counts.astype(np.int64, copy=False)


# ----------------------- Pair enumeration methods --------------------------- #

def dd_hist_kdtree_stream(points: np.ndarray,
                          bin_edges: np.ndarray,
                          r_max: float,
                          progress_every: int = 5000) -> Tuple[np.ndarray, int]:
    N = points.shape[0]
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    tree = cKDTree(points)
    n_pairs = 0

    for i in range(N):
        idxs = tree.query_ball_point(points[i], r=r_max)
        js = [j for j in idxs if j > i]
        if not js:
            if (i % progress_every) == 0:
                print(f"  KDTree progress: i={i}/{N} (pairs so far: {n_pairs})")
            continue

        diffs = points[js] - points[i]
        dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        dmin, dmax = bin_edges[0], bin_edges[-1]
        keep = (dists >= dmin) & (dists < dmax)
        dists = dists[keep]
        if dists.size > 0:
            counts += hist_from_distances(dists, bin_edges)
            n_pairs += int(dists.size)

        if (i % progress_every) == 0:
            print(f"  KDTree progress: i={i}/{N} (pairs so far: {n_pairs})")

    return counts, n_pairs


def dd_hist_grid(points: np.ndarray,
                 bin_edges: np.ndarray,
                 r_max: float,
                 progress_every: int = 100) -> Tuple[np.ndarray, int]:
    N = points.shape[0]
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    n_pairs = 0
    dmin, dmax = bin_edges[0], bin_edges[-1]

    mins = points.min(axis=0)
    rel = points - mins
    cell_size = r_max
    cell_idx = np.floor(rel / cell_size).astype(np.int64)
    from collections import defaultdict
    cell_map: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for idx, c in enumerate(cell_idx):
        cell_map[(int(c[0]), int(c[1]), int(c[2]))].append(idx)

    neighbor_offsets = []
    for dx in (0, 1, -1):
        for dy in (0, 1, -1):
            for dz in (0, 1, -1):
                if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0):
                    neighbor_offsets.append((dx, dy, dz))

    keys = list(cell_map.keys())
    for cidx, cell in enumerate(keys):
        pts_i = cell_map[cell]
        if not pts_i:
            continue

        for (dx, dy, dz) in neighbor_offsets:
            nbr = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
            pts_j = cell_map.get(nbr)
            if not pts_j:
                continue

            Ai = np.array(pts_i, dtype=np.int64)
            Aj = np.array(pts_j, dtype=np.int64)

            if (dx, dy, dz) == (0, 0, 0):
                if len(Ai) < 2:
                    continue
                for ii in range(len(Ai) - 1):
                    diffs = points[Ai[ii + 1:]] - points[Ai[ii]]
                    dists = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
                    keep = (dists >= dmin) & (dists < dmax)
                    dists = dists[keep]
                    if dists.size:
                        counts += hist_from_distances(dists, bin_edges)
                        n_pairs += int(dists.size)
            else:
                Pi = points[Ai]
                Pj = points[Aj]
                max_chunk = 20000
                if len(Ai) * len(Aj) <= max_chunk:
                    diffs = Pi[:, None, :] - Pj[None, :, :]
                    dists = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs)).ravel()
                    keep = (dists >= dmin) & (dists < dmax)
                    dists = dists[keep]
                    if dists.size:
                        counts += hist_from_distances(dists, bin_edges)
                        n_pairs += int(dists.size)
                else:
                    step = max(1, max_chunk // max(1, len(Ai)))
                    for s in range(0, len(Aj), step):
                        e = min(len(Aj), s + step)
                        Pjb = Pj[s:e]
                        diffs = Pi[:, None, :] - Pjb[None, :, :]
                        dists = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs)).ravel()
                        keep = (dists >= dmin) & (dists < dmax)
                        dists = dists[keep]
                        if dists.size:
                            counts += hist_from_distances(dists, bin_edges)
                            n_pairs += int(dists.size)

        if (cidx % progress_every) == 0:
            print(f"  Grid progress: cell {cidx}/{len(keys)} (pairs so far: {n_pairs})")

    return counts, n_pairs


def dd_hist(points: np.ndarray,
            bin_edges: np.ndarray,
            r_max: float,
            method_hint: str = "auto") -> Tuple[np.ndarray, int, str]:
    if method_hint not in {"auto", "kdtree", "grid"}:
        method_hint = "auto"

    if method_hint == "kdtree" and not HAVE_SCIPY:
        print("Requested method 'kdtree' but SciPy not available. Falling back to 'grid'.")

    if (method_hint == "kdtree" and HAVE_SCIPY) or (method_hint == "auto" and HAVE_SCIPY):
        counts, n_pairs = dd_hist_kdtree_stream(points, bin_edges, r_max)
        return counts, n_pairs, "kdtree"

    counts, n_pairs = dd_hist_grid(points, bin_edges, r_max)
    return counts, n_pairs, "grid"


# ---------------------------- Self-test ------------------------------------- #

def brute_force_hist(points: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    N = points.shape[0]
    if N < 2:
        return np.zeros(len(bin_edges) - 1, dtype=np.int64)
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    block = 1000
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        Pi = points[i0:i1]
        diffs = Pi[:, None, :] - points[None, :, :]
        dists = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
        I, J = np.ogrid[i0:i1, 0:N]
        mask_upper = (J > I)
        dists = dists[mask_upper]
        dmin, dmax = bin_edges[0], bin_edges[-1]
        keep = (dists >= dmin) & (dists < dmax)
        dists = dists[keep]
        if dists.size:
            counts += hist_from_distances(dists, bin_edges)
    return counts


# -------------------------- Robust globbing --------------------------------- #

def expand_tiers_glob(pattern: str, tiers_dir_hint: Optional[Path]) -> List[Path]:
    norm = pattern.replace("\\", "/")
    print(f"[glob] pattern: {norm}")
    matches = glob.glob(norm)
    paths = [Path(m) for m in matches if Path(m).is_file()]
    if paths:
        print(f"[glob] matched {len(paths)} file(s)")
        for m in paths[:8]:
            print(f"  - {m}")
        return sorted(paths)

    print("[glob] matched 0 file(s)")
    if tiers_dir_hint and tiers_dir_hint.exists():
        print(f"[hint] Listing files in: {tiers_dir_hint}")
        existing = sorted(tiers_dir_hint.glob("*"))
        for e in existing[:20]:
            tag = "DIR " if e.is_dir() else "FILE"
            print(f"  {tag}  {e.name}")
        if len(existing) > 20:
            print(f"  ... ({len(existing)-20} more)")
    else:
        print(f"[hint] Tiers directory missing/unknown: {tiers_dir_hint}")
    return []


# ------------------------------ Main ---------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Step 9A: Pairwise separations (DD) for ASTRODEEP tiers.")
    here = Path(__file__).resolve()
    project_root = here.parents[2]  # ...\JWST-Mature-Galaxies

    tiers_dir_default = (project_root / "data_processed" / "tiers")
    default_glob = str((tiers_dir_default / "*.csv").resolve())

    parser.add_argument("--tiers-glob",
                        default=default_glob,
                        help="Glob for tier CSVs (default: absolute path to ../data_processed/tiers/*.csv)")
    parser.add_argument("--bin-min", type=float, default=1.0, help="Lower edge of first distance bin [Mpc].")
    parser.add_argument("--bin-max", type=float, default=300.0, help="Upper edge of last distance bin [Mpc].")
    parser.add_argument("--bin-width", type=float, default=5.0, help="Bin width [Mpc].")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on rows per tier (0 = all).")
    parser.add_argument("--subsample", type=int, default=0, help="Optional subsample size before DD (0 = none).")
    parser.add_argument("--method", choices=["auto", "kdtree", "grid"], default="auto", help="Pair enumeration method.")
    parser.add_argument("--self-test", type=int, default=0, help="If >0, brute-force sanity check on this many points per tier.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling/self-test.")
    args = parser.parse_args()

    # Build output folders
    run_tag = timestamp_tag()
    results_dir = (project_root / "results" / "step9" / run_tag).resolve()
    figures_dir = (project_root / "figures" / "step9" / run_tag).resolve()
    ensure_dirs(results_dir, figures_dir)

    # Expand tiers glob robustly
    tiers = expand_tiers_glob(args.tiers_glob, tiers_dir_default)
    if not tiers:
        print(f"No files matched: {args.tiers_glob}")
        sys.exit(1)

    print(f"\nFound {len(tiers)} candidate CSV(s). Selecting those with usable coordinates ...")
    valid: List[Path] = []
    reasons: List[str] = []
    for p in tiers:
        try:
            df_head = pd.read_csv(p, nrows=2)  # small peek
            try:
                mode, c1, c2, c3 = select_columns(df_head)
                valid.append(p)
                reasons.append(f"{p.name}: using mode={mode}, cols=({c1},{c2},{c3})")
            except Exception as e:
                reasons.append(f"{p.name}: skipped — {e}")
        except Exception as e:
            reasons.append(f"{p.name}: skipped — read error: {e!r}")

    for r in reasons:
        print("  " + r)

    if not valid:
        print("\nNo usable files. Ensure each CSV has either X,Y,Z or (RA/DEC + zspec/zphot).")
        if not ASTROPY_OK:
            print("Astropy not detected; install with: pip install astropy")
        sys.exit(1)

    # Prepare bins
    bin_edges = make_bins(args.bin_min, args.bin_max, args.bin_width)
    rng = np.random.default_rng(args.seed)

    # Per-run summary rows
    summary_rows = []

    for csv_path in valid:
        t0 = time.time()
        tier_name = parse_tier_name(csv_path)
        print(f"\n=== Tier: {tier_name} ===")
        print(f"Loading: {csv_path}")

        pts, meta_load = load_points(csv_path,
                                     max_rows=args.max_rows,
                                     subsample=args.subsample if args.subsample > 0 else None,
                                     rng=rng)
        N = len(pts)
        print(f"Loaded {N} positions [{meta_load['mode']} via columns {meta_load['cols']}]")

        if N < 2:
            print("Not enough points for pairs; skipping.")
            continue

        # Optional self-test
        if args.self_test and args.self_test > 4:
            ntest = min(args.self_test, N)
            idx = rng.choice(N, size=ntest, replace=False)
            pts_test = pts[idx]
            print(f"Self-test on {ntest} points ...")
            bf_counts = brute_force_hist(pts_test, bin_edges)
            test_counts, test_pairs, method_used = dd_hist(pts_test, bin_edges,
                                                           r_max=bin_edges[-1], method_hint=args.method)
            diff = np.abs(bf_counts - test_counts).sum()
            tot = max(1, int(bf_counts.sum()))
            rel = diff / tot
            print(f"  Self-test method={method_used}: Δ(L1)/sum = {rel:.4f}")
            if rel > 0.01:
                print("  WARNING: >1% discrepancy in self-test. Inspect binning/params before full run.")

        # Full DD
        print(f"Computing DD with method={args.method} ...")
        counts, n_pairs_within, method_used = dd_hist(pts, bin_edges, r_max=bin_edges[-1], method_hint=args.method)
        t1 = time.time()
        elapsed = t1 - t0

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        out_csv = results_dir / f"DD_hist_{tier_name}.csv"
        out_df = pd.DataFrame({
            "bin_left_Mpc": bin_edges[:-1],
            "bin_right_Mpc": bin_edges[1:],
            "bin_center_Mpc": centers,
            "DD_count": counts
        })
        out_df.to_csv(out_csv, index=False)

        n_pairs_total = N * (N - 1) // 2
        meta = {
            "tier": tier_name,
            "input_csv": str(csv_path.resolve()),
            "coordinate_mode": meta_load["mode"],
            "columns_used": meta_load["cols"],
            "n_points": int(N),
            "n_pairs_total_all_distances": int(n_pairs_total),
            "n_pairs_within_rmax": int(n_pairs_within),
            "r_min_Mpc": float(bin_edges[0]),
            "r_max_Mpc": float(bin_edges[-1]),
            "bin_width_Mpc": float(bin_edges[1] - bin_edges[0]),
            "nbins": int(len(bin_edges) - 1),
            "method_used": method_used,
            "units": "Mpc (comoving)",
            "timestamp": run_tag,
            "runtime_sec": elapsed,
            "scipy_available": HAVE_SCIPY,
            "astropy_available": ASTROPY_OK
        }
        out_meta = results_dir / f"DD_meta_{tier_name}.json"
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        out_png = figures_dir / f"DD_hist_{tier_name}.png"
        plt.figure(figsize=(7, 5))
        plt.step(centers, counts, where="mid")
        plt.xlabel("Separation d  [Mpc (comoving)]")
        plt.ylabel("Pair counts (DD)")
        plt.title(f"DD separation histogram — {tier_name}")
        plt.axvline(150.0, linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"Done {tier_name}: n={N}, pairs<=rmax={n_pairs_within:,}, bins={len(bin_edges) - 1}, method={method_used}")
        print(f"  Wrote: {out_csv.name}, {out_meta.name}")
        print(f"  Figure: {out_png.name}")
        print(f"  Elapsed: {elapsed:.1f} s")

        summary_rows.append({
            "tier": tier_name,
            "n_points": N,
            "n_pairs_total_all_distances": n_pairs_total,
            "n_pairs_within_rmax": n_pairs_within,
            "r_min_Mpc": bin_edges[0],
            "r_max_Mpc": bin_edges[-1],
            "bin_width_Mpc": bin_edges[1] - bin_edges[0],
            "nbins": len(bin_edges) - 1,
            "method_used": method_used,
            "coordinate_mode": meta_load["mode"],
            "results_csv": str(out_csv.name),
            "meta_json": str(out_meta.name),
            "figure_png": str(out_png.name)
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = results_dir / "DD_run_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nRun summary saved: {summary_csv}")
    else:
        print("\nNo tiers processed. Check your inputs.")


if __name__ == "__main__":
    main()
