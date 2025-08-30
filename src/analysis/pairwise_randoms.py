#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 9B — Selection-matched random catalogs + RR/DR pair counts for JWST tiers.

Improvements vs. prior version:
  • (RA,Dec) are sampled JOINTLY by row ⇒ preserves the empirical sky mask exactly.
  • Optional stratification by 'field' (if present) so each field's mask and N(z) are preserved.
  • Optional smoothed N(z): KDE sampling (--z-kde) or discrete with Gaussian jitter (--z-jitter).

Outputs (per run tag):
  data_processed/randoms/step9b/<timestamp>/random_<tier>.csv
  results/step9b/<timestamp>/{RR_hist_<tier>.csv, RR_meta_<tier>.json}
  results/step9b/<timestamp>/{DR_hist_<tier>.csv, DR_meta_<tier>.json}
  figures/step9b/<timestamp>/{RR_hist_<tier>.png, DR_hist_<tier>.png}
  results/step9b/<timestamp>/step9b_run_summary.csv

Run examples (from src/):
  python analysis\\pairwise_randoms.py --tiers-glob "../data_processed/tiers/*.csv" --bin-min 1 --bin-max 250 --bin-width 5 --rand-mult 10
  # With field stratification + smoothed N(z)
  python analysis\\pairwise_randoms.py --tiers-glob "../data_processed/tiers/*.csv" --bin-min 1 --bin-max 250 --bin-width 5 --rand-mult 10 --stratify-by-field --z-kde

This script never edits inputs; it writes only to timestamped results/ and figures/ (and a randoms/ snapshot for inspection).
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

# Fast path for pair enumeration
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# KDE for smoothed N(z) (optional)
try:
    from scipy.stats import gaussian_kde
    HAVE_SCIPY_STATS = True
except Exception:
    HAVE_SCIPY_STATS = False

# Astropy for RA/Dec/z -> Cartesian
ASTROPY_OK = True
try:
    from astropy.cosmology import Planck18
    import astropy.units as u
except Exception:
    ASTROPY_OK = False

import glob
import os

# ---------- utilities ----------

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_tier_name(path: Path) -> str:
    stem = path.stem
    if "astrodeep_" in stem:
        s = stem.split("astrodeep_", 1)[1]
        if s.endswith("_coords"):
            s = s[:-7]
        if s:
            return s
    return stem

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def make_bins(r_min: float, r_max: float, dr: float) -> np.ndarray:
    if r_max <= r_min:
        raise ValueError("bin-max must be greater than bin-min")
    if dr <= 0:
        raise ValueError("bin-width must be positive")
    nbin = int(math.ceil((r_max - r_min) / dr))
    return r_min + np.arange(nbin + 1) * dr

def centers_from_edges(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])

def select_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, Optional[str]]:
    """
    Return (mode, RAcol, DECcol, Zcol, field_col) or (mode, Xcol, Ycol, Zcol, field_col) depending on mode.
    mode: 'xyz' (Cartesian present) or 'radecz' (need RA/DEC/z).
    field_col is returned if a column named 'field' (case-insensitive) exists; else None.
    """
    cols = {c.lower(): c for c in df.columns}
    field_col = cols.get("field", None)

    # Cartesian present?
    if all(k in cols for k in ("x", "y", "z")):
        return ("xyz", cols["x"], cols["y"], cols["z"], field_col)

    # RA/DEC/z candidates
    ra_c  = ["ra", "ra_optap", "ra_photoz"]
    dec_c = ["dec", "dec_optap", "dec_photoz"]
    z_c   = ["zspec", "zphot", "z", "z_best"]

    ra = next((cols[k] for k in ra_c if k in cols), None)
    dec = next((cols[k] for k in dec_c if k in cols), None)
    zz  = next((cols[k] for k in z_c if k in cols), None)
    if ra and dec and zz:
        return ("radecz", ra, dec, zz, field_col)

    raise ValueError("No usable coordinate columns found (need X,Y,Z or RA/DEC + z).")

def load_data_points(csv_path: Path, max_rows: int = 0) -> Tuple[np.ndarray, Dict, pd.DataFrame]:
    """
    Return (points_xyz[N,3], meta, df_used) where df_used always includes the coordinate columns
    and, if present, the 'field' column.
    """
    df = pd.read_csv(csv_path)
    mode, c1, c2, c3, field_col = select_columns(df)

    cols_needed = [c1, c2, c3]
    if field_col:
        cols_needed.append(field_col)
    used_df = df.loc[:, list(dict.fromkeys(cols_needed))].copy()

    if max_rows and max_rows > 0:
        used_df = used_df.iloc[:max_rows].copy()

    if mode == "xyz":
        X = pd.to_numeric(used_df[c1], errors="coerce").to_numpy(np.float64)
        Y = pd.to_numeric(used_df[c2], errors="coerce").to_numpy(np.float64)
        Z = pd.to_numeric(used_df[c3], errors="coerce").to_numpy(np.float64)
        pts = np.vstack([X, Y, Z]).T
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        df_used = used_df.loc[mask].reset_index(drop=True)
        meta = {"mode": "xyz", "cols": [c1, c2, c3], "field_col": field_col, "n_after": int(len(pts))}
        return pts, meta, df_used

    if not ASTROPY_OK:
        raise RuntimeError("Astropy is required for RA/DEC/z -> XYZ. Install with: pip install astropy")

    # radecz path
    ra_deg  = pd.to_numeric(used_df[c1], errors="coerce").to_numpy(np.float64)
    dec_deg = pd.to_numeric(used_df[c2], errors="coerce").to_numpy(np.float64)
    zz      = pd.to_numeric(used_df[c3], errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(ra_deg) & np.isfinite(dec_deg) & np.isfinite(zz) & (zz > 0)

    if field_col:
        df_used = pd.DataFrame({c1: ra_deg[finite], c2: dec_deg[finite], c3: zz[finite], field_col: used_df.loc[finite, field_col].to_numpy()})
    else:
        df_used = pd.DataFrame({c1: ra_deg[finite], c2: dec_deg[finite], c3: zz[finite]})

    ra  = np.deg2rad(df_used[c1].to_numpy())
    dec = np.deg2rad(df_used[c2].to_numpy())
    z   = df_used[c3].to_numpy()

    r = Planck18.comoving_distance(z).to(u.Mpc).value
    cosd = np.cos(dec)
    X = r * cosd * np.cos(ra)
    Y = r * cosd * np.sin(ra)
    Z = r * np.sin(dec)
    pts = np.vstack([X, Y, Z]).T

    meta = {"mode": "radecz", "cols": [c1, c2, c3], "field_col": field_col, "n_after": int(len(pts))}
    return pts, meta, df_used.reset_index(drop=True)

def draw_z_samples_basic(z_vals: np.ndarray, size: int, rng: np.random.Generator, jitter_sigma: float = 0.0) -> np.ndarray:
    """Discrete resample with optional Gaussian jitter and clipping to data range."""
    idx = rng.integers(0, len(z_vals), size=size)
    s = z_vals[idx].astype(np.float64, copy=True)
    if jitter_sigma and jitter_sigma > 0:
        s += rng.normal(0.0, jitter_sigma, size=size)
        zmin, zmax = float(np.min(z_vals)), float(np.max(z_vals))
        s = np.clip(s, zmin, zmax)
    return s

def draw_z_samples_kde(z_vals: np.ndarray, size: int, rng: np.random.Generator, bw_method: Optional[str|float] = "scott") -> np.ndarray:
    """Sample from a KDE estimate of N(z). Requires SciPy."""
    kde = gaussian_kde(z_vals.astype(np.float64), bw_method=bw_method)
    s = kde.resample(size, seed=rng).reshape(-1).astype(np.float64)  # shape (n_samples,)
    zmin, zmax = float(np.min(z_vals)), float(np.max(z_vals))
    return np.clip(s, zmin, zmax)

def sample_random_catalog(df_used: pd.DataFrame,
                          meta: Dict,
                          n_rand: int,
                          rng: np.random.Generator,
                          stratify_by_field: bool = False,
                          z_kde: bool = False,
                          z_kde_bw: Optional[str|float] = "scott",
                          z_jitter: float = 0.0) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Build a selection-matched random catalog.

    If meta['mode']=='xyz':
        - Resample existing XYZ rows with replacement; add tiny jitter to break ties.

    If meta['mode']=='radecz':
        - (RA,Dec) are sampled JOINTLY by row (preserves sky footprint exactly).
        - z is sampled independently from N(z):
            * stratify_by_field=True: use N(z) within each field for the chosen RA/Dec row's field.
            * else: use the overall N(z).
        - z sampling can be KDE-based (if --z-kde) or discrete with optional Gaussian jitter (--z-jitter).

    Returns: (random_df, random_xyz, stats_dict)
    """
    stats = {"mode": meta["mode"], "stratified": False, "z_sampler": "discrete", "z_kde_bw": None, "z_jitter": 0.0}

    if meta["mode"] == "xyz":
        idx = rng.integers(0, len(df_used), size=n_rand)
        X = pd.to_numeric(df_used.iloc[idx, 0], errors="coerce").to_numpy(np.float64)
        Y = pd.to_numeric(df_used.iloc[idx, 1], errors="coerce").to_numpy(np.float64)
        Z = pd.to_numeric(df_used.iloc[idx, 2], errors="coerce").to_numpy(np.float64)
        eps = rng.normal(0.0, 1e-6, size=(n_rand, 3))  # ~kpc jitter in Mpc units
        pts = np.vstack([X, Y, Z]).T + eps
        out_df = pd.DataFrame({"X": pts[:, 0], "Y": pts[:, 1], "Z": pts[:, 2]})
        return out_df, pts, stats

    # radecz path
    ra_col, dec_col, z_col = meta["cols"]
    field_col = meta.get("field_col")

    # We always sample (RA,Dec) jointly by row
    idx_rd = rng.integers(0, len(df_used), size=n_rand)
    ra_r = df_used[ra_col].to_numpy()[idx_rd]
    dec_r = df_used[dec_col].to_numpy()[idx_rd]

    # Decide how to draw z
    if stratify_by_field and field_col and (field_col in df_used.columns):
        stats["stratified"] = True
        # Pre-split z by field
        z_by_field: Dict[str, np.ndarray] = {}
        for key, grp in df_used.groupby(field_col):
            z_by_field[str(key)] = grp[z_col].to_numpy().astype(np.float64)

        # For each sampled RA/Dec row, get its field and sample z from that field distribution
        fields_for_rows = df_used[field_col].astype(str).to_numpy()[idx_rd]
        z_r = np.empty(n_rand, dtype=np.float64)
        for ufield in np.unique(fields_for_rows):
            mask = (fields_for_rows == ufield)
            z_vals = z_by_field[str(ufield)]
            if z_kde and HAVE_SCIPY_STATS and len(z_vals) > 5:
                stats["z_sampler"] = "kde"
                stats["z_kde_bw"] = z_kde_bw
                z_r[mask] = draw_z_samples_kde(z_vals, mask.sum(), rng, bw_method=z_kde_bw)
            else:
                stats["z_jitter"] = float(z_jitter)
                z_r[mask] = draw_z_samples_basic(z_vals, mask.sum(), rng, jitter_sigma=z_jitter)
    else:
        # Global N(z)
        z_vals = df_used[z_col].to_numpy().astype(np.float64)
        if z_kde and HAVE_SCIPY_STATS and len(z_vals) > 5:
            stats["z_sampler"] = "kde"
            stats["z_kde_bw"] = z_kde_bw
            z_r = draw_z_samples_kde(z_vals, n_rand, rng, bw_method=z_kde_bw)
        else:
            stats["z_jitter"] = float(z_jitter)
            z_r = draw_z_samples_basic(z_vals, n_rand, rng, jitter_sigma=z_jitter)

    # Convert to XYZ
    ra = np.deg2rad(ra_r)
    dec = np.deg2rad(dec_r)
    r = Planck18.comoving_distance(z_r).to(u.Mpc).value
    cosd = np.cos(dec)
    X = r * cosd * np.cos(ra)
    Y = r * cosd * np.sin(ra)
    Z = r * np.sin(dec)
    pts = np.vstack([X, Y, Z]).T

    out_cols = {"RA_deg": ra_r, "DEC_deg": dec_r, "z": z_r, "X": X, "Y": Y, "Z": Z}
    if meta.get("field_col") and (meta["field_col"] in df_used.columns):
        out_cols["field"] = df_used[meta["field_col"]].to_numpy()[idx_rd]
    out_df = pd.DataFrame(out_cols)

    return out_df, pts, stats

def hist_from_dists(dists: np.ndarray, edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(dists, bins=edges)
    return h.astype(np.int64)

def rr_hist(points: np.ndarray, edges: np.ndarray, r_max: float) -> Tuple[np.ndarray, int, str]:
    if HAVE_SCIPY:
        tree = cKDTree(points)
        N = len(points)
        counts = np.zeros(len(edges)-1, dtype=np.int64)
        n_pairs = 0
        for i in range(N):
            js = [j for j in tree.query_ball_point(points[i], r=r_max) if j > i]
            if not js:
                continue
            diffs = points[js] - points[i]
            d = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
            keep = (d >= edges[0]) & (d < edges[-1])
            d = d[keep]
            if d.size:
                counts += hist_from_dists(d, edges)
                n_pairs += int(d.size)
        return counts, n_pairs, "kdtree"

    # Fallback (exact, slower)
    pts = points
    N = len(pts)
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    n_pairs = 0
    block = 2000
    for i0 in range(0, N, block):
        i1 = min(N, i0+block)
        Pi = pts[i0:i1]
        diffs = Pi[:,None,:] - pts[None,:,:]
        d = np.sqrt(np.einsum("ijk,ijk->ij", diffs, diffs))
        I, J = np.ogrid[i0:i1, 0:N]
        mask_upper = (J > I)
        d = d[mask_upper]
        keep = (d >= edges[0]) & (d < edges[-1])
        d = d[keep]
        if d.size:
            counts += hist_from_dists(d, edges)
            n_pairs += int(d.size)
    return counts, n_pairs, "bruteforce"

def dr_hist(points_data: np.ndarray, points_rand: np.ndarray,
            edges: np.ndarray, r_max: float) -> Tuple[np.ndarray, int, str]:
    if HAVE_SCIPY:
        tree = cKDTree(points_rand)
        counts = np.zeros(len(edges)-1, dtype=np.int64)
        n_pairs = 0
        for i in range(len(points_data)):
            js = tree.query_ball_point(points_data[i], r=r_max)
            if not js:
                continue
            diffs = points_rand[js] - points_data[i]
            d = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
            keep = (d >= edges[0]) & (d < edges[-1])
            d = d[keep]
            if d.size:
                counts += hist_from_dists(d, edges)
                n_pairs += int(d.size)
        return counts, n_pairs, "kdtree"

    # Fallback
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    n_pairs = 0
    PR = points_rand
    for i in range(len(points_data)):
        diffs = PR - points_data[i]
        d = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        keep = (d >= edges[0]) & (d < edges[-1])
        d = d[keep]
        if d.size:
            counts += hist_from_dists(d, edges)
            n_pairs += int(d.size)
    return counts, n_pairs, "bruteforce"

def expand_glob(pattern: str) -> List[Path]:
    norm = pattern.replace("\\", "/")
    return sorted(Path(m) for m in glob.glob(norm) if Path(m).is_file())

def step_plot(edges: np.ndarray, counts: np.ndarray, title: str, out_png: Path):
    centers = centers_from_edges(edges)
    plt.figure(figsize=(7, 5))
    plt.step(centers, counts, where="mid")
    plt.xlabel("Separation d  [Mpc (comoving)]")
    plt.ylabel("Pair counts")
    plt.title(title)
    plt.axvline(150.0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------- main ----------

def main():
    here = Path(__file__).resolve()
    project_root = here.parents[2]

    parser = argparse.ArgumentParser(description="Step 9B: random catalogs + RR/DR histograms (mask-matched).")
    tiers_default = str((project_root / "data_processed" / "tiers" / "*.csv").resolve())
    parser.add_argument("--tiers-glob", default=tiers_default, help="Glob for tier CSVs.")
    parser.add_argument("--bin-min", type=float, default=1.0)
    parser.add_argument("--bin-max", type=float, default=300.0)
    parser.add_argument("--bin-width", type=float, default=5.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on data rows per tier (0=all).")
    parser.add_argument("--rand-mult", type=float, default=10.0, help="Randoms per data point (e.g., 10x).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by-field", action="store_true", help="Preserve sky mask + N(z) within each 'field' group if present.")
    parser.add_argument("--z-kde", action="store_true", help="Sample z from KDE-smoothed N(z) (requires SciPy).")
    parser.add_argument("--z-kde-bw", default="scott", help="KDE bandwidth ('scott', 'silverman', or float).")
    parser.add_argument("--z-jitter", type=float, default=0.0, help="Gaussian sigma for jitter when using discrete N(z).")
    args = parser.parse_args()

    run_tag = timestamp_tag()
    results_dir = (project_root / "results" / "step9b" / run_tag).resolve()
    figures_dir = (project_root / "figures" / "step9b" / run_tag).resolve()
    randoms_dir = (project_root / "data_processed" / "randoms" / "step9b" / run_tag).resolve()
    ensure_dirs(results_dir, figures_dir, randoms_dir)

    edges = make_bins(args.bin_min, args.bin_max, args.bin_width)
    r_max = float(edges[-1])
    rng = np.random.default_rng(args.seed)

    tier_files = expand_glob(args.tiers_glob)
    if not tier_files:
        print(f"No tier files matched: {args.tiers_glob}")
        sys.exit(1)

    print(f"Matched {len(tier_files)} tier file(s). SciPy KDTree: {HAVE_SCIPY}. Astropy: {ASTROPY_OK}. KDE: {HAVE_SCIPY_STATS}")

    summary = []

    for path in tier_files:
        t0 = time.time()
        tier = parse_tier_name(path)
        print(f"\n=== Tier {tier} ===")
        print(f"Loading data: {path}")
        D_xyz, meta_data, df_used = load_data_points(path, max_rows=args.max_rows)
        N_D = len(D_xyz)
        if N_D < 2:
            print("Too few data points; skipping.")
            continue
        print(f"  Data rows usable: {N_D} [{meta_data['mode']} cols={meta_data['cols']}, field={meta_data.get('field_col')}]")

        N_R = int(math.ceil(args.rand_mult * N_D))
        print(f"  Building random catalog: N_R = {N_R} (multiplier={args.rand_mult})")
        R_df, R_xyz, rand_stats = sample_random_catalog(
            df_used, meta_data, N_R, rng,
            stratify_by_field=args.stratify_by_field,
            z_kde=args.z_kde,
            z_kde_bw=args.z_kde_bw,
            z_jitter=args.z_jitter
        )

        # Write random catalog for inspection
        out_rand = randoms_dir / f"random_{tier}.csv"
        R_df.to_csv(out_rand, index=False)

        # RR
        print("  RR histogram ...")
        RR_counts, RR_pairs, rr_method = rr_hist(R_xyz, edges, r_max)
        # DR
        print("  DR histogram ...")
        DR_counts, DR_pairs, dr_method = dr_hist(D_xyz, R_xyz, edges, r_max)

        # Normalization factors
        pairs_DD_all = N_D * (N_D - 1) // 2
        pairs_RR_all = N_R * (N_R - 1) // 2
        pairs_DR_all = N_D * N_R

        RR_norm = RR_counts / max(1, pairs_RR_all)
        DR_norm = DR_counts / max(1, pairs_DR_all)

        # Save CSVs
        centers = centers_from_edges(edges)
        rr_df = pd.DataFrame({
            "bin_left_Mpc": edges[:-1],
            "bin_right_Mpc": edges[1:],
            "bin_center_Mpc": centers,
            "RR_count": RR_counts,
            "RR_norm": RR_norm
        })
        dr_df = pd.DataFrame({
            "bin_left_Mpc": edges[:-1],
            "bin_right_Mpc": edges[1:],
            "bin_center_Mpc": centers,
            "DR_count": DR_counts,
            "DR_norm": DR_norm
        })
        out_rr = results_dir / f"RR_hist_{tier}.csv"
        out_dr = results_dir / f"DR_hist_{tier}.csv"
        rr_df.to_csv(out_rr, index=False)
        dr_df.to_csv(out_dr, index=False)

        # Meta JSONs (record random generation strategy)
        base_meta = {
            "tier": tier,
            "input_csv": str(path.resolve()),
            "random_csv": str(out_rand.resolve()),
            "n_data": int(N_D),
            "n_random": int(N_R),
            "pairs_DD_all": int(pairs_DD_all),
            "pairs_RR_all": int(pairs_RR_all),
            "pairs_DR_all": int(pairs_DR_all),
            "r_min_Mpc": float(edges[0]),
            "r_max_Mpc": float(edges[-1]),
            "bin_width_Mpc": float(edges[1] - edges[0]),
            "nbins": int(len(edges) - 1),
            "coordinate_mode": meta_data["mode"],
            "columns_used": meta_data["cols"],
            "field_col": meta_data.get("field_col"),
            "random_strategy": rand_stats,
            "timestamp": timestamp_tag()
        }
        with open(results_dir / f"RR_meta_{tier}.json", "w", encoding="utf-8") as f:
            json.dump({**base_meta, "method_used": rr_method}, f, indent=2)
        with open(results_dir / f"DR_meta_{tier}.json", "w", encoding="utf-8") as f:
            json.dump({**base_meta, "method_used": dr_method}, f, indent=2)

        # Plots
        step_plot(edges, RR_counts, f"RR separation histogram — {tier}", figures_dir / f"RR_hist_{tier}.png")
        step_plot(edges, DR_counts, f"DR separation histogram — {tier}", figures_dir / f"DR_hist_{tier}.png")

        elapsed = time.time() - t0
        print(f"  Done {tier}: RR_pairs<=rmax={RR_pairs:,}, DR_pairs<=rmax={DR_pairs:,}  (elapsed {elapsed:.1f}s)")

        summary.append({
            "tier": tier,
            "n_data": N_D,
            "n_random": N_R,
            "pairs_RR_all": pairs_RR_all,
            "pairs_DR_all": pairs_DR_all,
            "r_min_Mpc": edges[0],
            "r_max_Mpc": edges[-1],
            "bin_width_Mpc": edges[1] - edges[0],
            "nbins": len(edges) - 1,
            "RR_csv": out_rr.name,
            "DR_csv": out_dr.name,
            "RR_meta": f"RR_meta_{tier}.json",
            "DR_meta": f"DR_meta_{tier}.json",
            "RR_fig": f"RR_hist_{tier}.png",
            "DR_fig": f"DR_hist_{tier}.png",
            "method_RR": rr_method,
            "method_DR": dr_method,
            "random_strategy": rand_stats
        })

    if summary:
        s = pd.DataFrame(summary)
        s.to_csv(results_dir / "step9b_run_summary.csv", index=False)
        print(f"\nRun summary: {results_dir / 'step9b_run_summary.csv'}")
    else:
        print("\nNo tiers processed.")

if __name__ == "__main__":
    main()
