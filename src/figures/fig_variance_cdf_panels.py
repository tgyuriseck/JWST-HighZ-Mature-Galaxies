# C:\JWST-Mature-Galaxies\src\figures\fig_variance_cdf_panels_v5.py
"""
Make 2-panel CDF plot of normalized inter-field variance (QC).

- Finds the *latest* step11c/run_* folder automatically.
- Loads mock normalized-variance arrays for z=8–10 and z=10–20.
- Loads observed normalized-variance scalars if present; otherwise uses
  your QC defaults (z8–10: 0.1011, z10–20: 0.004014) unless you override
  via CLI flags.
- Computes right-tail p = P(mock > observed).
- Saves to: results/figures_v2/variance/variance_cdf_panels_v1.{pdf,png}
  (filename kept stable so your LaTeX include keeps working).
"""

from __future__ import annotations
import os
import json
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- path helpers -----------------------------

def latest_run_dir() -> str:
    """Return newest 'run_*' under results/step11c (search a few roots)."""
    here = os.path.abspath(os.path.dirname(__file__))
    roots = [
        os.path.abspath(os.path.join(here, "..", "results", "step11c")),
        os.path.abspath("results/step11c"),
        os.path.abspath(os.path.join(here, "..", "..", "results", "step11c")),
    ]
    looked = []
    for root in roots:
        looked.append(root)
        if os.path.isdir(root):
            runs = sorted(glob(os.path.join(root, "run_*")), key=os.path.getmtime)
            if runs:
                return runs[-1]
    raise FileNotFoundError("No run_* folder found under any of:\n  " + "\n  ".join(looked))


def _first_numeric_column_from_csv(path: str) -> np.ndarray:
    # Try reading with header
    try:
        rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if getattr(rec, "dtype", None) is not None and rec.dtype.names:
            for name in rec.dtype.names:
                try:
                    col = np.asarray(rec[name], dtype=float)
                    col = col[np.isfinite(col)]
                    if col.size:
                        return col.ravel()
                except Exception:
                    continue
    except Exception:
        pass
    # Plain numeric CSV
    try:
        arr = np.loadtxt(path, delimiter=",")
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        # whitespace-delimited fallback
        arr = np.loadtxt(path)
        return np.asarray(arr, dtype=float).ravel()


def _load_array_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    if ext in (".csv", ".txt", ".tsv"):
        return _first_numeric_column_from_csv(path)
    if ext == ".json":
        with open(path, "r") as f:
            obj = json.load(f)
        for k in ("values", "mock", "data", "array"):
            if k in obj:
                return np.asarray(obj[k], dtype=float).ravel()
        try:
            return np.atleast_1d(float(obj)).ravel()
        except Exception as e:
            raise ValueError(f"JSON did not contain a numeric array: {path}") from e
    # last-ditch
    return np.asarray(np.loadtxt(path), dtype=float).ravel()


def _find_file(run_dir: str, slice_tag: str, kind: str) -> str | None:
    """
    kind in {'mock','obs'}; search recursively for plausible names.
    """
    if kind == "mock":
        patts = [
            f"**/*{slice_tag}*mock*norm*.npy",
            f"**/*{slice_tag}*mock*norm*.csv",
            f"**/*{slice_tag}*mock*norm*.txt",
            f"**/*{slice_tag}*mock*norm*.json",
            f"**/*mock*{slice_tag}*norm*.npy",
            f"**/*mock*{slice_tag}*norm*.csv",
        ]
    else:
        patts = [
            f"**/*{slice_tag}*obs*norm*.npy",
            f"**/*{slice_tag}*observed*norm*.npy",
            f"**/*{slice_tag}*obs*norm*.csv",
            f"**/*{slice_tag}*observed*norm*.csv",
            f"**/*{slice_tag}*obs*norm*.txt",
            f"**/*{slice_tag}*observed*norm*.txt",
            f"**/*{slice_tag}*obs*norm*.json",
            f"**/*{slice_tag}*observed*norm*.json",
        ]
    for patt in patts:
        hits = glob(os.path.join(run_dir, patt), recursive=True)
        if hits:
            return sorted(hits, key=os.path.getmtime)[-1]
    return None


def load_slice(run_dir: str, slice_tag: str, obs_override: float | None = None):
    """
    Return (mock_values_array, observed_scalar) for a slice_tag like 'z8_10'.
    """
    mock_file = _find_file(run_dir, slice_tag, "mock")
    if not mock_file:
        raise FileNotFoundError(f"Mock file for {slice_tag} not found under {run_dir}")
    mock_vals = _load_array_any(mock_file)

    if obs_override is not None:
        return mock_vals, float(obs_override)

    obs_file = _find_file(run_dir, slice_tag, "obs")
    if obs_file:
        obs_arr = _load_array_any(obs_file)
        if obs_arr.size != 1:
            raise ValueError(f"Observed file for {slice_tag} should contain a single scalar, got {obs_arr.shape}")
        return mock_vals, float(obs_arr.ravel()[0])

    # QC fallbacks you reported
    defaults = {"z8_10": 0.1011, "z10_20": 0.004014}
    if slice_tag in defaults:
        print(f"[{slice_tag}] WARNING: no observed file; using fallback {defaults[slice_tag]:g} "
              f"(override with --obs-{slice_tag.replace('_','-')}).")
        return mock_vals, defaults[slice_tag]

    raise FileNotFoundError(f"Observed normalized variance not found for {slice_tag}")


# ----------------------------- plotting -----------------------------

def main():
    # Parse optional overrides so you can freeze observed values explicitly
    ap = argparse.ArgumentParser(description="CDF panels of normalized inter-field variance (QC).")
    ap.add_argument("--obs-z8-10", type=float, default=None, help="Override observed normalized variance for z=8–10")
    ap.add_argument("--obs-z10-20", type=float, default=None, help="Override observed normalized variance for z=10–20")
    args = ap.parse_args()

    run_dir = latest_run_dir()
    print("Latest step11c run:", run_dir)

    # Load
    mock_8_10,  obs_8_10  = load_slice(run_dir, "z8_10",  obs_override=args.obs_z8_10)
    mock_10_20, obs_10_20 = load_slice(run_dir, "z10_20", obs_override=args.obs_z10_20)

    # Right-tail p = P(mock > observed)
    p_8_10  = float(np.mean(mock_8_10  > obs_8_10))
    p_10_20 = float(np.mean(mock_10_20 > obs_10_20))
    print(f"Computed p-values:  z8–10 = {p_8_10:.6g}   z10–20 = {p_10_20:.6g}")

    # Use your house style utilities
    from figures.jwst_plot_style import fig_ax, save_figure, ensure_dir

    fig, axs = fig_ax(1, 2, width_in=7.1, height_in=3.2, wspace=0.28)
    ax1, ax2 = axs

    def draw_panel(ax, xvals, x_obs, title, panel_tag, p_text, p_at="right"):
        xs = np.sort(np.asarray(xvals, dtype=float))
        cdf = np.linspace(0.0, 1.0, xs.size, endpoint=True)
        ax.plot(xs, cdf, lw=2)
        ax.axvline(x_obs, color="tab:red", ls="--", lw=2)
        ax.set_title(title)
        ax.set_xlabel("Inter-field variance (normalized)")
        ax.set_ylabel("Cumulative probability")
        ax.text(0.03, 0.93, f"({panel_tag})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        # p-label placement
        if p_at == "right":
            ax.text(0.94, 0.10, p_text, transform=ax.transAxes, ha="right", va="bottom")
        else:  # a touch left of center-bottom
            ax.text(0.70, 0.08, p_text, transform=ax.transAxes, ha="left", va="bottom")

    # Left panel: keep "p = 0" in the lower-right area (but inside the box)
    p_text_left = "p = 0" if p_8_10 == 0.0 else f"p = {p_8_10:.4f}"
    draw_panel(ax1, mock_8_10,  obs_8_10,  "z=8–10",  "a", p_text_left, p_at="right")

    # Right panel: show numeric p (formatted 4 decimals), sit near bottom-right but inside
    draw_panel(ax2, mock_10_20, obs_10_20, "z=10–20", "b", f"p = {p_10_20:.4f}", p_at="right")

    # Save alongside your other variance figures
    here = os.path.abspath(os.path.dirname(__file__))
    proj = os.path.abspath(os.path.join(here, ".."))
    outdir = os.path.join(proj, "results", "figures_v2", "variance")
    ensure_dir(outdir)
    outbase = os.path.join(outdir, "variance_cdf_panels_v5")  # stable name for LaTeX

    written = save_figure(fig, outbase)
    print("Saved:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
