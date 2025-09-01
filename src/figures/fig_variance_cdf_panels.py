# C:\JWST-HighZ-Mature-Galaxies\src\figures\fig_variance_cdf_panels.py
"""
Variance histogram panels with inset (QC, normalized), for Figure 4.

What this script does
---------------------
- Auto-discovers your newest `results\step11*\run_*` (works for step11, step11b, step11c).
- Loads the mock normalized-variance array and the observed normalized variance
  for each slice (z8_10 and z10_20).
- Plots a histogram of the mocks (with vertical lines for mock mean and mean+1σ),
  draws a vertical dashed line at the observed value, and includes a small inset zoom.
- Uses LaTeX-style titles:
      z = 8–10 (QC): normalized inter-field variance
      z = 10–20 (QC): normalized inter-field variance
- Saves **versionless** outputs under:
      results\figures\paper\variance_hist_z8_10_qc_linear_inset.(png|pdf)
      results\figures\paper\variance_hist_z10_20_qc_linear_inset.(png|pdf)

Run from VS Code terminal (working dir = ...\src):
    python figures\fig_variance_cdf_panels.py

Optional CLI overrides for observed values:
    --obs-z8-10  <float>
    --obs-z10-20 <float>
"""

from __future__ import annotations
import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- Path discovery ----------------------------- #

def _project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))          # ...\src\figures
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))  # repo root

def _latest_step11_run(results_root: str) -> str:
    """Return newest run_* dir under results\step11* (step11, step11b, step11c...)."""
    candidates = []
    for step in glob.glob(os.path.join(results_root, "step11*")):
        for run in glob.glob(os.path.join(step, "run_*")):
            if os.path.isdir(run):
                candidates.append(run)
    if not candidates:
        raise FileNotFoundError(
            f"No run_* folder found under '{results_root}\\step11*'. "
            "If you are pulling results from an older project, pass files explicitly or copy them in."
        )
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


# ----------------------------- Loading helpers ---------------------------- #

def _load_numeric_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=float).ravel()
    # try CSV then whitespace
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        arr = np.loadtxt(path)
    return np.asarray(arr, dtype=float).ravel()

def _first_numeric_col_from_csv(path: str) -> np.ndarray:
    # header-aware read; fall back to raw load
    try:
        rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if getattr(rec, "dtype", None) is not None and rec.dtype.names:
            for name in rec.dtype.names:
                try:
                    col = np.asarray(rec[name], dtype=float).ravel()
                    col = col[np.isfinite(col)]
                    if col.size:
                        return col
                except Exception:
                    pass
    except Exception:
        pass
    return _load_numeric_array(path)

def _find_file(run_dir: str, slice_tag: str, kind: str) -> str | None:
    """
    Locate a file under run_dir for a given slice_tag ('z8_10' or 'z10_20').
    kind ∈ {'mock','obs'} for mock distribution or observed scalar (in a 1-cell file).
    """
    patterns = []
    if kind == "mock":
        patterns = [
            f"**/*{slice_tag}*mock*norm*.npy",
            f"**/*{slice_tag}*mock*norm*.csv",
            f"**/*mock*{slice_tag}*norm*.csv",
            f"**/*{slice_tag}*mock*norm*.txt",
        ]
    else:
        patterns = [
            f"**/*{slice_tag}*obs*norm*.npy",
            f"**/*{slice_tag}*observed*norm*.npy",
            f"**/*{slice_tag}*obs*norm*.csv",
            f"**/*{slice_tag}*observed*norm*.csv",
            f"**/*{slice_tag}*obs*norm*.txt",
            f"**/*{slice_tag}*observed*norm*.txt",
        ]
    hits = []
    for patt in patterns:
        hits.extend(glob.glob(os.path.join(run_dir, patt), recursive=True))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime)
    return hits[-1]

def load_slice(run_dir: str, slice_tag: str, obs_override: float | None) -> tuple[np.ndarray, float]:
    """Return (mock_values, observed_scalar) for 'z8_10' or 'z10_20'."""
    mock_path = _find_file(run_dir, slice_tag, "mock")
    if not mock_path:
        raise FileNotFoundError(f"No mock normalized-variance file found for {slice_tag} under {run_dir}")

    # mock array (supports CSV first-column or raw numeric file)
    if mock_path.lower().endswith(".csv"):
        mocks = _first_numeric_col_from_csv(mock_path)
    else:
        mocks = _load_numeric_array(mock_path)

    # observed scalar
    if obs_override is not None:
        observed = float(obs_override)
    else:
        obs_path = _find_file(run_dir, slice_tag, "obs")
        if obs_path:
            # allow 1-cell CSV or 1-value text/npy
            if obs_path.lower().endswith(".csv"):
                vals = _first_numeric_col_from_csv(obs_path)
            else:
                vals = _load_numeric_array(obs_path)
            if vals.size != 1:
                raise ValueError(f"Observed file for {slice_tag} should contain a single number: {obs_path}")
            observed = float(vals[0])
        else:
            # conservative fallbacks (your QC values used in the paper)
            defaults = {"z8_10": 0.1011, "z10_20": 0.004014}
            if slice_tag not in defaults:
                raise FileNotFoundError(f"Observed normalized variance not found for {slice_tag}")
            print(f"[{slice_tag}] WARNING: no observed file; using fallback {defaults[slice_tag]:g}.")
            observed = defaults[slice_tag]

    return mocks, observed


# ----------------------------- Plotting ----------------------------------- #

def _pretty_title(slice_tag: str) -> str:
    if slice_tag == "z8_10":
        return r"$z = 8$–$10$ (QC): normalized inter-field variance\nObserved vs. mock ensemble (N=5000)"
    if slice_tag == "z10_20":
        return r"$z = 10$–$20$ (QC): normalized inter-field variance\nObserved vs. mock ensemble (N=5000)"
    return "Normalized inter-field variance"

def _nice_bins(x: np.ndarray, target: int = 50) -> int:
    n = max(10, min(150, int(np.sqrt(x.size) * 2)))
    return max(10, min(120, n if target is None else target))

def _draw_hist_with_inset(ax, x: np.ndarray, x_obs: float, title: str):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mu, sig = float(np.mean(x)), float(np.std(x, ddof=0))

    # main histogram
    bins = _nice_bins(x, target=50)
    ax.hist(x, bins=bins, color="#9ecae1", edgecolor="#5b8db8", alpha=0.9)
    ax.axvline(mu, color="#1f78b4", ls=":", lw=1.6)          # mock mean
    ax.axvline(mu + sig, color="#1f78b4", ls=":", lw=1.0)    # mock μ+1σ
    ax.axvline(x_obs, color="#d62728", ls="--", lw=1.8)      # observed

    ax.set_title(title)
    ax.set_xlabel("Inter-field variance (normalized)")
    ax.set_ylabel("Count of mocks")

    # inset zoom
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width="48%", height="42%", loc="center", borderpad=1.2)
    # choose a narrow window around the core of x
    lo, hi = np.percentile(x, [10, 90])
    ax_in.hist(x[(x >= lo) & (x <= hi)], bins=20, color="#c6dbef", edgecolor="#5b8db8", alpha=1.0)
    ax_in.axvline(mu, color="#1f78b4", ls=":", lw=1.4)
    ax_in.axvline(mu + sig, color="#1f78b4", ls=":", lw=1.0)
    ax_in.set_xticks([])
    ax_in.set_yticks([])


# ----------------------------- Main / CLI --------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description="Histogram panels of normalized inter-field variance (QC).")
    ap.add_argument("--obs-z8-10", type=float, default=None, help="Override observed normalized variance for z=8–10")
    ap.add_argument("--obs-z10-20", type=float, default=None, help="Override observed normalized variance for z=10–20")
    ap.add_argument("--out-dir", type=str, default=None, help="Custom output folder (default: results\\figures\\paper)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    root = _project_root()
    results_root = os.path.join(root, "results")
    run_dir = _latest_step11_run(results_root)

    # Load both slices
    mocks_8_10,  obs_8_10  = load_slice(run_dir, "z8_10",  args.obs_z8_10)
    mocks_10_20, obs_10_20 = load_slice(run_dir, "z10_20", args.obs_z10_20)

    # Style
    plt.rcParams.update({
        "figure.figsize": (5.1, 3.6),
        "figure.dpi": args.dpi,
        "savefig.dpi": args.dpi,
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
    })

    # z = 8–10 panel
    fig1, ax1 = plt.subplots()
    _draw_hist_with_inset(ax1, mocks_8_10, obs_8_10, _pretty_title("z8_10"))

    # z = 10–20 panel
    fig2, ax2 = plt.subplots()
    _draw_hist_with_inset(ax2, mocks_10_20, obs_10_20, _pretty_title("z10_20"))

    # Output (versionless names, in the new repo layout)
    out_dir = args.out_dir or os.path.join(root, "results", "figures", "paper")
    os.makedirs(out_dir, exist_ok=True)

    out_z8  = os.path.join(out_dir, "variance_hist_z8_10_qc_linear_inset")
    out_z10 = os.path.join(out_dir, "variance_hist_z10_20_qc_linear_inset")

    for base, fig in [(out_z8, fig1), (out_z10, fig2)]:
        fig.savefig(base + ".png", bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)

    print("[info] Wrote:")
    print(" ", out_z8  + ".png")
    print(" ", out_z8  + ".pdf")
    print(" ", out_z10 + ".png")
    print(" ", out_z10 + ".pdf")


if __name__ == "__main__":
    main()
