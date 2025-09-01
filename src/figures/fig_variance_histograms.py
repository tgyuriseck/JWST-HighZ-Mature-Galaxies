# C:\JWST-HighZ-Mature-Galaxies\src\figures\fig_variance_histograms.py
"""
Histogram panels (QC, normalized inter-field variance) with inset, for Figure 4.

What this script does
---------------------
- Reproduces the two histogram figures you used in the paper (z = 8–10, z = 10–20),
  matching the legacy look (bars, mean/σ lines, dashed observed line, inset zoom).
- Auto-discovers newest results\step11*\run_* in the new repo layout, OR accept --run-dir.
- Titles use LaTeX style:  z = 8–10 (QC)  and  z = 10–20 (QC).
- Saves versionless outputs under results\figures\paper\ :
    variance_hist_z8_10_qc_linear_inset.(png|pdf)
    variance_hist_z10_20_qc_linear_inset.(png|pdf)

Run from VS Code terminal (cwd = ...\src):
    python figures\fig_variance_histograms.py
    # or point at old results once:
    python figures\fig_variance_histograms.py --run-dir "D:\OldProject\results\step11c\run_2025-08-18_2314"
"""

from __future__ import annotations
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# ----------------------------- Paths -------------------------------------- #

def _root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))          # ...\src\figures
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))  # repo root

def _latest_run(results_root: str) -> str:
    cands = []
    for step in glob.glob(os.path.join(results_root, "step11*")):
        for run in glob.glob(os.path.join(step, "run_*")):
            if os.path.isdir(run):
                cands.append(run)
    if not cands:
        raise FileNotFoundError(f"No run_* under {results_root}\\step11*. Pass --run-dir if using old results.")
    cands.sort(key=os.path.getmtime)
    return cands[-1]


# ----------------------------- IO helpers --------------------------------- #

def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    try:
        return np.loadtxt(path, delimiter=",", dtype=float).ravel()
    except Exception:
        return np.loadtxt(path, dtype=float).ravel()

def _first_numeric_col_csv(path: str) -> np.ndarray:
    try:
        rec = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if getattr(rec, "dtype", None) is not None and rec.dtype.names:
            for name in rec.dtype.names:
                try:
                    col = np.asarray(rec[name], dtype=float).ravel()
                    col = col[np.isfinite(col)]
                    if col.size > 0:
                        return col
                except Exception:
                    pass
    except Exception:
        pass
    return _load_array(path)

def _find_file(run_dir: str, tag: str, kind: str) -> str | None:
    """kind in {'mock','obs'}; tag in {'z8_10','z10_20'}."""
    pats = []
    if kind == "mock":
        pats = [f"**/*{tag}*mock*norm*.csv", f"**/*{tag}*mock*norm*.npy", f"**/*mock*{tag}*norm*.csv", f"**/*{tag}*mock*norm*.txt"]
    else:
        pats = [f"**/*{tag}*obs*norm*.csv", f"**/*{tag}*observed*norm*.csv", f"**/*{tag}*obs*norm*.npy",
                f"**/*{tag}*observed*norm*.npy", f"**/*{tag}*obs*norm*.txt", f"**/*{tag}*observed*norm*.txt"]
    hits = []
    for p in pats:
        hits.extend(glob.glob(os.path.join(run_dir, p), recursive=True))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime)
    return hits[-1]

def load_slice(run_dir: str, tag: str, obs_override: float | None) -> tuple[np.ndarray, float]:
    mock_path = _find_file(run_dir, tag, "mock")
    if not mock_path:
        raise FileNotFoundError(f"No mock normalized-variance file found for {tag} under {run_dir}")
    mocks = _first_numeric_col_csv(mock_path) if mock_path.lower().endswith(".csv") else _load_array(mock_path)

    if obs_override is not None:
        observed = float(obs_override)
    else:
        obs_path = _find_file(run_dir, tag, "obs")
        if obs_path:
            vals = _first_numeric_col_csv(obs_path) if obs_path.lower().endswith(".csv") else _load_array(obs_path)
            if vals.size != 1:
                raise ValueError(f"Observed file for {tag} must contain a single number: {obs_path}")
            observed = float(vals[0])
        else:
            # Conservative fallbacks: the QC values used in your paper
            defaults = {"z8_10": 0.1011, "z10_20": 0.004014}
            observed = defaults[tag]
            print(f"[{tag}] WARNING: no observed file found; using fallback {observed:g}.")
    return mocks, observed


# ----------------------------- Plotting ----------------------------------- #

def _title(tag: str) -> str:
    return r"$z = 8$–$10$ (QC): normalized inter-field variance\nObserved vs. mock ensemble (N=5000)" if tag == "z8_10" \
        else r"$z = 10$–$20$ (QC): normalized inter-field variance\nObserved vs. mock ensemble (N=5000)"

def _nice_bins(x: np.ndarray) -> int:
    return max(10, min(120, int(np.sqrt(x.size) * 2)))

def _draw_hist(ax, x: np.ndarray, x_obs: float, title: str):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    mu, sig = float(np.mean(x)), float(np.std(x, ddof=0))

    bins = _nice_bins(x)
    ax.hist(x, bins=bins, color="#9ecae1", edgecolor="#5b8db8", alpha=0.9)
    ax.axvline(mu,      color="#1f78b4", ls=":",  lw=1.6)
    ax.axvline(mu+sig,  color="#1f78b4", ls=":",  lw=1.0)
    ax.axvline(x_obs,   color="#d62728", ls="--", lw=1.8)

    ax.set_title(title)
    ax.set_xlabel("Inter-field variance (normalized)")
    ax.set_ylabel("Count of mocks")

    # inset zoom
    ax_in = inset_axes(ax, width="48%", height="42%", loc="center", borderpad=1.2)
    lo, hi = np.percentile(x, [10, 90])
    core = x[(x >= lo) & (x <= hi)]
    ax_in.hist(core, bins=20, color="#c6dbef", edgecolor="#5b8db8", alpha=1.0)
    ax_in.axvline(mu,     color="#1f78b4", ls=":", lw=1.4)
    ax_in.axvline(mu+sig, color="#1f78b4", ls=":", lw=1.0)
    ax_in.set_xticks([]); ax_in.set_yticks([])


# ----------------------------- Main --------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description="Histogram panels of QC normalized inter-field variance with inset.")
    ap.add_argument("--run-dir", type=str, default=None, help="Path to results\\step11*\\run_* (default: auto-discover newest).")
    ap.add_argument("--obs-z8-10", type=float, default=None, help="Override z=8–10 observed value (normalized variance).")
    ap.add_argument("--obs-z10-20", type=float, default=None, help="Override z=10–20 observed value (normalized variance).")
    ap.add_argument("--out-dir", type=str, default=None, help="Output folder (default: results\\figures\\paper).")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    root = _root()
    results_root = os.path.join(root, "results")
    run_dir = args.run_dir or _latest_run(results_root)

    mocks_8_10,  obs_8_10  = load_slice(run_dir, "z8_10",  args.obs_z8_10)
    mocks_10_20, obs_10_20 = load_slice(run_dir, "z10_20", args.obs_z10_20)

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

    fig1, ax1 = plt.subplots()
    _draw_hist(ax1, mocks_8_10,  obs_8_10,  _title("z8_10"))
    fig2, ax2 = plt.subplots()
    _draw_hist(ax2, mocks_10_20, obs_10_20, _title("z10_20"))

    out_dir = args.out_dir or os.path.join(root, "results", "figures", "paper")
    os.makedirs(out_dir, exist_ok=True)
    out8  = os.path.join(out_dir, "variance_hist_z8_10_qc_linear_inset")
    out10 = os.path.join(out_dir, "variance_hist_z10_20_qc_linear_inset")
    for base, fig in [(out8, fig1), (out10, fig2)]:
        fig.savefig(base + ".png", bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)

    print("[info] wrote:\n ", out8 + ".png\n ", out8 + ".pdf\n ", out10 + ".png\n ", out10 + ".pdf")


if __name__ == "__main__":
    main()
