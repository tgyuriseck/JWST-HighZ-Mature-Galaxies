# C:\JWST-HighZ-Mature-Galaxies\src\figures\fig_variance_publication.py
"""
Figure 5 (QC normalized-variance significance) — public version

What this does
--------------
- Keeps dashed 3σ and 5σ guide lines (with small labels).
- Uses mathtext x-tick labels with spaces: "$z = 8$–$10$", "$z = 10$–$20$".
- Saves BOTH PNG and PDF to results\\figures\\paper with a **versionless** filename.
- Auto-discovers your latest Step 11 results under results\\step11*\\run_* (works
  for step11, step11b, step11c, etc.), or you can pass --run-dir.
- Also supports passing explicit CSVs via CLI if needed.

Run from VS Code terminal with working dir = ...\\src:
    python figures\\fig_variance_publication.py

Optional flags:
    --run-dir <path to results\\step11*\\run_*>
    --out-dir <custom output dir>  (default: results\\figures\\paper)
    --dpi     <int, default 300>
    --z8-observed <file>   (explicit observed per-field CSV for z8_10)
    --z8-mocks    <file>   (explicit mock normalized-variance CSV for z8_10)
    --z10-observed <file>  (explicit observed per-field CSV for z10_20)
    --z10-mocks    <file>  (explicit mock normalized-variance CSV for z10_20)
    --print-debug          (verbose discovery logging)
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Utilities ---------------------------------- #

def project_root_from_here() -> str:
    """Return project root (two levels up from this file)."""
    here = os.path.abspath(os.path.dirname(__file__))              # ...\src\figures
    root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))  # ...\
    return root

def find_candidate_run_dirs(results_root: str) -> list[str]:
    """
    Search results\\step11*\\run_* folders, newest last.
    Works for step11, step11b, step11c, etc.
    """
    candidates = []
    step11_globs = glob.glob(os.path.join(results_root, "step11*"))
    for step_dir in step11_globs:
        run_glob = glob.glob(os.path.join(step_dir, "run_*"))
        for rd in run_glob:
            if os.path.isdir(rd):
                candidates.append(rd)
    candidates.sort(key=os.path.getmtime)  # oldest -> newest
    return candidates

def latest_run_dir(results_root: str) -> str:
    cands = find_candidate_run_dirs(results_root)
    if not cands:
        raise FileNotFoundError(
            f"No run_* folder found under any '{os.path.join(results_root, 'step11*')}'. "
            "If you are preparing figures from a previous project, pass --run-dir or explicit CSVs."
        )
    return cands[-1]  # newest

def _debug(msg: str, enabled: bool):
    if enabled:
        print(f"[debug] {msg}")

def find_tables_in_run(run_dir: str, tag: str, debug: bool=False) -> tuple[str, str]:
    """
    tag ∈ {'z8_10', 'z10_20'}
    Returns (observed_per_field_csv, mock_norm_var_csv).

    We try a few flexible patterns because filenames vary slightly across runs.
    """
    # Common patterns
    obs_patterns = [
        f"{tag}_field_densities_raw_and_norm_qc.csv",
        f"{tag}_field_densities_norm_qc.csv",
        f"{tag}*field*norm*qc*.csv",
    ]
    mock_patterns = [
        f"{tag}_mock_variance_norm_qc.csv",
        f"{tag}*mock*variance*norm*qc*.csv",
    ]

    def first_match(patterns):
        for pat in patterns:
            hits = glob.glob(os.path.join(run_dir, pat))
            if hits:
                return hits[0]
        return None

    obs = first_match(obs_patterns)
    mvr = first_match(mock_patterns)

    # Fallback: broader search
    if obs is None:
        all_csv = glob.glob(os.path.join(run_dir, "*.csv"))
        obs_hits = [p for p in all_csv if tag in os.path.basename(p)
                    and "field" in os.path.basename(p) and "norm" in os.path.basename(p)]
        if obs_hits:
            obs = sorted(obs_hits)[0]

    if mvr is None:
        all_csv = glob.glob(os.path.join(run_dir, "*.csv"))
        mvr_hits = [p for p in all_csv if tag in os.path.basename(p)
                    and "mock" in os.path.basename(p) and "variance" in os.path.basename(p)]
        if mvr_hits:
            mvr = sorted(mvr_hits)[0]

    _debug(f"run_dir={run_dir} tag={tag} -> obs={obs}", debug)
    _debug(f"run_dir={run_dir} tag={tag} -> mvr={mvr}", debug)

    if obs is None:
        raise FileNotFoundError(f"Observed per-field CSV not found for {tag} in {run_dir}")
    if mvr is None:
        raise FileNotFoundError(f"Mock normalized-variance CSV not found for {tag} in {run_dir}")
    return obs, mvr

def observed_norm_variance(per_field_csv: str) -> float:
    """Variance across fields of normalized per-field density (rho_norm* column)."""
    df = pd.read_csv(per_field_csv)
    cols = [c for c in df.columns if 'rho_norm' in c]
    if not cols:
        raise KeyError(f"No rho_norm* column in {per_field_csv}")
    col = cols[0]
    vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float('nan')
    return float(np.var(vals, ddof=1))

def mock_norm_variances(mock_csv: str) -> np.ndarray:
    """Return mock distribution of normalized variances (first column)."""
    df = pd.read_csv(mock_csv)
    if df.shape[1] < 1:
        return np.array([], dtype=float)
    arr = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
    return arr[np.isfinite(arr)]

def compute_significance(run_dir: str, tag: str, debug: bool=False) -> dict:
    """
    Compute observed variance, mock mean/sigma, Z, and empirical right-tail p.
    Returns dict: {'tag','obs','mu','sigma','Z','p_empirical'}.
    """
    per_field_csv, mock_csv = find_tables_in_run(run_dir, tag, debug=debug)
    obs = observed_norm_variance(per_field_csv)
    mocks = mock_norm_variances(mock_csv)
    mu = float(np.mean(mocks)) if mocks.size else float('nan')
    sigma = float(np.std(mocks, ddof=0)) if mocks.size else float('nan')
    Z = float((obs - mu) / sigma) if np.isfinite(obs) and np.isfinite(mu) and np.isfinite(sigma) and sigma > 0 else float('nan')
    p = float(np.mean(mocks >= obs)) if mocks.size and np.isfinite(obs) else float('nan')
    return {'tag': tag, 'obs': obs, 'mu': mu, 'sigma': sigma, 'Z': Z, 'p_empirical': p}


# ----------------------------- Plotting ----------------------------------- #

def make_figure(z8_10_Z: float, z10_20_Z: float, out_dir: str, dpi: int = 300) -> list[str]:
    """
    Build and save the 2-bar significance plot with 3σ/5σ guides.
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.figsize": (7.0, 4.6),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.linewidth": 0.85,
    })

    fig, ax = plt.subplots()

    xs = np.arange(2)
    heights = [z8_10_Z, z10_20_Z]

    ax.bar(xs, heights, color="#1f77b4", alpha=0.70, edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs, [r"$z = 8$–$10$", r"$z = 10$–$20$"])

    for yval, label in [(3.0, r"$3\sigma$"), (5.0, r"$5\sigma$")]:
        ax.axhline(yval, color="gray", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(0.015, yval + 0.18, label, color="gray", fontsize=9,
                ha="left", va="bottom", transform=ax.get_yaxis_transform())

    ax.set_ylabel(r"z-score ($\sigma$)")
    ax.set_title("QC normalized-variance significance")

    ymax = np.nanmax([*heights, 5.2])
    ax.set_ylim(0, ymax * 1.06)
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # **Versionless** output name
    base = os.path.join(out_dir, "figure5_variance_significance_qc")
    paths = []
    for ext in (".png", ".pdf"):
        fname = base + ext
        fig.savefig(fname, bbox_inches="tight")
        paths.append(fname)

    plt.close(fig)
    return paths


# ----------------------------- CLI / Main --------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Figure 5 (variance significance) — softened public version")
    p.add_argument("--run-dir", type=str, default=None,
                   help="Path to specific results\\step11*\\run_* folder (default: latest discovered).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Custom output directory (default: results\\figures\\paper).")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    p.add_argument("--z8-observed", type=str, default=None, help="Explicit z8_10 observed per-field CSV.")
    p.add_argument("--z8-mocks",    type=str, default=None, help="Explicit z8_10 mock normalized-variance CSV.")
    p.add_argument("--z10-observed", type=str, default=None, help="Explicit z10_20 observed per-field CSV.")
    p.add_argument("--z10-mocks",    type=str, default=None, help="Explicit z10_20 mock normalized-variance CSV.")
    p.add_argument("--print-debug", action="store_true", help="Verbose discovery logging.")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    root = project_root_from_here()
    results_root = os.path.join(root, "results")
    out_dir = args.out_dir or os.path.join(root, "results", "figures", "paper")

    # Determine data sources
    if all([args.z8_observed, args.z8_mocks, args.z10_observed, args.z10_mocks]):
        print("[info] Using explicit CSV files passed via CLI.")
        z8_obs_csv, z8_mocks_csv = args.z8_observed, args.z8_mocks
        z10_obs_csv, z10_mocks_csv = args.z10_observed, args.z10_mocks
    else:
        run_dir = args.run_dir or latest_run_dir(results_root)
        print(f"[info] Using run directory: {run_dir}")
        z8_obs_csv, z8_mocks_csv = find_tables_in_run(run_dir, "z8_10", debug=args.print_debug)
        z10_obs_csv, z10_mocks_csv = find_tables_in_run(run_dir, "z10_20", debug=args.print_debug)

    # Compute Z-scores
    z8_obs = observed_norm_variance(z8_obs_csv)
    z8_mocks = mock_norm_variances(z8_mocks_csv)
    z8_mu = float(np.mean(z8_mocks)) if z8_mocks.size else float('nan')
    z8_sigma = float(np.std(z8_mocks, ddof=0)) if z8_mocks.size else float('nan')
    z8_Z = float((z8_obs - z8_mu) / z8_sigma) if np.isfinite(z8_obs) and np.isfinite(z8_mu) and np.isfinite(z8_sigma) and z8_sigma > 0 else float('nan')
    z8_p = float(np.mean(z8_mocks >= z8_obs)) if z8_mocks.size and np.isfinite(z8_obs) else float('nan')

    z10_obs = observed_norm_variance(z10_obs_csv)
    z10_mocks = mock_norm_variances(z10_mocks_csv)
    z10_mu = float(np.mean(z10_mocks)) if z10_mocks.size else float('nan')
    z10_sigma = float(np.std(z10_mocks, ddof=0)) if z10_mocks.size else float('nan')
    z10_Z = float((z10_obs - z10_mu) / z10_sigma) if np.isfinite(z10_obs) and np.isfinite(z10_mu) and np.isfinite(z10_sigma) and z10_sigma > 0 else float('nan')
    z10_p = float(np.mean(z10_mocks >= z10_obs)) if z10_mocks.size and np.isfinite(z10_obs) else float('nan')

    print("[info] Computed Z-scores (QC normalized-variance):")
    print(f"  z = 8–10  : Z = {z8_Z:.3f}  (empirical p ≈ {z8_p:.4g})")
    print(f"  z = 10–20 : Z = {z10_Z:.3f} (empirical p ≈ {z10_p:.4g})")

    saved = make_figure(z8_Z, z10_Z, out_dir=out_dir, dpi=args.dpi)
    print("[info] Saved softened Figure 5 to:")
    for path in saved:
        print("  -", path)

if __name__ == "__main__":
    sys.exit(main())
