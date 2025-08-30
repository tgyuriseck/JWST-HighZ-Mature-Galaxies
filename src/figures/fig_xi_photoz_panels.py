# C:\JWST-Mature-Galaxies\src\figures\fig_xi_photoz_panels_v2.py
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .jwst_plot_style import set_mpl_defaults, fig_ax, save_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT.parent / "results"
OUT_DIR      = PROJECT_ROOT.parent / "results" / "figures_v2" / "pz"

# Use plain text with true en-dash (–) and ≥ so Matplotlib renders correctly.
TIER_TITLES = {
    "z4_6":   "z = 4–6",
    "z6_8":   "z = 6–8",
    "z8_10":  "z = 8–10",
    "z10_20": "z = 10–20",
    "z4p":    "z ≥ 4",
    "z6p":    "z ≥ 6",
    "z8p":    "z ≥ 8",
    "z10p":   "z ≥ 10",
}

def find_latest_xi_mc_dir_for_tier(tier: str) -> Path | None:
    patterns = [
        str(RESULTS_DIR / "**" / "xi_mc" / tier / "xi_mean.csv"),
        str(RESULTS_DIR / "**" / tier / "xi_mc" / "xi_mean.csv"),
        str(RESULTS_DIR / "**" / "xi_mc" / tier / "xi_mean_*.csv"),
    ]
    candidates = []
    for pat in patterns:
        candidates += glob.glob(pat, recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return Path(candidates[0]).parent

def _pick_col(df: pd.DataFrame, regex: str, fallback_idx: int) -> str:
    for c in df.columns:
        if re.search(regex, c, re.I):
            return c
    return df.columns[fallback_idx]

def read_xi_mean_and_sd(tier_dir: Path):
    mean_csvs = sorted(tier_dir.glob("xi_mean*.csv"), key=os.path.getmtime, reverse=True)
    if not mean_csvs:
        raise FileNotFoundError(f"No xi_mean*.csv in {tier_dir}")
    mean_csv = mean_csvs[0]
    dfm = pd.read_csv(mean_csv)

    dcol  = _pick_col(dfm, r"d.*mpc|sep|dist", 0)
    meanc = _pick_col(dfm, r"xi[_\- ]?(mean|mu)$|^xi$", min(1, len(dfm.columns)-1))
    sdcol = None
    for cand in dfm.columns:
        if re.search(r"(xi[_\- ]?(sd|std|sigma)|sem)$", cand, re.I):
            sdcol = cand; break

    d  = pd.to_numeric(dfm[dcol], errors="coerce").to_numpy()
    mu = pd.to_numeric(dfm[meanc], errors="coerce").to_numpy()

    if sdcol is not None:
        sd = pd.to_numeric(dfm[sdcol], errors="coerce").to_numpy()
        note = f"mean±SD from {mean_csv.name}"
        m = np.isfinite(d) & np.isfinite(mu) & np.isfinite(sd)
        return d[m], mu[m], sd[m], note

    reals = sorted(tier_dir.glob("xi_realization*.csv"))
    if not reals:
        sd = np.zeros_like(mu)
        note = f"mean only from {mean_csv.name}"
        m = np.isfinite(d) & np.isfinite(mu)
        return d[m], mu[m], sd[m], note

    stacks, ref_d = [], None
    for rp in reals:
        dfr = pd.read_csv(rp)
        rd = _pick_col(dfr, r"d.*mpc|sep|dist", 0)
        rx = _pick_col(dfr, r"^xi($|[^a-z])", 1)
        if ref_d is None:
            ref_d = pd.to_numeric(dfr[rd], errors="coerce").to_numpy()
        stacks.append(pd.to_numeric(dfr[rx], errors="coerce").to_numpy())
    stack = np.vstack(stacks)
    mu2 = np.nanmean(stack, axis=0)
    sd2 = np.nanstd(stack, axis=0, ddof=1)
    note = f"SD from {len(stacks)} realizations"
    return ref_d, mu2, sd2, note

def add_panel_tag(ax, tag: str):
    ax.text(0.02, 0.96, tag, transform=ax.transAxes, va="top", ha="left",
            fontsize=12, weight="bold")

def plot_panel(ax, d, mu, sd, tier_label):
    ax.axhline(0.0, lw=1.0, color="k", alpha=0.5)
    ax.fill_between(d, mu - sd, mu + sd, alpha=0.25, linewidth=0)
    ax.plot(d, mu, lw=1.8)
    ax.set_xlim(np.nanmin(d), np.nanmax(d))
    ymin = np.nanmin(mu - sd); ymax = np.nanmax(mu + sd)
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel(r"Separation $d$ (Mpc)")
    ax.set_ylabel(r"$\xi(d)$")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_title(tier_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", default=["z6p", "z8_10"])
    parser.add_argument("--outfile", default="xi_photoz_panels_v2")
    args = parser.parse_args()

    set_mpl_defaults()
    found, tier_dirs = [], {}
    for t in args.tiers:
        td = find_latest_xi_mc_dir_for_tier(t)
        if td is not None:
            found.append(t); tier_dirs[t] = td
    if not found:
        raise FileNotFoundError("No xi_mean.csv found under results/**/xi_mc/<tier>/.")
    ncols = len(found)
    fig, axs = fig_ax(1, ncols, width_in=3.4*ncols, height_in=3.4, wspace=0.28)
    axs = [axs] if ncols == 1 else axs
    for i, t in enumerate(found):
        ax = axs[i]
        d, mu, sd, note = read_xi_mean_and_sd(tier_dirs[t])
        plot_panel(ax, d, mu, sd, TIER_TITLES.get(t, t))
        add_panel_tag(ax, f"({chr(ord('a')+i)})")
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, alpha=0.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outbase = OUT_DIR / args.outfile
    written = save_figure(fig, str(outbase))
    plt.close(fig)
    print("Saved:"); [print(" ", p) for p in written]

if __name__ == "__main__":
    main()
