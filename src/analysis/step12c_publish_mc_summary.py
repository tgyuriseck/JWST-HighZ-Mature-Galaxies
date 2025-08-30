# === step12c_publish_mc_summary.py ===
# Make paper-ready summary plots/tables from Step 12b outputs (no recompute).
#
# Inputs per tier (already produced by 12b):
#   results/step12/<runid>/xi_mc/<tier>/xi_mean.csv   (bin_right_Mpc, xi_mean, xi_std)
#   results/step12/<runid>/xi_mc/<tier>/power_mean.csv (lambda_Mpc, power_mean, power_std) [not used here]
#
# Outputs:
#   figures/step12/<runid>/xi_mc_panels.png
#   figures/step12/<runid>/xi_mc_points.png            (ξ at selected d with error bars)
#   results/step12/<runid>/xi_mc_summary.csv           (table of ξ_mean, ξ_std at selected d)
#
# Run from PS C:\JWST-Mature-Galaxies\src\analysis> :
#   python step12c_publish_mc_summary.py --runid run_20250819_173744 --tiers z6p z8_10
#   python step12c_publish_mc_summary.py --runid run_20250819_174526 --tiers z10_20
#
# Optional overlay of baseline (point-estimate) xi curve:
#   --baseline-xi <TIER>=<PATH>   (repeatable)
#   Example:
#   --baseline-xi z6p=C:\JWST-Mature-Galaxies\results\step9c\...\xi_z6p.csv
#
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES12_BASE = os.path.join(PROJ_ROOT, "results", "step12")
FIG12_BASE = os.path.join(PROJ_ROOT, "figures", "step12")

def load_xi(runid, tier):
    path = os.path.join(RES12_BASE, runid, "xi_mc", tier, "xi_mean.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Did Step 12b finish for {tier}?")
    df = pd.read_csv(path)
    # column names are exactly as written by 12b
    return df["bin_right_Mpc"].to_numpy(), df["xi_mean"].to_numpy(), df["xi_std"].to_numpy()

def load_baseline(path):
    # Accepts a CSV with columns [bin_right_Mpc, xi] or any two columns where the 1st is d and 2nd is xi
    df = pd.read_csv(path)
    d = df.iloc[:,0].to_numpy(dtype=float)
    xi = df.iloc[:,1].to_numpy(dtype=float)
    return d, xi

def nearest(your_d, target):
    idx = np.argmin(np.abs(your_d - target))
    return idx, your_d[idx]

def main():
    ap = argparse.ArgumentParser(description="Step 12c: publishable panels + summary table from MC outputs.")
    ap.add_argument("--runid", required=True)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--points", nargs="+", type=float, default=[40, 60, 100, 150, 200],
                    help="d [Mpc] where to report xi and error bars")
    ap.add_argument("--baseline-xi", action="append", default=[],
                    help="Optional overlays of baseline xi: <tier>=<path>. Repeatable.")
    args = ap.parse_args()

    # parse baseline map
    base_map = {}
    for item in args.baseline_xi:
        if "=" in item:
            t, p = item.split("=", 1)
            base_map[t.strip()] = p.strip()

    # ensure dirs
    out_fig_dir = os.path.join(FIG12_BASE, args.runid)
    os.makedirs(out_fig_dir, exist_ok=True)
    out_res_dir = os.path.join(RES12_BASE, args.runid)
    os.makedirs(out_res_dir, exist_ok=True)

    # 1) 3-panel ξ(d) with ±1σ
    nT = len(args.tiers)
    cols = 1
    rows = nT
    fig, axes = plt.subplots(rows, cols, figsize=(7, 3.2*rows), sharex=True)
    if nT == 1:
        axes = [axes]

    summary_rows = []
    for ax, tier in zip(axes, args.tiers):
        d, xi_mean, xi_std = load_xi(args.runid, tier)
        ax.plot(d, xi_mean, label="mean ξ", lw=2)
        ax.fill_between(d, xi_mean - xi_std, xi_mean + xi_std, alpha=0.3, label="±1σ (photo‑z)")
        if tier in base_map:
            bd, bxi = load_baseline(base_map[tier])
            ax.plot(bd, bxi, lw=1.5, alpha=0.8, label="baseline ξ (point z)")

        ax.axhline(0, lw=0.8, color="k", alpha=0.5)
        ax.set_ylabel("ξ(d)")
        ax.set_title(f"{tier}: ξ(d) across photo‑z realizations")
        ax.legend(loc="upper right", fontsize=9)

        # collect table values at requested points
        for p in args.points:
            idx, d_snap = nearest(d, p)
            summary_rows.append({"tier": tier, "d_Mpc": float(d_snap),
                                 "xi_mean": float(xi_mean[idx]),
                                 "xi_std": float(xi_std[idx])})

    axes[-1].set_xlabel("separation d [Mpc]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig_dir, "xi_mc_panels.png"), dpi=150)
    plt.close(fig)

    # 2) compact point chart (per scale, per tier)
    df_tab = pd.DataFrame(summary_rows)
    df_tab.to_csv(os.path.join(out_res_dir, "xi_mc_summary.csv"), index=False)

    # One plot per requested scale
    for p in args.points:
        sub = df_tab[np.isclose(df_tab["d_Mpc"], df_tab["d_Mpc"].unique()[np.argmin(np.abs(df_tab["d_Mpc"].unique()-p))])]
        tiers = sub["tier"].tolist()
        means = sub["xi_mean"].to_numpy()
        errs  = sub["xi_std"].to_numpy()
        plt.figure(figsize=(6,3.5))
        x = np.arange(len(tiers))
        plt.errorbar(x, means, yerr=errs, fmt="o", capsize=3)
        plt.xticks(x, tiers)
        plt.axhline(0, lw=0.8, color="k", alpha=0.5)
        plt.ylabel("ξ(d)")
        plt.title(f"ξ(d) at ~{p} Mpc (mean ± 1σ, photo‑z MC)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_fig_dir, f"xi_mc_points_{int(p)}Mpc.png"), dpi=150)
        plt.close()

    print(f"Wrote: {os.path.join(out_fig_dir, 'xi_mc_panels.png')}")
    print(f"Wrote: {os.path.join(out_res_dir, 'xi_mc_summary.csv')}")
    print(f"Also wrote: xi_mc_points_<scale>Mpc.png per scale")

if __name__ == "__main__":
    main()
