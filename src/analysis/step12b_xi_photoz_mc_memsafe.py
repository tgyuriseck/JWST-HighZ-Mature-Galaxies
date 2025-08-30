# === step12b_xi_photoz_mc_memsafe.py ===
# Step 12b (memory-safe, single-core): Compute xi(d) & wavelength spectrum across
# photo-z resampled realizations using cKDTree COUNT NEIGHBORS (cumulative) to avoid
# materializing pair lists. Same I/O as other 12b scripts.
#
# Run from PS C:\JWST-Mature-Galaxies\src\analysis> :
#   # Baby sanity: 5 realizations, z6p
#   python step12b_xi_photoz_mc_memsafe.py --runid run_20250819_173724 --tiers z6p --max-gal 1500 --rand-mult 8 --n-limit 5
#
#   # Scale (z6p + z8_10)
#   python step12b_xi_photoz_mc_memsafe.py --runid run_20250819_173744 --tiers z6p z8_10 --max-gal 6000 --rand-mult 10
#
# Outputs:
#   results/step12/<runid>/xi_mc/<tier>/ (xi_realizationNNN.csv, power_realizationNNN.csv, xi_mean/std, power_mean/std)
#   figures/step12/<runid>/ (xi_mc_<tier>.png, P_of_k_<tier>.png)

import os, argparse, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq
from scipy.stats import gaussian_kde

# cosmology
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception:
    COSMO = None; u = None

HERE = os.path.abspath(os.path.dirname(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES12_BASE = os.path.join(PROJ_ROOT, "results", "step12")
FIG12_BASE = os.path.join(PROJ_ROOT, "figures", "step12")

Z_ALTS     = ["zphot","z_phot","z","z_best","photoz","photo_z","z_b"]
FIELD_ALTS = ["field","FIELD","field_optap","field_photoz"]
RA_ALTS    = ["ra","RA","ra_optap","RA_optap","ra_photoz"]
DEC_ALTS   = ["dec","DEC","dec_optap","DEC_optap","dec_photoz"]

def pick_col(df, alts):
    for c in alts:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in alts:
        if c.lower() in low: return low[c.lower()]
    return None

def ensure_dirs(runid):
    out_xi = os.path.join(RES12_BASE, runid, "xi_mc")
    out_fig= os.path.join(FIG12_BASE, runid)
    os.makedirs(out_xi, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)
    return out_xi, out_fig

def resampled_dir(runid, tier):
    return os.path.join(RES12_BASE, runid, "resampled", tier)

def list_realization_csvs(runid, tier):
    d = resampled_dir(runid, tier)
    files = sorted(glob.glob(os.path.join(d, f"astrodeep_{tier}_realization*.csv")))
    if not files:
        raise FileNotFoundError(f"No realization CSVs in {d}")
    return files

def read_catalog(path, max_gal=None, rng=None):
    df = pd.read_csv(path)
    f = pick_col(df, FIELD_ALTS); r = pick_col(df, RA_ALTS)
    d = pick_col(df, DEC_ALTS);   z = pick_col(df, Z_ALTS)
    if any(x is None for x in [f,r,d,z]):
        raise ValueError(f"Column detection failed for {path}")
    df = df[[f,r,d,z]].rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})
    if max_gal is not None and len(df) > max_gal:
        if rng is None: rng = np.random.default_rng(7)
        idx = rng.choice(len(df), size=max_gal, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    return df

def comoving_xyz(ra_deg, dec_deg, z):
    if COSMO is None:
        raise RuntimeError("astropy not available. pip install astropy")
    chi = COSMO.comoving_distance(z).to(u.Mpc).value
    ra = np.deg2rad(ra_deg); dec = np.deg2rad(dec_deg)
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc= chi * np.sin(dec)
    return np.column_stack([x,y,zc])

def build_randoms(df, mult=10, rng=None):
    if rng is None: rng = np.random.default_rng(7)
    rows = []
    for fld, dff in df.groupby("field"):
        ra = dff["ra"].to_numpy(); dec = dff["dec"].to_numpy()
        z  = dff["zphot"].to_numpy()
        zf = z[np.isfinite(z)]
        if len(zf) < 2:
            zmed = float(np.nanmedian(z))
            zf = zmed + 0.01*np.random.standard_normal(1000)
        kde = gaussian_kde(zf)
        n_r = mult * len(dff)
        # RA/Dec bbox (simple; consistent with earlier steps)
        ra_min, ra_max = np.nanmin(ra), np.nanmax(ra)
        dec_min,dec_max= np.nanmin(dec),np.nanmax(dec)
        ra_r  = rng.uniform(ra_min, ra_max, size=n_r)
        dec_r = rng.uniform(dec_min,dec_max, size=n_r)
        z_r   = kde.resample(n_r).ravel()
        rows.append(pd.DataFrame({"field":fld, "ra":ra_r, "dec":dec_r, "zphot":z_r}))
    return pd.concat(rows, ignore_index=True)

def binner(min_d=1, max_d=251, step=5):
    edges = np.arange(min_d, max_d+step, step, dtype=float)
    centers = 0.5*(edges[:-1]+edges[1:])
    return edges, centers

# ---------- Memory-safe KD-Tree cumulative counting ----------
def auto_hist_from_cumulative(tree, edges, n_points):
    """Auto-pairs for one set: use cumulative neighbor counts and correct to unique pairs."""
    # cumulative counts at each edge
    cum = np.array([tree.count_neighbors(tree, r) for r in edges], dtype=np.float64)
    # unique cumulative pairs: subtract self-pairs (N) and divide by 2
    cum_unique = (cum - n_points) * 0.5
    # turn cumulative into histogram per bin
    H = np.diff(cum_unique, prepend=0.0)
    # first bin uses cum_unique[0] - 0
    # but prepend=0 means H[0] = cum_unique[0] - 0, H[i] = cum_unique[i] - cum_unique[i-1]
    # We actually need counts between edges[i],edges[i+1], so compute by differencing shifted cum:
    # Let's recompute explicitly to avoid confusion:
    Hbins = np.empty(len(edges)-1, dtype=np.float64)
    for i in range(len(Hbins)):
        Hbins[i] = cum_unique[i+1] - cum_unique[i]
    return Hbins

def cross_hist_from_cumulative(treeA, treeB, edges):
    """Cross-pairs between two sets: cumulative neighbors (already unique)."""
    cum = np.array([treeA.count_neighbors(treeB, r) for r in edges], dtype=np.float64)
    Hbins = np.empty(len(edges)-1, dtype=np.float64)
    for i in range(len(Hbins)):
        Hbins[i] = cum[i+1] - cum[i]
    return Hbins

# ---------- LS estimator & spectrum ----------
def landy_szalay_memsafe(dfD, rand_mult, edges, rng):
    xyzD = comoving_xyz(dfD["ra"].to_numpy(), dfD["dec"].to_numpy(), dfD["zphot"].to_numpy())
    dfR  = build_randoms(dfD, mult=rand_mult, rng=rng)
    xyzR = comoving_xyz(dfR["ra"].to_numpy(), dfR["dec"].to_numpy(), dfR["zphot"].to_numpy())

    tD = cKDTree(xyzD)
    tR = cKDTree(xyzR)
    DD = auto_hist_from_cumulative(tD, edges, n_points=len(xyzD))
    RR = auto_hist_from_cumulative(tR, edges, n_points=len(xyzR))
    DR = cross_hist_from_cumulative(tD, tR, edges)

    nD = len(xyzD); nR = len(xyzR)
    DDn = DD / max(nD*(nD-1)/2.0, 1.0)
    RRn = RR / max(nR*(nR-1)/2.0, 1.0)
    DRn = DR / max(nD*nR,         1.0)
    xi  = (DDn - 2.0*DRn + RRn) / np.maximum(RRn, 1e-12)
    return xi

def tapered_power(dcenters, xi, alpha=0.25, pad_pow2=4):
    x = xi - np.nanmean(xi)
    N = len(x)
    w = windows.tukey(N, alpha=alpha, sym=False)
    xw = x * w
    dd = dcenters[1] - dcenters[0]
    padN = int(2**np.ceil(np.log2(N))) * pad_pow2
    X = rfft(xw, n=padN)
    P = (X * np.conj(X)).real
    k = rfftfreq(padN, d=dd)
    k_phys = 2*np.pi*k
    with np.errstate(divide="ignore"):
        lam = 2*np.pi/np.maximum(k_phys, 1e-12)
    return lam, P

def process_tier(runid, tier, max_gal, rand_mult, edges, taper, pad_pow2, n_limit, seed, out_xi_dir, out_fig_dir):
    rng = np.random.default_rng(seed)
    tier_dir = os.path.join(RES12_BASE, runid, "resampled", tier)
    files = list_realization_csvs(runid, tier)
    if n_limit is not None:
        files = files[:n_limit]
    centers = 0.5*(edges[:-1]+edges[1:])
    tier_out = os.path.join(out_xi_dir, tier)
    os.makedirs(tier_out, exist_ok=True)

    xi_all, Pw_all, lam_ref = [], [], None

    for fpath in files:
        df = read_catalog(fpath, max_gal=max_gal, rng=rng)
        xi = landy_szalay_memsafe(df, rand_mult, edges, rng)
        lam, P = tapered_power(centers, xi, alpha=taper, pad_pow2=pad_pow2)

        tag = os.path.splitext(os.path.basename(fpath))[0].split("_realization")[-1]
        pd.DataFrame({"bin_right_Mpc": centers, "xi": xi}).to_csv(
            os.path.join(tier_out, f"xi_realization{tag}.csv"), index=False)
        pd.DataFrame({"lambda_Mpc": lam, "power": P}).to_csv(
            os.path.join(tier_out, f"power_realization{tag}.csv"), index=False)

        xi_all.append(xi)
        if lam_ref is None: lam_ref = lam
        Pw_all.append(P[:len(lam_ref)])

    X = np.vstack(xi_all)
    xi_mean = np.nanmean(X, axis=0)
    xi_std  = np.nanstd(X, axis=0, ddof=1)
    pd.DataFrame({"bin_right_Mpc": centers, "xi_mean": xi_mean, "xi_std": xi_std}).to_csv(
        os.path.join(tier_out, "xi_mean.csv"), index=False)

    Pstack = np.vstack([p[:len(lam_ref)] for p in Pw_all])
    P_mean = np.nanmean(Pstack, axis=0)
    P_std  = np.nanstd(Pstack, axis=0, ddof=1)
    pd.DataFrame({"lambda_Mpc": lam_ref, "power_mean": P_mean, "power_std": P_std}).to_csv(
        os.path.join(tier_out, "power_mean.csv"), index=False)

    # Plots
    plt.figure()
    plt.plot(centers, xi_mean, label="mean ξ")
    plt.fill_between(centers, xi_mean - xi_std, xi_mean + xi_std, alpha=0.3, label="±1σ (photo‑z)")
    plt.axhline(0, lw=0.8)
    plt.xlabel("separation d [Mpc]"); plt.ylabel("ξ(d)")
    plt.title(f"{tier}: ξ(d) across photo‑z realizations (mem‑safe KD‑Tree)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f"xi_mc_{tier}.png"), dpi=150); plt.close()

    plt.figure()
    plt.plot(lam_ref, P_mean, label="mean spectrum")
    plt.fill_between(lam_ref, P_mean - P_std, P_mean + P_std, alpha=0.3, label="±1σ (photo‑z)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("wavelength λ [Mpc]"); plt.ylabel("Power")
    plt.title(f"{tier}: spectrum of ξ(d) (mem‑safe KD‑Tree)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_fig_dir, f"P_of_k_{tier}.png"), dpi=150); plt.close()

def binner(min_d=1, max_d=251, step=5):
    edges = np.arange(min_d, max_d+step, step, dtype=float)
    centers = 0.5*(edges[:-1]+edges[1:])
    return edges, centers

def main():
    ap = argparse.ArgumentParser(description="Step 12b (memory-safe): ξ(d) & spectrum via KD‑Tree cumulative counts.")
    ap.add_argument("--runid", required=True)
    ap.add_argument("--tiers", nargs="+", required=True)
    ap.add_argument("--max-gal", type=int, default=3000)
    ap.add_argument("--rand-mult", type=int, default=8)
    ap.add_argument("--bins-min", type=float, default=1.0)
    ap.add_argument("--bins-max", type=float, default=251.0)
    ap.add_argument("--bins-d",   type=float, default=5.0)
    ap.add_argument("--taper", type=float, default=0.25)
    ap.add_argument("--pad-pow2", type=int, default=4)
    ap.add_argument("--n-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_xi, out_fig = ensure_dirs(args.runid)
    edges, _ = binner(args.bins_min, args.bins_max, args.bins_d)

    print("=== Step 12b (memory-safe, single-core / KD‑Tree) ===")
    print(f"Run: {args.runid}")
    print(f"Tiers: {args.tiers}")
    print(f"max_gal: {args.max_gal}, rand_mult: {args.rand_mult}")
    if args.n_limit: print(f"n_limit: {args.n_limit}")
    print("")

    for tier in args.tiers:
        process_tier(args.runid, tier, args.max_gal, args.rand_mult, edges,
                     args.taper, args.pad_pow2, args.n_limit, args.seed,
                     out_xi, out_fig)
        print(f"Processed tier {tier} → results/step12/{args.runid}/xi_mc/{tier}")

    print(f"\nFigures: {out_fig}")

if __name__ == "__main__":
    main()
