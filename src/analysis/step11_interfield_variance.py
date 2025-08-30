# === step11_interfield_variance.py ===
# Purpose:
#   Test whether per-field galaxy number densities vary more than expected
#   from cosmic variance + shot noise, using lognormal mock skies that match
#   the slice geometry and overall mean density.
#
# Inputs (from Step 7 tiers):
#   C:/JWST-Mature-Galaxies/data_processed/tiers/astrodeep_z8_10.csv
#   C:/JWST-Mature-Galaxies/data_processed/tiers/astrodeep_z10_20.csv
#
# Run from PS C:\JWST-Mature-Galaxies\src>:
#   python analysis\step11_interfield_variance.py
#   # or with options:
#   python analysis\step11_interfield_variance.py --slices z10_20 z8_10 --n-mocks 5000 --grid 256 --corr-pix 2.0 --seed 7
#
# Outputs (no overwrites; timestamped run folder):
#   results/step11/<runid>/
#       summary.txt
#       <slice>_observed_field_densities.csv
#       <slice>_mock_variance_samples.csv
#   figures/step11/<runid>/
#       <slice>_variance_hist.png
#
# Method (concise):
#   - For each slice (e.g., z10_20, z8_10):
#       1) For each field: make a small 2D histogram map (RA/Dec) on a grid.
#          Get per-pixel scale in deg and convert to Mpc using slice median χ(z).
#          Field area = (nx*ny) * (pix_mpc^2).
#          Observed density ρ_f = N_f / Area_f.
#       2) Pool all field maps to estimate the pixel Fano factor F = var/mean.
#          Infer a lognormal sigma_g via Poisson-lognormal relation:
#            Var[N_pix] ≈ mean + mean^2 * (exp(sigma_g^2)-1)
#            -> sigma_g^2 = ln(1 + max(0, (var-mean)/mean^2))
#          This sets clustering amplitude for the mocks.
#       3) For each mock (repeat n-mocks times):
#            a) Use a global mean density ρ_bar = (sum N_f) / (sum Area_f).
#            b) For each field: generate a correlated Gaussian field (corr length
#               = corr_pix pixels), rescale to sigma_g, exponentiate to lognormal,
#               set intensity λ_ij = μ_pix * exp(G - 0.5*sigma_g^2) with
#               μ_pix = ρ_bar * (pix_mpc^2), sample Poisson counts, sum to N_f^mock.
#            c) Compute mock densities ρ_f^mock = N_f^mock / Area_f, then variance
#               across fields. Store one variance per mock.
#       4) Compare observed variance vs the mock variance distribution:
#            - z-score: (var_obs - mean(mock)) / std(mock)
#            - p-value: fraction(mock_var >= var_obs)
#
# Notes:
#   - We use the same mapping function (small-angle tangent plane) as steps 09/10.
#   - Because fields have different sizes, we normalize to density (per Mpc^2).
#   - The "global mean" μ comes from the slice's total N / total Area so mocks
#     assume all fields sample the same parent distribution (null hypothesis).

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cosmology
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception:
    COSMO = None
    u = None

# Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RES_BASE  = os.path.join(PROJ_ROOT, "results", "step11")
FIG_BASE  = os.path.join(PROJ_ROOT, "figures", "step11")

# Column detection
Z_ALTS     = ["zphot","z_phot","z","z_best","photoz","photo_z","z_b"]
FIELD_ALTS = ["field","field_optap","field_photoz","FIELD"]
RA_ALTS    = ["ra","RA","ra_optap","RA_optap","ra_photoz","RA_photoz"]
DEC_ALTS   = ["dec","DEC","dec_optap","DEC_optap","dec_photoz","DEC_photoz"]

def pick_col(df, alts):
    for c in alts:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in alts:
        if c.lower() in low: return low[c.lower()]
    return None

# Geometry / mapping
def project_tangent(ra_deg, dec_deg):
    ra0 = np.nanmedian(ra_deg); dec0 = np.nanmedian(dec_deg)
    x = (ra_deg - ra0) * np.cos(np.deg2rad(dec0))
    y = (dec_deg - dec0)
    return x, y

def make_map(ra, dec, nx, ny, pad=0.05):
    x, y = project_tangent(ra, dec)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= pad*dx; xmax += pad*dx
    ymin -= pad*dy; ymax += pad*dy
    H, xe, ye = np.histogram2d(y, x, bins=[ny, nx], range=[[ymin,ymax],[xmin,xmax]])
    pix_deg = np.sqrt(((xmax-xmin)/nx) * ((ymax-ymin)/ny))
    return H.astype(float), pix_deg

def median_comoving_distance(df):
    if COSMO is None: return None
    z = pd.to_numeric(df["zphot"], errors="coerce").dropna()
    if len(z)==0: return None
    return float(COSMO.comoving_distance(np.nanmedian(z)).to(u.Mpc).value)

def gaussian_filter_fft(shape, sigma_pix):
    """Return a Fourier-space Gaussian filter for a given shape and sigma (pixels)."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    # In pixels^-1; convert to angular frequency in pixels
    k2 = (kx**2 + ky**2)
    # Real-space Gaussian with sigma_pix -> Fourier Gaussian exp(-2 pi^2 sigma^2 k^2)
    return np.exp(-2.0 * (np.pi**2) * (sigma_pix**2) * k2)

def make_correlated_gaussian(shape, sigma_pix, rng):
    """Generate a correlated Gaussian field with unit variance, zero mean."""
    ny, nx = shape
    white = rng.normal(0.0, 1.0, size=(ny, nx))
    Fwhite = np.fft.fft2(white)
    Gfilt = Fwhite * gaussian_filter_fft((ny, nx), sigma_pix)
    g = np.fft.ifft2(Gfilt).real
    # Normalize to zero mean, unit variance
    g -= np.mean(g)
    std = np.std(g)
    if std > 0:
        g /= std
    return g

def ensure_dirs(runid):
    res = os.path.join(RES_BASE, runid)
    fig = os.path.join(FIG_BASE, runid)
    os.makedirs(res, exist_ok=False)
    os.makedirs(fig, exist_ok=False)
    return res, fig

def load_slice(tag):
    path = os.path.join(TIERS_DIR, f"astrodeep_{tag}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run Step 7.")
    df = pd.read_csv(path)
    f = pick_col(df, FIELD_ALTS); r = pick_col(df, RA_ALTS); d = pick_col(df, DEC_ALTS); z = pick_col(df, Z_ALTS)
    if f is None or r is None or d is None or z is None:
        raise ValueError(f"Column detection failed for {path}")
    return df[[f,r,d,z]].rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})

def estimate_sigma_g_from_pixels(maps):
    """Estimate lognormal sigma_g from pooled pixel stats across all fields.
       maps: list of 2D arrays (counts per pixel).
       Returns sigma_g >= 0.
    """
    if len(maps)==0:
        return 0.0
    vals = np.concatenate([m.ravel() for m in maps])
    m = np.nanmean(vals)
    v = np.nanvar(vals)
    if not np.isfinite(m) or m<=0:
        return 0.0
    # Fano relation: v ≈ m + m^2 (exp(sigma_g^2)-1)
    excess = v - m
    if excess <= 0:
        return 0.0
    sigma2 = np.log(1.0 + excess / (m*m))
    return float(np.sqrt(max(0.0, sigma2)))

def run_slice(tag, grid, corr_pix, n_mocks, seed, res_dir, fig_dir):
    rng = np.random.default_rng(seed)

    # Load data
    df = load_slice(tag)
    fields = sorted(df["field"].unique().tolist())
    rbar = median_comoving_distance(df)

    # Build per-field maps and areas
    maps = []
    areas = []
    counts = []
    pix_mpc_list = []
    per_field_rows = []

    for fld in fields:
        dff = df[df["field"] == fld].copy()
        ra = pd.to_numeric(dff["ra"], errors="coerce").to_numpy()
        dec= pd.to_numeric(dff["dec"], errors="coerce").to_numpy()
        good = np.isfinite(ra) & np.isfinite(dec)
        ra, dec = ra[good], dec[good]
        H, pix_deg = make_map(ra, dec, grid, grid, pad=0.05)
        maps.append(H)
        # deg -> Mpc (small-angle s = chi * theta)
        if rbar is not None:
            pix_mpc = rbar * (pix_deg * np.pi/180.0)
        else:
            pix_mpc = np.nan
        pix_mpc_list.append(pix_mpc)
        area = (grid * grid) * (pix_mpc**2) if np.isfinite(pix_mpc) else np.nan
        areas.append(area)
        counts.append(float(len(ra)))
        per_field_rows.append({"slice": tag, "field": fld, "N_obs": int(len(ra)), "area_mpc2": float(area)})

    per_field_df = pd.DataFrame(per_field_rows)
    of_csv = os.path.join(res_dir, f"{tag}_observed_field_densities.csv")

    # Observed densities and inter-field variance
    per_field_df["rho_obs"] = per_field_df["N_obs"] / per_field_df["area_mpc2"]
    rho_obs = per_field_df["rho_obs"].to_numpy(dtype=float)
    var_obs = np.nanvar(rho_obs, ddof=1) if len(rho_obs)>=2 else np.nan

    # Global mean density
    A_tot = np.nansum(per_field_df["area_mpc2"].to_numpy(dtype=float))
    N_tot = np.nansum(per_field_df["N_obs"].to_numpy(dtype=float))
    rho_bar = N_tot / A_tot

    # Estimate sigma_g from pixel stats (pooled)
    sigma_g = estimate_sigma_g_from_pixels(maps)

    # Prepare mocks
    mock_vars = np.zeros(n_mocks, dtype=float)
    ny = nx = grid

    for m in range(n_mocks):
        rho_mock_fields = []
        for fld, pix_mpc, area in zip(fields, pix_mpc_list, areas):
            # Mean counts per pixel under null
            mu_pix = rho_bar * (pix_mpc**2)
            # Correlated Gaussian field, unit variance
            g = make_correlated_gaussian((ny, nx), corr_pix, rng)
            # Rescale to sigma_g
            g = g * sigma_g
            # Lognormal intensity with mean mu_pix per pixel
            lam = mu_pix * np.exp(g - 0.5*(sigma_g**2))
            # Poisson sample counts per pixel
            c = rng.poisson(lam).astype(float)
            N_mock = float(np.nansum(c))
            rho_mock = N_mock / area
            rho_mock_fields.append(rho_mock)
        rho_mock_fields = np.array(rho_mock_fields, dtype=float)
        # variance across fields for this mock
        if len(rho_mock_fields)>=2:
            mock_vars[m] = np.nanvar(rho_mock_fields, ddof=1)
        else:
            mock_vars[m] = np.nan

    # Drop non-finite mocks
    mock_vars = mock_vars[np.isfinite(mock_vars)]
    mu_mock = float(np.nanmean(mock_vars)) if len(mock_vars)>0 else np.nan
    sd_mock = float(np.nanstd(mock_vars, ddof=1)) if len(mock_vars)>1 else np.nan

    # z-score and p-value
    if np.isfinite(var_obs) and np.isfinite(mu_mock) and np.isfinite(sd_mock) and sd_mock>0:
        z_score = (var_obs - mu_mock) / sd_mock
    else:
        z_score = np.nan
    if len(mock_vars)>0 and np.isfinite(var_obs):
        p_tail = (np.sum(mock_vars >= var_obs) + 1) / (len(mock_vars) + 1)
    else:
        p_tail = np.nan

    # Save observed table and mock variance samples
    per_field_df.to_csv(of_csv, index=False)
    mv_csv = os.path.join(res_dir, f"{tag}_mock_variance_samples.csv")
    pd.DataFrame({"mock_var": mock_vars}).to_csv(mv_csv, index=False)

    # Plot histogram of mock variances with observed marker
    plt.figure()
    plt.hist(mock_vars, bins=40)
    plt.axvline(var_obs, linestyle="--")
    plt.xlabel("Inter-field variance of density (mock)")
    plt.ylabel("mocks")
    plt.title(f"{tag}: obs var={var_obs:.3e}, mu={mu_mock:.3e}, sd={sd_mock:.3e}, z={z_score:.2f}, p={p_tail:.3f}")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"{tag}_variance_hist.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Build summary lines
    lines = []
    lines.append(f"[Slice {tag}] fields={len(fields)}, rbar≈{rbar:.1f} Mpc" if rbar else f"[Slice {tag}] fields={len(fields)}")
    lines.append(f"  Areas (sum): {A_tot:.3e} Mpc^2, N_tot={int(N_tot)} → rho_bar={rho_bar:.3e} gal/Mpc^2")
    lines.append(f"  Observed inter-field variance: {var_obs:.3e}")
    lines.append(f"  Mocks: n={len(mock_vars)}, mean={mu_mock:.3e}, sd={sd_mock:.3e}")
    lines.append(f"  z-score: {z_score:.2f}, p-tail (>=obs): {p_tail:.3f}")
    lines.append(f"  Observed table: {of_csv}")
    lines.append(f"  Mock variances: {mv_csv}")
    lines.append(f"  Figure: {fig_path}")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Step 11: Inter-field variance vs lognormal mocks.")
    ap.add_argument("--slices", nargs="+", default=["z10_20","z8_10"], help="Tier tags to analyze.")
    ap.add_argument("--grid", type=int, default=256, help="Map grid size per field (NxN).")
    ap.add_argument("--corr-pix", type=float, default=2.0, help="Correlation length in pixels for Gaussian field.")
    ap.add_argument("--n-mocks", type=int, default=5000, help="Number of mock realizations.")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility.")
    args = ap.parse_args()

    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    res_dir, fig_dir = ensure_dirs(runid)

    summary = []
    summary.append("=== Step 11: Inter-field Variance vs Lognormal Mocks ===")
    summary.append(f"Run: {runid}")
    summary.append(f"Slices: {args.slices}")
    summary.append(f"Grid: {args.grid} x {args.grid}")
    summary.append(f"corr_pix: {args.corr_pix}")
    summary.append(f"n_mocks: {args.n_mocks}")
    summary.append("")

    for tag in args.slices:
        lines = run_slice(tag, args.grid, args.corr_pix, args.n_mocks, args.seed, res_dir, fig_dir)
        summary.append(lines)

    # Write summary
    os.makedirs(RES_BASE, exist_ok=True)
    os.makedirs(FIG_BASE, exist_ok=True)
    sum_path = os.path.join(res_dir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nSummary: {sum_path}")
    print(f"Figures: {fig_dir}")

if __name__ == "__main__":
    main()
