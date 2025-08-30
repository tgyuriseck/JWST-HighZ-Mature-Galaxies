# === step09_anisotropy_fft.py ===
# Purpose:
#   Per-slice, per-field 2D anisotropy & phase tests using a 2D FFT of galaxy
#   positions on the sky. Nulls via RA-scramble; jackknife across fields.
#
# Inputs (from Step 7):
#   C:/JWST-Mature-Galaxies/data_processed/tiers/astrodeep_z8_10.csv
#   C:/JWST-Mature-Galaxies/data_processed/tiers/astrodeep_z10_20.csv
#
# Run (from PS C:\JWST-Mature-Galaxies\src>):
#   python analysis\step09_anisotropy_fft.py
#   # options:
#   python analysis\step09_anisotropy_fft.py --slices z10_20 z8_10 --grid 256 --nscramble 50 --randmult 10
#
# Outputs (no overwrites; timestamped run folder):
#   results/step09/<runid>/
#       summary.txt
#       per_slice_<slice>_per_field_metrics.csv
#       per_slice_<slice>_jackknife.csv
#   figures/step09/<runid>/
#       <slice>_<field>_map.png
#       <slice>_<field>_fft_power.png
#       <slice>_<field>_anisotropy_polar.png
#       <slice>_<field>_phase_hist.png
#       <slice>_jackknife_anisotropy.png
#
# Notes:
#   - We detect columns robustly: field / RA / Dec.
#   - Angular → transverse Mpc scaling uses mean z of the slice (Planck18).
#   - "Band" of spatial frequencies defaults to scales ~5–40 Mpc transverse.
#   - Anisotropy metric: angular power modulation in the FFT plane in bandpass.
#   - Phase test: Rayleigh test of phase uniformity in bandpass.

import os
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- cosmology for transverse scaling ----
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception as e:
    COSMO = None
    u = None

# ------------- paths -------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RESULTS_BASE = os.path.join(PROJ_ROOT, "results", "step09")
FIG_BASE = os.path.join(PROJ_ROOT, "figures", "step09")

# ------------- helpers -------------
Z_ALTS = ["zphot", "z_phot", "z", "z_best", "photoz", "photo_z", "z_b"]
FIELD_ALTS = ["field", "field_optap", "field_photoz", "FIELD"]
RA_ALTS = ["ra", "RA", "ra_optap", "RA_optap", "ra_photoz", "RA_photoz"]
DEC_ALTS = ["dec", "DEC", "dec_optap", "DEC_optap", "dec_photoz", "DEC_photoz"]

def pick_col(df, alts):
    for c in alts:
        if c in df.columns:
            return c
    # case-insensitive fallback
    low = {c.lower(): c for c in df.columns}
    for c in alts:
        if c.lower() in low:
            return low[c.lower()]
    return None

def mean_comoving_distance(z):
    if COSMO is None:
        return None
    z = np.asarray(z)
    if z.size == 0:
        return None
    r = COSMO.comoving_distance(np.nanmedian(z)).to(u.Mpc).value
    return float(r)

def project_tangent_plane(ra_deg, dec_deg):
    """Small-angle tangent-plane projection about the field center.
       Returns x,y in degrees (not Mpc). We'll scale later."""
    ra0 = np.nanmedian(ra_deg)
    dec0 = np.nanmedian(dec_deg)
    dra = (ra_deg - ra0) * np.cos(np.deg2rad(dec0))
    ddec = dec_deg - dec0
    return dra, ddec, ra0, dec0

def hann2d(ny, nx):
    wy = 0.5 * (1 - np.cos(2*np.pi*np.arange(ny)/(ny-1)))
    wx = 0.5 * (1 - np.cos(2*np.pi*np.arange(nx)/(nx-1)))
    return np.outer(wy, wx)

def rayleigh_pvalue(phases):
    """Rayleigh test for non-uniformity of phases on [0, 2pi)."""
    n = len(phases)
    if n < 5:
        return 1.0
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C*C + S*S)
    z = (R**2) / n
    # Large-n approx for p-value
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    p = float(np.clip(p, 0, 1))
    return p

def band_from_scales_mpc(scales_mpc, pixscale_mpc, nx, ny):
    """Return boolean mask in k-space selecting wavelengths in [smin, smax] Mpc."""
    smin, smax = scales_mpc
    # Spatial frequencies for FFT assuming pixel size = pixscale_mpc
    kx = np.fft.fftfreq(nx, d=pixscale_mpc) * 2*np.pi
    ky = np.fft.fftfreq(ny, d=pixscale_mpc) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    k = np.sqrt(KX**2 + KY**2)
    # wavelength = 2*pi / k
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = 2*np.pi / k
    band = (lam >= smin) & (lam <= smax)
    band[0,0] = False
    return band, KX, KY, k

def angular_map(ra_deg, dec_deg, nx, ny, pad=0.05):
    """Return 2D histogram on a grid tightly bounding the points (+pad margin)."""
    x, y, ra0, dec0 = project_tangent_plane(ra_deg, dec_deg)
    # bounds
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= pad*dx; xmax += pad*dx
    ymin -= pad*dy; ymax += pad*dy
    # histogram
    H, xe, ye = np.histogram2d(y, x, bins=[ny, nx], range=[[ymin, ymax], [xmin, xmax]])
    # pixel scale (deg/pixel) along X (use geometric mean)
    pix_deg = np.sqrt(((xmax-xmin)/nx) * ((ymax-ymin)/ny))
    return H.astype(float), xe, ye, pix_deg, (ra0, dec0)

def map_to_power_phase(H, apodize=True):
    if apodize:
        win = hann2d(H.shape[0], H.shape[1])
        M = H * win
    else:
        M = H
    F = np.fft.fft2(M)
    P = np.abs(F)**2
    Phi = np.angle(F)
    return P, Phi

def anisotropy_metric(P, band, nang=18):
    """Angular modulation in band: compute power vs angle and return contrast."""
    ny, nx = P.shape
    # k-grid based on array indexing
    ky = np.fft.fftfreq(ny) * 2*np.pi
    kx = np.fft.fftfreq(nx) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    ang = (np.arctan2(KY, KX) + 2*np.pi) % (2*np.pi)
    mask = band & np.isfinite(P)
    if mask.sum() < 50:
        return np.nan, None, None
    bins = np.linspace(0, 2*np.pi, nang+1)
    idx = np.digitize(ang[mask], bins) - 1
    power = np.zeros(nang)
    for i in range(nang):
        sel = (idx == i)
        if np.any(sel):
            power[i] = np.nanmean(P[mask][sel])
        else:
            power[i] = np.nan
    m = np.nanmean(power)
    if not np.isfinite(m) or m <= 0:
        return np.nan, None, None
    contrast = (np.nanmax(power) - np.nanmin(power)) / m
    centers = 0.5*(bins[:-1] + bins[1:])
    return float(contrast), centers, power

def zscore_from_null(x, null_vals):
    mu = np.nanmean(null_vals)
    sd = np.nanstd(null_vals, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan, mu, sd
    z = (x - mu) / sd
    return float(z), float(mu), float(sd)

def ensure_dirs(runid):
    res_dir = os.path.join(RESULTS_BASE, runid)
    fig_dir = os.path.join(FIG_BASE, runid)
    os.makedirs(res_dir, exist_ok=False)
    os.makedirs(fig_dir, exist_ok=False)
    return res_dir, fig_dir

def load_slice_csv(tag):
    path = os.path.join(TIERS_DIR, f"astrodeep_{tag}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run Step 7 to create tiers.")
    df = pd.read_csv(path)
    fcol = pick_col(df, FIELD_ALTS)
    rcol = pick_col(df, RA_ALTS)
    dcol = pick_col(df, DEC_ALTS)
    zcol = pick_col(df, Z_ALTS)
    if fcol is None or rcol is None or dcol is None:
        raise ValueError(f"Could not detect field/RA/Dec columns in {path}.")
    return df[[fcol, rcol, dcol, zcol]].rename(columns={fcol:"field", rcol:"ra", dcol:"dec", zcol:"zphot"})

def slice_mean_distance_mpc(df):
    if COSMO is None:
        return None
    z = pd.to_numeric(df["zphot"], errors="coerce").dropna()
    if len(z) == 0:
        return None
    r = COSMO.comoving_distance(np.nanmedian(z)).to(u.Mpc).value
    return float(r)

def plot_map(H, title, outpng):
    plt.figure()
    plt.imshow(H, origin="lower", aspect="equal")
    plt.colorbar(label="counts")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def plot_power(P, title, outpng):
    plt.figure()
    plt.imshow(np.fft.fftshift(np.log10(P+1e-9)), origin="lower", aspect="equal")
    plt.colorbar(label="log10 power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def plot_aniso(angles, power, title, outpng):
    plt.figure()
    ax = plt.subplot(111, projection="polar")
    ax.plot(angles, power)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def plot_phase_hist(phases, title, outpng):
    plt.figure()
    plt.hist(phases, bins=30, range=(0, 2*np.pi))
    plt.xlabel("phase [rad]")
    plt.ylabel("modes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Step 09: 2D FFT anisotropy & phase tests with nulls and jackknife.")
    parser.add_argument("--slices", nargs="+", default=["z10_20", "z8_10"], help="Tier slice tags to analyze.")
    parser.add_argument("--grid", type=int, default=256, help="Grid size (NxN) for the 2D map.")
    parser.add_argument("--nscramble", type=int, default=40, help="Number of RA-scramble nulls per field.")
    parser.add_argument("--randmult", type=int, default=10, help="(Reserved) randoms multiplier for w(theta)-style checks.")
    parser.add_argument("--scale_mpc_min", type=float, default=5.0, help="Bandpass min scale in Mpc.")
    parser.add_argument("--scale_mpc_max", type=float, default=40.0, help="Bandpass max scale in Mpc.")
    args = parser.parse_args()

    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    res_dir, fig_dir = ensure_dirs(runid)

    # master summary text
    summary_lines = []
    summary_lines.append("=== Step 09: Anisotropy & Phase (2D FFT) ===")
    summary_lines.append(f"Run: {runid}")
    summary_lines.append(f"Slices: {args.slices}")
    summary_lines.append(f"Grid: {args.grid} x {args.grid}")
    summary_lines.append(f"Null scrambles per field: {args.nscramble}")
    summary_lines.append(f"Band scales (Mpc): [{args.scale_mpc_min}, {args.scale_mpc_max}]")
    summary_lines.append("")

    for tag in args.slices:
        df = load_slice_csv(tag)
        fields = sorted(df["field"].unique().tolist())

        # mean comoving distance for angular→Mpc conversion
        rbar_mpc = slice_mean_distance_mpc(df)
        # pixel scale in Mpc computed per field after we make the map (deg/pix -> Mpc/pix)
        per_field_rows = []

        summary_lines.append(f"[Slice {tag}] fields: {len(fields)}")
        for fld in fields:
            dff = df[df["field"] == fld].copy()
            ra = pd.to_numeric(dff["ra"], errors="coerce").to_numpy()
            dec = pd.to_numeric(dff["dec"], errors="coerce").to_numpy()
            ra = ra[np.isfinite(ra)]; dec = dec[np.isfinite(dec)]

            # Build map
            H, xe, ye, pix_deg, (ra0, dec0) = angular_map(ra, dec, args.grid, args.grid, pad=0.05)
            plot_map(H, f"{tag}:{fld} map (counts)", os.path.join(fig_dir, f"{tag}_{fld}_map.png"))

            # Convert deg/pixel -> Mpc/pixel using rbar_mpc
            if COSMO is not None and rbar_mpc is not None:
                # small-angle: transverse s = r * theta; 1 deg = pi/180 rad
                pixscale_mpc = rbar_mpc * (pix_deg * np.pi/180.0)
            else:
                pixscale_mpc = np.nan

            # FFT
            P, Phi = map_to_power_phase(H, apodize=True)
            plot_power(P, f"{tag}:{fld} FFT power (log10)", os.path.join(fig_dir, f"{tag}_{fld}_fft_power.png"))

            # Bandpass mask in k-space for given scales [smin, smax] Mpc
            if np.isfinite(pixscale_mpc) and pixscale_mpc > 0:
                band, KX, KY, K = band_from_scales_mpc([args.scale_mpc_min, args.scale_mpc_max], pixscale_mpc, args.grid, args.grid)
            else:
                # Fall back: use middle of array indices as a crude annulus
                band = np.zeros_like(P, dtype=bool)
                ny, nx = P.shape
                cy, cx = ny//2, nx//2
                yy, xx = np.ogrid[:ny, :nx]
                rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
                band[(rr >= 8) & (rr <= 40)] = True

            # Anisotropy metric + phases
            contrast_data, ang_centers, ang_power = anisotropy_metric(P, band, nang=18)
            phases_band = Phi[band]
            phases_band = phases_band[np.isfinite(phases_band)]
            p_phase = rayleigh_pvalue(phases_band)

            if ang_centers is not None and ang_power is not None:
                plot_aniso(ang_centers, ang_power, f"{tag}:{fld} anisotropy (bandpass)", os.path.join(fig_dir, f"{tag}_{fld}_anisotropy_polar.png"))
            if len(phases_band) > 0:
                plot_phase_hist(phases_band, f"{tag}:{fld} phase histogram (bandpass)", os.path.join(fig_dir, f"{tag}_{fld}_phase_hist.png"))

            # Null scrambles: permute RA (keeps Dec)
            null_contrasts = []
            rng = np.random.default_rng(42)
            for j in range(args.nscramble):
                ra_scr = rng.permutation(ra)
                Hn, *_ = angular_map(ra_scr, dec, args.grid, args.grid, pad=0.05)
                Pn, _ = map_to_power_phase(Hn, apodize=True)
                cnull, _, _ = anisotropy_metric(Pn, band, nang=18)
                null_contrasts.append(cnull)
            null_contrasts = np.array(null_contrasts, dtype=float)
            z_contrast, mu_null, sd_null = zscore_from_null(contrast_data, null_contrasts)

            per_field_rows.append({
                "slice": tag, "field": fld, "N": len(ra),
                "pixscale_mpc": pixscale_mpc,
                "anisotropy_contrast": contrast_data,
                "anisotropy_z_vs_null": z_contrast,
                "anisotropy_null_mean": mu_null,
                "anisotropy_null_sd": sd_null,
                "phase_rayleigh_p": p_phase
            })

        # write per-field metrics
        per_field_df = pd.DataFrame(per_field_rows)
        per_field_csv = os.path.join(res_dir, f"per_slice_{tag}_per_field_metrics.csv")
        per_field_df.to_csv(per_field_csv, index=False)

        # Jackknife (leave-one-field-out) on anisotropy z-scores
        # We summarize with the mean z-score across fields and jackknife its std.
        zvals = per_field_df["anisotropy_z_vs_null"].to_numpy(dtype=float)
        valid = np.isfinite(zvals)
        zvals = zvals[valid]
        fields_valid = np.array(fields)[valid]
        jk_vals = []
        for i in range(len(zvals)):
            mask = np.ones(len(zvals), dtype=bool); mask[i] = False
            jk_vals.append(np.nanmean(zvals[mask]))
        jk_vals = np.array(jk_vals, dtype=float)
        jk_mean = np.nanmean(zvals)
        # jackknife std: sqrt( (n-1)/n * sum( (theta_i - theta_bar)^2 ) )
        if len(jk_vals) > 1:
            jk_std = np.sqrt((len(jk_vals)-1)/len(jk_vals) * np.nansum((jk_vals - np.nanmean(jk_vals))**2))
        else:
            jk_std = np.nan

        jk_df = pd.DataFrame({
            "slice":[tag],
            "mean_anisotropy_z":[jk_mean],
            "jackknife_std":[jk_std],
            "n_fields":[len(fields_valid)]
        })
        jk_csv = os.path.join(res_dir, f"per_slice_{tag}_jackknife.csv")
        jk_df.to_csv(jk_csv, index=False)

        summary_lines.append(f"  Slice {tag}: fields={len(fields_valid)}, mean anisotropy z={jk_mean:.2f} ± {jk_std:.2f} (jackknife)")
        summary_lines.append(f"    per-field metrics: {per_field_csv}")
        summary_lines.append(f"    jackknife summary: {jk_csv}")

    # write summary
    os.makedirs(RESULTS_BASE, exist_ok=True)
    os.makedirs(FIG_BASE, exist_ok=True)
    sum_path = os.path.join(res_dir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    print(f"\nSummary: {sum_path}")
    print(f"Figures: {os.path.join(FIG_BASE, runid)}")

if __name__ == "__main__":
    main()
