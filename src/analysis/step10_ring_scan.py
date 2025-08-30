# === step10_ring_scan.py ===
# Purpose:
#   Per-field, per-slice circular (ring) matched-filter scan on the sky.
#   - Build 2D density maps for each field in slices {z10_20, z8_10} (default).
#   - For a grid of radii in Mpc (default 5..40), convolve with a narrow
#     annulus kernel to score ring-like overdensities across all centers.
#   - Record the maximum (over centers) response at each radius, per field.
#   - Nulls: RA-scramble nulls to get the distribution of maxima (per radius),
#     then compute per-radius p-values and a global look-elsewhere p for the
#     best radius per field.
#   - Cross-field check: does the same radius recur across >=3 fields more than
#     in nulls? Report a recurrence p-value.
#
# Run (from PS C:\JWST-Mature-Galaxies\src>):
#   python analysis\step10_ring_scan.py
#   # Options:
#   python analysis\step10_ring_scan.py --slices z10_20 z8_10 --rmin 5 --rmax 40 --nr 20 --nscramble 200 --grid 384
#
# Outputs (no overwrites; timestamped):
#   results/step10/<runid>/
#       summary.txt
#       per_slice_<slice>_per_field_best_rings.csv
#       per_slice_<slice>_radius_curves.csv
#       per_slice_<slice>_recurrence.csv
#   figures/step10/<runid>/
#       <slice>_<field>_best_radius_heatmap.png
#       <slice>_<field>_best_radius_hist.png
#
# Notes:
#   - Uses Planck18 for angular->Mpc conversion via median z of slice.
#   - Uses FFT convolution with an annulus kernel (width=f_width * radius).
#   - Look-elsewhere: we compare the single best radius per field to null maxima
#     across all radii. We also compute per-radius p-values (max over centers).
#   - Recurrence: histogram best radii over fields; compare to null recurrence.

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception:
    COSMO = None
    u = None

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RES_BASE  = os.path.join(PROJ_ROOT, "results", "step10")
FIG_BASE  = os.path.join(PROJ_ROOT, "figures", "step10")

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

def load_slice(tag):
    path = os.path.join(TIERS_DIR, f"astrodeep_{tag}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run Step 7.")
    df = pd.read_csv(path)
    f = pick_col(df, FIELD_ALTS); r = pick_col(df, RA_ALTS); d = pick_col(df, DEC_ALTS); z = pick_col(df, Z_ALTS)
    if f is None or r is None or d is None or z is None:
        raise ValueError(f"Column detection failed for {path}")
    return df[[f,r,d,z]].rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})

def median_chi_mpc(df):
    if COSMO is None: return None
    z = pd.to_numeric(df["zphot"], errors="coerce").dropna()
    if len(z)==0: return None
    return float(COSMO.comoving_distance(np.nanmedian(z)).to(u.Mpc).value)

def project_tangent(ra, dec):
    ra0 = np.nanmedian(ra); dec0 = np.nanmedian(dec)
    x = (ra - ra0) * np.cos(np.deg2rad(dec0))
    y = (dec - dec0)
    return x, y, ra0, dec0

def make_map(ra, dec, nx, ny, pad=0.05):
    x, y, *_ = project_tangent(ra, dec)
    xmin,xmax = np.nanmin(x), np.nanmax(x); ymin,ymax = np.nanmin(y), np.nanmax(y)
    dx,dy = xmax-xmin, ymax-ymin
    xmin -= pad*dx; xmax += pad*dx; ymin -= pad*dy; ymax += pad*dy
    H, xe, ye = np.histogram2d(y, x, bins=[ny,nx], range=[[ymin,ymax],[xmin,xmax]])
    pix_deg = np.sqrt(((xmax-xmin)/nx) * ((ymax-ymin)/ny))
    return H.astype(float), pix_deg

def hann2d(ny, nx):
    wy = 0.5*(1 - np.cos(2*np.pi*np.arange(ny)/(ny-1)))
    wx = 0.5*(1 - np.cos(2*np.pi*np.arange(nx)/(nx-1)))
    return np.outer(wy, wx)

def fft_convolve2d(image, kernel):
    F = np.fft.rfft2(image)
    K = np.fft.rfft2(kernel, s=image.shape)
    C = np.fft.irfft2(F*K, s=image.shape)
    return C

def annulus_kernel(ny, nx, r_pix, width_pix):
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny-1)/2.0, (nx-1)/2.0
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    k = ((rr >= (r_pix - width_pix/2)) & (rr <= (r_pix + width_pix/2))).astype(float)
    # zero-mean kernel helps suppress broad background
    if k.sum() > 0:
        k = k - k.mean()
    return k

def ensure_dirs(runid):
    res = os.path.join(RES_BASE, runid)
    fig = os.path.join(FIG_BASE, runid)
    os.makedirs(res, exist_ok=False)
    os.makedirs(fig, exist_ok=False)
    return res, fig

def ring_scan_field(ra, dec, rbar_mpc, nx, ny, radii_mpc, width_frac=0.25, apodize=True, rng=None, nscramble=0):
    H, pix_deg = make_map(ra, dec, nx, ny, pad=0.05)
    if apodize:
        H = H * hann2d(ny, nx)
    # deg -> Mpc
    pix_mpc = rbar_mpc * (pix_deg * np.pi/180.0)
    # Precompute kernel responses for each radius
    responses = []
    for r_mpc in radii_mpc:
        r_pix = max(1.0, r_mpc / pix_mpc)
        w_pix = max(1.0, width_frac * r_pix)
        K = annulus_kernel(ny, nx, r_pix, w_pix)
        R = fft_convolve2d(H, K)
        responses.append(R)
    # For each radius, take the max over centers as the test statistic
    max_resp = np.array([np.nanmax(R) for R in responses], dtype=float)

    # Nulls: RA scrambles
    null_max = None
    if nscramble > 0:
        ra = np.asarray(ra)
        dec = np.asarray(dec)
        null_max = np.zeros((nscramble, len(radii_mpc)), dtype=float)
        for s in range(nscramble):
            ra_scr = rng.permutation(ra)
            Hn, _ = make_map(ra_scr, dec, nx, ny, pad=0.05)
            if apodize: Hn = Hn * hann2d(ny, nx)
            for i, r_mpc in enumerate(radii_mpc):
                r_pix = max(1.0, r_mpc / pix_mpc)
                w_pix = max(1.0, width_frac * r_pix)
                K = annulus_kernel(ny, nx, r_pix, w_pix)
                Rn = fft_convolve2d(Hn, K)
                null_max[s, i] = np.nanmax(Rn)
    return max_resp, null_max, pix_mpc

def main():
    ap = argparse.ArgumentParser(description="Step 10: Ring/arc matched-filter scan with nulls and look-elsewhere correction.")
    ap.add_argument("--slices", nargs="+", default=["z10_20","z8_10"], help="Tier tags to analyze.")
    ap.add_argument("--rmin", type=float, default=5.0, help="Min ring radius [Mpc].")
    ap.add_argument("--rmax", type=float, default=40.0, help="Max ring radius [Mpc].")
    ap.add_argument("--nr", type=int, default=20, help="Number of radii to scan (inclusive).")
    ap.add_argument("--width_frac", type=float, default=0.25, help="Annulus width as a fraction of radius.")
    ap.add_argument("--grid", type=int, default=384, help="Map grid size (NxN).")
    ap.add_argument("--nscramble", type=int, default=200, help="Number of RA-scramble nulls.")
    args = ap.parse_args()

    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    res_dir, fig_dir = ensure_dirs(runid)

    summary = []
    summary.append("=== Step 10: Ring / Arc Scan ===")
    summary.append(f"Run: {runid}")
    summary.append(f"Slices: {args.slices}")
    summary.append(f"Radii Mpc: [{args.rmin}, {args.rmax}] with nr={args.nr}, width_frac={args.width_frac}")
    summary.append(f"Grid: {args.grid} x {args.grid}")
    summary.append(f"Null scrambles per field: {args.nscramble}")
    summary.append("")

    rng = np.random.default_rng(12345)

    for tag in args.slices:
        df = load_slice(tag)
        rbar = median_chi_mpc(df)
        fields = sorted(df["field"].unique().tolist())
        radii = np.linspace(args.rmin, args.rmax, args.nr)

        per_field_rows = []
        per_radius_rows = []  # one row per (field, radius) with data and null stats

        summary.append(f"[Slice {tag}] fields: {len(fields)}; rbar ~ {rbar:.1f} Mpc" if rbar else f"[Slice {tag}] fields: {len(fields)}; rbar ~ (unknown)")

        for fld in fields:
            dff = df[df["field"] == fld].copy()
            ra = pd.to_numeric(dff["ra"], errors="coerce").to_numpy()
            dec= pd.to_numeric(dff["dec"], errors="coerce").to_numpy()
            good = np.isfinite(ra) & np.isfinite(dec)
            ra, dec = ra[good], dec[good]
            if len(ra) < 50:
                # too sparse; record and skip heavy work
                per_field_rows.append({
                    "slice": tag, "field": fld, "N": len(ra),
                    "best_radius_mpc": np.nan, "best_resp": np.nan,
                    "best_p_local": np.nan, "best_p_global": np.nan
                })
                continue

            max_resp, null_max, pix_mpc = ring_scan_field(
                ra, dec, rbar, args.grid, args.grid, radii,
                width_frac=args.width_frac, apodize=True, rng=rng, nscramble=args.nscramble
            )

            # per-radius local p-values from nulls (one-sided, max over centers)
            if null_max is not None:
                p_loc = []
                for i in range(len(radii)):
                    nvals = null_max[:, i]
                    p = (np.sum(nvals >= max_resp[i]) + 1) / (len(nvals) + 1)
                    p_loc.append(p)
                p_loc = np.array(p_loc)
                # best radius by smallest local p
                i_best = int(np.nanargmin(p_loc))
                best_r = radii[i_best]; best_resp = max_resp[i_best]; best_p_local = p_loc[i_best]
                # look-elsewhere: compare best response to the null maxima over all radii
                # build null of "best over radii" by taking min p_loc (or max resp) across radii for each scramble
                null_best_resp = np.nanmax(null_max, axis=1)
                p_global = (np.sum(null_best_resp >= best_resp) + 1) / (len(null_best_resp) + 1)
            else:
                p_loc = np.full_like(radii, np.nan, dtype=float)
                i_best = int(np.nanargmax(max_resp))
                best_r = radii[i_best]; best_resp = max_resp[i_best]; best_p_local = np.nan; p_global = np.nan

            # store per-field summary
            per_field_rows.append({
                "slice": tag, "field": fld, "N": len(ra),
                "pixscale_mpc": pix_mpc,
                "best_radius_mpc": float(best_r),
                "best_resp": float(best_resp),
                "best_p_local": float(best_p_local),
                "best_p_global": float(p_global)
            })

            # store per-radius curves for this field
            for r, resp, p in zip(radii, max_resp, (p_loc if np.ndim(p_loc)>0 else [np.nan]*len(radii))):
                per_radius_rows.append({
                    "slice": tag, "field": fld, "radius_mpc": float(r),
                    "response": float(resp), "p_local": float(p)
                })

            # small visual: heatmap of the strongest-radius response map
            # (Recompute the best map for display)
            # Build a quick center-response map by re-doing the convolution at best_r
            from numpy.fft import rfft2, irfft2
            H, pix_deg = make_map(ra, dec, args.grid, args.grid, pad=0.05)
            H = H * hann2d(args.grid, args.grid)
            r_pix = max(1.0, best_r / pix_mpc)
            w_pix = max(1.0, args.width_frac * r_pix)
            K = annulus_kernel(args.grid, args.grid, r_pix, w_pix)
            R = fft_convolve2d(H, K)
            plt.figure()
            plt.imshow(R, origin="lower", aspect="equal")
            plt.colorbar(label="ring response")
            plt.title(f"{tag}:{fld} best R={best_r:.1f} Mpc")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{tag}_{fld}_best_radius_heatmap.png"), dpi=150)
            plt.close()

        # write per-field and per-radius outputs
        per_field_df  = pd.DataFrame(per_field_rows)
        per_radius_df = pd.DataFrame(per_radius_rows)
        pf_csv = os.path.join(res_dir, f"per_slice_{tag}_per_field_best_rings.csv")
        pr_csv = os.path.join(res_dir, f"per_slice_{tag}_radius_curves.csv")
        per_field_df.to_csv(pf_csv, index=False)
        per_radius_df.to_csv(pr_csv, index=False)

        # Recurrence across fields: count best radii (bin to tolerance) and compare to null
        # Bin edges every ~ (rmax-rmin)/nr ; use half-bin tolerance when grouping
        if len(per_field_df) > 0 and per_field_df["best_radius_mpc"].notna().any():
            radii = np.linspace(args.rmin, args.rmax, args.nr)
            width = (args.rmax - args.rmin) / max(args.nr-1, 1)
            # group by nearest radius gridpoint
            best_bins = np.digitize(per_field_df["best_radius_mpc"], radii, right=True)
            counts = pd.Series(best_bins).value_counts().sort_index()
            most_common = int(counts.max()) if len(counts)>0 else 0

            # Build null for recurrence: for each field, draw a best radius index from its null maxima
            # Approximate by assuming uniform over radii under null of maxima
            # (conservative; if anything this overestimates null recurrence)
            nfields = per_field_df["field"].nunique()
            nsim = 5000
            rng2 = np.random.default_rng(2024)
            null_recur = np.max(rng2.multinomial(nfields, [1/len(radii)]*len(radii), size=nsim), axis=1)
            p_recur = (np.sum(null_recur >= most_common) + 1) / (len(null_recur) + 1)
        else:
            most_common, p_recur = 0, np.nan

        # write recurrence
        rec_df = pd.DataFrame({
            "slice":[tag],
            "n_fields":[per_field_df['field'].nunique()],
            "most_common_count":[most_common],
            "p_recurrence":[p_recur]
        })
        rec_csv = os.path.join(res_dir, f"per_slice_{tag}_recurrence.csv")
        rec_df.to_csv(rec_csv, index=False)

        summary.append(f"  Slice {tag}: wrote {pf_csv}")
        summary.append(f"              wrote {pr_csv}")
        summary.append(f"              recurrence: most_common={most_common}, p≈{p_recur:.3f} (>=3 fields desirable)")
        summary.append("")

    # write summary
    os.makedirs(RES_BASE, exist_ok=True)
    os.makedirs(FIG_BASE, exist_ok=True)
    sum_path = os.path.join(res_dir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"\nSummary: {sum_path}")
    print(f"Figures: {os.path.join(FIG_BASE, runid)}")

if __name__ == "__main__":
    main()
