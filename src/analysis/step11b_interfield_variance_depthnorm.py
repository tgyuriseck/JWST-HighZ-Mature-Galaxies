# === step11b_interfield_variance_depthnorm.py ===
# Purpose:
#   Inter-field variance test with depth/completeness normalization.
#   - Estimate per-field "baseline" surface density from lower-z slices (e.g., z4_6,z6_8).
#   - For high-z slices (e.g., z8_10,z10_20), compute raw densities and depth-normalized densities:
#         rho_norm(field) = rho_highz(field) / baseline_rho(field)
#   - Build lognormal mocks as in Step 11. For the normalized case, divide each
#     mock field’s high-z density by the *observed* baseline factor of that field
#     (so mocks inherit the same depth/completeness pattern).
#   - Output CSVs and visual comparisons (raw vs normalized variance).
#
# Run from PS C:\JWST-Mature-Galaxies\src>:
#   python analysis\step11b_interfield_variance_depthnorm.py
#   # or with options:
#   python analysis\step11b_interfield_variance_depthnorm.py --high-slices z10_20 z8_10 --baseline-slices z4_6 z6_8 --n-mocks 5000 --grid 256 --corr-pix 2.0 --seed 11
#
# Outputs (timestamped, no overwrite):
#   results/step11b/<runid>/
#       summary.txt
#       <highslice>_field_densities_raw_and_norm.csv
#       <highslice>_mock_variance_raw.csv
#       <highslice>_mock_variance_norm.csv
#   figures/step11b/<runid>/
#       <highslice>_variance_comparison.png   # hist: mocks (raw vs norm) + observed markers
#       <highslice>_per_field_bar_raw_vs_norm.png   # bars: raw vs norm per field
#
# Notes:
#   - Baseline factors are computed from observed data only (z4_6,z6_8 by default).
#   - If a field lacks baseline data, we fall back to the *median baseline across fields* (conservative).

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
    COSMO, u = None, None

# Paths
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RES_BASE  = os.path.join(PROJ_ROOT, "results", "step11b")
FIG_BASE  = os.path.join(PROJ_ROOT, "figures", "step11b")

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

def load_slice(tag):
    path = os.path.join(TIERS_DIR, f"astrodeep_{tag}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run Step 7.")
    df = pd.read_csv(path)
    f = pick_col(df, FIELD_ALTS); r = pick_col(df, RA_ALTS); d = pick_col(df, DEC_ALTS); z = pick_col(df, Z_ALTS)
    if f is None or r is None or d is None or z is None:
        raise ValueError(f"Column detection failed for {path}")
    return df[[f,r,d,z]].rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})

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
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k2 = (kx**2 + ky**2)
    return np.exp(-2.0 * (np.pi**2) * (sigma_pix**2) * k2)

def make_correlated_gaussian(shape, sigma_pix, rng):
    ny, nx = shape
    white = rng.normal(0.0, 1.0, size=(ny, nx))
    F = np.fft.fft2(white)
    F *= gaussian_filter_fft((ny, nx), sigma_pix)
    g = np.fft.ifft2(F).real
    g -= np.mean(g)
    s = np.std(g)
    if s > 0: g /= s
    return g

def ensure_dirs(runid):
    res = os.path.join(RES_BASE, runid)
    fig = os.path.join(FIG_BASE, runid)
    os.makedirs(res, exist_ok=False)
    os.makedirs(fig, exist_ok=False)
    return res, fig

def estimate_sigma_g_from_maps(maps):
    if len(maps)==0: return 0.0
    vals = np.concatenate([m.ravel() for m in maps])
    m = np.nanmean(vals); v = np.nanvar(vals)
    if not np.isfinite(m) or m<=0: return 0.0
    excess = v - m
    if excess <= 0: return 0.0
    sigma2 = np.log(1.0 + excess/(m*m))
    return float(np.sqrt(max(0.0, sigma2)))

def per_field_density(df, grid, rbar_mpc):
    """Return dict field -> (N_obs, area_mpc2, rho_obs), plus map list and pix_mpc list."""
    fields = sorted(df["field"].unique().tolist())
    out = {}
    maps = []
    pix_mpc_list = []
    for fld in fields:
        dff = df[df["field"] == fld].copy()
        ra = pd.to_numeric(dff["ra"], errors="coerce").to_numpy()
        dec= pd.to_numeric(dff["dec"], errors="coerce").to_numpy()
        good = np.isfinite(ra) & np.isfinite(dec)
        ra, dec = ra[good], dec[good]
        H, pix_deg = make_map(ra, dec, grid, grid, pad=0.05)
        maps.append(H.astype(float))
        if rbar_mpc is not None:
            pix_mpc = rbar_mpc * (pix_deg * np.pi/180.0)
        else:
            pix_mpc = np.nan
        pix_mpc_list.append(pix_mpc)
        area = (grid*grid) * (pix_mpc**2) if np.isfinite(pix_mpc) else np.nan
        N = float(len(ra))
        rho = N / area if np.isfinite(area) and area>0 else np.nan
        out[fld] = (N, area, rho)
    return out, maps, pix_mpc_list, fields

def build_baseline_factors(baseline_tags, grid):
    """Compute baseline rho per field as mean over provided baseline slices."""
    per_field_rhos = {}
    counts = {}
    for tag in baseline_tags:
        dfb = load_slice(tag)
        rbar = median_comoving_distance(dfb)
        dens, _, _, fields = per_field_density(dfb, grid, rbar)
        for fld in fields:
            rho = dens[fld][2]
            if fld not in per_field_rhos:
                per_field_rhos[fld] = 0.0
                counts[fld] = 0
            if np.isfinite(rho):
                per_field_rhos[fld] += rho
                counts[fld] += 1
    baselines = {}
    vals = []
    for fld, s in per_field_rhos.items():
        if counts.get(fld, 0) > 0:
            baselines[fld] = s / counts[fld]
            vals.append(baselines[fld])
    # fallback for missing fields: median of available baselines
    fallback = float(np.nanmedian(vals)) if len(vals)>0 else 1.0
    return baselines, fallback

def run_highslice(tag, grid, corr_pix, n_mocks, seed, res_dir, fig_dir, baselines, fallback):
    rng = np.random.default_rng(seed)

    # Load high-z slice & compute per-field raw densities
    df = load_slice(tag)
    rbar = median_comoving_distance(df)
    dens_raw, maps, pix_mpc_list, fields = per_field_density(df, grid, rbar)

    # Estimate clustering amplitude sigma_g from pooled pixel stats
    sigma_g = estimate_sigma_g_from_maps(maps)

    # Build tables: raw and normalized densities
    rows = []
    rho_raw_list = []
    rho_norm_list = []
    baseline_used = []
    areas = []
    for i, fld in enumerate(fields):
        N, A, rho = dens_raw[fld]
        b = baselines.get(fld, fallback)
        rho_norm = rho / b if (np.isfinite(rho) and np.isfinite(b) and b>0) else np.nan
        rows.append({"slice": tag, "field": fld, "N_obs": int(N), "area_mpc2": float(A),
                     "rho_raw": float(rho), "baseline_rho": float(b), "rho_norm": float(rho_norm)})
        rho_raw_list.append(rho)
        rho_norm_list.append(rho_norm)
        baseline_used.append(b)
        areas.append(A)

    df_obs = pd.DataFrame(rows)
    obs_csv = os.path.join(res_dir, f"{tag}_field_densities_raw_and_norm.csv")
    df_obs.to_csv(obs_csv, index=False)

    # Observed inter-field variance (raw and normalized)
    rho_raw = np.array(rho_raw_list, dtype=float)
    rho_norm = np.array(rho_norm_list, dtype=float)
    var_raw_obs  = np.nanvar(rho_raw, ddof=1) if np.isfinite(rho_raw).sum()>=2 else np.nan
    var_norm_obs = np.nanvar(rho_norm, ddof=1) if np.isfinite(rho_norm).sum()>=2 else np.nan

    # Global mean density for mocks (raw case)
    A_tot = float(np.nansum(areas))
    N_tot = float(np.nansum([dens_raw[fld][0] for fld in fields]))
    rho_bar = N_tot / A_tot if A_tot>0 else np.nan

    # Mocks: raw variance (as in Step 11)
    mock_vars_raw = np.zeros(n_mocks, dtype=float)
    ny = nx = grid
    for m in range(n_mocks):
        rho_fields = []
        for fld, pix_mpc, A in zip(fields, pix_mpc_list, areas):
            mu_pix = rho_bar * (pix_mpc**2)
            g = make_correlated_gaussian((ny, nx), corr_pix, rng)
            g = g * sigma_g
            lam = mu_pix * np.exp(g - 0.5*(sigma_g**2))
            c = rng.poisson(lam).astype(float)
            N_mock = float(np.nansum(c))
            rho_fields.append(N_mock / A)
        rho_fields = np.array(rho_fields, dtype=float)
        mock_vars_raw[m] = np.nanvar(rho_fields, ddof=1)
    mock_vars_raw = mock_vars_raw[np.isfinite(mock_vars_raw)]
    mu_raw = float(np.nanmean(mock_vars_raw)) if len(mock_vars_raw)>0 else np.nan
    sd_raw = float(np.nanstd(mock_vars_raw, ddof=1)) if len(mock_vars_raw)>1 else np.nan
    z_raw = (var_raw_obs - mu_raw)/sd_raw if (np.isfinite(var_raw_obs) and np.isfinite(mu_raw) and np.isfinite(sd_raw) and sd_raw>0) else np.nan
    p_raw = (np.sum(mock_vars_raw >= var_raw_obs)+1)/(len(mock_vars_raw)+1) if (len(mock_vars_raw)>0 and np.isfinite(var_raw_obs)) else np.nan

    # Mocks: normalized variance (divide each mock field by its observed baseline factor)
    b_arr = np.array([baselines.get(fld, fallback) for fld in fields], dtype=float)
    mock_vars_norm = np.zeros(n_mocks, dtype=float)
    for m in range(n_mocks):
        rho_fields = []
        for fld, pix_mpc, A in zip(fields, pix_mpc_list, areas):
            mu_pix = rho_bar * (pix_mpc**2)
            g = make_correlated_gaussian((ny, nx), corr_pix, rng)
            g = g * sigma_g
            lam = mu_pix * np.exp(g - 0.5*(sigma_g**2))
            c = rng.poisson(lam).astype(float)
            N_mock = float(np.nansum(c))
            rho_fields.append(N_mock / A)
        rho_fields = np.array(rho_fields, dtype=float)
        rho_fields_norm = rho_fields / b_arr
        mock_vars_norm[m] = np.nanvar(rho_fields_norm, ddof=1)
    mock_vars_norm = mock_vars_norm[np.isfinite(mock_vars_norm)]
    mu_norm = float(np.nanmean(mock_vars_norm)) if len(mock_vars_norm)>0 else np.nan
    sd_norm = float(np.nanstd(mock_vars_norm, ddof=1)) if len(mock_vars_norm)>1 else np.nan
    z_norm = (var_norm_obs - mu_norm)/sd_norm if (np.isfinite(var_norm_obs) and np.isfinite(mu_norm) and np.isfinite(sd_norm) and sd_norm>0) else np.nan
    p_norm = (np.sum(mock_vars_norm >= var_norm_obs)+1)/(len(mock_vars_norm)+1) if (len(mock_vars_norm)>0 and np.isfinite(var_norm_obs)) else np.nan

    # Save mock variance samples
    mv_raw_csv  = os.path.join(res_dir, f"{tag}_mock_variance_raw.csv")
    mv_norm_csv = os.path.join(res_dir, f"{tag}_mock_variance_norm.csv")
    pd.DataFrame({"mock_var_raw":  mock_vars_raw}).to_csv(mv_raw_csv,  index=False)
    pd.DataFrame({"mock_var_norm": mock_vars_norm}).to_csv(mv_norm_csv, index=False)

    # Figures: variance hist comparison
    plt.figure(figsize=(8,5))
    bins = 40
    if len(mock_vars_raw)>0:
        plt.hist(mock_vars_raw, bins=bins, alpha=0.35, label="mocks (raw)")
    if len(mock_vars_norm)>0:
        plt.hist(mock_vars_norm, bins=bins, alpha=0.35, label="mocks (normalized)")
    if np.isfinite(var_raw_obs):
        plt.axvline(var_raw_obs, linestyle="--", linewidth=2, label=f"observed raw var")
    if np.isfinite(var_norm_obs):
        plt.axvline(var_norm_obs, linestyle="-.", linewidth=2, label=f"observed norm var")
    plt.xlabel("Inter-field variance of surface density")
    plt.ylabel("mocks")
    plt.title(f"{tag}: raw z={z_raw:.2f}, p={p_raw:.3f} | norm z={z_norm:.2f}, p={p_norm:.3f}")
    plt.legend()
    plt.tight_layout()
    comp_png = os.path.join(fig_dir, f"{tag}_variance_comparison.png")
    plt.savefig(comp_png, dpi=150)
    plt.close()

    # Figures: per-field bar chart (raw vs normalized)
    plt.figure(figsize=(9,4))
    x = np.arange(len(fields))
    width = 0.35
    plt.bar(x - width/2, rho_raw, width, label="raw")
    plt.bar(x + width/2, rho_norm, width, label="normalized")
    plt.xticks(x, fields, rotation=30, ha="right")
    plt.ylabel("surface density (arb units)")
    plt.title(f"{tag}: per-field densities (raw vs normalized)")
    plt.legend()
    plt.tight_layout()
    bars_png = os.path.join(fig_dir, f"{tag}_per_field_bar_raw_vs_norm.png")
    plt.savefig(bars_png, dpi=150)
    plt.close()

    # Summary lines
    lines = []
    lines.append(f"[High slice {tag}] fields={len(fields)}, rbar≈{rbar:.1f} Mpc" if rbar else f"[High slice {tag}] fields={len(fields)}")
    lines.append(f"  Raw variance:  obs={var_raw_obs:.3e}, mocks: mu={mu_raw:.3e}, sd={sd_raw:.3e}, z={z_raw:.2f}, p={p_raw:.3f}")
    lines.append(f"  Norm variance: obs={var_norm_obs:.3e}, mocks: mu={mu_norm:.3e}, sd={sd_norm:.3e}, z={z_norm:.2f}, p={p_norm:.3f}")
    lines.append(f"  Observed table: {obs_csv}")
    lines.append(f"  Mock raw vars:  {mv_raw_csv}")
    lines.append(f"  Mock norm vars: {mv_norm_csv}")
    lines.append(f"  Figures: {comp_png}; {bars_png}")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Step 11b: Inter-field variance with depth normalization and visuals.")
    ap.add_argument("--high-slices", nargs="+", default=["z10_20","z8_10"], help="High-z slices to analyze.")
    ap.add_argument("--baseline-slices", nargs="+", default=["z4_6","z6_8"], help="Baseline slices used to estimate per-field depth factors.")
    ap.add_argument("--grid", type=int, default=256, help="Map grid size per field (NxN).")
    ap.add_argument("--corr-pix", type=float, default=2.0, help="Correlation length in pixels for Gaussian field.")
    ap.add_argument("--n-mocks", type=int, default=5000, help="Number of mock realizations.")
    ap.add_argument("--seed", type=int, default=11, help="RNG seed.")
    args = ap.parse_args()

    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    res_dir, fig_dir = ensure_dirs(runid)

    summary = []
    summary.append("=== Step 11b: Inter-field Variance with Depth Normalization ===")
    summary.append(f"Run: {runid}")
    summary.append(f"High slices: {args.high_slices}")
    summary.append(f"Baseline slices: {args.baseline_slices}")
    summary.append(f"Grid: {args.grid} x {args.grid}, corr_pix={args.corr_pix}, n_mocks={args.n_mocks}")
    summary.append("")

    # Build baseline factors
    baselines, fallback = build_baseline_factors(args.baseline_slices, args.grid)
    summary.append(f"Baseline factors computed for {len(baselines)} fields; fallback={fallback:.3e}")
    summary.append("")

    # Run each high-z slice
    for tag in args.high_slices:
        lines = run_highslice(tag, args.grid, args.corr_pix, args.n_mocks, args.seed, res_dir, fig_dir, baselines, fallback)
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
