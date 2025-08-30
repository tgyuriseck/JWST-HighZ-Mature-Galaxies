# === step11c_variance_qualitycuts.py ===
# Purpose:
#   Essential robustness check: re-run inter-field variance (raw and depth-normalized)
#   AFTER applying quality cuts (SNR, de-star, photo-z quality when available).
#
# What it does:
#   - Auto-detects usable flux/error columns in common JWST bands (F444W, F356W, F277W, F200W).
#   - Computes SNR in the reddest available band; requires SNR >= threshold (default 10).
#   - Attempts to remove stars using any available of: CLASS_STAR/stellarity/star_flag.
#   - Applies optional photo-z quality if columns exist (e.g., zphot_chi2, odds).
#   - Applies the SAME cuts to baseline slices (z4_6, z6_8) used for depth normalization.
#   - Repeats Step 11b logic with 5000 mocks, timestamped outputs (no overwrites).
#
# Run from PS C:\JWST-Mature-Galaxies\src>:
#   python analysis\step11c_variance_qualitycuts.py
#   # or with options:
#   python analysis\step11c_variance_qualitycuts.py --high-slices z10_20 z8_10 --baseline-slices z4_6 z6_8 --snr-min 10 --destar --photoz-q --n-mocks 5000 --grid 256 --corr-pix 2.0
#
# Outputs (timestamped):
#   results/step11c/<runid>/
#       summary.txt
#       <slice>_qc_config.txt
#       <slice>_field_densities_raw_and_norm_qc.csv
#       <slice>_mock_variance_raw_qc.csv
#       <slice>_mock_variance_norm_qc.csv
#   figures/step11c/<runid>/
#       <slice>_variance_comparison_qc.png
#       <slice>_per_field_bar_raw_vs_norm_qc.png

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cosmology (for comoving distance)
try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception:
    COSMO, u = None, None

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RES_BASE  = os.path.join(PROJ_ROOT, "results", "step11c")
FIG_BASE  = os.path.join(PROJ_ROOT, "figures", "step11c")

# Column aliases
Z_ALTS     = ["zphot","z_phot","z","z_best","photoz","photo_z","z_b"]
FIELD_ALTS = ["field","field_optap","field_photoz","FIELD"]
RA_ALTS    = ["ra","RA","ra_optap","RA_optap","ra_photoz","RA_photoz"]
DEC_ALTS   = ["dec","DEC","dec_optap","DEC_optap","dec_photoz","DEC_photoz"]

# Potential band name roots (reddest first)
BAND_ROOTS = ["F444W","F356W","F277W","F200W","IRAC4","IRAC3"]
# Typical suffix patterns for flux/error
FLUX_SUFFIXES = ["_flux","_FLUX","_apflux","_APFLUX","_FLUX_AUTO","_FLUXAPER"]
ERR_SUFFIXES  = ["_err","_ERR","_aperr","_APERR","_FLUXERR","_ERRAPER"]

# Stellarity/star flags
STELLARITY_ALTS = ["CLASS_STAR","stellarity","stellarity_optap","stellarity_photoz","CLASSSTAR"]
STARFLAG_ALTS   = ["star_flag","STAR_FLAG","is_star","IS_STAR","STAR","star"]

# Photo-z quality
CHI2_ALTS  = ["zphot_chi2","chi2","CHI2","z_chi2"]
ODDS_ALTS  = ["odds","ODDS","z_odds","Z_ODDS"]

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
    df = df.rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})
    return df

def find_flux_err_pair(cols):
    """Return (flux_col, err_col) for the reddest available band, or (None,None)."""
    for root in BAND_ROOTS:
        # try combinations
        flux_candidates = [root + s for s in FLUX_SUFFIXES]
        err_candidates  = [root + s for s in ERR_SUFFIXES]
        fcol = next((c for c in flux_candidates if c in cols), None)
        ecol = next((c for c in err_candidates if c in cols), None)
        if fcol and ecol:
            return fcol, ecol, root
    # fallback: try any *_flux with matching *_err
    flux_guess = [c for c in cols if c.lower().endswith("flux")]
    for fcol in flux_guess:
        stem = fcol[:-4]  # drop 'flux'
        # look for 'err'
        e_candidates = [stem + "err", stem + "ERR", stem + "Err"]
        ecol = next((c for c in e_candidates if c in cols), None)
        if ecol:
            return fcol, ecol, fcol.replace("_flux","")
    return None, None, None

def apply_quality_cuts(df, snr_min=10.0, destar=True, photoz_q=True):
    """Return filtered df and a dict describing which cuts were applied and surviving fraction."""
    info = {"snr_used": None, "snr_min": snr_min, "destar": destar, "photoz_q": photoz_q,
            "flux_col": None, "err_col": None, "band_root": None,
            "stellar_col": None, "starflag_col": None, "chi2_col": None, "odds_col": None,
            "n_in": len(df), "n_out": None}

    # SNR cut
    fcol, ecol, broot = find_flux_err_pair(df.columns)
    if fcol and ecol:
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = pd.to_numeric(df[fcol], errors="coerce") / pd.to_numeric(df[ecol], errors="coerce")
        df = df[snr >= snr_min].copy()
        info.update({"snr_used": True, "flux_col": fcol, "err_col": ecol, "band_root": broot})
    else:
        info["snr_used"] = False  # proceed without SNR cut

    # de-star (if columns exist)
    if destar:
        stellar = pick_col(df, STELLARITY_ALTS)
        starfl  = pick_col(df, STARFLAG_ALTS)
        if stellar is not None:
            # Keep likely galaxies: CLASS_STAR or stellarity <= 0.8
            df = df[pd.to_numeric(df[stellar], errors="coerce") <= 0.8].copy()
            info["stellar_col"] = stellar
        elif starfl is not None:
            # Keep star_flag == 0
            df = df[pd.to_numeric(df[starfl], errors="coerce") == 0].copy()
            info["starflag_col"] = starfl
        else:
            # no star info; nothing applied
            pass

    # photo-z quality (gentle defaults, only if present)
    if photoz_q:
        cchi = pick_col(df, CHI2_ALTS)
        cods = pick_col(df, ODDS_ALTS)
        if cchi is not None:
            df = df[pd.to_numeric(df[cchi], errors="coerce") <= 10.0].copy()
            info["chi2_col"] = cchi
        if cods is not None:
            df = df[pd.to_numeric(df[cods], errors="coerce") >= 0.6].copy()
            info["odds_col"] = cods

    info["n_out"] = len(df)
    return df, info

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

def build_baseline_factors(baseline_tags, grid, snr_min, destar, photoz_q):
    baselines = {}
    counts = {}
    vals = []
    for tag in baseline_tags:
        dfb = load_slice(tag)
        dfb, _ = apply_quality_cuts(dfb, snr_min=snr_min, destar=destar, photoz_q=photoz_q)
        rbar = median_comoving_distance(dfb)
        dens, _, _, fields = per_field_density(dfb, grid, rbar)
        for fld in fields:
            rho = dens[fld][2]
            if fld not in baselines:
                baselines[fld] = 0.0; counts[fld] = 0
            if np.isfinite(rho):
                baselines[fld] += rho; counts[fld] += 1
    for fld in list(baselines.keys()):
        if counts[fld] > 0:
            baselines[fld] /= counts[fld]
            vals.append(baselines[fld])
    fallback = float(np.nanmedian(vals)) if len(vals)>0 else 1.0
    return baselines, fallback

def run_highslice(tag, grid, corr_pix, n_mocks, seed, res_dir, fig_dir, baselines, fallback,
                  snr_min, destar, photoz_q):
    rng = np.random.default_rng(seed)

    # Load & apply QC
    df = load_slice(tag)
    df_qc, qcinfo = apply_quality_cuts(df, snr_min=snr_min, destar=destar, photoz_q=photoz_q)
    rbar = median_comoving_distance(df_qc)

    # Per-field densities after QC
    dens_raw, maps, pix_mpc_list, fields = per_field_density(df_qc, grid, rbar)
    sigma_g = estimate_sigma_g_from_maps(maps)

    # Observed raw & normalized tables
    rows = []
    rho_raw_list, rho_norm_list, areas = [], [], []
    for fld in fields:
        N, A, rho = dens_raw[fld]
        b = baselines.get(fld, fallback)
        rho_norm = rho / b if (np.isfinite(rho) and np.isfinite(b) and b>0) else np.nan
        rows.append({"slice": tag, "field": fld, "N_obs": int(N), "area_mpc2": float(A),
                     "rho_raw": float(rho), "baseline_rho": float(b), "rho_norm": float(rho_norm)})
        rho_raw_list.append(rho); rho_norm_list.append(rho_norm); areas.append(A)

    df_obs = pd.DataFrame(rows)
    obs_csv = os.path.join(res_dir, f"{tag}_field_densities_raw_and_norm_qc.csv")
    df_obs.to_csv(obs_csv, index=False)

    rho_raw  = np.array(rho_raw_list, dtype=float)
    rho_norm = np.array(rho_norm_list, dtype=float)
    var_raw_obs  = np.nanvar(rho_raw,  ddof=1) if np.isfinite(rho_raw).sum() >= 2 else np.nan
    var_norm_obs = np.nanvar(rho_norm, ddof=1) if np.isfinite(rho_norm).sum() >= 2 else np.nan

    # Global mean for mocks (raw case)
    A_tot = float(np.nansum(areas))
    N_tot = float(np.nansum([dens_raw[f][0] for f in fields]))
    rho_bar = N_tot / A_tot if A_tot>0 else np.nan

    # Mocks: raw
    ny = nx = grid
    mock_vars_raw = np.zeros(n_mocks, dtype=float)
    for m in range(n_mocks):
        rho_fields = []
        for fld, pix_mpc, A in zip(fields, pix_mpc_list, areas):
            mu_pix = rho_bar * (pix_mpc**2)
            g = make_correlated_gaussian((ny, nx), corr_pix, rng) * sigma_g
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
    mv_raw_csv = os.path.join(res_dir, f"{tag}_mock_variance_raw_qc.csv")
    pd.DataFrame({"mock_var_raw_qc": mock_vars_raw}).to_csv(mv_raw_csv, index=False)

    # Mocks: normalized (divide each field by its observed baseline factor)
    b_arr = np.array([baselines.get(f, fallback) for f in fields], dtype=float)
    mock_vars_norm = np.zeros(n_mocks, dtype=float)
    for m in range(n_mocks):
        rho_fields = []
        for fld, pix_mpc, A in zip(fields, pix_mpc_list, areas):
            mu_pix = rho_bar * (pix_mpc**2)
            g = make_correlated_gaussian((ny, nx), corr_pix, rng) * sigma_g
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
    mv_norm_csv = os.path.join(res_dir, f"{tag}_mock_variance_norm_qc.csv")
    pd.DataFrame({"mock_var_norm_qc": mock_vars_norm}).to_csv(mv_norm_csv, index=False)

    # Figures
    # 1) variance comparison
    plt.figure(figsize=(8,5))
    bins = 40
    if len(mock_vars_raw)>0:  plt.hist(mock_vars_raw,  bins=bins, alpha=0.35, label="mocks (raw, QC)")
    if len(mock_vars_norm)>0: plt.hist(mock_vars_norm, bins=bins, alpha=0.35, label="mocks (norm, QC)")
    if np.isfinite(var_raw_obs):  plt.axvline(var_raw_obs,  linestyle="--", linewidth=2, label=f"obs raw var (QC)")
    if np.isfinite(var_norm_obs): plt.axvline(var_norm_obs, linestyle="-.", linewidth=2, label=f"obs norm var (QC)")
    plt.xlabel("Inter-field variance")
    plt.ylabel("mocks")
    plt.title(f"{tag}: raw z={z_raw:.2f}, p={p_raw:.3f} | norm z={z_norm:.2f}, p={p_norm:.3f}")
    plt.legend()
    plt.tight_layout()
    comp_png = os.path.join(fig_dir, f"{tag}_variance_comparison_qc.png")
    plt.savefig(comp_png, dpi=150); plt.close()

    # 2) per-field bars
    fields_sorted = fields
    x = np.arange(len(fields_sorted))
    plt.figure(figsize=(9,4))
    width = 0.35
    plt.bar(x - width/2, rho_raw,  width, label="raw (QC)")
    plt.bar(x + width/2, rho_norm, width, label="normalized (QC)")
    plt.xticks(x, fields_sorted, rotation=30, ha="right")
    plt.ylabel("surface density (arb units)")
    plt.title(f"{tag}: per-field densities after QC")
    plt.legend(); plt.tight_layout()
    bars_png = os.path.join(fig_dir, f"{tag}_per_field_bar_raw_vs_norm_qc.png")
    plt.savefig(bars_png, dpi=150); plt.close()

    # QC config file for traceability
    qc_lines = []
    for k,v in qcinfo.items():
        qc_lines.append(f"{k}: {v}")
    with open(os.path.join(res_dir, f"{tag}_qc_config.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(qc_lines) + "\n")

    # Summary
    lines = []
    lines.append(f"[High slice {tag}] (QC) fields={len(fields)}, rbar≈{rbar:.1f} Mpc" if rbar else f"[High slice {tag}] (QC) fields={len(fields)}")
    lines.append(f"  Raw variance (QC):  obs={var_raw_obs:.3e}, mocks: mu={mu_raw:.3e}, sd={sd_raw:.3e}, z={z_raw:.2f}, p={p_raw:.3f}")
    lines.append(f"  Norm variance (QC): obs={var_norm_obs:.3e}, mocks: mu={mu_norm:.3e}, sd={sd_norm:.3e}, z={z_norm:.2f}, p={p_norm:.3f}")
    lines.append(f"  Observed table: {obs_csv}")
    lines.append(f"  Mock raw vars:  {mv_raw_csv}")
    lines.append(f"  Mock norm vars: {mv_norm_csv}")
    lines.append(f"  Figures: {comp_png}; {bars_png}")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Step 11c: Inter-field variance with quality cuts (SNR / de-star / photo-z quality).")
    ap.add_argument("--high-slices", nargs="+", default=["z10_20","z8_10"], help="High-z slices to analyze.")
    ap.add_argument("--baseline-slices", nargs="+", default=["z4_6","z6_8"], help="Baseline slices for depth normalization.")
    ap.add_argument("--snr-min", type=float, default=10.0, help="Minimum SNR in reddest available band (if found).")
    ap.add_argument("--destar", action="store_true", help="Apply de-star cut when columns exist.")
    ap.add_argument("--photoz-q", action="store_true", help="Apply gentle photo-z quality (chi2<=10, odds>=0.6) when present.")
    ap.add_argument("--grid", type=int, default=256, help="Map grid size (NxN).")
    ap.add_argument("--corr-pix", type=float, default=2.0, help="Correlation length in pixels for Gaussian field.")
    ap.add_argument("--n-mocks", type=int, default=5000, help="Number of mock realizations.")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed.")
    args = ap.parse_args()

    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    res_dir, fig_dir = ensure_dirs(runid)

    header = []
    header.append("=== Step 11c: Inter-field Variance with Quality Cuts ===")
    header.append(f"Run: {runid}")
    header.append(f"High slices: {args.high_slices}")
    header.append(f"Baseline slices: {args.baseline_slices}")
    header.append(f"SNR min: {args.snr_min}, de-star: {args.destar}, photoz_q: {args.photoz_q}")
    header.append(f"Grid: {args.grid} x {args.grid}, corr_pix={args.corr_pix}, n_mocks={args.n_mocks}")
    header.append("")
    print("\n".join(header))

    # Build baseline factors with SAME QC
    baselines, fallback = build_baseline_factors(args.baseline_slices, args.grid, args.snr_min, args.destar, args.photoz_q)
    print(f"Baseline factors (QC) computed for {len(baselines)} fields; fallback={fallback:.3e}\n")

    summary = []
    summary.extend(header)
    summary.append(f"Baseline factors (QC) computed for {len(baselines)} fields; fallback={fallback:.3e}")
    summary.append("")

    for tag in args.high_slices:
        lines = run_highslice(tag, args.grid, args.corr_pix, args.n_mocks, args.seed,
                              res_dir, fig_dir, baselines, fallback,
                              args.snr_min, args.destar, args.photoz_q)
        summary.append(lines)

    # Write summary
    os.makedirs(RES_BASE, exist_ok=True)
    os.makedirs(FIG_BASE, exist_ok=True)
    sum_path = os.path.join(res_dir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    print(f"Summary: {sum_path}")
    print(f"Figures: {fig_dir}")

if __name__ == "__main__":
    main()
