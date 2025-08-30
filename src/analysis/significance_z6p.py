# -*- coding: utf-8 -*-
"""
Step 11: Statistical significance & null tests for the ~183 Mpc feature in z6p.

v1.3 (robust outputs + zero-padding):
- Creates output folders only when writing outputs (not at startup).
- Retries timestamped folder name with alphabetical suffix (_a, _b, ...) if a collision occurs.
- Keeps zero-padding (--zp-factor, default 8), auto-clamp, and FFT-bin diagnostics.

Typical run:
  python analysis\significance_z6p.py --n-sims 5000 --null-mode phase --d-col bin_right_Mpc --xi-col xi --zp-factor 8 --lambda-min 150 --lambda-max 200
"""

from __future__ import annotations
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Utility ----------
def project_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]  # <root>


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_unique_outdirs(root: Path, stage: str, ts: str) -> Tuple[Path, Path]:
    """
    Create results/figures/<stage>/<ts> with collision-safe suffix.
    Returns (out_results, out_figs).
    """
    base_res = root / "results" / stage / ts
    base_fig = root / "figures" / stage / ts
    out_res, out_fig = base_res, base_fig
    if out_res.exists() or out_fig.exists():
        # try suffixes _a, _b, ..., _z, then numeric
        suffixes = [f"_{chr(c)}" for c in range(ord('a'), ord('z') + 1)] + [f"_{i}" for i in range(1, 100)]
        for s in suffixes:
            cand_res = Path(str(base_res) + s)
            cand_fig = Path(str(base_fig) + s)
            if not cand_res.exists() and not cand_fig.exists():
                out_res, out_fig = cand_res, cand_fig
                break
    out_res.mkdir(parents=True, exist_ok=False)
    out_fig.mkdir(parents=True, exist_ok=False)
    return out_res, out_fig


# ---------- Data Loading ----------
def detect_columns(df: pd.DataFrame, d_col: Optional[str], xi_col: Optional[str]) -> Tuple[str, str]:
    if d_col is not None and xi_col is not None:
        if d_col not in df.columns or xi_col not in df.columns:
            raise ValueError(f"Provided columns not found. Available: {list(df.columns)}")
        return d_col, xi_col

    candidates_d, candidates_xi = [], []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["bin_right_mpc", "d_mpc", "r_mpc", "dist_mpc", "separation_mpc", "sep_mpc",
                                 "separation", "distance", "r (mpc)", "d (mpc)", "r", "d"]):
            candidates_d.append(c)
        if lc == "xi" or "xi(" in lc or "xi_" in lc or "correlation" in lc or lc == "corr":
            candidates_xi.append(c)
    if not candidates_d or not candidates_xi:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not candidates_d and numeric_cols:
            candidates_d.append(numeric_cols[0])
        if not candidates_xi and len(numeric_cols) >= 2:
            xic = [c for c in numeric_cols if c.lower() == "xi"]
            candidates_xi.append(xic[0] if xic else numeric_cols[1])
    if not candidates_d or not candidates_xi:
        raise ValueError(f"Could not auto-detect columns. Available: {list(df.columns)}. Use --d-col and --xi-col.")
    return candidates_d[0], candidates_xi[0]


def load_xi_csv(path: Path, d_col: Optional[str], xi_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    df = pd.read_csv(path)
    d_name, xi_name = detect_columns(df, d_col, xi_col)
    d = df[d_name].to_numpy(dtype=float)
    xi = df[xi_name].to_numpy(dtype=float)
    mask = np.isfinite(d) & np.isfinite(xi)
    d = d[mask]; xi = xi[mask]
    order = np.argsort(d)
    return d[order], xi[order], (d_name, xi_name)


# ---------- Preprocessing ----------
def make_uniform_grid(d: np.ndarray, xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if len(d) < 8:
        raise ValueError("Input series too short for FFT.")
    deltas = np.diff(d)
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        raise ValueError("Non-increasing distance vector.")
    delta_d = float(np.median(positive))
    d_u = np.arange(d[0], d[-1] + 0.5 * delta_d, delta_d)
    xi_u = np.interp(d_u, d, xi)
    return d_u, xi_u, delta_d


def poly_detrend(x: np.ndarray, d: np.ndarray, deg: int = 3) -> np.ndarray:
    coeff = np.polyfit(d, x, deg)
    trend = np.polyval(coeff, d)
    return x - trend


def tukey_window(n: int, alpha: float = 0.4) -> np.ndarray:
    if n < 2:
        return np.ones(n)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge > 0:
        w[:edge] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge, endpoint=False)))
        w[-edge:] = w[:edge][::-1]
    return w


# ---------- Spectrum ----------
def spectrum_power(x: np.ndarray, delta_d: float, zp_factor: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided power spectrum of a real signal with zero-padding.
    x: windowed, detrended series (length n)
    zp_factor: integer >=1, pad to n*zp_factor by appending zeros
    Returns (lambda_mpc, power)
    """
    n = len(x)
    if zp_factor < 1:
        zp_factor = 1
    n_pad = n * zp_factor
    X = np.fft.rfft(x, n=n_pad)
    freqs = np.fft.rfftfreq(n_pad, d=delta_d)  # cycles per Mpc
    power = (X.conjugate() * X).real
    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]
    lambda_mpc = 1.0 / freqs
    return lambda_mpc, power


def peak_in_lambda_range(lambda_mpc: np.ndarray, power: np.ndarray,
                         lam_min: float, lam_max: float) -> Tuple[float, float, int]:
    rng = (lambda_mpc >= lam_min) & (lambda_mpc <= lam_max)
    if not np.any(rng):
        raise ValueError("No spectral bins inside the requested wavelength range.")
    idxs = np.where(rng)[0]
    sub = power[rng]
    imax_local = int(np.argmax(sub))
    imax = idxs[imax_local]
    return float(power[imax]), float(lambda_mpc[imax]), int(imax)


# ---------- Nulls ----------
def phase_randomize_series(x_windowed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(x_windowed)
    kmax = len(X) - 1
    if kmax >= 2:
        phases = rng.uniform(0, 2 * np.pi, size=kmax - 1)
        amps = np.abs(X[1:kmax])
        X[1:kmax] = amps * np.exp(1j * phases)
    return np.fft.irfft(X, n=len(x_windowed))


def shuffle_bins(x_windowed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = x_windowed.copy()
    rng.shuffle(x)
    return x


# ---------- Jackknife ----------
def jackknife_leave_one_out(file_paths: List[Path],
                            base_d_u: np.ndarray,
                            lam_bounds: Tuple[float, float],
                            delta_d: float,
                            poly_deg: int,
                            alpha: float,
                            d_col: Optional[str],
                            xi_col: Optional[str],
                            zp_factor: int) -> pd.DataFrame:
    series = []
    for p in file_paths:
        d, xi, _ = load_xi_csv(p, d_col, xi_col)
        d_u2, xi_u2, _ = make_uniform_grid(d, xi)
        series.append(np.interp(base_d_u, d_u2, xi_u2))
    series = np.array(series)
    records = []
    lam_min, lam_max = lam_bounds
    window = tukey_window(len(base_d_u), alpha=alpha)
    for omit in range(series.shape[0]):
        keep = [i for i in range(series.shape[0]) if i != omit]
        xi_mean = series[keep].mean(axis=0)
        xi_detr = poly_detrend(xi_mean, base_d_u, deg=poly_deg)
        xw = xi_detr * window
        lam, P = spectrum_power(xw, delta_d, zp_factor)
        pmax, lam_at, _ = peak_in_lambda_range(lam, P, lam_min, lam_max)
        records.append({"omit_index": omit, "omit_file": str(file_paths[omit]),
                        "peak_lambda_mpc": lam_at, "peak_power": pmax})
    return pd.DataFrame.from_records(records)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Step 11: Null tests for ~183 Mpc feature in z6p.")
    ap.add_argument("--input", type=str, help="Path to xi_z6p.csv (default: results/step9c/20250816_174144/xi_z6p.csv)")
    ap.add_argument("--d-col", type=str, default=None, help="Distance column name override")
    ap.add_argument("--xi-col", type=str, default=None, help="Xi column name override")
    ap.add_argument("--n-sims", type=int, default=5000, help="Number of null simulations")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--lambda-min", type=float, default=150.0, help="Lower bound of wavelength range [Mpc]")
    ap.add_argument("--lambda-max", type=float, default=200.0, help="Upper bound of wavelength range [Mpc]")
    ap.add_argument("--auto-clamp", type=int, default=1, help="1=Clamp lambda range to available spectrum; 0=error if no overlap")
    ap.add_argument("--polydeg", type=int, default=3, help="Polynomial detrend degree")
    ap.add_argument("--alpha", type=float, default=0.4, help="Tukey window alpha in [0,1]")
    ap.add_argument("--null-mode", type=str, choices=["phase", "shuffle"], default="phase", help="Null generation mode")
    ap.add_argument("--jackknife-glob", type=str, default=None, help="Glob for subfield xi CSVs (optional)")
    ap.add_argument("--zp-factor", type=int, default=8, help="Zero-padding factor (>=1)")
    ap.add_argument("--diagnose", action="store_true", help="Print diagnostics and exit")
    args = ap.parse_args()

    root = project_root_from_this_file()
    if args.input is None:
        args.input = str(root / "results" / "step9c" / "20250816_174144" / "xi_z6p.csv")
    inp_path = Path(args.input).resolve()
    if not inp_path.exists():
        raise FileNotFoundError(f"Input file not found: {inp_path}")

    # Load & preprocess (no dirs created yet)
    d, xi, (d_name, xi_name) = load_xi_csv(inp_path, args.d_col, args.xi_col)
    d_u, xi_u, delta_d = make_uniform_grid(d, xi)
    xi_detr = poly_detrend(xi_u, d_u, deg=args.polydeg)
    window = tukey_window(len(d_u), alpha=args.alpha)
    xw_obs = xi_detr * window

    # Spectrum & available wavelength bounds (with zero-padding awareness)
    lam_obs, P_obs = spectrum_power(xw_obs, delta_d, args.zp_factor)
    lam_min_avail = float(np.min(lam_obs))
    lam_max_avail = float(np.max(lam_obs))
    print("=== Diagnostic ===")
    print(f"Input file: {inp_path}")
    print(f"Detected columns: distance='{d_name}', xi='{xi_name}'")
    print(f"Distance range: [{d_u[0]:.3f}, {d_u[-1]:.3f}] Mpc, N={len(d_u)}, Δd={delta_d:.6g} Mpc")
    print(f"Zero-padding factor: {args.zp_factor}x")
    print(f"Available wavelength span from spectrum: [{lam_min_avail:.3f}, {lam_max_avail:.3f}] Mpc")

    lam_lo_req, lam_hi_req = float(args.lambda_min), float(args.lambda_max)
    lam_lo_used, lam_hi_used = lam_lo_req, lam_hi_req

    if args.auto_clamp:
        lam_lo_used = max(lam_lo_req, lam_min_avail)
        lam_hi_used = min(lam_hi_req, lam_max_avail)
        if lam_lo_used >= lam_hi_used:
            print("\n[ERROR] Requested wavelength window has no overlap with available spectrum even after clamping.")
            print(f"Requested: [{lam_lo_req}, {lam_hi_req}] Mpc")
            print(f"Available: [{lam_min_avail:.3f}, {lam_max_avail:.3f}] Mpc")
            print("Action: Adjust --lambda-min/--lambda-max or increase --zp-factor.")
            return
        else:
            print(f"Using wavelength window: [{lam_lo_used:.3f}, {lam_hi_used:.3f}] Mpc (auto-clamped from [{lam_lo_req}, {lam_hi_req}])")
    else:
        if lam_lo_req < lam_min_avail or lam_hi_req > lam_max_avail:
            print("\n[ERROR] Requested wavelength window lies outside available spectrum and --auto-clamp=0.")
            print(f"Requested: [{lam_lo_req}, {lam_hi_req}] Mpc")
            print(f"Available: [{lam_min_avail:.3f}, {lam_max_avail:.3f}] Mpc")
            return
        else:
            print(f"Using wavelength window: [{lam_lo_used:.3f}, {lam_hi_used:.3f}] Mpc")

    # Print FFT bins near/within the window
    def print_nearby_bins(lam_array: np.ndarray, center_min: float, center_max: float, n_each: int = 6):
        idx_in = np.where((lam_array >= center_min) & (lam_array <= center_max))[0]
        if len(idx_in) == 0:
            center = 0.5 * (center_min + center_max)
            i = int(np.argmin(np.abs(lam_array - center)))
            lo = max(0, i - n_each)
            hi = min(len(lam_array), i + n_each + 1)
            sel = lam_array[lo:hi]
            print(f"Closest bins to center ~{center:.3f} Mpc:")
            print("  " + ", ".join(f"{v:.3f}" for v in sel))
        else:
            lo = max(0, idx_in[0] - n_each)
            hi = min(len(lam_array), idx_in[-1] + n_each + 1)
            sel = lam_array[lo:hi]
            print(f"FFT bins near/within [{center_min:.3f}, {center_max:.3f}] Mpc:")
            print("  " + ", ".join(f"{v:.3f}" for v in sel))

    print_nearby_bins(lam_obs, lam_lo_used, lam_hi_used)

    # Diagnose mode exits before creating any output directories
    if args.diagnose:
        ts = timestamp_now()
        out_results, _ = make_unique_outdirs(root, "step11", ts)
        readme = out_results / "README_step11.txt"
        with open(readme, "w", encoding="utf-8") as f:
            f.write("Diagnosis only (no nulls run):\n")
            f.write(f"Input: {inp_path}\n")
            f.write(f"Columns: distance='{d_name}', xi='{xi_name}'\n")
            f.write(f"d range: [{d_u[0]:.6f}, {d_u[-1]:.6f}] Mpc; N={len(d_u)}; Δd={delta_d:.6g}\n")
            f.write(f"Zero-padding factor: {args.zp_factor}\n")
            f.write(f"Wavelength available: [{lam_min_avail:.6f}, {lam_max_avail:.6f}] Mpc\n")
            f.write(f"Wavelength window used: [{lam_lo_used:.6f}, {lam_hi_used:.6f}] Mpc\n")
        print("\n[diagnose] Summary written. Exiting.")
        return

    # Observed peak
    pmax_obs, lam_at_obs, _ = peak_in_lambda_range(lam_obs, P_obs, lam_lo_used, lam_hi_used)

    # Nulls
    rng = np.random.default_rng(args.seed)
    null_peaks_power = np.empty(args.n_sims, dtype=float)
    null_peaks_lambda = np.empty(args.n_sims, dtype=float)
    for i in range(args.n_sims):
        if args.null_mode == "phase":
            x_surr = phase_randomize_series(xw_obs, rng)
            lam, P = spectrum_power(x_surr, delta_d, args.zp_factor)
        else:
            x_shuf = shuffle_bins(xw_obs, rng)
            lam, P = spectrum_power(x_shuf, delta_d, args.zp_factor)
        pmax, lam_at, _ = peak_in_lambda_range(lam, P, lam_lo_used, lam_hi_used)
        null_peaks_power[i] = pmax
        null_peaks_lambda[i] = lam_at

    p_value = float(np.mean(null_peaks_power >= pmax_obs))

    # Create unique output dirs now and write outputs
    ts = timestamp_now()
    out_results, out_figs = make_unique_outdirs(root, "step11", ts)

    pval_file = out_results / "pvalue_z6p.txt"
    with open(pval_file, "w", newline="") as f:
        f.write(f"Observed peak (lambda in [{lam_lo_used}, {lam_hi_used}] Mpc):\n")
        f.write(f"  lambda_obs_mpc = {lam_at_obs:.6f}\n")
        f.write(f"  power_obs      = {pmax_obs:.6g}\n")
        f.write(f"\nNull mode: {args.null_mode}\n")
        f.write(f"Simulations: {args.n_sims}\n")
        f.write(f"Zero-padding factor: {args.zp_factor}\n")
        f.write(f"p-value (one-sided, null peak >= observed peak): {p_value:.6g}\n")

    null_csv = out_results / "null_distribution.csv"
    with open(null_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["null_peak_power", "null_peak_lambda_mpc"])
        for ppk, ll in zip(null_peaks_power, null_peaks_lambda):
            w.writerow([f"{ppk:.6g}", f"{ll:.6f}"])

    readme = out_results / "README_step11.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Step 11: JWST z6p 183 Mpc significance test\n")
        f.write(f"Input: {inp_path}\n")
        f.write(f"Columns: distance='{d_name}', xi='{xi_name}'\n")
        f.write(f"N bins (uniform): {len(d_u)}; delta_d = {delta_d:.6g} Mpc\n")
        f.write(f"d range: [{d_u[0]:.6f}, {d_u[-1]:.6f}] Mpc\n")
        f.write(f"Detrend poly deg = {args.polydeg}\n")
        f.write(f"Tukey alpha      = {args.alpha}\n")
        f.write(f"Null mode        = {args.null_mode}\n")
        f.write(f"Simulations      = {args.n_sims}\n")
        f.write(f"Zero-padding     = {args.zp_factor}x\n")
        f.write(f"Wavelength available: [{lam_min_avail:.6f}, {lam_max_avail:.6f}] Mpc\n")
        f.write(f"Wavelength window used = [{lam_lo_used:.6f}, {lam_hi_used:.6f}] Mpc\n")
        f.write(f"Observed lambda  = {lam_at_obs:.6f} Mpc\n")
        f.write(f"Observed power   = {pmax_obs:.6g}\n")
        f.write(f"p_value          = {p_value:.6g}\n")

    fig_path = out_figs / "null_histogram.png"
    plt.figure(figsize=(7, 5))
    plt.hist(null_peaks_power, bins=50, edgecolor="black")
    plt.axvline(pmax_obs, linestyle="--", linewidth=2)
    plt.title(f"Null distribution of max peak power ({lam_lo_used:.0f}–{lam_hi_used:.0f} Mpc)\nObserved peak marked")
    plt.xlabel(f"Max peak power in [{lam_lo_used:.0f}, {lam_hi_used:.0f}] Mpc")
    plt.ylabel("Count")
    counts, _ = np.histogram(null_peaks_power, bins=50)
    y_annot = max(counts) * 0.75 if len(counts) else 1.0
    plt.text(pmax_obs, y_annot, f"Observed={pmax_obs:.3g}\np={p_value:.3g}", ha="left", va="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("\n=== Step 11 Results ===")
    print(f"Observed peak @ λ = {lam_at_obs:.3f} Mpc with power = {pmax_obs:.6g}")
    print(f"Null mode = {args.null_mode}, simulations = {args.n_sims}, zero-padding = {args.zp_factor}x")
    print(f"Empirical p-value (null peak >= observed peak) = {p_value:.6g}")
    print("\nOutputs:")
    print(f"  {pval_file}")
    print(f"  {null_csv}")
    print(f"  {readme}")
    print(f"  {fig_path}")

if __name__ == "__main__":
    main()
