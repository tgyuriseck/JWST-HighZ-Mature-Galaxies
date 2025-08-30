# power_from_xi.py
# Step 10: Detection-only periodicity check from xi(d)
# - Reads per-tier xi_*.csv from latest results\step9c\<timestamp>\ (or --xi-dir)
# - Prepares xi(d): detrend (mean|linear|none), Tukey window, zero-pad
# - FFT -> P(k) = |FFT[xi]|^2 (one-sided, no DC), convert to wavelength λ = 2π/k
# - Saves CSVs + labeled figures per tier, optional stacked spectrum
#
# Run examples (from C:\JWST-Mature-Galaxies\src):
#   python analysis\power_from_xi.py
#   python analysis\power_from_xi.py --stack
#   python analysis\power_from_xi.py --xi-dir "C:\JWST-Mature-Galaxies\results\step9c\20250816_174144" --stack

import argparse
import datetime as dt
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Optional SciPy usage (nice-to-have). Fallbacks provided if unavailable. ---
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    _scipy_find_peaks = None


def _project_root() -> str:
    """Return project root assuming this file lives in src/analysis/."""
    here = os.path.abspath(os.path.dirname(__file__))              # ...\src\analysis
    return os.path.abspath(os.path.join(here, "..", ".."))         # -> ...\ (project root)


def _latest_step9c_dir(root: str) -> str:
    """Pick newest results\step9c\<timestamp>\ directory that contains xi_*.csv."""
    base = os.path.join(root, "results", "step9c")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Missing folder: {base}")

    cand = sorted(
        [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    )
    cand = [d for d in cand if glob.glob(os.path.join(d, "xi_*.csv"))]
    if not cand:
        raise FileNotFoundError(f"No xi_*.csv found under {base}\\<timestamp>\\")
    return cand[-1]


def _tukey_window(N: int, alpha: float) -> np.ndarray:
    """Tukey window (alpha in [0,1]). Rectangular at 0, Hann at 1."""
    if alpha <= 0:
        return np.ones(N, dtype=float)
    if alpha >= 1:
        n = np.arange(N)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    w = np.ones(N, dtype=float)
    edge = int(np.floor(alpha * (N - 1) / 2.0))
    if edge > 0:
        n = np.arange(0, edge + 1)
        w[: edge + 1] = 0.5 * (1 + np.cos(np.pi * ((2 * n) / (alpha * (N - 1)) - 1)))
        n2 = np.arange(N - edge - 1, N)
        w[N - edge - 1 :] = 0.5 * (1 + np.cos(np.pi * ((2 * (n2 - (N - 1))) / (alpha * (N - 1)) + 1)))
    return w


def _simple_find_peaks(y: np.ndarray, min_distance: int = 3, z_thresh: float = 2.0) -> np.ndarray:
    if y.size < 3:
        return np.array([], dtype=int)
    med = np.median(y)
    std = np.std(y) if np.std(y) > 0 else 1.0
    thr = med + z_thresh * std
    candidates = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    candidates = [i for i in candidates if y[i] >= thr]
    kept: List[int] = []
    for i in sorted(candidates, key=lambda j: y[j], reverse=True):
        if all(abs(i - j) >= min_distance for j in kept):
            kept.append(i)
    kept.sort()
    return np.array(kept, dtype=int)


def _detrend(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x.copy()
    if mode == "mean":
        return x - np.mean(x)
    if mode == "linear":
        n = np.arange(x.size)
        coeff = np.polyfit(n, x, 1)
        trend = np.polyval(coeff, n)
        return x - trend
    raise ValueError(f"Unknown detrend mode: {mode}")


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def _read_xi_csv(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(path)
    col_center = None
    for cand in ["bin_center_Mpc", "r_center_Mpc", "r_Mpc", "r"]:
        if cand in df.columns:
            col_center = cand
            break
    if col_center is None:
        raise ValueError(f"{os.path.basename(path)} missing a bin-center column.")
    col_xi = None
    for cand in ["xi", "xi_val", "xi_value"]:
        if cand in df.columns:
            col_xi = cand
            break
    if col_xi is None:
        raise ValueError(f"{os.path.basename(path)} missing xi column.")
    col_sig = "sigma_xi" if "sigma_xi" in df.columns else None
    r = df[col_center].to_numpy(dtype=float)
    xi = df[col_xi].to_numpy(dtype=float)
    sig = df[col_sig].to_numpy(dtype=float) if col_sig else None
    return r, xi, sig


def _infer_tier_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.match(r"^xi_(.+)\.csv$", base)
    return m.group(1) if m else os.path.splitext(base)[0]


def _fft_power(
    r_centers: np.ndarray,
    xi: np.ndarray,
    tukey_alpha: float,
    detrend_mode: str,
    pad_pow2: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    diffs = np.diff(r_centers)
    delta_d = float(np.median(diffs))
    if not np.allclose(diffs, delta_d, rtol=1e-3, atol=1e-6):
        print("[WARN] bin spacing not perfectly uniform; using median Δd =", delta_d)
    x = _detrend(xi, detrend_mode)
    win = _tukey_window(x.size, tukey_alpha)
    xw = x * win
    if pad_pow2:
        N_fft = _next_pow2(xw.size) * 4
    else:
        N_fft = xw.size
    X = np.fft.rfft(xw, n=N_fft)
    f_cyc = np.fft.rfftfreq(N_fft, d=delta_d)
    X = X[1:]
    f_cyc = f_cyc[1:]
    P = (np.abs(X) ** 2)
    k_rad = 2.0 * np.pi * f_cyc
    return f_cyc, k_rad, P, delta_d, N_fft


def _peaks_from_power(
    k_rad: np.ndarray,
    P: np.ndarray,
    delta_d: float,
    N_bins: int,
    use_scipy: bool,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (2.0 * np.pi) / k_rad
    lam_min = 2.0 * delta_d
    lam_max = 0.9 * (N_bins * delta_d)
    valid = (lam >= lam_min) & (lam <= lam_max) & np.isfinite(lam)
    idx_valid = np.where(valid)[0]
    if idx_valid.size < 3:
        return np.array([], dtype=int)
    Pv = P[valid]
    if use_scipy and _scipy_find_peaks is not None:
        peaks, _ = _scipy_find_peaks(Pv, prominence=np.median(Pv) * 0.25, distance=2)
    else:
        peaks = _simple_find_peaks(Pv, min_distance=2, z_thresh=2.0)
    return idx_valid[peaks]


def _ensure_dirs(root: str, ts: str) -> Tuple[str, str]:
    res_dir = os.path.join(root, "results", "step10", ts)
    fig_dir = os.path.join(root, "figures", "step10", ts)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return res_dir, fig_dir


def _save_power_csv(path: str, f_cyc: np.ndarray, k_rad: np.ndarray, P: np.ndarray):
    df = pd.DataFrame({
        "f_cyc_per_Mpc": f_cyc,
        "k_rad_per_Mpc": k_rad,
        "lambda_Mpc": np.where(f_cyc > 0, 1.0 / f_cyc, np.nan),
        "power": P,
    })
    df.to_csv(path, index=False)


def _save_peaks_csv(path: str, k_rad: np.ndarray, P: np.ndarray, peak_idx: np.ndarray):
    if peak_idx.size == 0:
        pd.DataFrame(columns=["k_rad_per_Mpc", "lambda_Mpc", "power"]).to_csv(path, index=False)
        return
    k = k_rad[peak_idx]
    lam = (2.0 * np.pi) / k
    Pw = P[peak_idx]
    df = pd.DataFrame({
        "k_rad_per_Mpc": k,
        "lambda_Mpc": lam,
        "power": Pw,
    }).sort_values("power", ascending=False)
    df.to_csv(path, index=False)


def _plot_power(
    out_png: str,
    title: str,
    k_rad: np.ndarray,
    P: np.ndarray,
    tukey_alpha: float,
    detrend_mode: str,
    delta_d: float,
    N_bins: int,
    peak_idx: np.ndarray,
    annotate_top: int = 5,
    bao_lambda: float = 150.0,
    kmax: Optional[float] = None,
):
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (2.0 * np.pi) / k_rad
    valid = np.isfinite(lam)
    lam = lam[valid]
    Pp = P[valid]
    if kmax is not None:
        lam_min_allowed = (2.0 * np.pi) / kmax
        keep = lam >= lam_min_allowed
        lam = lam[keep]
        Pp = Pp[keep]
    order = np.argsort(lam)
    lam_plot = lam[order]
    P_plot = Pp[order]
    plt.figure(figsize=(9, 5.5))
    plt.plot(lam_plot, P_plot)
    plt.xlabel("Wavelength λ (Mpc)")
    plt.ylabel("Power P(k) = |FFT[ξ]|² (arb. units)")
    plt.title(title)
    if bao_lambda is not None and bao_lambda > 0:
        plt.axvline(bao_lambda, linestyle="--", alpha=0.6)
        plt.text(bao_lambda, np.nanmax(P_plot) * 0.85, "BAO ~150 Mpc", rotation=90, va="top", ha="right")
    if peak_idx.size > 0:
        order_peaks = np.argsort(P[peak_idx])[::-1][:annotate_top]
        for j in order_peaks:
            i = peak_idx[j]
            lam_i = (2.0 * np.pi) / k_rad[i]
            Pi = P[i]
            if not np.isfinite(lam_i):
                continue
            plt.scatter([lam_i], [Pi], s=30)
            plt.text(lam_i, Pi, f"λ≈{lam_i:.1f} Mpc", va="bottom", ha="center", fontsize=9)
    meta = f"Tukey α={tukey_alpha}, detrend={detrend_mode}, Δd={delta_d:.3g} Mpc, Nbins={N_bins}"
    plt.gcf().text(0.99, 0.01, meta, ha="right", va="bottom", fontsize=8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _stack_xi(
    per_tier: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    weights_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tiers = list(per_tier.keys())
    r0 = per_tier[tiers[0]][0]
    N = r0.size
    X = []
    W = []
    for t in tiers:
        r, xi, sig = per_tier[t]
        if r.size != N or not np.allclose(r, r0, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Tier {t} has mismatched bin centers; ensure Step 9C used identical bins.")
        X.append(xi)
        if weights_mode == "ivar":
            if sig is None:
                raise ValueError(f"Tier {t} missing sigma_xi for inverse-variance weights. Use --weights equal or regenerate.")
            w = np.where(sig > 0, 1.0 / (sig ** 2), 0.0)
        else:
            w = np.ones_like(xi)
        W.append(w)
    X = np.vstack(X)
    W = np.vstack(W)
    Wsum = np.sum(W, axis=0)
    Wsum_safe = np.where(Wsum > 0, Wsum, np.nan)
    xi_stack = np.nansum(W * X, axis=0) / Wsum_safe
    sigma_eff = np.where(Wsum > 0, 1.0 / np.sqrt(Wsum), np.inf)
    return r0, xi_stack, sigma_eff


def main():
    parser = argparse.ArgumentParser(description="Step 10: FFT-based periodicity detection from xi(d).")
    parser.add_argument("--xi-dir", type=str, default=None, help="Path to results\\step9c\\<timestamp> folder with xi_*.csv.")
    parser.add_argument("--stack", action="store_true", help="Also compute stacked spectrum across tiers.")
    parser.add_argument("--weights", type=str, default="ivar", choices=["ivar", "equal"], help="Stacking weights (ivar|equal).")
    parser.add_argument("--tukey-alpha", type=float, default=0.25, help="Tukey window alpha (0=rect, 1=Hann).")
    parser.add_argument("--detrend", type=str, default="mean", choices=["mean", "linear", "none"], help="Detrend mode.")
    parser.add_argument("--no-pad-pow2", action="store_true", help="Disable zero-padding to 4× next power-of-two.")
    parser.add_argument("--kmax", type=float, default=None, help="Optional x-limit: max k (rad/Mpc) to annotate/plot.")
    args = parser.parse_args()

    root = _project_root()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir, fig_dir = _ensure_dirs(root, ts)

    if args.xi_dir is not None:
        xi_dir = args.xi_dir
    else:
        xi_dir = _latest_step9c_dir(root)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Reading per-tier ξ(d) from: {xi_dir}")
    print(f"[INFO] Writing CSV to: {res_dir}")
    print(f"[INFO] Writing figures to: {fig_dir}")
    if not _HAVE_SCIPY:
        print("[INFO] SciPy not detected; using a simple local-max peak finder (z≈2 threshold).")

    paths = sorted(glob.glob(os.path.join(xi_dir, "xi_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No xi_*.csv in {xi_dir}")

    per_tier: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
    for p in paths:
        tier = _infer_tier_from_filename(p)
        r, xi, sig = _read_xi_csv(p)
        per_tier[tier] = (r, xi, sig)

        pad_pow2 = not args.no_pad_pow2
        f_cyc, k_rad, P, delta_d, N_fft = _fft_power(
            r, xi, args.tukey_alpha, args.detrend, pad_pow2
        )
        N_bins = r.size
        peak_idx = _peaks_from_power(k_rad, P, delta_d, N_bins, use_scipy=_HAVE_SCIPY)

        out_power_csv = os.path.join(res_dir, f"power_{tier}.csv")
        _save_power_csv(out_power_csv, f_cyc, k_rad, P)

        out_peaks_csv = os.path.join(res_dir, f"peaks_{tier}.csv")
        _save_peaks_csv(out_peaks_csv, k_rad, P, peak_idx)

        out_png = os.path.join(fig_dir, f"power_{tier}.png")
        _plot_power(
            out_png=out_png,
            title=f"Power spectrum from ξ(d) — {tier}",
            k_rad=k_rad,
            P=P,
            tukey_alpha=args.tukey_alpha,
            detrend_mode=args.detrend,
            delta_d=delta_d,
            N_bins=N_bins,
            peak_idx=peak_idx,
            annotate_top=5,
            bao_lambda=150.0,
            kmax=args.kmax,
        )

        if peak_idx.size:
            k_top = k_rad[peak_idx]
            lam_top = (2.0 * np.pi) / k_top
            P_top = P[peak_idx]
            order = np.argsort(P_top)[::-1][:3]
            print(f"[{tier}] top peaks (λ Mpc, power):")
            for j in order:
                print(f"   λ≈{lam_top[j]:.1f}  P={P_top[j]:.3g}")
        else:
            print(f"[{tier}] no peaks detected under default thresholds.")

    if args.stack:
        print("[INFO] Computing stacked ξ(d) across tiers ", ", ".join(per_tier.keys()))
        r0, xi_stack, sigma_eff = _stack_xi(per_tier, weights_mode=args.weights)
        stack_xi_csv = os.path.join(res_dir, "stack_xi.csv")
        pd.DataFrame({
            "bin_center_Mpc": r0,
            "xi_stack": xi_stack,
            "sigma_eff": sigma_eff
        }).to_csv(stack_xi_csv, index=False)

        pad_pow2 = not args.no_pad_pow2
        f_cyc, k_rad, P, delta_d, N_fft = _fft_power(
            r0, xi_stack, args.tukey_alpha, args.detrend, pad_pow2
        )
        N_bins = r0.size
        peak_idx = _peaks_from_power(k_rad, P, delta_d, N_bins, use_scipy=_HAVE_SCIPY)

        out_power_csv = os.path.join(res_dir, "power_stacked.csv")
        _save_power_csv(out_power_csv, f_cyc, k_rad, P)

        out_peaks_csv = os.path.join(res_dir, "peaks_stacked.csv")
        _save_peaks_csv(out_peaks_csv, k_rad, P, peak_idx)

        out_png = os.path.join(fig_dir, "power_stacked.png")
        _plot_power(
            out_png=out_png,
            title=f"Power spectrum from ξ(d) — STACKED ({args.weights})",
            k_rad=k_rad,
            P=P,
            tukey_alpha=args.tukey_alpha,
            detrend_mode=args.detrend,
            delta_d=delta_d,
            N_bins=N_bins,
            peak_idx=peak_idx,
            annotate_top=6,
            bao_lambda=150.0,
            kmax=args.kmax,
        )

        if peak_idx.size:
            k_top = k_rad[peak_idx]
            lam_top = (2.0 * np.pi) / k_top
            P_top = P[peak_idx]
            order = np.argsort(P_top)[::-1][:5]
            print("[stacked] top peaks (λ Mpc, power):")
            for j in order:
                print(f"   λ≈{lam_top[j]:.1f}  P={P_top[j]:.3g}")
        else:
            print("[stacked] no peaks detected under default thresholds.")

    with open(os.path.join(res_dir, "README_step10.txt"), "w", encoding="utf-8") as f:
        f.write("Step 10: FFT-based periodicity detection from xi(d)\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"xi_dir: {xi_dir}\n")
        f.write(f"detrend: {args.detrend}\n")
        f.write(f"tukey_alpha: {args.tukey_alpha}\n")
        f.write(f"pad_pow2: {not args.no_pad_pow2}\n")
        f.write(f"stack: {args.stack}\n")
        if args.stack:
            f.write(f"weights: {args.weights}\n")
        f.write("Files written: power_*.csv, peaks_*.csv, power_*.png\n")

    print("\n[OK] Step 10 finished.")
    print(f"Open figures here: {fig_dir}")
    print(f"Peak tables here: {res_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
