# make_harmonics_leakage_figure.py
# Purpose: Create a 4-panel figure explaining harmonics and spectral leakage,
#          with an overlay of your real stacked spectrum from Step 10.
#
# Panels:
#   A) Ideal periodic signal (λ0 ~ 150 Mpc) -> sharp FFT peak at λ0
#   B) Harmonics example (add small components at λ0/2, λ0/3) -> peaks at 150, 75, 50 Mpc
#   C) Finite-windowed signal (truncate to 1–250 Mpc, Tukey window) -> leakage and smeared power
#   D) Real data overlay: latest results\step10\...\power_stacked.csv plotted vs λ with vertical guides
#
# Outputs (auto-timestamped):
#   figures\methods\<timestamp>\harmonics_leakage.png
#   figures\methods\<timestamp>\harmonics_leakage.pdf
#
# Run from: C:\JWST-Mature-Galaxies\src
#   python analysis\make_harmonics_leakage_figure.py

import os
import glob
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Utilities
# ---------------------------

def project_root():
    """Assume this file is at src/analysis; go up two levels to project root."""
    here = os.path.abspath(os.path.dirname(__file__))        # ...\src\analysis
    return os.path.abspath(os.path.join(here, "..", ".."))   # -> ...\ (project root)

def ensure_out_dirs(root, ts):
    out_dir = os.path.join(root, "figures", "methods", ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def latest_power_stacked_csv(root):
    """
    Find the newest results\\step10\\<timestamp>\\power_stacked.csv.
    Returns path or None.
    """
    step10_base = os.path.join(root, "results", "step10")
    if not os.path.isdir(step10_base):
        return None
    candidates = []
    for d in sorted(glob.glob(os.path.join(step10_base, "*"))):
        p = os.path.join(d, "power_stacked.csv")
        if os.path.isfile(p):
            candidates.append(p)
    return candidates[-1] if candidates else None

def tukey_window(N, alpha):
    """Tukey window; alpha in [0,1]."""
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        n = np.arange(N)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    w = np.ones(N)
    edge = int(np.floor(alpha * (N - 1) / 2.0))
    if edge > 0:
        n = np.arange(0, edge + 1)
        w[: edge + 1] = 0.5 * (1 + np.cos(np.pi * ((2 * n) / (alpha * (N - 1)) - 1)))
        n2 = np.arange(N - edge - 1, N)
        w[N - edge - 1 :] = 0.5 * (1 + np.cos(np.pi * ((2 * (n2 - (N - 1))) / (alpha * (N - 1)) + 1)))
    return w

def one_sided_power_from_signal(x, delta_d):
    """
    Real FFT one-sided power (drop DC):
      returns (lambda_Mpc_sorted, power_sorted)
    """
    # Zero-mean to avoid a giant DC spike obscuring the plot
    x = x - np.mean(x)

    X = np.fft.rfft(x)
    f_cyc = np.fft.rfftfreq(x.size, d=delta_d)  # cycles/Mpc
    # drop DC
    X = X[1:]
    f_cyc = f_cyc[1:]
    P = np.abs(X) ** 2

    # convert to wavelength λ = 1/f_cyc (Mpc)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = 1.0 / f_cyc
    ok = np.isfinite(lam) & (lam > 0)
    lam = lam[ok]
    P = P[ok]

    # sort by λ ascending for plotting
    order = np.argsort(lam)
    return lam[order], P[order]

# ---------------------------
# Synthetic signals
# ---------------------------

def make_signal_ideal(lambda0, length_mpc=5000.0, delta=1.0, amp=1.0):
    """
    Long, near-infinite signal to show a sharp fundamental.
    """
    d = np.arange(0.0, length_mpc, delta)
    x = amp * np.sin(2.0 * np.pi * d / lambda0)
    return d, x

def make_signal_with_harmonics(lambda0, length_mpc=5000.0, delta=1.0, amps=(1.0, 0.35, 0.2)):
    """
    Add 1st and 2nd harmonics (λ0/2, λ0/3) to show peaks at 75 and 50 Mpc.
    amps: tuple for [fundamental, first_harmonic, second_harmonic]
    """
    d = np.arange(0.0, length_mpc, delta)
    a0, a1, a2 = amps
    x = (
        a0 * np.sin(2.0 * np.pi * d / lambda0) +
        a1 * np.sin(2.0 * np.pi * d / (lambda0 / 2.0)) +
        a2 * np.sin(2.0 * np.pi * d / (lambda0 / 3.0))
    )
    return d, x

def make_windowed_signal(lambda0, r_min=1.0, r_max=250.0, delta=5.0, tukey_alpha=0.25, noise_amp=0.0):
    """
    Truncate to the same finite range and spacing as your xi(d), add a mild window,
    optionally small noise to mimic measurement variance.
    """
    d = np.arange(r_min, r_max + 1e-9, delta)
    x = np.sin(2.0 * np.pi * d / lambda0)
    if noise_amp > 0:
        rng = np.random.default_rng(12345)
        x = x + noise_amp * rng.normal(size=x.size)
    win = tukey_window(x.size, tukey_alpha)
    xw = x * win
    return d, xw, win

# ---------------------------
# Figure generation
# ---------------------------

def main():
    root = project_root()
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_out_dirs(root, ts)

    # --- Define scales
    lambda0 = 150.0  # BAO-like scale
    # For "ideal", use very long series & fine sampling -> sharp spectral line
    dA, xA = make_signal_ideal(lambda0=lambda0, length_mpc=5000.0, delta=1.0, amp=1.0)
    lamA, PA = one_sided_power_from_signal(xA, delta_d=1.0)

    # Harmonics
    dB, xB = make_signal_with_harmonics(lambda0=lambda0, length_mpc=5000.0, delta=1.0, amps=(1.0, 0.35, 0.2))
    lamB, PB = one_sided_power_from_signal(xB, delta_d=1.0)

    # Finite-windowed example (match your bins ~5 Mpc from 1..250)
    dC, xC, winC = make_windowed_signal(lambda0=lambda0, r_min=1.0, r_max=250.0, delta=5.0, tukey_alpha=0.25, noise_amp=0.0)
    lamC, PC = one_sided_power_from_signal(xC, delta_d=5.0)

    # Real data overlay (latest stacked spectrum)
    stacked_path = latest_power_stacked_csv(root)
    have_stacked = stacked_path is not None

    if have_stacked:
        dfS = pd.read_csv(stacked_path)
        # The Step 10 CSV includes columns:
        #   f_cyc_per_Mpc, k_rad_per_Mpc, lambda_Mpc, power
        lamS = dfS["lambda_Mpc"].to_numpy(dtype=float)
        PS = dfS["power"].to_numpy(dtype=float)
        ok = np.isfinite(lamS) & (lamS > 0) & np.isfinite(PS)
        lamS = lamS[ok]
        PS = PS[ok]
        order = np.argsort(lamS)
        lamS = lamS[order]
        PS = PS[order]

    # --- Build the 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    axA, axB = axes[0]
    axC, axD = axes[1]

    # Panel A: Ideal fundamental only
    axA.plot(lamA, PA)
    axA.set_title("A. Ideal periodicity (λ₀ ≈ 150 Mpc)")
    axA.set_xlabel("Wavelength λ (Mpc)")
    axA.set_ylabel("Power (arb.)")
    axA.axvline(150.0, linestyle="--", alpha=0.7)
    axA.text(150.0, np.nanmax(PA)*0.8, "Fundamental ~150", rotation=90, va="top", ha="right", fontsize=9)

    # Panel B: Harmonics visible (λ₀, λ₀/2, λ₀/3)
    axB.plot(lamB, PB)
    axB.set_title("B. Harmonics at λ₀/2 and λ₀/3")
    axB.set_xlabel("Wavelength λ (Mpc)")
    axB.set_ylabel("Power (arb.)")
    for L in (150.0, 75.0, 50.0):
        axB.axvline(L, linestyle="--", alpha=0.7)
        axB.text(L, np.nanmax(PB)*0.85, f"{int(L)}", rotation=90, va="top", ha="right", fontsize=9)

    # Panel C: Finite window -> leakage/smearing
    axC.plot(lamC, PC)
    axC.set_title("C. Finite window (1–250 Mpc, Δ=5, Tukey α=0.25) → leakage")
    axC.set_xlabel("Wavelength λ (Mpc)")
    axC.set_ylabel("Power (arb.)")
    for L in (150.0, 90.0, 80.0, 56.0, 50.0):
        axC.axvline(L, linestyle=":", alpha=0.6)
    axC.text(150.0, np.nanmax(PC)*0.88, "150", rotation=90, va="top", ha="right", fontsize=9)

    # Panel D: Real stacked spectrum overlay (if available)
    if have_stacked:
        axD.plot(lamS, PS)
        axD.set_title("D. Real stacked spectrum (latest Step 10)")
        axD.set_xlabel("Wavelength λ (Mpc)")
        axD.set_ylabel("Power (arb.)")
        # Visual guides at the scales we've been discussing
        for L in (50.0, 80.0, 90.0, 150.0):
            axD.axvline(L, linestyle="--", alpha=0.6)
            axD.text(L, np.nanmax(PS)*0.85, f"{int(L)}", rotation=90, va="top", ha="right", fontsize=9)
        # Annotate file used
        axD.text(0.98, 0.02, os.path.basename(os.path.dirname(stacked_path)), transform=axD.transAxes,
                 ha="right", va="bottom", fontsize=8, alpha=0.8)
    else:
        axD.set_title("D. Real stacked spectrum (not found)")
        axD.set_xlabel("Wavelength λ (Mpc)")
        axD.set_ylabel("Power (arb.)")
        axD.text(0.5, 0.5, "No power_stacked.csv found.\nRun Step 10 first.", ha="center", va="center",
                 transform=axD.transAxes)

    plt.tight_layout()

    # Save
    png_path = os.path.join(out_dir, "harmonics_leakage.png")
    pdf_path = os.path.join(out_dir, "harmonics_leakage.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print("[OK] Wrote figure:")
    print(" ", png_path)
    print(" ", pdf_path)

if __name__ == "__main__":
    main()
