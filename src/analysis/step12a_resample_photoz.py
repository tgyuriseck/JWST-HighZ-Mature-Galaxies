# === step12a_resample_photoz.py ===
# Step 12a: Generate Monte Carlo realizations by perturbing photo-z per galaxy.
# - Tiers can be provided positionally (e.g., "z6p z8_10") or via --tiers.
# - If no tiers are provided, the script lists available tiers and exits cleanly.
#
# Outputs (timestamped, no overwrite):
#   results/step12/<runid>/resampled/<tier>/astrodeep_<tier>_realizationNNN.csv
#   results/step12/<runid>/summary_resampling.txt
#
# Run from PS C:\JWST-Mature-Galaxies\src\analysis> :
#   # Quick 5-sample sanity test for one tier
#   python step12a_resample_photoz.py z6p --n 5 --sigma-frac 0.06 --seed 7
#
#   # Two tiers, 100 realizations each (paper-level)
#   python step12a_resample_photoz.py z6p z8_10 --n 100 --sigma-frac 0.06 --seed 7 --zmin 4
#
# Notes:
# - Uses per-object error column if present (e.g., zerr/z_err/dz/sigma_z/etc).
# - Fallback sigma: sigma_z = sigma_frac * (1+z).
# - RA/Dec are unchanged; only z is perturbed (radial smear).
# - Clamps z to [zmin, zmax] if provided.

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

# --- Project paths (relative to this file) ---
HERE = os.path.abspath(os.path.dirname(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TIERS_DIR = os.path.join(PROJ_ROOT, "data_processed", "tiers")
RES_BASE  = os.path.join(PROJ_ROOT, "results", "step12")

# --- Column aliases we’ll try to detect automatically ---
Z_ALTS     = ["zphot","z_phot","z","z_best","photoz","photo_z","z_b"]
ZERR_ALTS  = ["zerr","z_err","dz","sigma_z","photoz_err","pz_sigma"]
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

def list_available_tiers():
    if not os.path.isdir(TIERS_DIR):
        return []
    out = []
    for fn in os.listdir(TIERS_DIR):
        if fn.startswith("astrodeep_") and fn.endswith(".csv"):
            tag = fn[len("astrodeep_"):-4]
            out.append(tag)
    return sorted(out)

def ensure_dirs(runid):
    res_dir = os.path.join(RES_BASE, runid, "resampled")
    os.makedirs(res_dir, exist_ok=False)
    return res_dir

def load_tier(tag):
    path = os.path.join(TIERS_DIR, f"astrodeep_{tag}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}.")
    df = pd.read_csv(path)
    f = pick_col(df, FIELD_ALTS); r = pick_col(df, RA_ALTS)
    d = pick_col(df, DEC_ALTS);   z = pick_col(df, Z_ALTS)
    if any(x is None for x in [f,r,d,z]):
        raise ValueError(f"Column detection failed for {path}")
    err = pick_col(df, ZERR_ALTS)
    keep = [f,r,d,z] + ([err] if err else [])
    df = df[keep].rename(columns={f:"field", r:"ra", d:"dec", z:"zphot"})
    if err: df = df.rename(columns={err:"zerr"})
    return df, bool(err)

def resample_z(z, rng, have_err, zerr, sigma_frac, zmin=None, zmax=None):
    if have_err:
        sig = np.asarray(zerr, dtype=float)
        sig = np.where(np.isfinite(sig) & (sig>0), sig, sigma_frac*(1.0+z))
    else:
        sig = sigma_frac*(1.0+z)
    z_new = rng.normal(z, sig)
    if zmin is not None: z_new = np.maximum(z_new, zmin)
    if zmax is not None: z_new = np.minimum(z_new, zmax)
    return z_new

def parse_args_or_help():
    parser = argparse.ArgumentParser(description="Step 12a: resample photo-z by tier.")
    # Allow tiers as either positional or via --tiers
    parser.add_argument("positional_tiers", nargs="*", help="Tier tags (e.g., z6p z8_10)")
    parser.add_argument("--tiers", nargs="+", help="Tier tags (alternative to positional)")
    parser.add_argument("--n", type=int, default=100, help="Realizations per tier")
    parser.add_argument("--sigma-frac", type=float, default=0.06, help="Fallback sigma_z = frac*(1+z)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--zmin", type=float, default=None, help="Clamp: min z")
    parser.add_argument("--zmax", type=float, default=None, help="Clamp: max z")
    args = parser.parse_args()

    # Resolve tier list
    tiers = args.positional_tiers if args.positional_tiers else (args.tiers or [])
    if not tiers:
        avail = list_available_tiers()
        print("No tiers provided.\n")
        if avail:
            print("Available tiers in data_processed/tiers:")
            for t in avail: print(f"  - {t}")
            print("\nExample:")
            print("  python step12a_resample_photoz.py z6p --n 5 --sigma-frac 0.06 --seed 7")
        else:
            print("No tier CSVs found in data_processed/tiers. Make sure Step 7 created them.")
        sys.exit(0)
    return args, tiers

def main():
    args, tiers = parse_args_or_help()
    runid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_resampled_root = ensure_dirs(runid)
    rng = np.random.default_rng(args.seed)

    summary = []
    summary.append("=== Step 12a: Photo-z Resampling ===")
    summary.append(f"Run: {runid}")
    summary.append(f"Tiers: {tiers}")
    summary.append(f"Realizations per tier: {args.n}")
    summary.append(f"Fallback sigma_frac: {args.sigma_frac}")
    if args.zmin is not None or args.zmax is not None:
        summary.append(f"Clamp: [{args.zmin}, {args.zmax}]")
    summary.append("")

    for tag in tiers:
        df, have_err = load_tier(tag)
        tier_dir = os.path.join(out_resampled_root, tag)
        os.makedirs(tier_dir, exist_ok=False)

        z = pd.to_numeric(df["zphot"], errors="coerce").to_numpy()
        zerr = df["zerr"].to_numpy() if "zerr" in df.columns else None

        for i in range(1, args.n+1):
            z_new = resample_z(z, rng, have_err, zerr, args.sigma_frac, args.zmin, args.zmax)
            out = df.copy()
            out["zphot"] = z_new
            ofile = os.path.join(tier_dir, f"astrodeep_{tag}_realization{str(i).zfill(3)}.csv")
            out.to_csv(ofile, index=False)

        summary.append(f"[{tag}] N={len(df)}  per-object σz column: {'yes' if have_err else 'no'}")
        summary.append(f"  → wrote {args.n} files under {tier_dir}")
        summary.append("")

    os.makedirs(os.path.join(RES_BASE, runid), exist_ok=True)
    sum_path = os.path.join(RES_BASE, runid, "summary_resampling.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    print("\n".join(summary))
    print(f"\nSummary: {sum_path}")
    print(f"Next: run Step 12b with --runid {runid}")

if __name__ == "__main__":
    main()
