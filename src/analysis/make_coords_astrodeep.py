# === make_coords_astrodeep.py ===
# Purpose:
#   Convert (RA, Dec, zphot) to comoving Cartesian (X, Y, Z) for *all* tier CSVs.
#   - Scans data_processed/tiers/ for astrodeep_*.csv (≥ tiers AND range bins)
#   - Robustly detects RA/Dec/field/zphot columns (handles RA/DEC and *_optap/_photoz)
#   - Uses Planck18 cosmology for comoving distances
#   - Writes outputs to a unique run folder to avoid overwrites:
#       data_processed/coords/run_YYYYmmdd_HHMMSS/<input_basename>_coords.csv
#   - Produces a manifest + summary:
#       results/step8_coord_summary_<runid>.txt
#       results/step8_coord_manifest_<runid>.csv
#
# Run (from PS C:\JWST-Mature-Galaxies\src>):
#   python analysis\make_coords_astrodeep.py
#
# Requirements:
#   pip install astropy numpy pandas

import os
import sys
import argparse
import textwrap
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.cosmology import Planck18 as COSMO
    import astropy.units as u
except Exception as e:
    raise SystemExit(
        "Astropy (with cosmology) is required.\n"
        "Install with: pip install astropy\n"
        f"Details: {e}"
    )

# ---- Paths (project-root aware) ----
PROJ_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TIERS_DIR   = os.path.join(PROJ_ROOT, "data_processed", "tiers")
COORDS_DIR  = os.path.join(PROJ_ROOT, "data_processed", "coords")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
FIG_DIR     = os.path.join(PROJ_ROOT, "figures")  # reserved for later; not used here

def ensure_dirs():
    for d in [TIERS_DIR, COORDS_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

# --- Column detection helpers ---
Z_ALTS      = ["zphot", "z_phot", "z", "z_best", "photoz", "photo_z", "z_b"]
FIELD_ALTS  = ["field", "field_optap", "field_photoz", "FIELD"]
RA_ALTS     = ["ra", "RA", "ra_optap", "RA_optap", "ra_photoz", "RA_photoz"]
DEC_ALTS    = ["dec", "DEC", "dec_optap", "DEC_optap", "dec_photoz", "DEC_photoz"]

def pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive scan if exact failed
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    zcol     = pick_col(df, Z_ALTS)
    fieldcol = pick_col(df, FIELD_ALTS)
    racol    = pick_col(df, RA_ALTS)
    deccol   = pick_col(df, DEC_ALTS)
    missing = [name for name, col in [("zphot", zcol), ("field", fieldcol), ("RA", racol), ("Dec", deccol)] if col is None]
    if missing:
        # Provide a short preview of the header to help debugging
        header_preview = ", ".join(list(df.columns)[:20])
        raise ValueError(f"Missing required columns: {missing}. Header preview: {header_preview}")
    return zcol, fieldcol, racol, deccol

# --- Coordinate math ---
def r_comoving_mpc(z: np.ndarray) -> np.ndarray:
    """Comoving line-of-sight distance in Mpc for an array of redshifts (Planck18)."""
    # astropy returns Quantity; convert to Mpc and to ndarray
    with np.errstate(invalid="ignore"):
        r = COSMO.comoving_distance(z)  # Quantity
    return r.to(u.Mpc).value

def sph_to_cart(r_mpc: np.ndarray, ra_deg: np.ndarray, dec_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical (r, ra, dec) to Cartesian (X,Y,Z) in Mpc.
       RA, Dec in degrees. Uses standard astronomy convention."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = r_mpc * cosd * np.cos(ra)
    y = r_mpc * cosd * np.sin(ra)
    z = r_mpc * np.sin(dec)
    return x, y, z

def process_file(in_csv: str, out_dir: str) -> dict:
    """Read one tier CSV, compute coordinates, and write coords CSV. Returns stats dict."""
    base = os.path.splitext(os.path.basename(in_csv))[0]  # e.g., astrodeep_z6p
    out_csv = os.path.join(out_dir, f"{base}_coords.csv")

    df = pd.read_csv(in_csv)

    zcol, fieldcol, racol, deccol = detect_columns(df)

    # Coerce numerics and drop rows with invalid values
    z = pd.to_numeric(df[zcol], errors="coerce")
    ra = pd.to_numeric(df[racol], errors="coerce")
    dec= pd.to_numeric(df[deccol], errors="coerce")

    valid = np.isfinite(z) & np.isfinite(ra) & np.isfinite(dec) & (z > 0)
    n_in = len(df)
    n_ok = int(valid.sum())
    if n_ok == 0:
        # Write an empty file with header for traceability (no overwrite behavior)
        empty = pd.DataFrame(columns=["field","ra_deg","dec_deg","zphot","r_mpc","x_mpc","y_mpc","z_mpc"])
        empty.to_csv(out_csv, index=False)
        return {
            "input": in_csv,
            "output": out_csv,
            "rows_in": n_in,
            "rows_ok": n_ok,
            "r_min": np.nan, "r_median": np.nan, "r_max": np.nan
        }

    dfv = df.loc[valid, [fieldcol, racol, deccol, zcol]].copy()
    # Normalize column names in output
    dfv.columns = ["field", "ra_deg", "dec_deg", "zphot"]

    # Compute comoving distances and Cartesian coords
    r = r_comoving_mpc(dfv["zphot"].to_numpy())
    x, y, zc = sph_to_cart(r, dfv["ra_deg"].to_numpy(), dfv["dec_deg"].to_numpy())

    dfv["r_mpc"] = r
    dfv["x_mpc"] = x
    dfv["y_mpc"] = y
    dfv["z_mpc"] = zc

    # Save
    dfv.to_csv(out_csv, index=False)

    return {
        "input": in_csv,
        "output": out_csv,
        "rows_in": n_in,
        "rows_ok": n_ok,
        "r_min": float(np.nanmin(r)),
        "r_median": float(np.nanmedian(r)),
        "r_max": float(np.nanmax(r)),
    }

def main():
    parser = argparse.ArgumentParser(
        description="Convert (RA, Dec, zphot) to comoving XYZ for all tiers without overwriting prior runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tiers-dir", type=str, default=TIERS_DIR,
                        help="Folder containing astrodeep_* tier CSVs.")
    args = parser.parse_args()

    ensure_dirs()

    # Unique run folder to avoid overwrites
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(COORDS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=False)

    # Discover inputs
    csvs = [os.path.join(args.tiers_dir, f) for f in os.listdir(args.tiers_dir)
            if f.lower().endswith(".csv") and f.lower().startswith("astrodeep_")]
    csvs.sort()

    if not csvs:
        raise SystemExit(f"No tier CSVs found in {args.tiers_dir}. Run Step 7 first.")

    print(f"[Step 8] Writing coordinates to: {out_dir}")
    print(f"Found {len(csvs)} tier files.")

    manifest_rows = []
    for path in csvs:
        stats = process_file(path, out_dir)
        manifest_rows.append(stats)
        print(f"  - {os.path.basename(path)}: in={stats['rows_in']:,}, ok={stats['rows_ok']:,} -> {os.path.basename(stats['output'])}")

    # Save manifest CSV
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(RESULTS_DIR, f"step8_coord_manifest_{run_id}.csv")
    manifest_df.to_csv(manifest_csv, index=False)

    # Write summary TXT
    lines = []
    lines.append("=== Step 8: Coordinate Conversion Summary ===\n")
    lines.append(f"Run folder: {out_dir}")
    lines.append(f"Tiers dir:  {args.tiers_dir}")
    lines.append(f"Files processed: {len(csvs)}\n")
    for s in manifest_rows:
        lines.append(f"{os.path.basename(s['input']):<30}  in={s['rows_in']:<8} ok={s['rows_ok']:<8} "
                     f"r[min/med/max] ~ [{s['r_min']:.1f}, {s['r_median']:.1f}, {s['r_max']:.1f}] Mpc"
                     if s['rows_ok'] > 0 else
                     f"{os.path.basename(s['input']):<30}  in={s['rows_in']:<8} ok=0 (no valid rows)")
    lines.append("")
    summary_txt = os.path.join(RESULTS_DIR, f"step8_coord_summary_{run_id}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nManifest: {manifest_csv}")
    print(f"Summary:  {summary_txt}")

if __name__ == "__main__":
    main()
