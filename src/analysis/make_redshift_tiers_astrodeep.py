# === make_redshift_tiers_astrodeep.py ===
# Purpose:
#   Produce tiered redshift slices from the master catalog (z-only pipeline).
#   - Loads data_processed/astrodeep_master.csv
#   - Normalizes 'zphot' and 'field' columns
#   - Writes per-tier CSVs for z ≥ {4,6,8,10}
#   - Writes binned range CSVs: [4,6), [6,8), [8,10), [10,20]
#   - Saves per-tier per-field counts + a summary TXT
#   - Saves simple histograms (overall) per tier
#
# Run (from PS C:\JWST-Mature-Galaxies\src>):
#   python analysis\make_redshift_tiers_astrodeep.py
#   # Optional: custom tiers (comma-separated):
#   python analysis\make_redshift_tiers_astrodeep.py --tiers 4,5,6,8,10
#
# Outputs:
#   data_processed/tiers/astrodeep_z4p.csv
#   data_processed/tiers/astrodeep_z6p.csv
#   data_processed/tiers/astrodeep_z8p.csv
#   data_processed/tiers/astrodeep_z10p.csv
#   data_processed/tiers/astrodeep_z4_6.csv
#   data_processed/tiers/astrodeep_z6_8.csv
#   data_processed/tiers/astrodeep_z8_10.csv
#   data_processed/tiers/astrodeep_z10_20.csv
#
#   results/step7_tier_summary.txt
#   results/step7_counts_z4p.csv, step7_counts_z6p.csv, step7_counts_z8p.csv, step7_counts_z10p.csv
#
#   figures/step7_hist_z4p.png, ... (one per ≥ tier)
#
# Notes:
#   - This is redshift-only (no SNR/stellarity yet). That’s intentional at this stage.

import os
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths (project-root aware) ----
PROJ_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROC   = os.path.join(PROJ_ROOT, "data_processed")
TIERS_DIR   = os.path.join(DATA_PROC, "tiers")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
FIG_DIR     = os.path.join(PROJ_ROOT, "figures")

MASTER_CSV  = os.path.join(DATA_PROC, "astrodeep_master.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "step7_tier_summary.txt")

DEFAULT_TIERS = [4.0, 6.0, 8.0, 10.0]   # define z ≥ thresholds
RANGE_TOP     = 20.0                    # upper bound for the last range (10–20)

def ensure_dirs():
    for d in [TIERS_DIR, RESULTS_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

def normalize_master(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}. Run Step 4 first.")

    df = pd.read_csv(path)

    # Normalize redshift column to 'zphot'
    z_alts = ["zphot", "z_phot", "z", "z_best", "photoz", "photo_z", "z_b"]
    z_col = next((c for c in z_alts if c in df.columns), None)
    if z_col is None:
        raise ValueError("No redshift column found (expected one of: " + ", ".join(z_alts) + ").")
    if z_col != "zphot":
        df = df.rename(columns={z_col: "zphot"})

    # Normalize field column to 'field'
    field_alts = ["field", "field_optap", "field_photoz", "FIELD"]
    f_col = next((c for c in field_alts if c in df.columns), None)
    if f_col is None:
        raise ValueError("No field column present (looked for field, field_optap, field_photoz, FIELD).")
    if f_col != "field":
        df = df.rename(columns={f_col: "field"})

    # Clean z
    df["zphot"] = pd.to_numeric(df["zphot"], errors="coerce")
    df = df.dropna(subset=["zphot"]).reset_index(drop=True)
    return df

def save_histogram(zvals: np.ndarray, title: str, out_png: str, bins: int = 50):
    if len(zvals) == 0:
        return
    plt.figure()
    plt.hist(zvals, bins=bins)
    plt.xlabel("zphot")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def per_field_counts(df: pd.DataFrame) -> pd.DataFrame:
    # expects 'field' and 'zphot' present
    return df.groupby("field", as_index=False)["zphot"].count().rename(columns={"zphot": "count"})

def write_counts_and_hist(df: pd.DataFrame, label: str):
    """Save per-field counts CSV and histogram for a given ≥ tier."""
    counts = per_field_counts(df)
    counts_path = os.path.join(RESULTS_DIR, f"step7_counts_{label}.csv")
    counts.to_csv(counts_path, index=False)

    # Overall histogram
    hist_png = os.path.join(FIG_DIR, f"step7_hist_{label}.png")
    save_histogram(df["zphot"].to_numpy(), f"z distribution ({label})", hist_png)

    return counts_path, hist_png, counts

def main():
    parser = argparse.ArgumentParser(
        description="Create tiered redshift slices from astrodeep_master.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tiers", type=str, default="",
                        help="Comma-separated z thresholds for '≥' tiers (e.g., 4,6,8,10). If empty, defaults are used.")
    parser.add_argument("--range-top", type=float, default=RANGE_TOP,
                        help="Upper bound for the last [z_last, range_top] bin.")
    args = parser.parse_args()

    ensure_dirs()
    df = normalize_master(MASTER_CSV)

    # Parse tiers
    if args.tiers.strip():
        try:
            tiers = [float(x) for x in args.tiers.split(",")]
        except Exception:
            raise SystemExit("Could not parse --tiers. Use a comma-separated list of numbers, e.g. 4,6,8,10")
    else:
        tiers = DEFAULT_TIERS[:]

    tiers = sorted(set(tiers))
    if len(tiers) < 2:
        # we need at least two to form 3 range bins later; but ≥ tiers still fine
        pass

    # --- Create ≥ tier CSVs ---
    summary_lines = []
    summary_lines.append("=== Step 7: Redshift Tier Slices ===\n")
    summary_lines.append(f"Master input: {MASTER_CSV}")
    summary_lines.append(f"Tiers (≥): {tiers}")
    summary_lines.append(f"Range top for last bin: {args.range_top}")
    summary_lines.append("")

    ge_outputs = []   # (label, path_csv, path_counts, path_hist, nrows)
    for thr in tiers:
        df_ge = df[df["zphot"] >= thr].copy()
        label = f"z{int(thr)}p" if thr.is_integer() else f"z{thr}p".replace(".","p")
        out_csv = os.path.join(TIERS_DIR, f"astrodeep_{label}.csv")
        df_ge.to_csv(out_csv, index=False)
        counts_csv, hist_png, counts = write_counts_and_hist(df_ge, label)
        ge_outputs.append((label, out_csv, counts_csv, hist_png, len(df_ge)))

    # --- Create range bins between consecutive tiers and last bin to top ---
    range_outputs = []  # (label, path_csv, nrows)
    if len(tiers) >= 2:
        for a, b in zip(tiers[:-1], tiers[1:]):
            df_bin = df[(df["zphot"] >= a) & (df["zphot"] < b)].copy()
            label = f"z{int(a)}_{int(b)}" if a.is_integer() and b.is_integer() else f"z{a}_{b}".replace(".","p")
            out_csv = os.path.join(TIERS_DIR, f"astrodeep_{label}.csv")
            df_bin.to_csv(out_csv, index=False)
            range_outputs.append((label, out_csv, len(df_bin)))
        # tail bin
        last = tiers[-1]
        top = float(args.range_top)
        df_tail = df[(df["zphot"] >= last) & (df["zphot"] <= top)].copy()
        label_tail = f"z{int(last)}_{int(top)}" if last.is_integer() and top.is_integer() else f"z{last}_{top}".replace(".","p")
        out_tail = os.path.join(TIERS_DIR, f"astrodeep_{label_tail}.csv")
        df_tail.to_csv(out_tail, index=False)
        range_outputs.append((label_tail, out_tail, len(df_tail)))

    # --- Write summary TXT ---
    summary_lines.append("≥ Tier outputs:")
    for label, csv_path, counts_path, hist_path, nrows in ge_outputs:
        summary_lines.append(f"  - {label:<8}  rows={nrows:,}  csv={csv_path}")
        summary_lines.append(f"               counts={counts_path}")
        summary_lines.append(f"               hist={hist_path}")
    summary_lines.append("")

    if range_outputs:
        summary_lines.append("Range-bin outputs:")
        for label, csv_path, nrows in range_outputs:
            summary_lines.append(f"  - {label:<10} rows={nrows:,}  csv={csv_path}")
        summary_lines.append("")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
