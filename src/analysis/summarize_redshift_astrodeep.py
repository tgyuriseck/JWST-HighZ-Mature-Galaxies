# === summarize_redshift_astrodeep.py ===
# Purpose:
#   Summarize the high-z sample by field using zphot only.
#   - Loads data_processed/astrodeep_master.csv (from Step 4)
#   - Filters by zphot >= zmin (default 4.0)
#   - Writes per-field counts and global stats
#   - Saves basic histograms
#
# Run (from C:\JWST-Mature-Galaxies\src):
#   python analysis\summarize_redshift_astrodeep.py
#   python analysis\summarize_redshift_astrodeep.py --zmin 5.0 --bins 60
#
# Outputs:
#   results/step6_redshift_summary.txt
#   results/step6_redshift_counts_by_field.csv
#   figures/step6_z_hist_overall.png
#   figures/step6_z_hist_by_field.png  (simple stacked histogram by field)

import os
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths (project-root aware) ----
PROJ_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROC   = os.path.join(PROJ_ROOT, "data_processed")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
FIG_DIR     = os.path.join(PROJ_ROOT, "figures")

MASTER_CSV  = os.path.join(DATA_PROC, "astrodeep_master.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "step6_redshift_summary.txt")
COUNTS_CSV  = os.path.join(RESULTS_DIR, "step6_redshift_counts_by_field.csv")
HIST_ALL    = os.path.join(FIG_DIR, "step6_z_hist_overall.png")
HIST_BYF    = os.path.join(FIG_DIR, "step6_z_hist_by_field.png")

def ensure_dirs():
    for d in [RESULTS_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

def load_master():
    if not os.path.isfile(MASTER_CSV):
        raise FileNotFoundError(f"Missing {MASTER_CSV}. Run Step 4 first.")
    df = pd.read_csv(MASTER_CSV)

    # Normalize redshift column
    z_alternates = ["zphot", "z_phot", "z", "z_best", "photoz", "photo_z", "z_b"]
    z_col = next((c for c in z_alternates if c in df.columns), None)
    if z_col is None:
        raise ValueError("No redshift column found (expected one of: " + ", ".join(z_alternates) + ").")
    if z_col != "zphot":
        df = df.rename(columns={z_col: "zphot"})

    # Normalize field column (merge created field_optap/field_photoz)
    field_alternates = ["field", "field_optap", "field_photoz"]
    field_col = next((c for c in field_alternates if c in df.columns), None)
    if field_col is None:
        # Last resort: if a column literally named 'FIELD' exists
        if "FIELD" in df.columns:
            field_col = "FIELD"
        else:
            raise ValueError("No field column present (looked for field, field_optap, field_photoz).")
    if field_col != "field":
        df = df.rename(columns={field_col: "field"})

    return df

def summarize(df, zmin: float, bins: int):
    # Filter by z
    z = pd.to_numeric(df["zphot"], errors="coerce")
    df = df.assign(zphot=z).dropna(subset=["zphot"])
    highz = df[df["zphot"] >= zmin].copy()

    # Per-field counts
    counts = highz.groupby("field", as_index=False)["zphot"].count().rename(columns={"zphot": "count"})

    # Global stats
    total_all = len(df)
    total_sel = len(highz)
    frac = (total_sel / total_all) if total_all > 0 else np.nan
    stats = {
        "zmin": zmin,
        "total_rows_all": total_all,
        "total_rows_selected": total_sel,
        "fraction_selected": frac,
        "z_min_selected": float(highz["zphot"].min()) if total_sel > 0 else np.nan,
        "z_median_selected": float(highz["zphot"].median()) if total_sel > 0 else np.nan,
        "z_mean_selected": float(highz["zphot"].mean()) if total_sel > 0 else np.nan,
        "z_max_selected": float(highz["zphot"].max()) if total_sel > 0 else np.nan,
    }

    # Histograms (selected only)
    if total_sel > 0:
        plt.figure()
        plt.hist(highz["zphot"], bins=bins)
        plt.xlabel("zphot")
        plt.ylabel("Count")
        plt.title(f"Redshift distribution (zphot ≥ {zmin})")
        plt.tight_layout()
        plt.savefig(HIST_ALL, dpi=150)
        plt.close()

        plt.figure()
        fields = sorted(highz["field"].unique().tolist())
        data_series = [highz.loc[highz["field"] == f, "zphot"].values for f in fields]
        plt.hist(data_series, bins=bins, stacked=True, label=fields)
        plt.xlabel("zphot")
        plt.ylabel("Count (stacked)")
        plt.title(f"Redshift by field (zphot ≥ {zmin})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(HIST_BYF, dpi=150)
        plt.close()

    return highz, counts, stats

def write_outputs(counts: pd.DataFrame, stats: dict):
    counts.to_csv(COUNTS_CSV, index=False)

    summary = textwrap.dedent(f"""
    === Step 6: Redshift Summary (zphot ≥ {stats['zmin']}) ===

    Input master: {MASTER_CSV}

    Global:
      total rows (all):     {stats['total_rows_all']:,}
      total rows (selected):{stats['total_rows_selected']:,}
      fraction selected:    {stats['fraction_selected']:.4f}

    Selected z stats:
      z_min:    {stats['z_min_selected']:.4f}
      z_median: {stats['z_median_selected']:.4f}
      z_mean:   {stats['z_mean_selected']:.4f}
      z_max:    {stats['z_max_selected']:.4f}

    Per-field counts written to:
      {COUNTS_CSV}

    Figures:
      overall histogram:    {HIST_ALL if os.path.isfile(HIST_ALL) else '(not created)'}
      per-field histogram:  {HIST_BYF if os.path.isfile(HIST_BYF) else '(not created)'}
    """).strip("\n")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(summary)

def main():
    parser = argparse.ArgumentParser(
        description="Summarize high-z sample by field (zphot-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--zmin", type=float, default=4.0, help="Minimum zphot (inclusive).")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histograms.")
    args = parser.parse_args()

    ensure_dirs()
    df = load_master()
    _, counts, stats = summarize(df, zmin=args.zmin, bins=args.bins)
    write_outputs(counts, stats)

if __name__ == "__main__":
    main()
