# === preprocess_astrodeep.py ===
# Purpose:
#   Assemble ASTRODEEP–JWST per-field FITS (optap + photoz) into per-field merged CSVs,
#   then concatenate to a master CSV and produce a filtered high-z galaxy candidate set.
#
# Inputs (fixed to your layout):
#   data_raw/astrodeep-jwst/ASTRODEEP-JWST_optap/*.fits
#   data_raw/astrodeep-jwst/ASTRODEEP-JWST_photoz/*.fits
#
# Outputs:
#   data_processed/fields/<FIELD>_merged.csv
#   data_processed/astrodeep_master.csv
#   data_processed/astrodeep_candidates.csv
#   results/step4_summary.txt
#   figures/step4_z_hist.png
#   figures/step4_mass_hist.png
#
# Usage (from project root C:\JWST-Mature-Galaxies):
#   python src\analysis\preprocess_astrodeep.py --remove-stars
#   python src\analysis\preprocess_astrodeep.py --zmin 5.0 --snrmin 5 --remove-stars
#   python src\analysis\preprocess_astrodeep.py --scan-only   # just show what files/fields are detected

import os
import glob
import argparse
import textwrap
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from astropy.table import Table
except ImportError:
    raise SystemExit(
        "Astropy is required for reading FITS. Install it with:\n"
        "    pip install astropy\n"
    )

# ---------------- Paths (project-root aware) ----------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_ROOT       = os.path.join(PROJ_ROOT, "data_raw", "astrodeep-jwst")
RAW_OPTAP_DIR  = os.path.join(RAW_ROOT, "ASTRODEEP-JWST_optap")
RAW_PHOTOZ_DIR = os.path.join(RAW_ROOT, "ASTRODEEP-JWST_photoz")

DATA_PROCESSED = os.path.join(PROJ_ROOT, "data_processed")
FIELDS_DIR     = os.path.join(DATA_PROCESSED, "fields")
RESULTS_DIR    = os.path.join(PROJ_ROOT, "results")
FIGURES_DIR    = os.path.join(PROJ_ROOT, "figures")

MASTER_CSV    = os.path.join(DATA_PROCESSED, "astrodeep_master.csv")
CAND_CSV      = os.path.join(DATA_PROCESSED, "astrodeep_candidates.csv")
SUMMARY_TXT   = os.path.join(RESULTS_DIR, "step4_summary.txt")
Z_HIST_PNG    = os.path.join(FIGURES_DIR, "step4_z_hist.png")
MASS_HIST_PNG = os.path.join(FIGURES_DIR, "step4_mass_hist.png")

# ---------------- Helpers ----------------
def ensure_dirs():
    for d in [DATA_PROCESSED, FIELDS_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

def read_fits(path: str) -> pd.DataFrame:
    tbl = Table.read(path, format="fits")
    return tbl.to_pandas()

def guess_key_columns(df_optap: pd.DataFrame, df_photoz: pd.DataFrame) -> Optional[Tuple[str, str]]:
    candidates = ["ID","Id","id","OBJID","objid","SOURCE_ID","source_id","NUMBER","Number","number"]
    optap_map  = {c.lower(): c for c in df_optap.columns}
    photoz_map = {c.lower(): c for c in df_photoz.columns}

    for c in candidates:
        cl = c.lower()
        if cl in optap_map and cl in photoz_map:
            return optap_map[cl], photoz_map[cl]

    common  = set(optap_map.keys()).intersection(photoz_map.keys())
    id_like = [x for x in common if x in {k.lower() for k in candidates}]
    if len(id_like) == 1:
        x = id_like[0]
        return optap_map[x], photoz_map[x]
    if "id" in optap_map and "id" in photoz_map:
        return optap_map["id"], photoz_map["id"]
    return None

def merge_field(field_name: str, optap_path: str, photoz_path: str) -> pd.DataFrame:
    df_optap  = read_fits(optap_path);  df_optap["field"]  = field_name
    df_photoz = read_fits(photoz_path); df_photoz["field"] = field_name

    keys = guess_key_columns(df_optap, df_photoz)
    if keys is None:
        df_optap  = df_optap.reset_index().rename(columns={"index": "row_index_optap"})
        df_photoz = df_photoz.reset_index().rename(columns={"index": "row_index_photoz"})
        merged = pd.merge(df_optap, df_photoz, on="field", how="outer", suffixes=("_optap","_photoz"))
        merged["merge_note"] = "no_id_key_outer_join_on_field"
    else:
        lk, rk = keys
        merged = pd.merge(df_optap, df_photoz, left_on=lk, right_on=rk, how="left", suffixes=("_optap","_photoz"))
        merged["merge_note"] = f"left_join_on_{lk}__{rk}"
    return merged

def find_fields() -> list[str]:
    optap  = {os.path.basename(p).replace("_optap.fits","")   for p in glob.glob(os.path.join(RAW_OPTAP_DIR,  "*_optap.fits"))}
    photoz = {os.path.basename(p).replace("_photoz.fits","")  for p in glob.glob(os.path.join(RAW_PHOTOZ_DIR, "*_photoz.fits"))}
    return sorted(optap.intersection(photoz))

def resolve_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lut = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lut:
            return lut[name.lower()]
    return None

def detect_columns(df: pd.DataFrame) -> dict:
    z_candidates    = ["zphot","z_phot","z_best","z","photoz","photo_z","zphot_best","z_b"]
    mass_candidates = ["logm","logmass","logmstar","log_mstar","mstar_log","mass_log","stellar_mass_log"]
    star_candidates = ["class_star","stellarity","star_flag","star","sg","is_star"]
    snr_candidates  = ["snr_f200w","snr_f150w","snr_f277w","snr_best","snr","f200w_snr","f150w_snr"]
    mag_candidates  = ["mag_f200w","mag_f150w","f200w_mag","f150w_mag","mag"]
    return {
        "z": resolve_column(df, z_candidates),
        "logm": resolve_column(df, mass_candidates),
        "star_flag": resolve_column(df, star_candidates),
        "snr": resolve_column(df, snr_candidates),
        "mag": resolve_column(df, mag_candidates),
    }

def apply_filters(df: pd.DataFrame, cols: dict, zmin: float, snrmin: Optional[float],
                  remove_stars: bool, max_stellarity: Optional[float]) -> tuple[pd.DataFrame, dict]:
    stats = {}; mask = np.ones(len(df), dtype=bool)

    if cols["z"] and cols["z"] in df.columns:
        z = pd.to_numeric(df[cols["z"]], errors="coerce")
        z_mask = np.isfinite(z) & (z >= zmin)
        stats["z_col"] = cols["z"]; stats["zmin"] = zmin; stats["z_pass"] = int(z_mask.sum())
        mask &= z_mask
    else:
        stats["z_col"] = None; stats["zmin"] = zmin; stats["z_pass"] = None

    if snrmin is not None and cols["snr"] and cols["snr"] in df.columns:
        snr = pd.to_numeric(df[cols["snr"]], errors="coerce")
        snr_mask = np.isfinite(snr) & (snr >= snrmin)
        stats["snr_col"] = cols["snr"]; stats["snrmin"] = snrmin; stats["snr_pass"] = int((mask & snr_mask).sum())
        mask &= snr_mask
    else:
        stats["snr_col"] = cols["snr"]; stats["snrmin"] = snrmin; stats["snr_pass"] = None

    removed_by_star = None
    if remove_stars and cols["star_flag"] and cols["star_flag"] in df.columns:
        series = df[cols["star_flag"]]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.between(0, 1, inclusive="both").sum() > 0.5 * len(df):
            thr = 0.9 if max_stellarity is None else float(max_stellarity)
            keep_mask = numeric <= thr
        else:
            as_str = series.astype(str).str.lower()
            keep_mask = ~as_str.isin(["1","true","t","yes"])
        before = int(mask.sum()); mask &= keep_mask.fillna(True).to_numpy(); after = int(mask.sum())
        removed_by_star = before - after

    out = df.loc[mask].copy()

    head = [c for c in ["field","ID","id","objid","SOURCE_ID","ra","dec","RA","DEC"] if c in out.columns]
    ordered, seen = [], set()
    for c in head + [cols["z"], cols["logm"], cols["snr"], cols["mag"], cols["star_flag"]]:
        if c and c in out.columns and c not in seen:
            ordered.append(c); seen.add(c)
    out = out[ordered + [c for c in out.columns if c not in seen]]

    stats.update({
        "input_rows": len(df),
        "output_rows": len(out),
        "logm_col": cols["logm"],
        "mag_col": cols["mag"],
        "star_col": cols["star_flag"],
        "star_removed": removed_by_star,
    })
    return out, stats

def save_histograms(df: pd.DataFrame, cols: dict):
    if cols["z"] and cols["z"] in df.columns:
        z = pd.to_numeric(df[cols["z"]], errors="coerce"); z = z[np.isfinite(z)]
        if len(z) > 0:
            plt.figure(); plt.hist(z, bins=40)
            plt.xlabel(cols["z"]); plt.ylabel("Count"); plt.title("Redshift distribution (filtered)")
            plt.tight_layout(); plt.savefig(Z_HIST_PNG, dpi=150); plt.close()
    if cols["logm"] and cols["logm"] in df.columns:
        m = pd.to_numeric(df[cols["logm"]], errors="coerce"); m = m[np.isfinite(m)]
        if len(m) > 0:
            plt.figure(); plt.hist(m, bins=40)
            plt.xlabel(cols["logm"]); plt.ylabel("Count"); plt.title("log10(M*/Msun) distribution (filtered)")
            plt.tight_layout(); plt.savefig(MASS_HIST_PNG, dpi=150); plt.close()

def write_summary(path: str, msg: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

def scan_inputs() -> tuple[list[str], list[str], list[str]]:
    optap_files  = glob.glob(os.path.join(RAW_OPTAP_DIR,  "*_optap.fits"))
    photoz_files = glob.glob(os.path.join(RAW_PHOTOZ_DIR, "*_photoz.fits"))
    optap_fields  = sorted({os.path.basename(p).replace("_optap.fits","")  for p in optap_files})
    photoz_fields = sorted({os.path.basename(p).replace("_photoz.fits","") for p in photoz_files})
    overlap = sorted(set(optap_fields).intersection(photoz_fields))

    print("=== Input scan ===")
    print(f"optap dir:   {RAW_OPTAP_DIR}")
    print(f"photoz dir:  {RAW_PHOTOZ_DIR}")
    print(f"optap files:  {len(optap_files)}")
    print(f"photoz files: {len(photoz_files)}")
    if len(optap_files) > 0:
        print("  optap fields:", ", ".join(optap_fields))
    if len(photoz_files) > 0:
        print("  photoz fields:", ", ".join(photoz_fields))
    print("  overlap fields:", ", ".join(overlap) if overlap else "(none)")
    print("------------------")
    return optap_fields, photoz_fields, overlap

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Assemble per-field ASTRODEEP FITS (optap+photoz), then filter high-z galaxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--zmin", type=float, default=4.0, help="Minimum photometric redshift to keep (>=).")
    parser.add_argument("--snrmin", type=float, default=None, help="Minimum SNR (>=) if an SNR column exists.")
    parser.add_argument("--remove-stars", action="store_true", help="Attempt to remove stars by star/stellarity column.")
    parser.add_argument("--max-stellarity", type=float, default=0.9, help="If stellarity in [0..1], keep <= this value.")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and report detected inputs, then exit.")
    args = parser.parse_args()

    ensure_dirs()

    # Always show a quick scan so path issues are obvious
    _, _, overlap = scan_inputs()
    if args.scan_only:
        return

    if not overlap:
        raise SystemExit("No matching fields found across *_optap.fits and *_photoz.fits in data_raw/astrodeep-jwst.")

    merged_pieces = []
    per_field_counts = []
    for fld in overlap:
        optap_path  = os.path.join(RAW_OPTAP_DIR,  f"{fld}_optap.fits")
        photoz_path = os.path.join(RAW_PHOTOZ_DIR, f"{fld}_photoz.fits")
        print(f"[Assemble] {fld}"); print(f"  - {optap_path}"); print(f"  - {photoz_path}")
        merged = merge_field(fld, optap_path, photoz_path)
        out_csv = os.path.join(FIELDS_DIR, f"{fld}_merged.csv")
        merged.to_csv(out_csv, index=False)
        print(f"  -> wrote {out_csv}  ({len(merged):,} rows, {len(merged.columns)} cols)")
        merged_pieces.append(merged); per_field_counts.append((fld, len(merged)))

    master = pd.concat(merged_pieces, ignore_index=True)
    master.to_csv(MASTER_CSV, index=False)
    print(f"[Master] wrote {MASTER_CSV}  ({len(master):,} rows, {len(master.columns)} cols)")

    cols = detect_columns(master)
    filtered, stats = apply_filters(master, cols, args.zmin, args.snrmin, args.remove_stars, args.max_stellarity)
    filtered.to_csv(CAND_CSV, index=False)
    print(f"[Candidates] wrote {CAND_CSV}  ({len(filtered):,} rows)")

    # Figures
    save_histograms(filtered, cols)

    # Summary
    per_field_block = "\n".join([f"  - {fld}: {cnt:,} rows" for fld, cnt in per_field_counts])
    summary = textwrap.dedent(f"""
    === Step 4: Per-field Assembly & Global Filtering Summary ===

    Project root: {PROJ_ROOT}

    Inputs:
      optap dir:   {RAW_OPTAP_DIR}
      photoz dir:  {RAW_PHOTOZ_DIR}

    Fields assembled:
    {per_field_block if per_field_block else '  (none)'}

    Master:
      file:   {MASTER_CSV}
      shape:  {len(master):,} rows x {len(master.columns)} cols

    Detected columns (global):
      redshift:   {cols.get('z')}
      log mass:   {cols.get('logm')}
      SNR:        {cols.get('snr')}
      magnitude:  {cols.get('mag')}
      star_flag:  {cols.get('star_flag')}

    Filters applied (global candidates):
      z >= {stats.get('zmin')}
      SNR >= {stats.get('snrmin')}
      remove_stars: {args.remove_stars}  (star col: {stats.get('star_col')}, removed: {stats.get('star_removed')})

    Output files:
      per-field CSVs:  {FIELDS_DIR}\\<FIELD>_merged.csv
      master CSV:      {MASTER_CSV}
      candidates CSV:  {CAND_CSV}
      z histogram:     {Z_HIST_PNG if os.path.isfile(Z_HIST_PNG) else '(not created)'}
      mass histogram:  {MASS_HIST_PNG if os.path.isfile(MASS_HIST_PNG) else '(not created)'}
    """).strip("\n")
    write_summary(SUMMARY_TXT, summary)

if __name__ == "__main__":
    main()
