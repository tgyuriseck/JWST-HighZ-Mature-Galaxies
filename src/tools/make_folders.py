#!/usr/bin/env python3
"""
Build the minimal folder tree a reviewer needs.

Run from:
PS C:\JWST-HighZ-Mature-Galaxies\src>  python tools\make_folders.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ...\JWST-HighZ-Mature-Galaxies
dirs = [
    ROOT / "data_raw" / "astrodeep-jwst" / "ASTRODEEP-JWST_optap",
    ROOT / "data_raw" / "astrodeep-jwst" / "ASTRODEEP-JWST_photoz",
    ROOT / "data_processed" / "coords",
    ROOT / "data_processed" / "fields",
    ROOT / "data_processed" / "tiers",
    ROOT / "results" / "step9",
    ROOT / "results" / "step9b",
    ROOT / "results" / "step9c",
    ROOT / "results" / "step10",
    ROOT / "results" / "step11",
    ROOT / "results" / "step11b",
    ROOT / "results" / "step11c",
    ROOT / "results" / "step12",
    ROOT / "figures" / "publication",
    ROOT / "figures" / "methods",
    ROOT / "docs",
]

def main():
    made = 0
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        made += 1
    print(f"[OK] Ensured {made} folders.")
    print(r"[NEXT] Put ASTRODEEP FITS into: data_raw\astrodeep-jwst\ASTRODEEP-JWST_optap\ and ASTRODEEP-JWST_photoz\.")
    print(r"[NEXT] Then follow REVIEW_GUIDE.md (Step 2: ingest).")

if __name__ == "__main__":
    main()
