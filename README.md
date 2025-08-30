# JWST High Redshift Galaxy Clustering

This repository contains the analysis pipeline and scripts accompanying the paper:

**JWST High Redshift Galaxy Clustering: Evidence for Accelerated Early Structure Formation**

---

## Contents

- analysis/ – Core analysis scripts (catalog ingestion, pair counts, correlation functions, variance, photo-z Monte Carlo, null tests).
- figures/ – Scripts to generate publication-quality figures.
- tools/ – Utilities (make_folders.py to set up the folder tree, rename_for_release.py for maintainers).
- REVIEW_GUIDE.md – Step-by-step instructions to reproduce the results from scratch.
- .gitignore – Keeps large or temporary files out of version control.

---

## Quick Start (Reviewers)

1. Clone the repo:

       git clone https://github.com/<your-username>/JWST-HighZ-Mature-Galaxies.git
       cd JWST-HighZ-Mature-Galaxies/src

2. Set up a virtual environment and install dependencies:

       python -m venv ..\.venv
       ..\.venv\Scripts\activate
       pip install -r requirements.txt
       pip install graphviz

3. Create the folder tree:

       python tools\make_folders.py

4. Download ASTRODEEP–JWST public catalogs:

   ASTRODEEP-JWST data portal:
   https://www.astrodeep.eu/astrodeep-jwst/

   Required subfolders:
     ASTRODEEP-JWST_optap/   → place all photometry catalog FITS here
     ASTRODEEP-JWST_photoz/  → place all photo-z catalog FITS here

   Place these under:
     data_raw\astrodeep-jwst\

5. Follow the REVIEW_GUIDE.md for the complete pipeline (Steps 2–9).

---

## Notes

- Only scripts are included in the repo. Large intermediate products (results/, data_processed/) are ignored by design.
- Optional / appendix-style figures (anisotropy, ring scans, leakage tests) are available in analysis/ and documented in the guide.
- All results are reproducible from scratch following REVIEW_GUIDE.md.
