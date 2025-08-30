# REVIEW_GUIDE.md

## Purpose
This guide documents the exact sequence of steps to reproduce the results and figures in the paper:

**JWST High Redshift Galaxy Clustering: Evidence for Accelerated Early Structure Formation**

The workflow is deterministic and modular: each script produces outputs archived in `results/` and figures in `figures/`.  
Optional diagnostics and appendix-style plots are noted clearly.

---

## 0) One-time environment setup (deterministic)

    python -V
    python -m venv ..\.venv
    ..\.venv\Scripts\activate
    pip install -r requirements.txt
    pip install graphviz

Determinism knobs (per session):

    set PYTHONHASHSEED=0
    set NUMEXPR_MAX_THREADS=1

---

## 0.5) Create folders (one-time)

    python tools\make_folders.py

This will build the required tree:

- data_raw\astrodeep-jwst\ASTRODEEP-JWST_optap\
- data_raw\astrodeep-jwst\ASTRODEEP-JWST_photoz\
- data_processed\{coords,fields,tiers}\
- results\{step9,step9b,step9c,step10,step11,step11b,step11c,step12}\
- figures\{publication,methods}\
- docs\

---

## 1) Data placement

Download the public ASTRODEEP–JWST catalogs from:

    https://www.astrodeep.eu/astrodeep-jwst/

You need both:
- ASTRODEEP-JWST_optap/   (photometry catalogs, FITS files)
- ASTRODEEP-JWST_photoz/  (photometric redshift catalogs, FITS files)

Place them into:

    data_raw\astrodeep-jwst\ASTRODEEP-JWST_optap\*.fits
    data_raw\astrodeep-jwst\ASTRODEEP-JWST_photoz\*.fits
    
---

## 2) Build master catalog + redshift tiers

    python analysis\step4_prepare_catalogs.py
    python analysis\step6_z_hists.py
    python analysis\step7_hist_tiers.py

Outputs: data_processed\astrodeep_master.csv, tier CSVs, histograms in figures\.

---

## 3) (Optional) Comoving coordinates

    python analysis\make_coords_astrodeep.py

Adds XYZ columns for speedup.

---

## 4) Pair counts + correlation function ξ(d)

    python analysis\pairwise_astrodeep.py --all-bins
    python analysis\pairwise_randoms.py --all-bins
    python analysis\xi_astrodeep.py --all-bins

Outputs: results\step9c\<run>\xi_<tier>.csv

Optional pretty panels:

    python analysis\plot_xi_tiers.py

---

## 5) Fourier power / periodicity check

    python analysis\fft_xi_power.py --all

Optional methods / appendix:

    python analysis\step09_anisotropy_fft.py
    python analysis\step10_ring_scan.py
    python analysis\make_harmonics_leakage_figure.py

---

## 6) Inter-field variance suite

    python analysis\step11_interfield_variance.py
    python analysis\step11b_interfield_variance_depthnorm.py
    python analysis\step11c_variance_qualitycuts.py

---

## 7) Photo-z perturbation (Monte Carlo)

    python analysis\step12a_resample_photoz.py
    python analysis\step12b_xi_photoz_mc_memsafe.py
    python analysis\step12c_publish_mc_summary.py

---

## 8) Null tests / significance

    python analysis\significance_z6p.py --n-sims 5000 --null-mode phase --d-col bin_right_Mpc --xi-col xi --zp-factor 8 --lambda-min 140 --lambda-max 220

---

## 9) Publication-quality figures

    python figures\fig_xi_photoz_panels.py
    python figures\fig_variance_cdf_panels.py
    python figures\fig_variance_publication.py
    python figures\fig_perfield_rank_qc.py
    python figures\plot_variance_figures.py

Outputs in figures\publication\.

---

## Notes
- **Appendix / optional figures**: anisotropy FFTs, ring scans, leakage demo, and xi-tier panels live in `analysis\` (Step 5).  
- **Main paper figures**: only those in Step 9 (above).  
- Reviewers can safely ignore `results/` and regenerate from scratch.
