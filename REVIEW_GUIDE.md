# JWST High-Redshift Galaxy Clustering: Reproduction Guide

**Complete step-by-step instructions to reproduce all results from the paper**

---

## Overview

**What this reproduces:** All analysis, figures, and statistical results from the MNRAS paper  
**Total runtime:** 4-6 hours on a standard laptop  
**Disk space required:** ~60 GB (50 GB data + 10 GB outputs)  
**Platforms:** Windows (fully tested), Linux/macOS (with adaptations noted)

---

## Table of Contents

- [0. Environment Setup](#0-environment-setup) (5 minutes)
- [1. Data Download](#1-data-download) (1-2 hours)
- [2. Master Catalog Creation](#2-master-catalog-creation) (30 minutes)
- [3. Pair Counts & Correlation Functions](#3-pair-counts--correlation-functions) (1-2 hours)
- [4. Fourier Power Spectrum](#4-fourier-power-spectrum) (15 minutes)
- [5. Inter-Field Variance Analysis](#5-inter-field-variance-analysis) (2 hours)
- [6. Photo-z Monte Carlo](#6-photo-z-monte-carlo) (1 hour)
- [7. Significance Testing](#7-significance-testing) (30 minutes)
- [8. Publication Figures](#8-publication-figures) (15 minutes)
- [Verification Checklist](#verification-checklist)
- [Troubleshooting](#troubleshooting)

---

## 0. Environment Setup

**Runtime:** ~5 minutes

### 0.1 Create Virtual Environment

**Linux/macOS:**
```bash
cd JWST-HighZ-Mature-Galaxies
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
cd JWST-HighZ-Mature-Galaxies
python -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
cd JWST-HighZ-Mature-Galaxies
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 0.2 Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed astropy-7.0.1 numpy-1.25.1 scipy-1.15.2 ...
```

**Verify installation:**
```bash
python -c "import numpy, scipy, pandas, astropy; print('All imports successful')"
```

### 0.3 Create Directory Structure
```bash
cd src
python tools/make_folders.py
```

**Expected output:**
```
Created: ../data_raw/astrodeep-jwst/ASTRODEEP-JWST_photoz/
Created: ../data_raw/astrodeep-jwst/ASTRODEEP-JWST_optap/
Created: ../data_processed/fields/
Created: ../data_processed/tiers/
Created: ../results/step9/
...
```

---

## 1. Data Download

**Runtime:** 1-2 hours (depending on connection speed)  
**Download size:** ~45-50 GB

### 1.1 Download ASTRODEEP Catalogs

Visit: [https://www.astrodeep.eu/astrodeep-jwst/](https://www.astrodeep.eu/astrodeep-jwst/)

Download **both catalog types** for **all seven fields**:
- `*-photoz.fits` (photometric redshift catalogs)
- `*-optap.fits` (aperture-optimized photometry catalogs)

**Required files (14 total):**
- abell2744-photoz.fits & abell2744-optap.fits
- ceers-photoz.fits & ceers-optap.fits
- jades-gn-photoz.fits & jades-gn-optap.fits
- jades-gs-photoz.fits & jades-gs-optap.fits
- ngdeep-photoz.fits & ngdeep-optap.fits
- primer-cosmos-photoz.fits & primer-cosmos-optap.fits
- primer-uds-photoz.fits & primer-uds-optap.fits

### 1.2 Place Files in Correct Structure
```
data_raw/astrodeep-jwst/
├── ASTRODEEP-JWST_photoz/
│   ├── abell2744-photoz.fits
│   ├── ceers-photoz.fits
│   ├── jades-gn-photoz.fits
│   ├── jades-gs-photoz.fits
│   ├── ngdeep-photoz.fits
│   ├── primer-cosmos-photoz.fits
│   └── primer-uds-photoz.fits
└── ASTRODEEP-JWST_optap/
    ├── abell2744-optap.fits
    ├── ceers-optap.fits
    ├── jades-gn-optap.fits
    ├── jades-gs-optap.fits
    ├── ngdeep-optap.fits
    ├── primer-cosmos-optap.fits
    └── primer-uds-optap.fits
```

**Verify files are in place:**

**Linux/macOS:**
```bash
ls -lh ../data_raw/astrodeep-jwst/ASTRODEEP-JWST_photoz/*.fits | wc -l
ls -lh ../data_raw/astrodeep-jwst/ASTRODEEP-JWST_optap/*.fits | wc -l
```

**Windows (PowerShell):**
```powershell
(Get-ChildItem ..\data_raw\astrodeep-jwst\ASTRODEEP-JWST_photoz\*.fits).Count
(Get-ChildItem ..\data_raw\astrodeep-jwst\ASTRODEEP-JWST_optap\*.fits).Count
```

Both commands should output: `7`

---

## 2. Master Catalog Creation

**Runtime:** ~30 minutes  
**Working directory:** `src/`

### 2.1 Build Master Catalog
```bash
python analysis/preprocess_astrodeep.py
```

**Expected output:**
```
Processing ABELL2744...
Processing CEERS...
...
Master catalog saved: ../data_processed/astrodeep_master.csv
Total galaxies: 531173
```

### 2.2 Generate Redshift Summary
```bash
python analysis/summarize_redshift_astrodeep.py
```

**Expected output:**
```
Total objects with zphot >= 0: 531173
Redshift range: 0.00 to 20.45
Figures saved to: ../figures/
```

### 2.3 Create Redshift Tiers
```bash
python analysis/make_redshift_tiers_astrodeep.py
```

**Expected output:**
```
Created tier: z4_6 (61848 galaxies)
Created tier: z6_8 (47548 galaxies)
Created tier: z8_10 (29294 galaxies)
Created tier: z10_20 (7214 galaxies)
Created tier: z6p (84056 galaxies)
Tier CSVs saved to: ../data_processed/tiers/
```

**Verify tier files exist:**

**Linux/macOS:**
```bash
ls -lh ../data_processed/tiers/astrodeep_z*.csv
```

**Windows:**
```cmd
dir ..\data_processed\tiers\astrodeep_z*.csv
```

You should see 5 CSV files.

---

## 3. Pair Counts & Correlation Functions

**Runtime:** 1-2 hours (z4_6 tier is longest)  
**Working directory:** `src/`

### 3.1 Prepare Tiers for Photo-z Analysis

We need to ensure all tiers use `zphot` consistently by renaming any `zspec` columns to `zspec_unused`.

**Windows (PowerShell) - TESTED METHOD:**
```powershell
$tiers = @(
  "..\data_processed\tiers\astrodeep_z4_6.csv",
  "..\data_processed\tiers\astrodeep_z6_8.csv",
  "..\data_processed\tiers\astrodeep_z8_10.csv",
  "..\data_processed\tiers\astrodeep_z10_20.csv",
  "..\data_processed\tiers\astrodeep_z6p.csv"
)
foreach ($f in $tiers) {
  if (Test-Path $f) {
    Copy-Item $f "$f.bak" -Force
    $lines = Get-Content $f
    $lines[0] = $lines[0] -replace '\bzspec\b','zspec_unused'
    Set-Content -Path $f -Encoding UTF8 -NoNewline -Value ($lines -join "`r`n")
    Write-Host "Header updated in $f"
  } else {
    Write-Host "Missing $f" -ForegroundColor Yellow
  }
}
```

**Expected output:**
```
Header updated in ..\data_processed\tiers\astrodeep_z4_6.csv
Header updated in ..\data_processed\tiers\astrodeep_z6_8.csv
Header updated in ..\data_processed\tiers\astrodeep_z8_10.csv
Header updated in ..\data_processed\tiers\astrodeep_z10_20.csv
Header updated in ..\data_processed\tiers\astrodeep_z6p.csv
```

---

**Linux/Mac - UNTESTED ALTERNATIVE:**

If you don't have access to Windows PowerShell, use one of these alternatives:

**Option A: Using sed (one-liner):**
```bash
for f in ../data_processed/tiers/astrodeep_z*.csv; do
  sed -i.bak '1s/\bzspec\b/zspec_unused/' "$f"
done
```

**Option B: Using Python (one-liner):**
```bash
python -c "import pandas as pd; import glob; [pd.read_csv(f).rename(columns={'zspec':'zspec_unused'}).to_csv(f,index=False) for f in glob.glob('../data_processed/tiers/astrodeep_z*.csv')]"
```

**Note:** These Linux/Mac alternatives are provided for convenience but have not been validated by the author. They should produce equivalent results (renaming the `zspec` column header to `zspec_unused` in all tier CSV files).

If you encounter issues, you can also manually edit the CSV files: open each tier file in a text editor and change `zspec` to `zspec_unused` in the first line (header row).

---

### 3.2 Compute Data-Data (DD) Pairs

**Binning:** 1 → 250 Mpc in 5 Mpc steps (50 bins)

**Run z6p tier first** (needed for significance testing later):
```bash
python analysis/pairwise_astrodeep.py --tiers-glob "../data_processed/tiers/astrodeep_z6p.csv" --bin-min 1 --bin-max 250 --bin-width 5
```

**Then run the four range tiers:**
```bash
python analysis/pairwise_astrodeep.py --tiers-glob "../data_processed/tiers/astrodeep_z*_*.csv" --bin-min 1 --bin-max 250 --bin-width 5
```

**Expected output:**
```
Processing tier: z4_6 (61848 galaxies)
Computing DD pairs...
Progress: [########################################] 100%
Saved: ../results/step9/run_YYYYMMDD_HHMMSS/DD_hist_z4_6.csv
...
```

**Runtime note:** z4_6 takes longest (~30-45 min on typical laptop)

### 3.3 Generate Random Catalogs (DR/RR)

**Settings:** 10x oversampling, KDE redshift sampling, field stratification

**Run four range tiers:**
```bash
python analysis/pairwise_randoms.py --tiers-glob "../data_processed/tiers/astrodeep_z*_*.csv" --bin-min 1 --bin-max 250 --bin-width 5 --rand-mult 10 --stratify-by-field --z-kde
```

**Run z6p separately:**
```bash
python analysis/pairwise_randoms.py --tiers-glob "../data_processed/tiers/astrodeep_z6p.csv" --bin-min 1 --bin-max 250 --bin-width 5 --rand-mult 10 --stratify-by-field --z-kde
```

**Expected output:**
```
Processing tier: z4_6
Generating 618480 random points...
Computing RR pairs...
Computing DR pairs...
Saved: ../results/step9b/run_YYYYMMDD_HHMMSS/RR_hist_z4_6.csv
...
```

**Runtime note:** z4_6 and z6_8 are slow (~30-60 min each)

### 3.4 Compute Correlation Functions xi(d)

**Uses:** Latest DD (step9) and DR/RR (step9b) folders automatically

**Run four range tiers:**
```bash
python analysis/xi_astrodeep.py --smooth-window 5
```

**Run z6p with explicit folder paths:**

First, find your z6p timestamps:

**Linux/macOS:**
```bash
ls -td ../results/step9/run_* | head -1    # Most recent DD folder
ls -td ../results/step9b/run_* | head -1   # Most recent RR/DR folder
```

**Windows (PowerShell):**
```powershell
Get-ChildItem ..\results\step9\run_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-ChildItem ..\results\step9b\run_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

Then run with those paths:
```bash
python analysis/xi_astrodeep.py --smooth-window 5 --dd-dir "../results/step9/run_YYYYMMDD_HHMMSS" --rdir "../results/step9b/run_YYYYMMDD_HHMMSS"
```

**Expected output:**
```
Computing xi(d) for tier: z4_6
Applying 5-bin smoothing...
Saved: ../results/step9c/run_YYYYMMDD_HHMMSS/xi_z4_6.csv
...
```

**Verify:** Check that `xi_z6p.csv` appears in `results/step9c/<timestamp>/`

---

## 4. Fourier Power Spectrum

**Runtime:** ~15 minutes  
**Working directory:** `src/`

### 4.1 Compute Stacked Power Spectrum

**Default settings** (Tukey alpha=0.25, mean detrending):
```bash
python analysis/power_from_xi.py --stack
```

**Expected output:**
```
Loading xi(d) from: ../results/step9c/run_YYYYMMDD_HHMMSS/
Processing tier: z4_6
Processing tier: z6_8
Processing tier: z8_10
Processing tier: z10_20
Computing stacked spectrum...
Saved: ../results/step10/run_YYYYMMDD_HHMMSS/peaks_stacked.csv
Figure: ../figures/step10/run_YYYYMMDD_HHMMSS/power_spectrum_stacked.png
```

### 4.2 Optional: Robustness Checks

Test different preprocessing parameters:
```bash
# Stronger tapering
python analysis/power_from_xi.py --stack --tukey-alpha 0.40

# Weaker tapering
python analysis/power_from_xi.py --stack --tukey-alpha 0.10

# Linear detrending
python analysis/power_from_xi.py --stack --detrend linear

# Equal weights (instead of inverse-variance)
python analysis/power_from_xi.py --stack --weights equal
```

**What to look for:** Modest bands at ~50-56 Mpc and 80-90 Mpc may appear. Short-wavelength spikes (~11-31 Mpc) are expected from survey geometry.

---

## 5. Inter-Field Variance Analysis

**Runtime:** ~2 hours total  
**Working directory:** `src/`

This section quantifies field-to-field fluctuations and is the **primary result** of the paper.

### 5.1 Raw Inter-Field Variance
```bash
python analysis/step11_interfield_variance.py --n-mocks 5000
```

**Expected output:**
```
Processing tier: z10_20
Observed variance: 0.585
Mock mean: 0.0023
Mock std: 0.0023
Empirical p-value: < 0.0002

Processing tier: z8_10
Observed variance: 15.55
Mock mean: 0.0105
Empirical p-value: < 0.0002

Saved: ../results/step11/run_YYYYMMDD_HHMMSS/summary.txt
```

**Runtime:** ~10 minutes

### 5.2 Depth-Normalized Variance
```bash
python analysis/step11b_interfield_variance_depthnorm.py --n-mocks 5000 --baseline-slices z4_6 z6_8
```

**Expected output:**
```
Normalizing z10_20 using baseline: z4_6, z6_8
Raw variance: 0.585 → Normalized: 0.004
Mock mean (norm): 0.001
Empirical p ~ 0.001

Normalizing z8_10 using baseline: z4_6, z6_8
Raw variance: 15.55 → Normalized: 0.101
Mock mean (norm): 0.027
Empirical p < 0.0002

Saved: ../results/step11b/run_YYYYMMDD_HHMMSS/summary.txt
Figures: ../figures/step11b/run_YYYYMMDD_HHMMSS/
```

**Runtime:** ~20 minutes

**What this shows:** Variance remains extreme even after correcting for survey depth differences.

### 5.3 Quality-Cut Variance
```bash
python analysis/step11c_variance_qualitycuts.py --snr-min 10 --destar --photoz-q --n-mocks 5000
```

**Expected output:**
```
Applying quality cuts:
  - SNR >= 10
  - Star removal
  - Photo-z quality flags

z10_20 (QC): 7214 → 4821 galaxies after cuts
Raw variance (QC): 0.585
Normalized variance (QC): 0.004
Empirical p ~ 0.001

z8_10 (QC): 29294 → 19847 galaxies after cuts
Raw variance (QC): 15.55
Normalized variance (QC): 0.101
Empirical p < 0.0002

Saved: ../results/step11c/run_YYYYMMDD_HHMMSS/summary.txt
```

**Runtime:** ~20 minutes

**Interpretation:** The overdensities persist after aggressive quality filtering, ruling out contamination.

---

## 6. Photo-z Monte Carlo

**Runtime:** ~1 hour  
**Working directory:** `src/`

This tests the stability of xi(d) under photo-z uncertainties.

### 6.1 Generate Resampled Catalogs
```bash
python analysis/step12a_resample_photoz.py z6p z8_10 z10_20 --n 100 --sigma-frac 0.06 --seed 7 --zmin 4
```

**Expected output:**
```
Run ID: run_YYYYMMDD_HHMMSS
Resampling tier: z6p (100 realizations)
Resampling tier: z8_10 (100 realizations)
Resampling tier: z10_20 (100 realizations)
Saved to: ../results/step12/run_YYYYMMDD_HHMMSS/resampled/
```

**Capture the Run ID for next step:**

**Linux/macOS/Git Bash:**
```bash
RUNID=$(ls -t ../results/step12/ | head -1)
echo $RUNID
```

**Windows (PowerShell):**
```powershell
$RUNID = (Get-ChildItem ..\results\step12 -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
echo $RUNID
```

### 6.2 Compute xi(d) for Resampled Catalogs

**Quick test** (5 realizations per tier):

**Linux/macOS:**
```bash
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z6p --max-gal 1500 --rand-mult 8 --n-limit 5
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z8_10 --max-gal 1500 --rand-mult 8 --n-limit 5
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z10_20 --max-gal 1500 --rand-mult 8 --n-limit 5
```

**Windows (PowerShell) - use the $RUNID you captured:**
```powershell
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z6p --max-gal 1500 --rand-mult 8 --n-limit 5
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z8_10 --max-gal 1500 --rand-mult 8 --n-limit 5
python analysis/step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z10_20 --max-gal 1500 --rand-mult 8 --n-limit 5
```

**Expected output:**
```
Processing realization 1/5 for tier z6p...
Processing realization 2/5 for tier z6p...
...
Saved: ../results/step12/run_YYYYMMDD_HHMMSS/xi_mc/z6p/xi_mc_mean.csv
Figures: ../figures/step12/run_YYYYMMDD_HHMMSS/
```

**Optional full run** (100 realizations, takes ~1 hour):
```bash
# Increase --n-limit to 100 and --max-gal to 5000
```

---

## 7. Significance Testing

**Runtime:** ~30 minutes  
**Working directory:** `src/`

Null hypothesis testing for the z6p feature at lambda ~ 183 Mpc.

### 7.1 Find Your z6p xi(d) File

**Linux/macOS:**
```bash
ls -t ../results/step9c/run_*/xi_z6p.csv | head -1
```

**Windows (PowerShell):**
```powershell
Get-ChildItem ..\results\step9c\run_*\xi_z6p.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### 7.2 Run Significance Test
```bash
python analysis/significance_z6p.py --input "../results/step9c/run_YYYYMMDD_HHMMSS/xi_z6p.csv" --n-sims 5000 --null-mode phase --d-col bin_right_Mpc --xi-col xi --zp-factor 8 --lambda-min 140 --lambda-max 220
```

**Expected output:**
```
Loading xi(d) from: ../results/step9c/run_YYYYMMDD_HHMMSS/xi_z6p.csv
Peak wavelength: 183.2 Mpc
Peak power: 0.042
Running 5000 phase-scrambled simulations...
Empirical p-value: 0.059
Conclusion: Not statistically significant (p > 0.05)
```

**Interpretation:** The 183 Mpc feature does not exceed significance thresholds.

---

## 8. Publication Figures

**Runtime:** ~15 minutes  
**Working directory:** `src/`

Generate all publication-quality figures:
```bash
python figures/fig_xi_photoz_panels.py
python figures/fig_variance_cdf_panels.py
python figures/fig_variance_publication.py
python figures/fig_perfield_rank_qc.py
python figures/plot_variance_figures.py
```

**Expected output:**
```
Saved: ../figures/publication/xi_photoz_panels_v2.pdf
Saved: ../figures/publication/variance_cdf_panels_v1.pdf
Saved: ../figures/publication/per_field_bars_qc_v3.pdf
Saved: ../figures/publication/perfield_rank_qc_v1.pdf
Saved: ../figures/publication/variance_hist_z8_10_qc_linear_inset_v5b.pdf
Saved: ../figures/publication/variance_hist_z10_20_qc_v4.pdf
```

**Verify figures:**

**Linux/macOS:**
```bash
ls -lh ../figures/publication/*.pdf
```

**Windows:**
```cmd
dir ..\figures\publication\*.pdf
```

You should see 6+ PDF files.

---

## Verification Checklist

After completing the pipeline, verify the following:

- [ ] `data_processed/astrodeep_master.csv` exists (~531k rows)
- [ ] Five tier CSVs exist in `data_processed/tiers/`
- [ ] `results/step9c/*/xi_z8_10.csv` shows positive xi(d) at small scales
- [ ] `results/step11c/*/summary.txt` shows p < 0.001 for both tiers
- [ ] `figures/publication/` contains 6+ PDF files
- [ ] Total runtime was 4-6 hours
- [ ] Disk usage is ~60 GB (50 GB data + 10 GB outputs)

If any check fails, see [Troubleshooting](#troubleshooting) below.

---

## Troubleshooting

### "Module not found: astropy"

**Solution:** Activate virtual environment first
```bash
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### "No such file or directory: data_raw/..."

**Solution:** Run `make_folders.py` first, then check file placement
```bash
python tools/make_folders.py
ls ../data_raw/astrodeep-jwst/ASTRODEEP-JWST_photoz/
```

### "DD folder not found"

**Solution:** Check timestamp and use most recent folder
```bash
# Linux/macOS:
ls -td ../results/step9/run_* | head -1

# Windows (PowerShell):
Get-ChildItem ..\results\step9 -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Runtime > 10 hours

**Normal behavior:** The z4_6 tier has 61k galaxies and can take 30-60 minutes alone on older laptops. Consider reducing `--rand-mult` from 10 to 5 if speed is critical (at cost of slightly larger uncertainties).

### "Memory error" during pair counting

**Solution:** Close other applications or reduce sample size with `--max-gal` flag
```bash
python analysis/pairwise_astrodeep.py --tiers-glob "..." --max-gal 10000 ...
```

### Figures look different from paper

**Check:** Make sure you're using the timestamped folders from the SAME analysis run. Mixing DD from one run with RR from another will produce incorrect xi(d).

### PowerShell script permission error (Windows)

**Solution:** Run PowerShell as Administrator or change execution policy temporarily
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

---

## Optional: Appendix Figures

The following analyses are documented in the paper but not required for reproduction:

- **Anisotropy FFTs:** `analysis/step*anisotropy*.py`
- **Ring scans:** `analysis/step*ring*.py`  
- **Leakage tests:** Additional Fourier parameter sweeps

These are exploratory and not included in the main reproduction workflow.

---

**Last updated:** February 2026  
**Questions?** Open an issue at [github.com/tgyuriseck/JWST-HighZ-Mature-Galaxies](https://github.com/tgyuriseck/JWST-HighZ-Mature-Galaxies/issues)
