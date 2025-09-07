# REVIEW_GUIDE.md

## Purpose

This guide documents the exact sequence to reproduce the results and figures in:

**JWST High Redshift Galaxy Clustering: Evidence for Accelerated Early Structure Formation**

Everything runs locally. Each step prints where outputs were saved (CSV under `results\`, figures under `figures\`). Optional/appendix items are clearly marked.

---

## 0) One-time environment setup (deterministic)

*(Open a VS Code terminal at `C:\JWST-HighZ-Mature-Galaxies\src`.)*

    python -V
    python -m venv ..\.venv
    ..\.venv\Scripts\activate
    python -m pip install -r ..\requirements.txt
    python -m pip install graphviz

Determinism knobs (PowerShell; re-run each new session):

    $env:PYTHONHASHSEED = '0'
    $env:NUMEXPR_MAX_THREADS = '1'

---

## 0.5) Create folders (one-time; skip if already present)

    python tools\make_folders.py

Creates:

- `data_raw\astrodeep-jwst\ASTRODEEP-JWST_optap\`, `..._photoz\`
- `data_processed\{fields,tiers}\`
- `results\{step9,step9b,step10,step11,step11b,step11c,step12}\`
- `figures\{publication,methods}\`, `docs\`

---

## 1) Data placement - - Skip if already setup from README or Step-By-Step Setup Guide.

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

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\preprocess_astrodeep.py
    python analysis\summarize_redshift_astrodeep.py
    python analysis\make_redshift_tiers_astrodeep.py

What to expect:
It reads *_photoz.fits and *_optap.fits from data_raw\astrodeep-jwst\....

It writes a merged master CSV under C:\JWST-HighZ-Mature-Galaxies\data_processed\

You should see progress messages; if a folder/file is missing, the script will complain.

When it finishes, let me know what it printed (or confirm you see a new CSV under data_processed). Then we’ll do Step 26 (redshift summaries) with:

Outputs: 
data_processed\astrodeep_master.csv, tier CSVs, histograms in figures\.

---

## 3) Pair counts + correlation function ξ(d)  (paper path)

### 3.0 Pre-check (force **photo-z**)

The pairwise script prefers `zspec` if present. To reproduce the paper, run on **photo-z** (`zphot`) for the **four range tiers**. Rename the header `zspec` → `zspec_unused` in these four files:

- `..\data_processed\tiers\astrodeep_z4_6.csv`
- `..\data_processed\tiers\astrodeep_z6_8.csv`
- `..\data_processed\tiers\astrodeep_z8_10.csv`
- `..\data_processed\tiers\astrodeep_z10_20.csv`

PowerShell one-liner (header-only edit; makes a `.bak` backup of each file):

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    $files = @(
      "..\data_processed\tiers\astrodeep_z4_6.csv",
      "..\data_processed\tiers\astrodeep_z6_8.csv",
      "..\data_processed\tiers\astrodeep_z8_10.csv",
      "..\data_processed\tiers\astrodeep_z10_20.csv"
    )
    foreach ($f in $files) {
      Copy-Item $f "$f.bak" -Force
      $lines = Get-Content $f
      $lines[0] = $lines[0] -replace '\bzspec\b','zspec_unused'
      Set-Content -Path $f -Encoding UTF8 -NoNewline -Value ($lines -join "`r`n")
      Write-Host "Header updated in $f"
    }

Expected in the DD logs after this change:

    using mode=radecz, cols=(RA_optap, DEC_optap, zphot)
    Loaded 61848 ... 47548 ... 29294 ... 7214 ...

### Binning (use these exact edges everywhere)

Use the same bin edges for DD, DR/RR, and ξ:

- `--bin-min 1  --bin-max 250  --bin-width 5`  → 50 bins (5, 10, …, 250 Mpc)

---

### 3a. Data–data (DD) pair counts  — **range tiers only**

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\pairwise_astrodeep.py --tiers-glob "../data_processed/tiers/astrodeep_z*_*.csv" --bin-min 1 --bin-max 250 --bin-width 5

Outputs (timestamped run folder under `results\step9\YYYYMMDD_HHMMSS\`):

- `DD_hist_<tier>.csv`, `DD_meta_<tier>.json`, `DD_hist_<tier>.png`
- `DD_run_summary.csv`

Runtime & resources (important):

- On a typical laptop, large tiers may take **tens of seconds to a few minutes** (single-core KD-tree).
- Progress lines show `KDTree progress: i=...` and cumulative pairs; lack of files mid-run is normal.

---

#### 3b. Randoms (DR/RR) — exact replication path

To reproduce the historical run you used (heavier, matches your archived summary), run with all tiers (range + cumulative), KDE redshift sampling, and field stratification.

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\pairwise_randoms.py --tiers-glob "../data_processed/tiers/*.csv" --bin-min 1 --bin-max 250 --bin-width 5 --rand-mult 10 --stratify-by-field --z-kde

Runtime warning (read this):

--rand-mult 10 means ~10× more random points than data; RR scales like 
(𝑘𝑁)^2. Expect hours on the largest tier (z4_6) on a laptop.

If you need a quicker check first, use --rand-mult 3 and then rerun at 10× once you’ve verified outputs.

Expected run summary (CSV-style; your run produced lines like):
tier,n_data,n_random,pairs_RR_all,pairs_DR_all,r_min_Mpc,r_max_Mpc,bin_width_Mpc,nbins,RR_csv,DR_csv,RR_meta,DR_meta,RR_fig,DR_fig,method_RR,method_DR,random_strategy

---

##### 3c. Correlation function ξ(d) (Landy–Szalay)

xi_astrodeep.py does not take bin flags; it consumes the most recent DD and DR/RR runs (their binning is already baked into the histograms). You can optionally smooth the ξ(d) curve.

Typical command (uses latest DD under results\step9\... and latest randoms under results\step9b\...):

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
python analysis\xi_astrodeep.py --smooth-window 5

Outputs:

Timestamped folder under results\step9c\YYYYMMDD_HHMMSS\

xi_<tier>.csv, xi_meta_<tier>.json, and xi_<tier>.png (for z4_6, z6_8, z8_10, z10_20)

step9c_run_summary.csv

Optional pretty panels:

    python analysis\plot_xi_tiers.py

---

## 4) Fourier power / periodicity check

   ## Inputs
- ξ(d): latest under `results\step9c\<timestamp>\`
- Power runs inspected:

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
  - `power_from_xi.py --stack`

  **Optional - Additional Commands** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
  - `power_from_xi.py --stack --tukey-alpha 0.4`
  - `power_from_xi.py --stack --tukey-alpha 0.1`
  - `power_from_xi.py --stack --detrend linear`
  - `power_from_xi.py --stack --weights equal`
  - `power_from_xi.py --stack --no-pad-pow2`

## Snapshot of tier peaks reported by the script
- Short λ (≈11–31 Mpc): strong in many tiers (z10_20, z10p, z8_10, z8p), repeatedly around ≈12–15–20–25–30 Mpc.
- Medium λ:
  - **≈50–56 Mpc**: appears in z4_6 and z4p and shows up in stacked runs across settings.
  - **≈80–90 Mpc**: appears in z6_8 / z6p and shows up in stacked runs across settings.
- Long λ (BAO-like): a **~183 Mpc** feature appears mainly in **z6p**; it weakens under stronger windowing and does not dominate in the stacked spectrum.

## Stack-level behavior across reruns
- **Stable-but-modest bands**:
  - **~50–56 Mpc**: persists across α=0.1/0.4, linear detrend, no-pad; power is modest.
  - **~80–90 Mpc**: likewise persists with modest power.
- **Short-λ spikes (≈11–31 Mpc)**: always strong but likely **artifacts** (bin width = 5 Mpc → harmonics/ringing at multiples; also high-k noise in high-z tiers).
- **~183 Mpc (z6p)**: not robust—fades with small processing changes and does not strengthen when stacked.

## Interpretation (working)
- The persistent ~50–56 and ~80–90 Mpc bands are consistent with **harmonics/leakage** when any fundamental near BAO scale is weak in ξ and the data are windowed over a finite range.
- No compelling BAO (~150 Mpc) detection in the stack at this stage.
- Short-λ peaks are not interpreted as astrophysical.

## What to open (for visual confirmation)
- Figures: `figures\step10\<timestamp>\`
  - `power_stacked.png` (look for ~50–56 and ~80–90 Mpc bands)
  - `power_z6p.png` (inspect the ~183 Mpc spike’s stability)
- Tables: `results\step10\<timestamp>\`
  - `peaks_stacked.csv`, `peaks_<tier>.csv`

Optional methods / appendix:

    python analysis\step09_anisotropy_fft.py
    python analysis\step10_ring_scan.py 
    python analysis\make_harmonics_leakage_figure.py

- On a typical laptop, 'ring_scan' may take **10 to 25 minutes**
---

## 5) Inter-field variance suite

This section quantifies field-to-field fluctuations at high-z using lognormal mocks, then applies depth normalization and quality cuts.  

---

**5a) Inter-field variance (raw)**

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\step11_interfield_variance.py --n-mocks 5000

- Uses the high-z **range tiers** by default: `z10_20` and `z8_10`.
- Defaults (for reference): grid `256×256`, `corr_pix=2.0`, seed fixed in script.
- Outputs:
  - `results\step11\run_YYYYMMDD_HHMMSS\summary.txt`
  - `results\step11\run_YYYYMMDD_HHMMSS\z*_observed_field_densities.csv`
  - `results\step11\run_YYYYMMDD_HHMMSS\z*_mock_variance_samples.csv`
  - Figures in `figures\step11\run_YYYYMMDD_HHMMSS\` (variance histograms)
- Runtime: ~10 minutes on a typical laptop.
- What to look for: very large z-scores (observed variance ≫ mock mean).  
  *Example from a successful run:*  
  `z10_20: z ≈ 201` (raw), `z8_10: z ≈ 2059` (raw).

---

**5b) Inter-field variance with depth normalization**

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\step11b_interfield_variance_depthnorm.py --n-mocks 5000 --baseline-slices z4_6 z6_8

- Normalizes high-z slices (`z10_20`, `z8_10`) by **baseline** slices (`z4_6`, `z6_8`) to correct for field depth differences.
- Outputs:
  - `results\step11b\run_YYYYMMDD_HHMMSS\summary.txt`
  - `results\step11b\run_YYYYMMDD_HHMMSS\z*_field_densities_raw_and_norm.csv`
  - `results\step11b\run_YYYYMMDD_HHMMSS\z*_mock_variance_raw.csv`
  - `results\step11b\run_YYYYMMDD_HHMMSS\z*_mock_variance_norm.csv`
  - Figures in `figures\step11b\run_YYYYMMDD_HHMMSS\` (raw vs norm comparison histograms, per-field bars)
- Runtime: ~20 minutes on a typical laptop.
- Expected behavior: variance **drops** after normalization but remains **highly significant**.  
  *Example from a successful run:*  
  `z10_20: raw z ≈ 153 → norm z ≈ 9.4`  
  `z8_10:  raw z ≈ 1876 → norm z ≈ 34.5`

---

**5c) Inter-field variance with quality cuts**

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\step11c_variance_qualitycuts.py --snr-min 10 --destar --photoz-q --n-mocks 5000

- Applies QC filters (S/N ≥ 10, remove stars, require good photo-z) before computing inter-field variance and depth-normalized variance.
- Outputs:
  - `results\step11c\run_YYYYMMDD_HHMMSS\summary.txt`
  - `results\step11c\run_YYYYMMDD_HHMMSS\z*_field_densities_raw_and_norm_qc.csv`
  - `results\step11c\run_YYYYMMDD_HHMMSS\z*_mock_variance_raw_qc.csv`
  - `results\step11c\run_YYYYMMDD_HHMMSS\z*_mock_variance_norm_qc.csv`
  - Figures in `figures\step11c\run_YYYYMMDD_HHMMSS\` (QC versions of 5b plots)
- Runtime: typically similar to 5b on a laptop (~20 minutes).
- Interpretation: check whether excess variance **persists** after QC.  
  *Example from a successful run:*  
  `z10_20 (QC): raw z ≈ 252, norm z ≈ 10.6`  
  `z8_10  (QC): raw z ≈ 2018, norm z ≈ 32.1`  
  (Exact values vary with random seed; persistence of large z after QC is the key check.)
---

## 6) Photo-z perturbation (Monte Carlo)

Stability check of ξ(d) under photo-z noise. 

---

**6a) Generate resampled photo-z catalogs (one Run ID for all tiers)**  
Creates 100 re-draws per tier and prints a `Run: run_YYYYMMDD_HHMMSS`.

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\step12a_resample_photoz.py z6p z8_10 z10_20 --n 100 --sigma-frac 0.06 --seed 7 --zmin 4

After 6a, capture the newest Run ID automatically (no manual copy/paste needed):
**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    $RUNID = (Get-ChildItem ..\results\step12 -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name

Outputs:
- Resampled CSVs per tier: `results\step12\$RUNID\resampled\<tier>\resample_000.csv ... resample_099.csv`
- Summary: `results\step12\$RUNID\summary_resampling.txt`

Notes:
- `--sigma-frac 0.06` is a fallback σz/z when no per-object σz column exists.
- `--zmin 4` clamps draws to z ≥ 4.

---

**6b) ξ(d) for the resampled catalogs (memory-safe)**  
Quick sanity MC: 5 realizations per tier, 1.5k galaxies each, randoms ×8 (fast, single-core KD-Tree). Run all three tiers under the same `$RUNID`:

**Command** (from `C:\JWST-HighZ-Mature-Galaxies\src`):
    python analysis\step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z6p    --max-gal 1500 --rand-mult 8 --n-limit 5
    python analysis\step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z8_10  --max-gal 1500 --rand-mult 8 --n-limit 5
    python analysis\step12b_xi_photoz_mc_memsafe.py --runid $RUNID --tiers z10_20 --max-gal 1500 --rand-mult 8 --n-limit 5

Outputs:
- Per-tier MC products: `results\step12\$RUNID\xi_mc\<tier>\...`
- Combined summary table: `results\step12\$RUNID\xi_mc_summary.csv`
- Figures: `figures\step12\$RUNID\` (e.g., `xi_mc_panels.png`, per-scale point plots)

Scale-up (optional): increase `--n-limit` (e.g., 50 or 100) and/or `--max-gal` (e.g., 5000). Runtime grows accordingly.

---

## 7) Null tests / significance

    python analysis\significance_z6p.py --n-sims 5000 --null-mode phase --d-col bin_right_Mpc --xi-col xi --zp-factor 8 --lambda-min 140 --lambda-max 220

---

## 8) Publication-quality figures

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
