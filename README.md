# JWST High-Redshift Galaxy Clustering Pipeline

**Reproducible analysis of large-scale structure and inter-field variance in public JWST photometric catalogs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MNRAS](https://img.shields.io/badge/MNRAS-Under%20Review-orange)](https://academic.oup.com/mnras)

---

## 📄 Paper

**JWST High-Redshift Galaxy Clustering: A Reproducible Photometric Pipeline and Inter-Field Variance Analysis at z > 8**  
Timothy Gyuriseck (2025)

- **Status:** Submitted to *Monthly Notices of the Royal Astronomical Society* (MNRAS)
- **Preprint:** *[Zenodo DOI will be added upon upload]*
- **arXiv:** *[Will be added if/when endorsed]*

---

## 🔬 Summary

This repository contains a fully reproducible clustering analysis pipeline applied to 531,173 galaxies from the public [ASTRODEEP–JWST photometric catalogs](https://www.astrodeep.eu/astrodeep-jwst/) across seven deep fields (ABELL2744, CEERS, JADES-GN, JADES-GS, NGDEEP, PRIMER-COSMOS, PRIMER-UDS).

### Key Finding

Analysis reveals **extreme field-to-field variance** in galaxy surface densities at z=8–10 and z=10–20 that persists after:
- Depth normalization using lower-z baselines (z=4–8)
- Strict quality cuts (SNR ≥ 10, star removal, photo-z quality filters)

The normalized variance far exceeds expectations from 5,000 lognormal mock realizations (empirical p < 2×10⁻⁴ at z=8–10; p ≃ 10⁻³ at z=10–20), suggesting the presence of rare, physically overdense regions consistent with proto-cluster or large-scale filamentary structure forming earlier than baseline ΛCDM predictions.

No evidence is found for universal periodic clustering patterns or BAO-scale features at these redshifts.

### Methods

- **Two-point correlation function:** Landy–Szalay estimator with geometry-matched random catalogs
- **Fourier analysis:** Periodicity search with null hypothesis testing (phase-scrambling, 5,000 realizations)
- **Monte Carlo validation:** Lognormal mock catalogs with field-specific geometry
- **Photo-z uncertainty propagation:** 100-realization stability analysis
- **Variance testing:** Raw, depth-normalized, and quality-filtered comparisons

---

## 🚀 Quick Start

### Prerequisites

- **Python:** 3.8 or higher
- **Disk space:** ~50 GB for ASTRODEEP catalogs
- **RAM:** 8 GB minimum (16 GB recommended for full pipeline)
- **Time:** 4–6 hours total runtime on a standard laptop

### Installation
bash
# Clone repository
git clone https://github.com/tgyuriseck/JWST-HighZ-Mature-Galaxies.git
cd JWST-HighZ-Mature-Galaxies

# Create virtual environment (Linux/macOS)
python3 -m venv .venv
source .venv/bin/activate

# Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Download Data

The pipeline requires public ASTRODEEP–JWST photometric catalogs:

1. Visit: [https://www.astrodeep.eu/astrodeep-jwst/](https://www.astrodeep.eu/astrodeep-jwst/)
2. Download both catalog types for all seven fields:
   - `*-photoz.fits` (photometric redshift catalogs)
   - `*-optap.fits` (aperture-optimized photometry catalogs)
3. Place downloaded files in the following structure:

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

**Total download size:** ~45–50 GB

### Run Pipeline
bash
# Create directory structure
python src/tools/make_folders.py

# See REVIEW_GUIDE.md for complete step-by-step instructions

For full reproduction instructions, see **[REVIEW_GUIDE.md](REVIEW_GUIDE.md)**.

---

## 📁 Repository Structure
JWST-HighZ-Mature-Galaxies/
├── src/
│   ├── analysis/              # Core clustering analysis scripts
│   │   ├── preprocess_astrodeep.py
│   │   ├── pairwise_astrodeep.py
│   │   ├── xi_astrodeep.py
│   │   ├── power_from_xi.py
│   │   ├── step11*_interfield_variance*.py
│   │   └── step12*_photoz*.py
│   ├── figures/               # Publication figure generation
│   │   ├── fig_variance_publication.py
│   │   ├── fig_xi_photoz_panels.py
│   │   └── ...
│   └── tools/                 # Utility scripts
│       └── make_folders.py
├── data_raw/                  # ASTRODEEP catalogs (user-provided)
├── data_processed/            # Generated tier catalogs
├── results/                   # Analysis outputs (generated)
├── figures/                   # Generated figures
├── REVIEW_GUIDE.md            # Step-by-step reproduction guide
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file

**Note:** `data_processed/`, `results/`, and `figures/` directories are generated by the pipeline and excluded from version control.

---

## 📊 Pipeline Overview

### Analysis Steps

| Step | Script | Runtime* | Output |
|------|--------|----------|--------|
| 1 | Master catalog creation | 30 min | `astrodeep_master.csv` (531k galaxies) |
| 2 | Redshift tier slicing | 5 min | Tier CSVs (z=4–6, 6–8, 8–10, 10–20) |
| 3 | Pair counting (DD/DR/RR) | 1–2 hrs | Pair count histograms |
| 4 | Correlation function ξ(d) | 30 min | `xi_*.csv` files |
| 5 | Fourier power spectrum | 15 min | `peaks_stacked.csv` |
| 6 | Inter-field variance | 2 hrs | Mock ensemble comparisons |
| 7 | Photo-z Monte Carlo | 1 hr | 100-realization ξ(d) bands |
| 8 | Publication figures | 15 min | PDF figures |

*Runtime estimates for a typical laptop (4-core, 16 GB RAM)

### Key Outputs

- **Correlation functions:** `results/step9c/*/xi_<tier>.csv`
- **Variance statistics:** `results/step11c/*/summary.txt`
- **Publication figures:** `figures/publication/*.pdf`
- **Full reproduction log:** Generated by following [REVIEW_GUIDE.md](REVIEW_GUIDE.md)

---

## 📖 Citation

If you use this pipeline or findings in your research, please cite:
bibtex
@article{Gyuriseck2025,
  author = {Gyuriseck, Timothy},
  title = {{JWST High-Redshift Galaxy Clustering: A Reproducible Photometric 
           Pipeline and Inter-Field Variance Analysis at z > 8}},
  journal = {MNRAS (submitted)},
  year = {2025},
  note = {GitHub: https://github.com/tgyuriseck/JWST-HighZ-Mature-Galaxies}
}

*BibTeX entry will be updated with journal reference and DOI upon publication.*

---

## 🔗 Related Work

This analysis uses the public **ASTRODEEP–JWST photometric catalogs**:
- Merlin et al. (2024), *Astronomy & Astrophysics*, 691, A240  
  DOI: [10.1051/0004-6361/202450514](https://doi.org/10.1051/0004-6361/202450514)

Key papers cited in the methodology:
- Landy & Szalay (1993) — Two-point correlation estimator
- Coles & Jones (1991) — Lognormal density field models
- Dalmasso et al. (2024, MNRAS) — JADES clustering to z~11
- Li et al. (2025, MNRAS) — EPOCHS photometric overdensity candidates

---

## 🛠️ Requirements

### Software Dependencies

See [requirements.txt](requirements.txt) for exact versions. Key packages:

- `numpy` (≥1.24)
- `scipy` (≥1.10)
- `pandas` (≥2.0)
- `matplotlib` (≥3.7)
- `astropy` (≥5.2)
- `h5py` (≥3.8)

### System Requirements

- **OS:** Linux, macOS, or Windows with Python 3.8+
- **Memory:** 8 GB minimum; 16 GB recommended for z=4–6 tier processing
- **Storage:** 60 GB total (50 GB data + 10 GB outputs)

---

## 📧 Contact

**Timothy Gyuriseck**  
Independent Researcher  
Dallas, TX, United States

- 📧 Email: tgyuriseck@gmail.com
- 🔗 GitHub: [@tgyuriseck](https://github.com/tgyuriseck)

For questions about the pipeline or reproduction issues, please [open an issue](https://github.com/tgyuriseck/JWST-HighZ-Mature-Galaxies/issues).

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work uses public data from the ASTRODEEP–JWST photometric catalogs ([Merlin et al. 2024](https://doi.org/10.1051/0004-6361/202450514)). AI-assisted tools were used for pipeline code development and manuscript editing; all scientific decisions, analysis design, and interpretations are the author's own.

---

## ⚠️ Reproducibility Notes

- **All results are reproducible** from the public ASTRODEEP catalogs following [REVIEW_GUIDE.md](REVIEW_GUIDE.md)
- Intermediate data products are generated locally and excluded from version control due to size
- Random seeds are fixed in analysis scripts for deterministic results
- Full reproduction takes 4–6 hours on a standard laptop
- Photo-z uncertainties are propagated via Monte Carlo resampling (100 realizations per tier)

---

## 🔄 Version History

- **v1.2.0** (2025-02-18): Initial public release accompanying MNRAS submission

---

*Last updated: February 2026*
