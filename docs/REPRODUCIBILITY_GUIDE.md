# Reproducibility Guide

How to reproduce the results in the MNRAS paper "The morphological barrier: quantifying the injection–realism gap for CNN strong lens finders in DESI Legacy Survey DR10" using data stored in `s3://darkhaloscope/`.

Last updated: 2026-02-18

## Prerequisites

- AWS CLI configured with access to the `darkhaloscope` bucket (us-east-2)
- Python 3.12+ with packages listed in `requirements.txt`
- GPU with ≥16 GB VRAM for training (A100/H100 recommended)
- ~500 GB local disk for a full checkout of relevant data

## Quick start: download everything needed

```bash
# Code
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/code/ ./code/
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/configs/ ./configs/
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/dhs/ ./dhs/
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/scripts/ ./scripts/

# Pre-trained model (primary)
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/ ./checkpoints/paperIV_efficientnet_v2_s_v4_finetune/

# Training/validation manifests
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/manifests/ ./manifests/

# Final results (D06)
aws s3 sync s3://darkhaloscope/stronglens-realism-gap/results/D06_corrected_priors/ ./results/D06_corrected_priors/
```

---

## 1. Training data

### 1.1 Positive cutouts (real strong lenses)

```
s3://darkhaloscope/stronglens-realism-gap/cutouts/positives/
```

~9,600 `.npz` files. Each contains a `(3, 101, 101)` array in `grz` bands, sourced from DESI Legacy Survey DR10 coadds. Includes Tier-A (high-confidence) and Tier-B (candidate) lenses.

### 1.2 Negative cutouts (non-lenses)

```
s3://darkhaloscope/stronglens-realism-gap/cutouts/negatives/
```

~1,279,000 `.npz` files organized in timestamped batches. Random LRG-centred cutouts from DR10 with no known lensing features.

### 1.3 Training manifests

```
s3://darkhaloscope/stronglens-realism-gap/manifests/training_v1.parquet          # full training manifest
s3://darkhaloscope/stronglens-realism-gap/manifests/training_parity_70_30_v1.parquet  # 70/30 train/val split
s3://darkhaloscope/stronglens-realism-gap/manifests/training_parity_v1.json      # split metadata
```

These define which cutouts are in the train vs. validation set, along with labels and metadata.

### 1.4 DR10 coadd tiles

```
s3://darkhaloscope/dr10/coadd_cache/{brickname}/
```

Cached FITS tiles (image, invvar, psfsize) per band (`g`, `r`, `z`) for each brick used in cutout extraction. The brick list comes from `dr10/survey-bricks-dr10-south.fits.gz`.

---

## 2. Model training

### 2.1 Training config

The primary model config:

```
s3://darkhaloscope/stronglens-realism-gap/configs/paperIV_bottlenecked_resnet.yaml
```

### 2.2 Injection priors

The corrected injection prior distributions (used in D06):

```
s3://darkhaloscope/stronglens-realism-gap/configs/injection_priors.yaml
```

This defines the Sérsic parameter distributions for parametric lens injection: `theta_E`, `n_sersic`, `R_eff`, `axis_ratio`, `source_mag`, `beta_frac`.

### 2.3 Training command

```bash
cd .
export PYTHONPATH=.
python3 dhs/scripts/run_experiment.py --config configs/paperIV_bottlenecked_resnet.yaml
```

### 2.4 Pre-trained checkpoints

```
s3://darkhaloscope/stronglens-realism-gap/checkpoints/paperIV_efficientnet_v2_s_v4_finetune/best.pt
```

This is the model used to produce all results in the paper. To skip training, download `best.pt` and point the evaluation scripts at it.

All model checkpoints (all architectures):

```
s3://darkhaloscope/stronglens-realism-gap/checkpoints/
```

---

## 3. Injection engine

### 3.1 Core code

```
s3://darkhaloscope/stronglens-realism-gap/dhs/injection_engine.py   # Sérsic injection code
s3://darkhaloscope/stronglens-realism-gap/dhs/data.py               # data loading, on-the-fly injection
s3://darkhaloscope/stronglens-realism-gap/dhs/train.py              # training loop
```

### 3.2 How injections work

The injection engine draws parameters from `injection_priors.yaml`, renders a Sérsic source lensed by an SIS profile, and adds the result to a real DR10 host cutout. The D06 experiments test the CNN's ability to distinguish these injections from real lenses.

---

## 4. Reproducing the paper results (D06)

All D06 results are at:

```
s3://darkhaloscope/stronglens-realism-gap/results/D06_corrected_priors/
```

### 4.1 Grid completeness (Tables 3–4, Figures 1–2)

```
grid_no_poisson/          # 110,000 injections scored without Poisson noise
  selection_function.csv       # aggregate completeness stats
  cutouts/cell*.npz       # individual injection cutouts with scores

grid_poisson/             # 110,000 injections scored with Poisson noise
  selection_function.csv
  cutouts/cell*.npz
```

The grid spans `theta_E` × `source_mag` × `beta_frac` × `n_sersic` × `R_eff` × `axis_ratio` as defined in `injection_priors.yaml`.

Key numbers from the paper:
- No-Poisson completeness at p > 0.3: **5.18%** (5,697 / 110,000)
- Poisson completeness at p > 0.3: **3.79%** (4,174 / 110,000)

### 4.2 Bright-arc experiments (Table 6, Figure 4)

Six variants, each with 1,605 files (1,600 cutouts + results JSON + metadata):

```
ba_baseline/              # no augmentation
ba_poisson/               # Poisson noise added
ba_clip20/                # clipped at mag 20
ba_gain_1e12/             # extreme gain (1e12)
ba_poisson_clip20/        # Poisson + clip
ba_unrestricted/          # no restrictions
```

### 4.3 Linear probe (Table 5, Figure 3)

```
linear_probe/             # AUC results, feature embeddings
  linear_probe_results.json
  embeddings.npz          # CNN embeddings for UMAP/Fréchet analysis
```

Key numbers:
- Tier-A vs low-bf injections AUC: **0.997 ± 0.003**
- Tier-A vs Tier-B AUC: **0.778 ± 0.062**

### 4.4 Tier-A scoring

```
tier_a_scoring/           # CNN scores for all Tier-A validation lenses
```

### 4.5 Poisson diagnostics

```
poisson_diagnostics/      # noise-level diagnostic plots
```

### 4.6 Visual comparison (Figure 5)

```
comparison_figure/        # real vs injected gallery figure
gallery/                  # full HTML gallery with thumbnails (273 files)
```

### 4.7 Provenance

```
provenance.json           # git SHA, config checksums, timestamps
```

---

## 5. Generating paper figures

The figure generation script is in the git repo (not S3):

```
paper/generate_all_figures.py    # Figures 1–4
scripts/generate_comparison_figure.py  # Figure 5
```

These read from the D06 results directory and output PDF figures.

---

## 6. Paper source

The LaTeX source files (`mnras_merged_draft_v1.tex` through `v14.tex`) are in the **git repository only**, not in S3:

```
paper/mnras_merged_draft.tex   # latest version
```

Build with:

```bash
cd paper
latexmk -pdf mnras_merged_draft_v14.tex
```

---

## 7. Earlier experiment generations (for reference)

These are not needed to reproduce the paper but document the experimental history:

| S3 path | Description |
|---------|-------------|
| `models/gen2_50epochs/` | Gen2 model |
| `models/gen3_moffat_v2/` | Gen3 with Moffat PSF |
| `models/gen4_hardneg/` | Gen4 with hard negatives |
| `models/gen5_cosmos/` | Gen5 with COSMOS stamps |
| `models/gen5_prime/` | Gen5 prime |
| `results/D01_*` through `D05_*` | Earlier diagnostic runs |
| `scores/` | Inference scores from earlier generations |
| `hard_negatives/` | Hard-negative mining outputs |
| `anchor_cutouts/`, `anchor_baseline/` | Known-lens anchor evaluations |

---

## 8. Infrastructure artifacts (not needed for science)

| S3 path | Description |
|---------|-------------|
| `emr-code/`, `emr_code/`, `emr-scripts/`, `emr_bootstrap/` | AWS EMR Spark pipeline code |
| `emr-logs/`, `emr_logs/` | EMR cluster logs |
| `phase1p5_*/`, `phase2_*/`, `phase3*/`, `phase4/`, `phase5*/` | Data engineering pipeline outputs |
| `runs/` | Training run logs (TensorBoard) |
| `logs/` | Pipeline execution logs |
| `.pytest_cache/` | Test cache (ignore) |
| `planb/` | Plan B paired/unpaired experiments |

---

## 9. File format reference

| Extension | Format | Typical contents |
|-----------|--------|------------------|
| `.npz` | NumPy compressed | `(3, 101, 101)` float32 cutout in `grz` bands |
| `.pt` | PyTorch | Model state dict |
| `.parquet` | Apache Parquet | Tabular data (manifests, scores, catalogs) |
| `.json` | JSON | Experiment configs, results summaries, provenance |
| `.yaml` | YAML | Training/injection configs |
| `.fits`, `.fits.fz` | FITS (compressed) | DR10 coadd images, catalogs |
| `.csv` | CSV | Evaluation metrics, catalogs |

---

## 10. Checksums and verification

The D06 run includes `provenance.json` with SHA-256 checksums of the config files and git commit hash used to produce the results. To verify integrity:

```bash
aws s3 cp s3://darkhaloscope/stronglens-realism-gap/results/D06_corrected_priors/provenance.json .
cat provenance.json
```

Compare the recorded checksums against the current config files to confirm nothing has drifted.
