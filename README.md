# stronglens-realism-gap

**A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10**

This repository contains all code, data, results, and the manuscript for the paper submitted to RAS Techniques and Instruments (RASTI). The injection-recovery pipeline measures how well a CNN strong-lens finder can distinguish real gravitational lenses from synthetic injections based on Sersic profiles, and quantifies the resulting "realism gap" that limits completeness estimates.

## Key results

- A logistic regression linear probe on CNN penultimate-layer embeddings achieves **AUC = 0.997 +/- 0.003** (five-fold cross-validation; bootstrap 95% CI: [0.996, 1.000]) for separating real lenses from injections, demonstrating that the CNN trivially distinguishes them.
- Marginal completeness at the CNN's operating threshold (p > 0.3) is **5.18%** (5697/110,000 injections detected) over the full parameter grid — implying a severe injection realism gap.
- The gap is consistent with a **morphological** barrier: Frechet distance analysis shows the separation emerges in learned mid-level features, not in simple photometry.
- Adding Poisson noise to injections does not close the gap (marginal completeness drops to 3.79%), though paired McNemar tests reveal a statistically significant effect at intermediate source magnitudes (source mag 21--22: p = 3.2e-4; source mag 22--23: p = 2.8e-4), with pooled McNemar chi-squared p = 1.2e-5 across mag 20--23.

## Repository structure

```
├── paper/                        # RASTI manuscript and figures
│   ├── balaji_johnson_2026_rasti.tex  # LaTeX source
│   ├── balaji_johnson_2026_rasti.pdf  # Compiled PDF
│   ├── fig1–fig5*.pdf            # Publication figures
│   └── generate_all_figures.py   # Regenerate the completeness, UMAP, score, and bright-arc figures
├── dhs/                          # Core Python library
│   ├── injection_engine.py       # SIE lens model + Sersic sources + ray-tracing
│   ├── model.py                  # EfficientNetV2-S architecture
│   ├── data.py                   # Dataset and dataloader
│   ├── train.py                  # Training loop
│   ├── preprocess.py             # Cutout preprocessing (annulus normalisation)
│   └── scripts/                  # Training and evaluation runners
├── scripts/                      # Analysis and diagnostic scripts
│   ├── feature_space_analysis.py # Linear probe + Frechet distance
│   ├── selection_function_grid.py# Completeness grid over (theta_E, mag)
│   ├── permutation_bootstrap_auc.py  # Statistical significance tests
│   ├── mcnemar_bright_arc.py     # McNemar test for Poisson effect
│   └── ...                       # Additional diagnostics
├── configs/                      # Experiment configuration
│   ├── injection_priors.yaml     # Injection parameter distributions
│   └── paperIV_efficientnet_v2_s_v4_finetune.yaml  # CNN training config
├── tests/                        # Unit and integration tests
├── sim_to_real_validations/      # Sim-to-real validation experiments
├── results/                      # D06 experiment outputs
│   └── D06_corrected_priors/     # Final run with K-corrected priors
│       ├── analysis/             # Aggregate summary JSON
│       ├── linear_probe/         # Embeddings, probe weights, AUC
│       ├── grid_*/               # Completeness grids (with/without Poisson)
│       ├── ba_*/                 # Bright-arc experiment variants
│       ├── poisson_diagnostics/  # Paired Poisson analysis + McNemar
│       ├── gallery/              # 271 lens thumbnails + HTML gallery
│       └── provenance.json       # Reproducibility checksums
├── data/
│   └── positives/
│       └── desi_candidates.csv   # DESI Strong Lensing catalogue cross-match
└── docs/                         # Technical documentation
    ├── REPRODUCIBILITY_GUIDE.md
    ├── TECHNICAL_SPECIFICATIONS.md
    └── EXPERIMENT_REGISTRY.md
```

## Reproducing the paper

### Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies: Python 3.12+, PyTorch 2.1+, torchvision, numpy, scipy, scikit-learn, matplotlib, pandas, astropy, umap-learn.

### Regenerating figures

The completeness, UMAP, score-distribution, and bright-arc figures can be regenerated from the included result files:

```bash
cd stronglens-realism-gap
python paper/generate_all_figures.py
```

The real-vs-injected comparison figure:

```bash
python scripts/generate_comparison_figure.py
```

### Running the full pipeline

The full pipeline requires:
1. Access to DESI Legacy Survey DR10 cutouts (via the Legacy Survey cutout service or local brick files)
2. A trained CNN checkpoint (available on request or via the training scripts)
3. GPU compute for training and inference

See `docs/REPRODUCIBILITY_GUIDE.md` for detailed instructions.

## Data

- **Real lenses**: 112 Tier-A spectroscopically confirmed lenses (validation set; 389 Tier-A in total) from the DESI Strong Lensing catalogue, cross-matched with DR10 imaging.
- **Injections**: Synthetic lensed arcs generated by `dhs/injection_engine.py` using SIE lens models and Sersic source profiles, injected into real DR10 galaxy cutouts.
- **Negatives**: ~447,000 non-lens cutouts (312,744 training, 134,149 validation) from the DR10 Tractor catalogue, selected with magnitude and colour cuts spanning the full range of galaxy morphologies.

## Citation

If you use this code or the results in your work, please cite:

```bibtex
@unpublished{balaji2026morphological,
  title={A morphological barrier: quantifying the injection realism gap for CNN strong lens finders in DESI Legacy Survey DR10},
  author={Balaji, Aditrivaishnavi and Johnson, Parker T.},
  note={Submitted to RAS Techniques and Instruments (under review)},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
