# Methods and limitations

This repository supports a representation-level comparison between selected
observed strong lenses and controlled source injections, together with
separate detector-response experiments. The analyses retain their own
populations and denominators; they are not combined into one realism score.

## Data and primary detector

The detector is an ImageNet-initialized EfficientNetV2-S trained for 160
epochs with unweighted binary cross-entropy. The primary checkpoint is seed 42
at epoch 160. Training uses 286,679 catalogue negatives and 3,633 noisy
catalogue-positive examples. Validation uses 61,437 negatives and 778 noisy
positives.

Literature review identifies 63 confirmed systems. Forty-three form the
selected observed endpoint; 20 systems used during exploratory feature work
are excluded from all reported observed-reference denominators. Separate
catalogue-negative samples set the threshold (20,002 rows) and measure its
achieved exceedance rate (12,512 rows).

## Injection and representation analyses

### \(G_{\rm rec}\): conditional recovery

The recovery grid contains 49,500 injections on 4,500 distinct DR10 targets:
nine PSF--depth cells, 500 targets per cell, and 11 Einstein radii per target.
No post-render visibility or signal-to-noise cut is applied. The detector
recovers 1,710 scenes (3.4545 per cent). This is conditional response over the
declared grid, not survey completeness.

### \(G_{\rm domain}\): observed--injected separation

The domain comparison uses 43 selected observed systems and 6,336 injections
on 576 targets. Five dependency-closed folds keep linked targets, scenes,
sources, lens parameters, pixels, and physical groups together. Linear
readouts are evaluated in four representations:

- operational penultimate features: AUC 0.991340;
- Stein self-supervised survey features: AUC 0.958187;
- generic ImageNet features: AUC 0.978528;
- 18 pixel summaries: AUC 0.980745.

These finite-cohort AUCs show that population identity is accessible in each
feature protocol. They do not identify a unique physical cause.

### \(G_{\rm match}\): matched controls

Each observed system is matched without replacement to one target using
observed \(z\)-band magnitude, Sérsic index, axis ratio, \(r\)-band PSF, and
\(r\)-band depth. The exact bare target and its stored 1.5-arcsec injection
form a 43-triplet comparison.

Mean injected-minus-bare movement toward the observed side is +0.023183 in
operational features, +0.235027 in Stein features, +0.024792 in ImageNet
features, and -0.029207 in pixel statistics. Matching reduces imbalance on
five recorded quantities; it does not establish physical equivalence.

### \(P_{\rm noise}\): source-photon sampling

Every recovery-grid scene receives eight source-photon realizations under the
declared \(g=150\ {\rm e^{-}\,nmgy^{-1}}\) conversion. Existing target, sky,
detector, and coaddition noise are unchanged. Recovery changes from 3.4545 to
2.2515 per cent, a paired change of -1.203030 percentage points. The
conversion is a modelling choice, not a calibrated gain for every exposure.

### \(P_{\rm src}\): source-construction contrast

The 1,584 paired comparisons share target, lens mapping, source position,
intrinsic per-band flux, effective PSF, renderer, and photon index. One arm
uses a prepared COSMOS F814W spatial map; the other uses the fitted
single-Sérsic counterpart of the same object. The Sérsic arm exceeds the
structured arm by +0.560290 percentage points, with interval
[+0.283912, +0.932231].

This is a comparison of complete source-preparation pathways. Measurement
uncertainty, regularization, support restriction, and projection remain part
of the structured arm.

### Fixed-target brightness control

Forty-three targets retain the same lens geometry and source shape while the
intrinsic source magnitude varies from 18.5 to 25.5. One expectation image and
eight photon realizations per target--magnitude cell produce 3,096 scored
images. This is a bounded detector-response control, not a population
visibility analysis.

## Principal limitations

- Injection targets are not population-matched to the observed deflectors.
- Lens mass parameters are assigned independently of target redshift,
  luminosity, velocity dispersion, stellar mass, and environment.
- The source experiment is limited to selected high-S/N COSMOS F814W
  constructions and one spatial map per band.
- The renderer uses an effective circular Gaussian coadd PSF and inserts
  source light after coaddition.
- The \(g=150\) source-photon conversion is not an exposure-level gain model.
- Complete development-footprint polygons and row-level upstream \(g/r\)
  exposure columns are unavailable.
- The Stein pretraining set may overlap present cutouts at an unknown level.
- The checkpoint panel contains correlated variants of one detector design,
  not independent model replications.
