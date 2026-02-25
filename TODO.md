# Future Work

## 1. Gabor filter analysis for injection realism diagnostics

Investigate whether Gabor filters (oriented bandpass filters sensitive to texture at specific spatial frequencies and orientations) can provide an interpretable, non-neural diagnostic for the injection realism gap. Gabor filter banks could:

- Quantify textural differences between real arcs and parametric injections at specific spatial scales
- Provide a CNN-independent validation of the morphological barrier finding
- Help decompose the feature-space separation into interpretable frequency/orientation components
- Potentially serve as a lightweight realism gate that does not require a trained CNN

Starting point: apply a Gabor filter bank (e.g. 4 scales x 6 orientations) to real Tier-A cutouts and brightness-matched injections, then compare the filter response distributions.

## 2. Real galaxy stamps from HUDF as source-plane objects

Replace parametric Sersic sources with real galaxy stamps from the Hubble Ultra Deep Field (HUDF), following the approach of Canameras et al. (2024, HOLISMOKES XI) who demonstrated this for HSC.

Key challenges for adapting to DESI DR10:
- HST-to-DESI bandpass transformation (F435W/F606W/F775W/F850LP to g/r/z)
- 8.7x pixel scale difference (HUDF at 0.03 arcsec/pixel vs DESI at 0.262 arcsec/pixel)
- PSF matching between HST diffraction-limited and DESI ground-based seeing
- Sufficient sample size of high-z source galaxies at appropriate magnitudes

Success criterion: if real-stamp injections bring the linear probe AUC materially closer to 0.5 (chance level), this confirms the morphological barrier is a property of the parametric source model rather than the injection-recovery framework. The paper proposes using the linear probe AUC as a quantitative "realism gate" for this purpose (Section 5.3).
