# P_src paired source-construction protocol

## Question

How does the specified structured F814W source-construction pathway compare
with its fitted single-Sérsic counterpart under shared targets and lens
mappings?

## Sources and pairs

The experiment uses 144 score-blindly selected COSMOS 25.2 F814W sources,
balanced over sixteen source strata. Each empirical source is used eleven
times, producing 1,584 target--source pairs on distinct targets.

The structured arm uses the regularized GalSim RealGalaxy preparation
described in the manuscript. The deterministic comparison arm uses the
single-Sérsic fit to the same COSMOS galaxy. Both are reconvolved to the same
0.15-arcsec circular resolution, receive the same source-specific rotation,
and are projected under the stated support and retained-flux rules.

Within a pair, both arms share the target, SIE-plus-shear mapping, nominal
source position, aligned source-plane centroid, effective per-band PSF,
intrinsic g/r/z fluxes, renderer, detector, threshold, and coupled
source-photon scheme. Eight photon draws are scored per arm and pair.

## Estimation

The primary contrast is the Sérsic-minus-structured photon-averaged detection
risk difference. Multiway multiplier intervals preserve target-spatial-group
dependence and the reuse of empirical sources within their fixed strata.

The estimand compares complete source-construction pathways. Equal intrinsic
band flux does not force equal post-lensing in-stamp flux or peak intensity,
and the structured arm also retains measurement and regularization effects
from its F814W preparation. The result is therefore not a universal
morphology effect.
