# G_rec recovery-grid protocol

## Question

How often does the specified detector respond over the designed target,
observing-condition, source, lens, and Einstein-radius grid?

## Population

The design contains 4,500 distinct catalogue-negative DR10 targets, with 500
targets in each of nine occupied r-band PSF--depth cells. Each target receives
one intrinsic source and lens-nuisance draw that is reused across eleven
Einstein radii from 0.50 to 3.00 arcsec in 0.25-arcsec steps. The resulting
bank contains 49,500 scenes.

Target selection and observing-cell assignment do not use detector scores.
Targets retain their recorded object and spatial-group identities.

## Source and lens construction

The source is one elliptical Sérsic profile with intrinsic r-band magnitude
from 23 to 26, Sérsic index from 0.5 to 2.0, effective radius from 0.15 to
0.50 arcsec, axis ratio from 0.3 to 1.0, and the specified colour and
source-offset distributions. The lens mapping is an SIE plus external shear
with the distributions stated in the manuscript.

The intrinsic source, normalized source offset, lens axis ratio and
orientation, shear, mass-centre offset, and base random key remain fixed
within a target's eleven-radius sequence. Einstein radius and the
corresponding angular source offset change with radius.

## Visibility and scoring

No rendered-arc signal-to-noise or visibility admission rule is applied after
rendering. Faint and blended scenes remain in the denominator. All scenes use
the primary detector, its training-consistent preprocessing, and the common
threshold.

## Estimation

Recovery is averaged equally over occupied cells, targets within cells, and
the eleven radii. The balanced design makes this equal to the scene mean.
Target-spatial-group multiplier intervals preserve recorded spatial
dependence. The result is conditional recovery over the designed grid, not
survey-population completeness.
