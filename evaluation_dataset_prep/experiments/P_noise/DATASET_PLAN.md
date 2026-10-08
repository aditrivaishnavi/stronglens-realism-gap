# P_noise source-photon protocol

## Question

How does the adopted source-photon sampling change detector recovery when the
target, expected lensed source, geometry, and detector rule remain fixed?

## Paired design

The experiment uses all 49,500 deterministic G_rec expectation scenes. Each
scene receives eight keyed source-photon realizations at the adopted effective
conversion of 150 electrons per nanomaggy. The draw count was selected by the
documented pre-score Monte Carlo precision pilot.

The observed DR10 target pixels already contain their realized foreground,
sky, detector, resampling, and coaddition noise. The intervention draws only
the photon term associated with the inserted source; it does not add a second
sky or target-noise realization.

Every row retains the target, parent-scene, geometry, source, detector,
threshold, and photon-draw identities. The expectation and photon arms use the
same detector preprocessing and common threshold.

## Estimation

Detection indicators are first averaged over the eight draws within a scene.
The paired photon-minus-expectation difference is then averaged with equal
cell, target, and radius weighting. Target-spatial-group multiplier intervals
preserve shared-target and spatial dependence. The finite-draw Monte Carlo
standard error is reported separately.

The result is conditional on the adopted source-photon law. It is not an
exposure-level calibration of the DR10 gain or correlated-noise process.
