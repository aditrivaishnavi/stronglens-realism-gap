# Public study ground rules

This document summarizes the final data-role and analysis rules used by the
reported study. The detailed definitions and estimators remain in the
manuscript.

## Detector and operating point

The primary detector is the seed-42, epoch-160 EfficientNetV2-S model with its
training-consistent preprocessing. A separate sample of 20,002
catalogue-negative objects sets the operating threshold by the specified
order-statistic rule. The threshold is then applied without recalibration to
the 12,512-object achieved-FPR sample, the 43 selected confirmed systems, and
the controlled injection experiments.

## Data roles

Detector development, threshold calibration, achieved-FPR evaluation,
selected-system evaluation, and injection-target roles are separated by the
recorded object, physical-system, spatial-component, HEALPix, and sky-neighbour
links. Confirmed systems enter neither detector fitting nor validation. The 20
systems used during exploratory feature work are excluded from all final
endpoint and observed-reference denominators.

The available identities and grouping records show no recorded overlap
between development and evaluation samples. Exact footprint polygons are
unavailable for part of the development set, so broader geometric overlap
cannot be tested.

## Related experiments

The 49,500-scene recovery grid is the parent bank for the source-photon
experiment and for the 6,336-row injected subset used by the domain probes.
The matched analysis uses exact bare targets and their stored
1.5-arcsec injections. The paired source-construction experiment uses a
separate balanced fixed-scene design drawn from the same eligible target
universe.

Selection, matching, fold construction, and eligibility do not use detector
outcomes. Related rows remain in the same dependency component, and repeated
rows do not receive disproportionate fitting or evaluation weight.

## Interpretation

Recovery, observed--injected distinguishability, matched movement,
source-photon response, source-construction response, and fixed-target
brightness response are different measurements. They do not form a causal
decomposition of one scalar realism gap. Reported conclusions are conditional
on the selected populations, detector, threshold, renderer, source and mass
models, and observing-support definitions.
