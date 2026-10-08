# G_match matched-control protocol

## Question

How does adding one specified simulated source move the same target along an
axis that separates selected observed systems from matched bare targets?

## References and target universe

The observed roster is the exact 43-system selected confirmed cohort used by
the domain analysis. Candidate targets are the 4,500 G_rec targets. Matching
does not use detector scores, representation features, probe outcomes, or
post-render source visibility.

## Matching

Each observed reference is matched without replacement to one target using:

- observed z-band magnitude;
- catalogue Sérsic index;
- catalogue axis ratio;
- local r-band PSF FWHM; and
- local r-band depth.

The complete target roster supplies robust centres and interquartile-range
scales. Every component has an absolute standardized caliper of 2 and the
five-dimensional Euclidean distance has a caliper of 3. Assignment maximizes
cardinality and then minimizes total distance, with the registered
identifier-hash order resolving ties. All 43 references receive unique
targets.

## Triplets and readout

Each component contains one observed reference, the matched target's exact
stored bare cutout, and that target's exact stored 1.5-arcsec G_rec
expectation image. There is no rerendering, target replacement, flux
rescaling, imputation, or post-render selection.

For each representation and fold, the linear axis is fitted only from
training-fold observed and bare rows. Injected rows fit neither the axis nor
its scale. The primary movement is the equal-target injected-minus-bare
change in training-bare standard-deviation units toward the observed side.
Pooled AUCs are separate finite-cohort diagnostics.

Matching reduces differences only in the five recorded quantities. It does
not establish equivalence in redshift, mass, environment, source properties,
or unrecorded image characteristics.
