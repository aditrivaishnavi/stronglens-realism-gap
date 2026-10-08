# G_match public protocol

This document records the final score-blind preparation and analysis contract
for the 43 observed--bare--injected matched triplets.

## Population and matching

The observed references are the selected 43 confirmed systems used by
G_domain. Candidate targets are the exact 4,500 G_rec targets. Matching uses
only observed z-band magnitude, Sérsic index, axis ratio, r-band PSF FWHM,
and r-band depth.

Target-based robust centres and interquartile ranges define the standardized
coordinates. Component calipers are 2, total-distance caliper is 3, assignment
is without replacement, and identifier hashes define deterministic tie order.
The assignment matches all 43 references.

No detector score, outcome, feature representation, fitted probe, rendered
source flux, or target-total-flux floor enters eligibility or assignment.

## Pair construction

Each selected target is linked to its exact 1.5-arcsec G_rec scene. The bare
row uses the stored target cutout; the injected row uses the stored
target-plus-source expectation. Both retain the registered target, source,
lens, PSF, renderer, random, and intrinsic-flux identities.

The public preparation records distinguish hashes of complete files from
hashes of their scientific pixel arrays. Validators reopen the tracked or
declared external artifacts and verify the expected layout, dtype, shape,
identities, and source addition.

## Dependency and analysis contract

Each observed--bare--injected triplet belongs to one dependency component.
Recorded direct and transitive links to other analysis rows remain within one
probe fold.

In each feature protocol, the observed--bare linear axis and its scale are
fitted without injected rows. Reported movement is the within-target
injected-minus-bare change along that held-out axis. Observed--bare,
observed--injected, and injected--bare AUCs are reported separately and are
not interpreted as causal effects.

## Scope

The protocol supports a matched finite-cohort diagnostic. It does not make
the targets physical deflector analogues, establish population equivalence,
or identify the physical cause of observed--injected separation.
