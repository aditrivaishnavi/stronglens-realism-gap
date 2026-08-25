# Evaluation-dataset preparation

This directory contains the code and compact records used to construct the
detector-development, calibration, evaluation, and injection-target roles.
Detector scores are not used to select the canonical target roster.

## Development and test manifests

The portable manifests under `data/manifests/` contain:

- 290,312 training rows;
- 62,215 validation rows;
- 61,498 test rows, including 61,435 catalogue negatives and 63 confirmed
  systems.

Rows are grouped before assignment using retained object, spatial, and
physical-system identifiers. The manifest validator checks object and path
separation, HEALPix cells, physical groups, spatial components, and the
declared 38-arcsec relation.

## Catalogue-negative preparation

The preparation code:

1. inventories available cutouts and canonicalizes repeated object keys;
2. excludes detector training and validation identities;
3. verifies the payload schema and finite three-band pixels;
4. joins the required Tractor quantities;
5. applies catalogue, photometry, mask, exposure, and proximity rules;
6. forms spatial groups and applies role exclusions; and
7. selects targets deterministically within PSF--depth cells.

The source S3 bucket name is retained in the acquisition utilities because it
identifies the existing project data location. AWS credentials are never
stored in the repository; callers provide their profile or environment
configuration locally.

## Canonical target roster

The final roster contains 4,500 distinct catalogue-negative targets: 500 in
each of nine retained \(r\)-band PSF--depth cells. Selection uses a
namespaced hash, a maximum of five targets per spatial group, and at least 100
spatial groups per cell.

Every selected target belongs to the canonical test-negative population and
is disjoint from the threshold-calibration and achieved-FPR roles in both
object identity and retained evaluation spatial group.

The compact selection records are:

- `final_selection/final_grec_selection.parquet`;
- `final_selection/FINAL_SELECTION_RECEIPT.json`;
- `results_v2/canonical_test_host_repair_v1/selection/`;
- `results_v2/canonical_test_host_repair_v1/grec/hosts/`.

## Observed references

`reference_cohort/` records the 63 literature-confirmed systems and the
catalogue associations used for the selected 43-system endpoint. Object-level
confirmation evidence is published in
`manuscript/tables/confirmed_43_provenance.csv`.

The selected roster is a finite heterogeneous cohort. The repository does not
claim a general eligibility rule that independently selects those 43 systems
from all 63 confirmed systems.

## External pixels

Raw DR10 cutouts are not redistributed through Git. Compact manifests and
receipts record their identifiers and expected hashes. Pixel-level
reproduction requires the external data described in
`data/downloads/artifacts.yaml` and `docs/REPRODUCIBILITY.md`.
