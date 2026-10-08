# Licensing and data policy

## Software

Source code in `dhs/`, `scripts/`, `experiments_v2/`, and `tests/` is released
under the repository's MIT licence unless a file states otherwise.
Aditrivaishnavi Balaji is the software author listed in `CITATION.cff`.

## Manuscript and figures

The manuscript and manuscript figures are authored research materials. The
software MIT licence should not be interpreted as a separate grant for
journal-formatted text or figures. Their reuse remains subject to the authors'
and publisher's applicable terms.

## Project-generated data products

Committed manifests, score tables, feature matrices, out-of-fold predictions,
and result summaries are project-generated research products. Their hashes and
scientific roles are recorded in the public authority and evidence index.
The archival record accompanying publication should state their explicit data
licence.

## Third-party data

DESI Legacy Imaging Surveys DR10 and GalSim COSMOS 25.2 pixels remain subject
to their source archives and licences. This repository does not redistribute
those datasets as original project data.

The `darkhaloscope` S3 bucket name identifies the existing project data
location. The repository contains no AWS credentials, access keys, or signed
download URLs.

## Model checkpoints and large image banks

The primary detector checkpoint, Stein checkpoint, generated image banks, and
source-photon pixel banks are not stored in Git. Their expected hashes and
availability states are listed in `data/downloads/artifacts.yaml`. Public
archive records should provide the applicable licence and retrieval URL for
each redistributable artifact.
