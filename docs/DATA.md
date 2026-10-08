# Data

Implementation field names used in the files below are defined in
[`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## Committed data products

`data/manifests/` contains portable row-level split assignments. Paths are
relative to the separately obtained `stronglens_calibration` data root.

`results_v2/canonical_test_host_repair_v1/gdomain/` contains the canonical
6,379-row \(G_{\rm domain}\) manifest, dependency folds, operational
penultimate, Stein, generic-ImageNet and pixel features, OOF outputs, fixed-head
baseline, and result authority.

`results_v2/canonical_test_host_repair_v1/gmatch/` contains the canonical 43
observed--bare--injected triplets and matched-axis movement outputs for all four
representations.

`results_v2/canonical_test_host_repair_v1/revision_controls/` contains the 99
cell--radius summaries and the reported fixed-target brightness/source-photon
products. The scientific brightness authority has 3,096 clip-10 score rows.

The public authority and derived-reporting reconciliation are
`results_v2/canonical_test_host_repair_v1/PUBLIC_STUDY_AUTHORITY.json` and
`PUBLIC_REPORTING_SUMMARY.json` in the same directory.

These are derived feature arrays, not survey image pixels.

`data/catalogues/lens_provenance.csv` gives the object-level literature
classification and evidence review for the candidate catalogue.
The 43-system table and its sources are under `manuscript/tables/`.

## External inputs

Pixel-level reproduction additionally requires:

- DESI Legacy Imaging Surveys DR10 \(g,r,z\) cutouts.
- The GalSim COSMOS 25.2 real-galaxy sample
  (`https://doi.org/10.5281/zenodo.3242143`).
- The primary EfficientNetV2-S checkpoint.
- The stored 49,500-image recovery-grid archive and source-photon banks for
  reconstruction.

DR10 and COSMOS data remain subject to their original distribution terms.
This repository does not redistribute their raw pixels.

The model and recovery-grid checksums are listed in
`data/downloads/artifacts.yaml`. Their public archival copies are pending and
the private project S3 locations are intentionally not published. Until public
copies are active, the committed score tables and resampling products reproduce
the manuscript statistics without redistributing the original image archives.
