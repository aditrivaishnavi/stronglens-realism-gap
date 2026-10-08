# Public supporting-evidence index

This index identifies the repository-relative records supporting the reported
manuscript values. The canonical public authority is
`results_v2/canonical_test_host_repair_v1/PUBLIC_STUDY_AUTHORITY.json`
(SHA-256
`eeabfce1eeb8632a92b75731e1121f807ceb2d6a5c937bf347779f030a2a3dbe`).

The public checkout uses repository-relative and declared external-artifact
paths. Original execution hashes and their sanitized public counterparts are
bound separately in
`results_v2/canonical_test_host_repair_v1/PUBLIC_PATH_SANITIZATION.json`.

Final-state public protocols are provided in
`evaluation_dataset_prep/GROUND_RULES.md`,
`evaluation_dataset_prep/experiments/`, and
`docs/G_MATCH_V2_AUTHORITY.md`.

## Detector and endpoints

- Threshold calibration:
  `results_v2/evaluation_v2/PUBLIC_THRESHOLD_PROTOCOL.json`
- Catalogue-negative endpoint:
  `results_v2/evaluation_v2/achieved_fpr_scores/`
- Selected 43-system endpoint:
  `results_v2/evaluation_v2/terminal_confirmed_scores/`
- Object-level confirmation provenance:
  `manuscript/tables/confirmed_43_provenance.csv`

## Recovery and interventions

- Recovery grid:
  `results_v2/canonical_test_host_repair_v1/grec/`
- Source-photon experiment:
  `results_v2/canonical_test_host_repair_v1/pnoise/`
- Paired source-construction experiment:
  `results_v2/canonical_test_host_repair_v1/psrc/`
- Fixed-target brightness response:
  `results_v2/canonical_test_host_repair_v1/revision_controls/`

## Representation analyses

- Observed–injected feature comparison:
  `results_v2/canonical_test_host_repair_v1/gdomain/`
- Matched target assignments and movement:
  `results_v2/canonical_test_host_repair_v1/gmatch/`
- Visual and tabular matching supplement:
  `manuscript/supplementary_matching/`

## Robustness and numerical checks

- Checkpoint panel:
  `results_v2/canonical_test_host_repair_v1/checkpoint_sensitivity/`
  and
  `results_v2/canonical_test_host_repair_v1/checkpoint_sensitivity_terminal/`
- Renderer comparison cohort:
  `results_v2/grec/renderer_reconciliation_v1/`
  with public hash verification in
  `scripts/verify_public_renderer_evidence.py`
- Reporting summary:
  `results_v2/canonical_test_host_repair_v1/PUBLIC_REPORTING_SUMMARY.json`
  (SHA-256
  `fa1f0730dc2e146a7721e324fa484d01bcb31baa1a474eb6eef714c2c514d7f5`)

## Figures and tables

Figure and table generators, inputs, and output hashes are recorded in:

- `manuscript/figures/sources.json`
- `manuscript/tables/sources.json`

Large third-party or project-generated pixel payloads are not stored in Git.
Their expected hashes remain recorded in the relevant source registries and
receipts.
