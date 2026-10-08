# Linear-probe diagnostics of strong-lens source injection

Code, compact data products, results, and manuscript sources for:

> **Linear-probe diagnostics of observed and source-injected strong-lens
> images in DESI Legacy Imaging Surveys DR10**

The primary investigation applies one dependency-aware linear-probe protocol
to the operational detector's penultimate features, Stein self-supervised
survey features, generic ImageNet features, and elementary pixel statistics.
Matched controls constrain interpretation; recovery, source/photon,
and fixed-target brightness experiments separately characterize detector response.

## Reported results

- **Observed endpoint:** the primary EfficientNetV2-S detector flags 39 of 43
  selected confirmed systems at the primary threshold.
- **\(G_{\rm domain}\):** component-weighted OOF AUC is 0.991340 in
  operational penultimate features, 0.958187 in Stein features, 0.978528 in
  ImageNet features, and 0.980745 in pixel statistics. The existing
  operational lens score separately ranks the same populations with AUC
  0.980306. These are not independent replications or physical-realism ranks.
- **\(G_{\rm match}\):** mean injected-minus-bare movement is +0.023183 in
  operational features, +0.235027 in Stein features, +0.024792 in ImageNet
  features, and -0.029207 in pixel statistics. Operational movement is
  positive for 27 targets and negative for 16.
- **\(G_{\rm rec}\):** 1,710 of 49,500 declared recovery-grid injections are
  detected (3.4545%). This is conditional recovery on the grid, not population
  completeness.
- **\(P_{\rm src}\):** the single-Sérsic arm exceeds the structured F814W arm
  by +0.560290 percentage points, with interval [+0.283912, +0.932231].
  This compares complete source constructions rather than morphology alone.
- **\(P_{\rm noise}\):** adding the declared \(g=150\) arc-associated Poisson
  term changes recovery from 3.4545% to 2.2515%, a paired change of -1.203030
  percentage points, interval [-1.387747, -1.027768].
- **Fixed-target brightness and source-photon response:** 344 expectation
  images and 2,752 photon images give 3,096 retained clip-10 score rows over
  magnitudes 18.5--25.5. This is not a visibility-completeness analysis.

Together these results do not assign the broad simulation-to-observation
difference to one cause. They do not support a large fitted-Sérsic suppression
under the tested seeing-limited constructions; target selection, mass
assignment, complex configurations, source-noise/pedestal construction, and
survey processing remain important limitations.

## Repository layout

- `dhs/` — detector, rendering, pairing, matching, and statistics code.
- `scripts/` — reproducible command-line analyses.
- `configs/` — split, rendering, and Poisson configurations.
- `data/manifests/` — separate portable train, validation, and test manifests.
- `results_v2/` — compact scores, features, results, and verification
  records; the public authority below identifies the paper-bound artifacts.
- `experiments_v2/` — declared protocols and analysis packages.
- `manuscript/` — paper source, compiled manuscript, figures, and scientific supplement.
- `docs/` — methods, data, and reproduction instructions.
- `tests/` — physics, statistics, split, and result regression tests.

Public-data boundaries and repository size choices are documented in
`docs/LICENSING_AND_DATA_POLICY.md` and `docs/LARGE_FILE_POLICY.md`.

## Quick start

```bash
uv sync --locked --extra dev
make test-all PYTHON="uv run python"
```

Tests needing external checkpoints or image payloads report explicit skips
when those artifacts are unavailable. The reported outputs and submission
documents are regression-tested by:

```bash
uv run pytest -q tests/test_public_canonical_authority.py
uv run pytest -q tests/test_submission_documents.py
uv run pytest -q tests/test_public_manifests.py
```

The public study authority and reporting reconciliation are:

- `results_v2/canonical_test_host_repair_v1/PUBLIC_STUDY_AUTHORITY.json`;
- `results_v2/canonical_test_host_repair_v1/PUBLIC_REPORTING_SUMMARY.json`.

The complete pixel-level rendering and detector rescoring require DR10 cutouts,
the COSMOS source sample, and the primary detector checkpoint. See
`docs/REPRODUCIBILITY.md`.

## Citation

Please cite the accompanying RASTI manuscript. Citation metadata will be
updated with the journal DOI after publication.

## Licence

Software is MIT-licensed; see `LICENSE`. Manuscript, figure, derived-data, and
third-party-data terms are described in `docs/LICENSING_AND_DATA_POLICY.md`.
