# Reproducibility

## Environment

The tested public environment uses Python 3.12 and the committed `uv.lock`:

```bash
uv sync --locked --extra dev --extra s3
```

## Split validation

```bash
uv run python scripts/validate_manifests.py \
  --train data/manifests/train.parquet \
  --validation data/manifests/validation.parquet \
  --test data/manifests/test.parquet
```

This checks identifiers, relative paths, HEALPix cells, physical groups,
spatial components, and the 38-arcsec cross-split separation.

The primary training configuration is `configs/models/primary_seed42.yaml`:

```bash
uv run python dhs/scripts/run_experiment.py \
  --config configs/models/primary_seed42.yaml
```

## Verification with committed inputs

```bash
uv run pytest -q tests/test_public_canonical_authority.py
uv run pytest -q tests/test_public_manifests.py
uv run pytest -q tests/test_submission_documents.py
uv run python manuscript/verify_submission.py
```

These checks use the committed compact score tables, feature matrices,
out-of-fold predictions, result records, figure/table registries, and active
submission documents.

The analysis implementations are retained under `experiments_v2/`. Their
production configurations record the original input bindings and are not
standalone entry points for a checkout without the external checkpoint and
pixel payloads. The default public workflow verifies the committed compact
products rather than attempting to recreate unavailable image banks.

## Pixel-level reproduction

Complete pixel regeneration requires the primary checkpoint, DR10 cutouts,
and the COSMOS 25.2 source sample. Their expected destinations and hashes are
listed in `data/downloads/artifacts.yaml`. Large pixel banks and checkpoints
are intentionally not stored in Git.

The committed compact products are sufficient to inspect and recompute the
reported statistical summaries. Renderer and feature-extraction workflows
validate their external inputs against the recorded hashes before scoring.

Public records use repository-relative paths and the external destinations in
`data/downloads/artifacts.yaml`. The dual execution/public hash mapping is
recorded in
`results_v2/canonical_test_host_repair_v1/PUBLIC_PATH_SANITIZATION.json`.

## Tests

```bash
make test-all PYTHON="uv run python"
```

Tests requiring optional COSMOS, lens-modelling, or GPU inputs skip when those
inputs are unavailable. This command runs both the root regression suite and
the eight experiment-package suites. The result-regression tests use committed
compact artifacts and should run on CPU.

## Manuscript

Build the manuscript from the repository root:

```bash
uv run python manuscript/build_documents.py
```

The manuscript build also requires Tectonic. PDF verification requires
Poppler's `pdfinfo` and `pdftotext` utilities.
