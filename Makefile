.PHONY: test test-packages test-all validate verify manuscript

PYTHON ?= python3

test:
	$(PYTHON) -m pytest -q

test-packages:
	PYTHONPATH=experiments_v2/grec_pnoise/src $(PYTHON) -m pytest -q experiments_v2/grec_pnoise/tests
	PYTHONPATH=experiments_v2/grec_pnoise/src:experiments_v2/pnoise_pilot_v2/src $(PYTHON) -m pytest -q experiments_v2/pnoise_pilot_v2/tests
	PYTHONPATH=experiments_v2/psrc/src $(PYTHON) -m pytest -q experiments_v2/psrc/tests
	PYTHONPATH=experiments_v2/probes/src $(PYTHON) -m pytest -q experiments_v2/probes/tests
	PYTHONPATH=experiments_v2/gmatch_canonical_repair_v1/src $(PYTHON) -m pytest -q experiments_v2/gmatch_canonical_repair_v1/tests
	PYTHONPATH=experiments_v2/gmatch_analysis_v2/src:experiments_v2/probes/src $(PYTHON) -m pytest -q experiments_v2/gmatch_analysis_v2/tests
	PYTHONPATH=experiments_v2/grec_pnoise/src:experiments_v2/evaluation_endpoints_v2/src $(PYTHON) -m pytest -q experiments_v2/evaluation_endpoints_v2/tests
	PYTHONPATH=experiments_v2/grec_pnoise/src:experiments_v2/checkpoint_sensitivity_v2/src:experiments_v2/evaluation_endpoints_v2/src:experiments_v2/psrc/src $(PYTHON) -m pytest -q experiments_v2/checkpoint_sensitivity_v2/tests

test-all: test test-packages

validate:
	$(PYTHON) scripts/validate_manifests.py \
		--train data/manifests/train.parquet \
		--validation data/manifests/validation.parquet \
		--test data/manifests/test.parquet

verify:
	$(PYTHON) -m pytest -q tests/test_public_canonical_authority.py
	$(PYTHON) -m pytest -q tests/test_public_documentation.py
	$(PYTHON) -m pytest -q tests/test_public_path_sanitization.py
	$(PYTHON) -m pytest -q tests/test_public_renderer_evidence.py
	$(PYTHON) -m pytest -q tests/test_public_manifests.py
	$(PYTHON) -m pytest -q tests/test_submission_documents.py
	$(PYTHON) manuscript/verify_submission.py

manuscript:
	mkdir -p manuscript/build
	cd manuscript && tectonic -X compile main.tex --outdir build
