.PHONY: run train test build-ui report verify crisislens leafguard help

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

run:
	$(PYTHON) scripts/run_projects.py

train:
	$(PYTHON) -m crisislens.ml.train
	$(PYTHON) -m leafguard.ml.train

test:
	$(PYTHON) -m unittest crisislens.tests.test_ml_core crisislens.tests.test_api_explain leafguard.tests.test_ml_core leafguard.tests.test_api_upload

build-ui:
	npm --prefix crisislens/ui run build
	npm --prefix leafguard/ui run build

report:
	$(PYTHON) scripts/build_leafguard_report.py

verify: train test build-ui
	$(PYTHON) -m compileall -q crisislens leafguard scripts

crisislens:
	$(PYTHON) -m crisislens.api.server

leafguard:
	$(PYTHON) -m leafguard.api.server

help:
	@echo "make run      Train both projects and start both APIs + UIs"
	@echo "make train    Regenerate model/report artifacts"
	@echo "make test     Run Python unit tests"
	@echo "make build-ui Build both SvelteKit UIs"
	@echo "make report   Build the LeafGuard A4 research PDF"
	@echo "make verify   Train, test, build UIs, and compile-check"
	@echo "make crisislens  Start only CrisisLens API on http://127.0.0.1:8011"
	@echo "make leafguard   Start only LeafGuard API on http://127.0.0.1:8022"
