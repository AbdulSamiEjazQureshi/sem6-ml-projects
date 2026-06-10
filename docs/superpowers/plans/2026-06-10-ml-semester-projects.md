# ML Semester Projects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CrisisLens and LeafGuard as offline-first ML semester projects with datasets, APIs, SvelteKit UIs, models, experiments, and research-report material.

**Architecture:** Each project is self-contained. The ML layer uses pure Python modules and serialized JSON model artifacts; the API layer uses `http.server`; the frontend is SvelteKit source that calls the local API. Generated datasets and experiment outputs live inside each project.

**Tech Stack:** Python standard library, `unittest`, JSON/CSV datasets, BMP images, stdlib HTTP API, SvelteKit source, CSS.

---

## File Structure

- `crisislens/ml/text.py`: tokenization, vocabulary, vectorization.
- `crisislens/ml/models.py`: Naive Bayes, perceptron, clustering, serialization.
- `crisislens/ml/metrics.py`: accuracy, precision/recall/F1, confusion matrix.
- `crisislens/ml/train.py`: dataset loading, training, evaluation, report artifact writing.
- `crisislens/api/server.py`: stdlib JSON API.
- `crisislens/data/tweets.csv`: bundled offline dataset.
- `crisislens/report/research_report.md`: report-ready writeup.
- `crisislens/ui/*`: SvelteKit source.
- `leafguard/ml/image_io.py`: BMP generation/parsing.
- `leafguard/ml/features.py`: interpretable feature extraction.
- `leafguard/ml/models.py`: nearest-centroid and KNN classifiers.
- `leafguard/ml/metrics.py`: metrics.
- `leafguard/ml/generate_dataset.py`: offline image dataset generator.
- `leafguard/ml/train.py`: training, evaluation, report artifact writing.
- `leafguard/api/server.py`: stdlib JSON API.
- `leafguard/data/images/*`: bundled/generated BMP samples.
- `leafguard/report/research_report.md`: report-ready writeup.
- `leafguard/ui/*`: SvelteKit source.

## Tasks

### Task 1: CrisisLens tests and ML core

- [ ] Write `unittest` coverage for tokenization, Naive Bayes prediction, perceptron prediction, and metrics.
- [ ] Run the test file and confirm it fails because modules are missing.
- [ ] Implement `text.py`, `models.py`, and `metrics.py`.
- [ ] Run tests and confirm they pass.

### Task 2: CrisisLens dataset, training, report, and API

- [ ] Add a local labeled crisis tweet CSV.
- [ ] Add training script that writes model JSON and report artifacts.
- [ ] Add stdlib HTTP API exposing health, predict, batch, experiments, and clusters.
- [ ] Run training and API-adjacent tests.

### Task 3: LeafGuard tests and image ML core

- [ ] Write `unittest` coverage for BMP roundtrip, feature extraction, nearest-centroid, KNN, and metrics.
- [ ] Run the test file and confirm it fails because modules are missing.
- [ ] Implement `image_io.py`, `features.py`, `models.py`, and `metrics.py`.
- [ ] Run tests and confirm they pass.

### Task 4: LeafGuard dataset, training, report, and API

- [ ] Add dataset generator that creates bundled BMP samples.
- [ ] Add training script that writes model JSON and report artifacts.
- [ ] Add stdlib HTTP API exposing health, predict, predict-image, experiments, and samples.
- [ ] Run generator, training, and tests.

### Task 5: SvelteKit UIs and documentation

- [ ] Add SvelteKit source for CrisisLens and LeafGuard.
- [ ] Add strong visual design using local CSS and no remote assets.
- [ ] Add READMEs with setup, training, API, UI, and demo instructions.
- [ ] Run final Python verification commands.

## Self-Review

- All spec requirements map to a task.
- No placeholder tasks remain.
- Project boundaries are independent.
- The plan acknowledges the missing package environment and keeps ML/API runtime dependency-free.
