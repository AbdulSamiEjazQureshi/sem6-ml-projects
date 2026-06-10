# ML Semester Projects Design

## Goal

Build two offline-first machine learning semester projects that include APIs, SvelteKit UIs, local datasets, training/evaluation scripts, saved model artifacts, and research-report material.

## Project 1: CrisisLens

CrisisLens is a disaster tweet intelligence lab. It classifies short social posts into crisis categories, compares simple NLP models, clusters unlabeled messages into event themes, and exposes explanations that are easy to discuss in a viva.

### Research Questions

- How well can bag-of-words models classify disaster reports when the dataset is small?
- Which model performs better on short crisis text: Multinomial Naive Bayes or a linear perceptron?
- Can keyword-weight explanations make predictions understandable to non-technical emergency staff?
- Can lightweight clustering discover meaningful event themes without labels?

### ML Scope

- Local CSV dataset with labeled crisis messages.
- Tokenization, stopword removal, vocabulary building, TF counts, TF-IDF features.
- Multinomial Naive Bayes classifier.
- Linear perceptron classifier.
- KMeans-style text clustering over TF-IDF vectors.
- Metrics: accuracy, macro-F1, confusion matrix, per-class precision/recall.
- Report artifacts: model comparison CSV, cluster summary CSV, confusion matrix JSON.

### API Scope

The Python standard-library HTTP API provides:

- `GET /health`
- `POST /predict` with `{ "text": "..." }`
- `POST /batch` with `{ "texts": [...] }`
- `GET /experiments`
- `GET /clusters`

### UI Scope

The SvelteKit UI uses an emergency operations aesthetic. It includes a prediction console, batch input panel, model comparison table, cluster board, and explanation chips. The source is kept offline and does not require remote assets.

## Project 2: LeafGuard

LeafGuard is an offline crop disease vision lab. It generates a small bundled BMP image dataset with visually distinct disease patterns, extracts interpretable image features, trains classical image classifiers, and provides a diagnosis API/UI.

### Research Questions

- Can interpretable color, texture, and lesion features classify crop disease patterns in small-data settings?
- How do nearest-centroid and KNN models compare when the image dataset is compact?
- Do augmentation-like pattern variations improve robustness?
- Which features most clearly separate healthy, rust, blight, and leaf spot classes?

### ML Scope

- Local generated BMP image dataset with four classes: `healthy`, `rust`, `blight`, `leaf_spot`.
- Pure-Python BMP generation and parsing.
- Feature extraction: green ratio, brown ratio, yellow ratio, dark lesion ratio, brightness, edge density, spot count proxy.
- Nearest-centroid classifier.
- K-nearest-neighbor classifier.
- Metrics: accuracy, macro-F1, confusion matrix, per-class precision/recall.
- Report artifacts: model comparison CSV, feature summary CSV, confusion matrix JSON.

### API Scope

The Python standard-library HTTP API provides:

- `GET /health`
- `POST /predict` with `{ "features": {...} }`
- `POST /predict-image` with `{ "path": "data/images/...bmp" }`
- `GET /experiments`
- `GET /samples`

### UI Scope

The SvelteKit UI uses a field-diagnostics lab aesthetic. It includes sample selector, feature evidence panel, model comparison table, and diagnosis result card. It is designed for a classroom demo with bundled sample images.

## Offline Constraint

Both projects avoid mandatory pip/npm downloads for the ML/API layer. Python code uses the standard library. SvelteKit source is provided with package manifests; if Node dependencies are installed later, the frontend can run normally. The APIs and ML scripts remain runnable offline immediately.

## Research Report Deliverables

Each project includes:

- `report/research_report.md`
- `report/experiments.csv`
- `report/confusion_matrix.json`
- dataset notes
- methodology
- results
- limitations
- future work
- presentation demo script

## Testing

Tests use Python `unittest` and cover tokenization, model training/prediction, metrics, dataset generation, feature extraction, and API handler helpers where practical.

## Spec Review

- No placeholders remain.
- The two projects are independent and can be evaluated separately.
- The offline constraint is explicit.
- SvelteKit is scoped as source code while API/ML execution remains dependency-free.
