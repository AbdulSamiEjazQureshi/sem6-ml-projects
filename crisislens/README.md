# CrisisLens

Offline disaster tweet intelligence lab for a semester ML project.

## Run ML Training

```bash
python3 -m crisislens.ml.train
```

This writes:

- `models/model.json`
- `report/experiments.csv`
- `report/confusion_matrix.json`
- `report/clusters.json`

## Run API

```bash
python3 -m crisislens.api.server
```

API URL: `http://127.0.0.1:8011`

Endpoints:

- `GET /health`
- `POST /predict` with `{ "text": "wildfire smoke evacuation order" }`
- `POST /batch` with `{ "texts": ["...", "..."] }`
- `GET /experiments`
- `GET /clusters`

## Run UI

The UI source is SvelteKit. If SvelteKit dependencies are installed:

```bash
cd crisislens/ui
npm install
npm run dev
```

The ML/API layer does not require pip packages.

## Demo Script

1. Train models and show `report/experiments.csv`.
2. Start the API.
3. In the UI, classify: `wildfire smoke near homes evacuation needed`.
4. Open clusters and explain how unlabeled crisis messages can be grouped.
5. Discuss why Naive Bayes beats perceptron on this small dataset.
