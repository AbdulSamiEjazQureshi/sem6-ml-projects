# LeafGuard

Offline crop disease image classifier for a semester ML project.

## Generate Dataset

```bash
python3 -m leafguard.ml.generate_dataset
```

This creates BMP samples under `data/images/`.

## Run ML Training

```bash
python3 -m leafguard.ml.train
```

This writes:

- `models/model.json`
- `report/experiments.csv`
- `report/feature_summary.csv`
- `report/confusion_matrix.json`

## Run API

```bash
python3 -m leafguard.api.server
```

API URL: `http://127.0.0.1:8022`

Endpoints:

- `GET /health`
- `POST /predict` with extracted feature values
- `POST /predict-image` with `{ "path": "data/images/rust/rust_00.bmp" }`
- `GET /experiments`
- `GET /samples`

## Input Images

The bundled training/demo images are BMP because the project can generate and parse BMP files using only the Python standard library. That keeps the offline ML pipeline dependency-free.

For presentation, the UI also accepts external images from the browser:

- JPG
- PNG
- BMP
- any browser-readable `image/*` file

When you upload a normal photo, the browser previews it, extracts the same color/lesion/texture features on a canvas, and sends those numeric features to the ML API.

## Kaggle Datasets

The project now supports real Kaggle-style datasets through:

```text
leafguard/data/kaggle/
```

Use folders like:

```text
leafguard/data/kaggle/Apple___healthy/
leafguard/data/kaggle/Apple___rust/
leafguard/data/kaggle/Tomato___leaf_spot/
```

Then retrain:

```bash
python3 -m leafguard.ml.train
```

If the folders include `Species___Condition`, LeafGuard trains:

- a disease/condition classifier
- a KNN species/tree identity classifier

If no Kaggle species dataset is loaded, the UI honestly reports tree identity as `unknown` instead of faking it.

JPG/PNG Kaggle training requires Pillow in the Python environment. BMP works without Pillow. External JPG/PNG upload in the UI still works because the browser decodes the image and sends extracted features.

## ML Algorithms

LeafGuard uses two classical ML algorithms:

1. **Nearest Centroid**
   - Builds an average feature profile for each disease class.
   - Predicts the class whose average profile is closest to the uploaded leaf.

2. **K-Nearest Neighbors, KNN**
   - Stores training feature vectors.
   - For a new leaf, finds the closest training leaves.
   - Votes among those neighbors.
   - The UI shows the neighbor labels and distances.

Current bundled offline classes:

- `healthy`
- `rust`
- `blight`
- `leaf_spot`
- `senescent_leaf` for autumn/aging leaves, not disease

## Run UI

The UI source is SvelteKit. If SvelteKit dependencies are installed:

```bash
cd leafguard/ui
npm install
npm run dev
```

The ML/API layer does not require pip packages.

## Demo Script

1. Generate the image dataset.
2. Train both classifiers.
3. Start the API.
4. Diagnose `data/images/rust/rust_00.bmp`.
5. Upload an external leaf image from the laptop.
6. Explain that Nearest Centroid and KNN both classify the same extracted features.
7. Discuss interpretable features such as brown ratio, spot ratio, and edge density.
