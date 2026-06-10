import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from leafguard.ml.features import extract_features
from leafguard.ml.image_io import read_bmp, read_bmp_bytes
from leafguard.ml.kaggle_dataset import read_image_features
from leafguard.ml.models import KNearestNeighbors, NearestCentroid


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.json"


def load_payload():
    if not MODEL_PATH.exists():
        from leafguard.ml.train import train

        train()
    with open(MODEL_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def regenerate_assets():
    from leafguard.ml.train import train

    train()
    return load_payload()


def save_payload(payload):
    """Write the current payload to model.json."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_features(features, label):
    if label == "healthy":
        return "No disease predicted: the leaf is mostly green with low lesion signals."
    if label == "senescent_leaf":
        return "No infectious disease predicted: the leaf looks like an aging/autumn leaf with high yellow-orange color and low disease lesion structure."
    cues = []
    if features["brown_ratio"] > 0.15:
        cues.append("brown/orange disease patches are visible")
    if features["spot_ratio"] > 0.08:
        cues.append("spot lesions are above healthy level")
    if features["dark_ratio"] > 0.2:
        cues.append("dark damaged tissue is prominent")
    if features["green_ratio"] > 0.9:
        cues.append("leaf remains mostly green")
    if not cues:
        cues.append("feature pattern is subtle and needs model comparison")
    return f"Predicted {label} because " + ", ".join(cues) + "."


def label_metadata(label, features=None, species_model=None):
    disease_labels = {"rust", "blight", "leaf_spot"}
    condition = {
        "healthy": "Healthy leaf",
        "senescent_leaf": "Aging/autumn leaf, not a crop disease",
        "rust": "Fungal rust-like disease pattern",
        "blight": "Blight-like damaged leaf tissue",
        "leaf_spot": "Leaf spot-like lesion pattern",
    }.get(label, "Unknown leaf condition")
    metadata = {
        "is_disease": label in disease_labels,
        "leaf_condition": condition,
        "tree_identity": {
            "status": "not_trained",
            "label": "unknown",
            "message": "Tree/species identity needs a separate Kaggle species dataset; this model currently classifies disease/condition only.",
        },
    }
    if species_model and features:
        species = species_model.predict_one(features)
        metadata["tree_identity"] = {
            "status": "predicted",
            "label": species["label"],
            "confidence": species["confidence"],
            "message": f"Predicted species/tree identity using Kaggle-style species labels: {species['label']}.",
        }
    return metadata


def prediction_from_features(model, features, challenger=None, species_model=None):
    centroid_result = model.predict_one(features)
    challenger_result = challenger.predict_one(features) if challenger else None
    result = challenger_result or centroid_result
    payload = {
        "features": features,
        "human_summary": summarize_features(features, result["label"]),
        **label_metadata(result["label"], features=features, species_model=species_model),
        **result,
    }
    payload["centroid"] = centroid_result
    if challenger_result:
        payload["knn"] = challenger_result
        payload["model_agreement"] = challenger_result["label"] == centroid_result["label"]
        payload["algorithm_note"] = (
            "KNN is the displayed diagnosis because it compares the uploaded leaf with the closest real training leaves. "
            "Nearest Centroid is still shown as a baseline that compares against average disease profiles."
        )
    return payload


def prediction_from_image_bytes(model, image_bytes, challenger=None, species_model=None):
    features = extract_features(read_bmp_bytes(image_bytes))
    return prediction_from_features(model, features, challenger=challenger, species_model=species_model)


class LeafGuardHandler(BaseHTTPRequestHandler):
    payload = None
    model = None
    challenger = None
    species_model = None

    def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send(200, {"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, {"status": "ok", "project": "LeafGuard"})
        elif parsed.path == "/experiments":
            self._send(200, self.payload["experiments"])
        elif parsed.path == "/samples":
            self._send(200, self.payload["samples"][:24])
        elif parsed.path == "/image":
            query = parse_qs(parsed.query)
            requested = query.get("path", [""])[0]
            path = (ROOT / requested).resolve()
            if ROOT not in path.parents or path.suffix.lower() != ".bmp":
                self._send(400, {"error": "invalid image path"})
                return
            self._send_bytes(200, "image/bmp", path.read_bytes())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/predict":
            self._send(200, prediction_from_features(self.model, body.get("features", {}), challenger=self.challenger, species_model=self.species_model))
        elif self.path == "/predict-image":
            path = ROOT / body.get("path", "")
            features = read_image_features(path)
            self._send(200, prediction_from_features(self.model, features, challenger=self.challenger, species_model=self.species_model))
        elif self.path == "/predict-upload":
            data_url = body.get("image", "")
            encoded = data_url.split(",", 1)[-1]
            self._send(200, prediction_from_image_bytes(self.model, base64.b64decode(encoded), challenger=self.challenger, species_model=self.species_model))
        elif self.path == "/feedback":
            features = body.get("features")
            corrected_label = body.get("corrected_label")
            if not features or not corrected_label:
                self._send(400, {"error": "features and corrected_label are required"})
                return
            # Add the user-corrected sample to the KNN model
            self.challenger.add_training_sample(features, corrected_label)
            # Rebuild NearestCentroid from all training data
            all_samples, all_labels = self.challenger.get_all_training_data()
            self.model = NearestCentroid().fit(all_samples, all_labels)
            # Update model entries in payload
            type(self).payload["primary"] = self.model.to_dict()
            type(self).payload["challenger"] = self.challenger.to_dict()
            # Persist to disk
            save_payload(self.payload)
            # Return updated prediction
            result = prediction_from_features(
                self.model, features,
                challenger=self.challenger,
                species_model=self.species_model,
            )
            result["feedback_applied"] = True
            result["corrected_label"] = corrected_label
            result["total_training_samples"] = len(self.challenger.samples)
            self._send(200, result)
        elif self.path == "/regenerate":
            payload = regenerate_assets()
            type(self).payload = payload
            type(self).model = NearestCentroid.from_dict(payload["primary"])
            type(self).challenger = KNearestNeighbors.from_dict(payload["challenger"])
            type(self).species_model = KNearestNeighbors.from_dict(payload["species_model"]) if payload.get("species_model") else None
            self._send(200, {"status": "regenerated", "samples": len(payload["samples"])})
        else:
            self._send(404, {"error": "not found"})


def run(host="127.0.0.1", port=8022):
    payload = load_payload()
    LeafGuardHandler.payload = payload
    LeafGuardHandler.model = NearestCentroid.from_dict(payload["primary"])
    LeafGuardHandler.challenger = KNearestNeighbors.from_dict(payload["challenger"])
    LeafGuardHandler.species_model = KNearestNeighbors.from_dict(payload["species_model"]) if payload.get("species_model") else None
    server = ThreadingHTTPServer((host, port), LeafGuardHandler)
    print(f"LeafGuard API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
