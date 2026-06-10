import math
from collections import Counter, defaultdict


FEATURES = [
    "green_ratio",
    "brown_ratio",
    "yellow_ratio",
    "orange_ratio",
    "dark_ratio",
    "spot_ratio",
    "brightness",
    "edge_density",
]

FEATURE_SCALES = {
    "brightness": 255.0,
}


def _distance(left, right):
    return math.sqrt(
        sum(
            ((left.get(name, 0.0) - right.get(name, 0.0)) / FEATURE_SCALES.get(name, 1.0)) ** 2
            for name in FEATURES
        )
    )


class NearestCentroid:
    def __init__(self):
        self.centroids = {}
        self.labels = []

    def fit(self, samples, labels):
        self.labels = sorted(set(labels))
        grouped = defaultdict(list)
        for sample, label in zip(samples, labels):
            grouped[label].append(sample)
        self.centroids = {}
        for label, rows in grouped.items():
            self.centroids[label] = {
                name: sum(row.get(name, 0.0) for row in rows) / len(rows) for name in FEATURES
            }
        return self

    def predict_one(self, sample):
        distances = {label: _distance(sample, centroid) for label, centroid in self.centroids.items()}
        best = min(distances, key=distances.get)
        inverse = {label: 1 / (distance + 1e-9) for label, distance in distances.items()}
        total = sum(inverse.values()) or 1.0
        return {
            "label": best,
            "confidence": inverse[best] / total,
            "distances": distances,
            "evidence": sorted(FEATURES, key=lambda name: abs(sample.get(name, 0.0) - self.centroids[best].get(name, 0.0)), reverse=True)[:4],
        }

    def predict(self, samples):
        return [self.predict_one(sample)["label"] for sample in samples]

    def to_dict(self):
        return {"kind": "nearest_centroid", "centroids": self.centroids, "labels": self.labels}

    @classmethod
    def from_dict(cls, payload):
        model = cls()
        model.centroids = payload["centroids"]
        model.labels = payload["labels"]
        return model


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.samples = []
        self.labels = []

    def fit(self, samples, labels):
        self.samples = samples
        self.labels = labels
        return self

    def predict_one(self, sample):
        neighbors = sorted(
            ((_distance(sample, train), label) for train, label in zip(self.samples, self.labels)),
            key=lambda row: row[0],
        )[: self.k]
        votes = Counter(label for _, label in neighbors)
        best = votes.most_common(1)[0][0]
        return {
            "label": best,
            "confidence": votes[best] / self.k,
            "neighbors": [{"distance": distance, "label": label} for distance, label in neighbors],
        }

    def predict(self, samples):
        return [self.predict_one(sample)["label"] for sample in samples]

    def add_training_sample(self, sample, label):
        """Append one labeled sample for online/active learning."""
        self.samples.append(sample)
        self.labels.append(label)
        return self

    def get_all_training_data(self):
        """Return (samples, labels) for refitting other models."""
        return self.samples, self.labels

    def to_dict(self):
        return {"kind": "knn", "k": self.k, "samples": self.samples, "labels": self.labels}

    @classmethod
    def from_dict(cls, payload):
        model = cls(payload["k"])
        model.samples = payload["samples"]
        model.labels = payload["labels"]
        return model
