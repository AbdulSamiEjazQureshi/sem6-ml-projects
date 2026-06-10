import tempfile
import unittest
from pathlib import Path

from leafguard.ml.features import extract_features
from leafguard.ml.image_io import generate_leaf_bitmap, read_bmp, write_bmp
from leafguard.ml.metrics import accuracy, confusion_matrix, macro_f1
from leafguard.ml.models import KNearestNeighbors, NearestCentroid


class LeafGuardCoreTests(unittest.TestCase):
    def test_bmp_roundtrip_preserves_dimensions(self):
        pixels = generate_leaf_bitmap("rust", seed=3, size=24)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.bmp"
            write_bmp(path, pixels)
            loaded = read_bmp(path)

        self.assertEqual(len(loaded), 24)
        self.assertEqual(len(loaded[0]), 24)
        self.assertEqual(loaded[12][12], pixels[12][12])

    def test_feature_extraction_detects_rust_coloration(self):
        rust = generate_leaf_bitmap("rust", seed=5, size=32)
        healthy = generate_leaf_bitmap("healthy", seed=5, size=32)

        rust_features = extract_features(rust)
        healthy_features = extract_features(healthy)

        self.assertGreater(rust_features["brown_ratio"], healthy_features["brown_ratio"])
        self.assertGreater(rust_features["spot_ratio"], healthy_features["spot_ratio"])

    def test_autumn_leaf_is_not_classified_as_rust(self):
        samples = []
        labels = []
        for label in ["healthy", "rust", "blight", "leaf_spot", "senescent_leaf"]:
            for seed in range(6):
                samples.append(extract_features(generate_leaf_bitmap(label, seed=seed, size=32)))
                labels.append(label)
        model = KNearestNeighbors(k=3).fit(samples, labels)

        prediction = model.predict_one(extract_features(generate_leaf_bitmap("senescent_leaf", seed=99, size=32)))

        self.assertEqual(prediction["label"], "senescent_leaf")

    def test_nearest_centroid_classifies_generated_leaf_pattern(self):
        samples = []
        labels = []
        for label in ["healthy", "rust", "blight", "leaf_spot"]:
            for seed in range(4):
                samples.append(extract_features(generate_leaf_bitmap(label, seed=seed, size=28)))
                labels.append(label)
        model = NearestCentroid()
        model.fit(samples, labels)

        prediction = model.predict_one(extract_features(generate_leaf_bitmap("blight", seed=99, size=28)))

        self.assertEqual(prediction["label"], "blight")
        self.assertGreater(prediction["confidence"], 0.25)

    def test_knn_classifies_generated_leaf_pattern(self):
        samples = []
        labels = []
        for label in ["healthy", "rust", "blight", "leaf_spot"]:
            for seed in range(5):
                samples.append(extract_features(generate_leaf_bitmap(label, seed=seed, size=28)))
                labels.append(label)
        model = KNearestNeighbors(k=3)
        model.fit(samples, labels)

        prediction = model.predict_one(extract_features(generate_leaf_bitmap("leaf_spot", seed=42, size=28)))

        self.assertEqual(prediction["label"], "leaf_spot")

    def test_metrics_report_macro_f1_and_confusion_matrix(self):
        truth = ["healthy", "rust", "rust", "blight"]
        pred = ["healthy", "rust", "blight", "blight"]

        self.assertEqual(accuracy(truth, pred), 0.75)
        self.assertGreater(macro_f1(truth, pred), 0.7)
        self.assertEqual(confusion_matrix(truth, pred)["rust"]["blight"], 1)


if __name__ == "__main__":
    unittest.main()
