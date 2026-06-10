import unittest
from pathlib import Path

from leafguard.api.server import prediction_from_features, prediction_from_image_bytes, regenerate_assets
from leafguard.ml.kaggle_dataset import TEST_LABELS, load_demo_test_images
from leafguard.ml.image_io import generate_leaf_bitmap, write_bmp_bytes
from leafguard.ml.models import KNearestNeighbors, NearestCentroid
from leafguard.ml.features import extract_features


class LeafGuardUploadTests(unittest.TestCase):
    def test_prediction_from_uploaded_bmp_bytes(self):
        samples = []
        labels = []
        for label in ["healthy", "rust", "blight", "leaf_spot"]:
            for seed in range(4):
                samples.append(extract_features(generate_leaf_bitmap(label, seed=seed, size=28)))
                labels.append(label)
        model = NearestCentroid().fit(samples, labels)
        image_bytes = write_bmp_bytes(generate_leaf_bitmap("rust", seed=99, size=28))

        result = prediction_from_image_bytes(model, image_bytes)

        self.assertEqual(result["label"], "rust")
        self.assertIn("features", result)
        self.assertIn("human_summary", result)

    def test_prediction_from_external_canvas_features(self):
        model = NearestCentroid().fit(
            [
                {"green_ratio": 0.95, "brown_ratio": 0.01, "yellow_ratio": 0.0, "dark_ratio": 0.02, "spot_ratio": 0.0, "brightness": 90.0, "edge_density": 0.05},
                {"green_ratio": 0.65, "brown_ratio": 0.25, "yellow_ratio": 0.02, "dark_ratio": 0.08, "spot_ratio": 0.22, "brightness": 82.0, "edge_density": 0.12},
            ],
            ["healthy", "rust"],
        )
        features = {"green_ratio": 0.63, "brown_ratio": 0.27, "yellow_ratio": 0.01, "dark_ratio": 0.09, "spot_ratio": 0.2, "brightness": 80.0, "edge_density": 0.11}

        result = prediction_from_features(model, features)

        self.assertEqual(result["label"], "rust")
        self.assertIn("brown", result["human_summary"])

    def test_prediction_can_compare_centroid_and_knn(self):
        samples = [
            {"green_ratio": 0.95, "brown_ratio": 0.01, "yellow_ratio": 0.0, "dark_ratio": 0.02, "spot_ratio": 0.0, "brightness": 90.0, "edge_density": 0.05},
            {"green_ratio": 0.92, "brown_ratio": 0.02, "yellow_ratio": 0.0, "dark_ratio": 0.03, "spot_ratio": 0.0, "brightness": 88.0, "edge_density": 0.06},
            {"green_ratio": 0.65, "brown_ratio": 0.25, "yellow_ratio": 0.02, "dark_ratio": 0.08, "spot_ratio": 0.22, "brightness": 82.0, "edge_density": 0.12},
            {"green_ratio": 0.62, "brown_ratio": 0.28, "yellow_ratio": 0.01, "dark_ratio": 0.09, "spot_ratio": 0.24, "brightness": 80.0, "edge_density": 0.13},
        ]
        labels = ["healthy", "healthy", "rust", "rust"]
        centroid = NearestCentroid().fit(samples, labels)
        knn = KNearestNeighbors(k=3).fit(samples, labels)

        result = prediction_from_features(centroid, samples[-1], challenger=knn)

        self.assertEqual(result["label"], "rust")
        self.assertTrue(result["is_disease"])
        self.assertEqual(result["tree_identity"]["status"], "not_trained")
        self.assertEqual(result["knn"]["label"], "rust")
        self.assertTrue(result["model_agreement"])
        self.assertIn("neighbors", result["knn"])

    def test_senescent_leaf_response_is_not_a_disease(self):
        features = {"green_ratio": 0.05, "brown_ratio": 0.35, "yellow_ratio": 0.45, "orange_ratio": 0.6, "dark_ratio": 0.05, "spot_ratio": 0.02, "brightness": 118.0, "edge_density": 0.05}
        model = NearestCentroid().fit([features], ["senescent_leaf"])

        result = prediction_from_features(model, features)

        self.assertFalse(result["is_disease"])
        self.assertIn("aging", result["leaf_condition"].lower())

    def test_species_model_adds_tree_identity(self):
        features = {"green_ratio": 0.8, "brown_ratio": 0.05, "yellow_ratio": 0.02, "orange_ratio": 0.0, "dark_ratio": 0.02, "spot_ratio": 0.01, "brightness": 95.0, "edge_density": 0.05}
        disease_model = NearestCentroid().fit([features], ["healthy"])
        species_model = KNearestNeighbors(k=1).fit([features], ["Apple"])

        result = prediction_from_features(disease_model, features, species_model=species_model)

        self.assertEqual(result["tree_identity"]["status"], "predicted")
        self.assertEqual(result["tree_identity"]["label"], "Apple")

    def test_regenerate_assets_returns_samples_and_models(self):
        payload = regenerate_assets()

        self.assertGreaterEqual(len(payload["samples"]), 40)
        self.assertIn("primary", payload)
        self.assertIn("challenger", payload)

    def test_local_demo_jpgs_match_expected_labels(self):
        payload = regenerate_assets()
        centroid = NearestCentroid.from_dict(payload["primary"])
        knn = KNearestNeighbors.from_dict(payload["challenger"])
        samples, _labels, paths, _species = load_demo_test_images()

        self.assertGreaterEqual(len(samples), 4)
        for sample, path in zip(samples, paths):
            expected = TEST_LABELS[Path(path).stem][1]
            with self.subTest(path=path):
                result = prediction_from_features(centroid, sample, challenger=knn)
                self.assertEqual(result["label"], expected)


if __name__ == "__main__":
    unittest.main()
