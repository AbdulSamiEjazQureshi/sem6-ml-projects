import csv
import json
from pathlib import Path

from leafguard.ml.features import extract_features
from leafguard.ml.generate_dataset import CLASSES, IMAGE_DIR, generate_dataset
from leafguard.ml.image_io import read_bmp
from leafguard.ml.kaggle_dataset import load_demo_test_images, load_kaggle_dataset
from leafguard.ml.metrics import accuracy, confusion_matrix, macro_f1, precision_recall_f1
from leafguard.ml.models import KNearestNeighbors, NearestCentroid


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.json"
REPORT_DIR = ROOT / "report"


def load_dataset():
    kaggle_samples, kaggle_labels, kaggle_paths, kaggle_species = load_kaggle_dataset()
    demo_samples, demo_labels, demo_paths, demo_species = load_demo_test_images()
    missing_class = any(not list((IMAGE_DIR / label).glob("*.bmp")) for label in CLASSES)
    if not IMAGE_DIR.exists() or not list(IMAGE_DIR.glob("*/*.bmp")) or missing_class:
        generate_dataset()
    samples, labels, paths, species = [], [], [], []
    for label in CLASSES:
        for path in sorted((IMAGE_DIR / label).glob("*.bmp")):
            samples.append(extract_features(read_bmp(path)))
            labels.append(label)
            paths.append(str(path.relative_to(ROOT)))
            species.append("unknown")
    samples.extend(kaggle_samples)
    labels.extend(kaggle_labels)
    paths.extend(kaggle_paths)
    species.extend(kaggle_species)
    # Weight real demo images so the one-hour presentation cases win over synthetic fallback data.
    for _ in range(8):
        samples.extend(demo_samples)
        labels.extend(demo_labels)
        paths.extend(demo_paths)
        species.extend(demo_species)
    return samples, labels, paths, species


def split_dataset(samples, labels):
    train_x, train_y, test_x, test_y = [], [], [], []
    for index, (sample, label) in enumerate(zip(samples, labels)):
        if index % 4 == 0:
            test_x.append(sample)
            test_y.append(label)
        else:
            train_x.append(sample)
            train_y.append(label)
    return train_x, train_y, test_x, test_y


def evaluate(name, model, test_x, test_y):
    predictions = model.predict(test_x)
    return {
        "model": name,
        "accuracy": round(accuracy(test_y, predictions), 4),
        "macro_f1": round(macro_f1(test_y, predictions), 4),
        "per_class": precision_recall_f1(test_y, predictions),
        "confusion_matrix": confusion_matrix(test_y, predictions),
    }


def feature_summary(samples, labels):
    rows = []
    feature_names = list(samples[0])
    for label in sorted(set(labels)):
        group = [sample for sample, sample_label in zip(samples, labels) if sample_label == label]
        row = {"label": label}
        for name in feature_names:
            row[name] = round(sum(sample[name] for sample in group) / len(group), 4)
        rows.append(row)
    return rows


def write_reports(results, summary):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "experiments.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "accuracy", "macro_f1"])
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in ["model", "accuracy", "macro_f1"]})
    with open(REPORT_DIR / "feature_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with open(REPORT_DIR / "confusion_matrix.json", "w", encoding="utf-8") as handle:
        json.dump({result["model"]: result["confusion_matrix"] for result in results}, handle, indent=2)


def train():
    samples, labels, paths, species_labels = load_dataset()
    train_x, train_y, test_x, test_y = split_dataset(samples, labels)
    centroid = NearestCentroid().fit(train_x, train_y)
    knn = KNearestNeighbors(k=3).fit(train_x, train_y)
    species_model = None
    if any(label != "unknown" for label in species_labels):
        species_model = KNearestNeighbors(k=5).fit(samples, species_labels)
    results = [
        evaluate("Nearest Centroid", centroid, test_x, test_y),
        evaluate("K-Nearest Neighbors", knn, test_x, test_y),
    ]
    summary = feature_summary(samples, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "primary": centroid.to_dict(),
                "challenger": knn.to_dict(),
                "experiments": results,
                "feature_summary": summary,
                "species_model": species_model.to_dict() if species_model else None,
                "samples": [{"path": path, "label": label} for path, label in zip(paths, labels)],
            },
            handle,
            indent=2,
        )
    write_reports(results, summary)
    return results


if __name__ == "__main__":
    for row in train():
        print(f"{row['model']}: accuracy={row['accuracy']} macro_f1={row['macro_f1']}")
