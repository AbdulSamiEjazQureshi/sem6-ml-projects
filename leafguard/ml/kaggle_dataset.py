from pathlib import Path

from leafguard.ml.features import extract_features
from leafguard.ml.image_io import read_bmp


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "data" / "kaggle"
TEST_IMAGE_DIR = ROOT / "test_images"
SUPPORTED = {".bmp", ".jpg", ".jpeg", ".png"}
TEST_LABELS = {
    "autumn": ("unknown", "senescent_leaf"),
    "disease": ("unknown", "blight"),
    "disease2": ("unknown", "rust"),
    "healthy": ("unknown", "healthy"),
    "healthy2": ("unknown", "healthy"),
    "healthy_l": ("unknown", "healthy"),
    "disease_pore": ("unknown", "rust"),
    "disease_red": ("unknown", "rust"),
    "disease_spot": ("unknown", "leaf_spot"),
}


def parse_class_name(name):
    if "___" in name:
        species, condition = name.split("___", 1)
        return species.replace("_", " "), condition.replace("_", " ")
    return "unknown", name.replace("_", " ")


def read_image_features(path):
    if path.suffix.lower() == ".bmp":
        return extract_features(read_bmp(path))
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("JPG/PNG Kaggle training requires Pillow in the active Python environment.") from exc
    image = Image.open(path).convert("RGB").resize((96, 96))
    pixels = []
    raw = list(image.getdata())
    for y in range(image.height):
        row = []
        for x in range(image.width):
            row.append(raw[y * image.width + x])
        pixels.append(row)
    return extract_features(pixels)


def load_kaggle_dataset(limit_per_class=80):
    if not KAGGLE_DIR.exists():
        return [], [], [], []
    samples, labels, paths, species_labels = [], [], [], []
    for class_dir in sorted(path for path in KAGGLE_DIR.iterdir() if path.is_dir()):
        species, condition = parse_class_name(class_dir.name)
        count = 0
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED or count >= limit_per_class:
                continue
            samples.append(read_image_features(path))
            labels.append(condition)
            species_labels.append(species)
            paths.append(str(path.relative_to(ROOT)))
            count += 1
    return samples, labels, paths, species_labels


def load_demo_test_images():
    if not TEST_IMAGE_DIR.exists():
        return [], [], [], []
    samples, labels, paths, species_labels = [], [], [], []
    for path in sorted(TEST_IMAGE_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        species, condition = TEST_LABELS.get(path.stem, ("unknown", path.stem))
        samples.append(read_image_features(path))
        labels.append(condition)
        species_labels.append(species)
        paths.append(str(path.relative_to(ROOT)))
    return samples, labels, paths, species_labels
