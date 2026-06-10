from pathlib import Path

from leafguard.ml.image_io import generate_leaf_bitmap, write_bmp


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
CLASSES = ["healthy", "rust", "blight", "leaf_spot", "senescent_leaf"]


def generate_dataset(samples_per_class=12, size=48):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label in CLASSES:
        class_dir = IMAGE_DIR / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(samples_per_class):
            path = class_dir / f"{label}_{seed:02d}.bmp"
            write_bmp(path, generate_leaf_bitmap(label, seed=seed, size=size))
            manifest.append({"path": str(path.relative_to(ROOT)), "label": label})
    return manifest


if __name__ == "__main__":
    for row in generate_dataset():
        print(f"{row['label']},{row['path']}")
