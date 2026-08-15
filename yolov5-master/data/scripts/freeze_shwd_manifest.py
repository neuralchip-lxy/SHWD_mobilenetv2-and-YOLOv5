"""Freeze a YOLO-format SHWD dataset split and write a reproducibility manifest."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_path_for(image_path, dataset_root):
    """Map an image path under images/ to its corresponding YOLO label path."""
    relative = image_path.relative_to(dataset_root)
    return dataset_root / "labels" / relative.relative_to("images").with_suffix(".txt")


def inspect_split(split, image_dir, dataset_root, class_names, output_dir):
    """Write a frozen image list and return label, corruption, and checksum statistics."""
    image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    class_counts = {name: 0 for name in class_names}
    empty_labels = 0
    missing_labels = 0
    corrupt_images = []
    invalid_labels = []
    labels_digest = hashlib.sha256()
    frozen_list = []

    for image_path in image_paths:
        relative = image_path.relative_to(dataset_root).as_posix()
        frozen_list.append(relative)
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            corrupt_images.append(relative)

        label_path = label_path_for(image_path, dataset_root)
        if not label_path.exists():
            missing_labels += 1
            continue

        label_bytes = label_path.read_bytes()
        labels_digest.update(relative.encode("utf-8"))
        labels_digest.update(label_bytes)
        rows = [row.strip() for row in label_bytes.decode("utf-8", errors="replace").splitlines() if row.strip()]
        if not rows:
            empty_labels += 1
            continue

        for row_number, row in enumerate(rows, start=1):
            values = row.split()
            try:
                class_index = int(values[0])
                coordinates = [float(value) for value in values[1:]]
                if len(values) != 5 or class_index not in range(len(class_names)) or any(
                    value < 0 or value > 1 for value in coordinates
                ):
                    raise ValueError
            except (ValueError, IndexError):
                invalid_labels.append(f"{relative}:{row_number}:{row}")
                continue
            class_counts[class_names[class_index]] += 1

    (output_dir / f"{split}_images.txt").write_text("\n".join(frozen_list) + "\n", encoding="utf-8")
    (output_dir / f"{split}_corrupt_images.txt").write_text("\n".join(corrupt_images) + "\n", encoding="utf-8")
    (output_dir / f"{split}_invalid_labels.txt").write_text("\n".join(invalid_labels) + "\n", encoding="utf-8")
    return {
        "images": len(image_paths),
        "instances_by_class": class_counts,
        "total_instances": sum(class_counts.values()),
        "empty_label_images": empty_labels,
        "missing_label_images": missing_labels,
        "corrupt_images": len(corrupt_images),
        "invalid_label_rows": len(invalid_labels),
        "labels_aggregate_sha256": labels_digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description="Freeze a SHWD YOLO dataset split for reproducible experiments.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the SHWD dataset YAML file.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the frozen manifest.")
    args = parser.parse_args()

    data_path = args.data.resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(data["path"]).resolve()
    class_names = list(data["names"])
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copy2(data_path, output_dir / "shwd_frozen.yaml")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_yaml": str(data_path),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "data_yaml_sha256": sha256_file(data_path),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        image_dir = dataset_root / data[split]
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} image directory: {image_dir}")
        summary["splits"][split] = inspect_split(split, image_dir, dataset_root, class_names, output_dir)

    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = (
        "This directory freezes the SHWD split used for a paper experiment.\n"
        "The image lists are relative to dataset_root in dataset_manifest.json.\n"
        "Do not change the source data YAML, image lists, or labels after this manifest is created.\n"
    )
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nManifest written to: {output_dir}")


if __name__ == "__main__":
    main()
