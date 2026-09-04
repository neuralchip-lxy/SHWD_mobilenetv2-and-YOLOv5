"""Summarize per-class normalized bounding-box scales in a YOLO dataset split."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml


def percentile(values, q):
    """Return an interpolated percentile from an already sorted nonempty list."""
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def parse_args():
    """Parse dataset-scale analysis options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML file")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument(
        "--small-side",
        type=float,
        default=32 / 640,
        help="normalized equivalent-side threshold used to report small-target proportion",
    )
    return parser.parse_args()


def label_directory(dataset, split):
    """Resolve the labels directory for a standard images/<split> YOLO dataset layout."""
    split_path = Path(dataset[split])
    if split_path.parts[:1] != ("images",):
        raise ValueError(f"Expected images/<split> layout, received {dataset[split]!r}")
    return Path(dataset["path"]) / "labels" / Path(*split_path.parts[1:])


def main():
    """Collect class counts and normalized box-scale summaries from label files."""
    options = parse_args()
    dataset = yaml.safe_load(options.data.read_text(encoding="utf-8"))
    names = dataset["names"]
    labels_dir = label_directory(dataset, options.split)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    values = defaultdict(lambda: defaultdict(list))
    image_counts = defaultdict(set)
    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            class_id = int(fields[0])
            _, _, width, height = map(float, fields[1:])
            if not 0 <= class_id < len(names) or width <= 0 or height <= 0:
                continue
            values[class_id]["width"].append(width)
            values[class_id]["height"].append(height)
            values[class_id]["area"].append(width * height)
            values[class_id]["equivalent_side"].append(math.sqrt(width * height))
            image_counts[class_id].add(label_path.stem)

    report = {
        "data": str(options.data.resolve()),
        "split": options.split,
        "label_files": len(label_files),
        "small_side_threshold": options.small_side,
        "classes": {},
    }
    for class_id, class_name in enumerate(names):
        class_values = values[class_id]
        if not class_values["area"]:
            continue
        summary = {"instances": len(class_values["area"]), "images": len(image_counts[class_id])}
        for metric, metric_values in class_values.items():
            metric_values.sort()
            summary[metric] = {
                "p10": percentile(metric_values, 0.10),
                "p25": percentile(metric_values, 0.25),
                "p50": percentile(metric_values, 0.50),
                "p75": percentile(metric_values, 0.75),
                "p90": percentile(metric_values, 0.90),
            }
        sides = class_values["equivalent_side"]
        summary["small_target_proportion"] = sum(side < options.small_side for side in sides) / len(sides)
        report["classes"][class_name] = summary

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
