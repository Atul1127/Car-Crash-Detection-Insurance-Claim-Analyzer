"""Audit the YOLO dataset before retraining.

Usage:
    python scripts/audit_yolo_dataset.py

Reports class balance, missing label/image pairs, malformed labels, and very
small bounding boxes. This should be run before changing the model because the
current test metrics show substantial per-class variation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(raw: str, yaml_path: Path) -> Path:
    p = Path(raw)
    return (yaml_path.parent / p).resolve() if not p.is_absolute() else p


def audit_split(name: str, image_dir: Path, label_dir: Path, class_names: list[str]) -> None:
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    image_stems = {p.stem for p in images}
    labels = list(label_dir.rglob("*.txt")) if label_dir.exists() else []
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    counts = Counter()
    malformed = 0
    tiny_boxes = 0
    total_boxes = 0

    for label_file in labels:
        for line_no, raw in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                malformed += 1
                continue
            if cls < 0 or cls >= len(class_names) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                malformed += 1
                continue
            counts[cls] += 1
            total_boxes += 1
            if w * h < 0.01:
                tiny_boxes += 1

    print(f"\n=== {name.upper()} ===")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Boxes:  {total_boxes}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Malformed labels: {malformed}")
    print(f"Very small boxes (<1% image area): {tiny_boxes}")
    print("Class distribution:")
    for idx, class_name in enumerate(class_names):
        print(f"  {idx}: {class_name:<18} {counts[idx]:>5} boxes")

    if missing_labels:
        print("  First missing labels:", ", ".join(missing_labels[:5]))
    if missing_images:
        print("  First missing images:", ", ".join(missing_images[:5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/damage_dataset.yaml")
    args = parser.parse_args()

    yaml_path = (PROJECT_ROOT / args.data).resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = config["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    for split in ("train", "val", "test"):
        raw_images = config.get(split)
        if not raw_images:
            continue
        image_dir = resolve_path(raw_images, yaml_path)
        label_dir = image_dir.parent / "labels"
        audit_split(split, image_dir, label_dir, list(names))


if __name__ == "__main__":
    main()
