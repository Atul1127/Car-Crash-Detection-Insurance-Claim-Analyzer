"""Build a clean 7-class YOLO dataset from the current 8-class dataset.

Class 7 (legacy ``unknown``) is treated as an ambiguous catch-all and is not
used as a supervised class. Images that contain at least one real class
(0-6) are kept, with class-7 boxes removed. Images containing only class 7
annotations are skipped rather than converted into negative examples.

The original train/valid/test directories are never modified.

Run from repository root:
    python scripts/build_clean_yolo_dataset.py

Output:
    data/clean_7class/{train,valid,test}/{images,labels}
    data/clean_7class.yaml
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "clean_7class"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CLASS_NAMES = [
    "bumper_dent",
    "bumper_scratch",
    "door_dent",
    "door_scratch",
    "glass_shatter",
    "head_lamp",
    "tail_lamp",
]
KNOWN_CLASSES = set(range(len(CLASS_NAMES)))


def image_for(label_path: Path, images_dir: Path) -> Path | None:
    return next(
        (p for p in (images_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTS) if p.exists()),
        None,
    )


def clean_split(split: str) -> tuple[int, int, int]:
    labels_dir = ROOT / split / "labels"
    images_dir = ROOT / split / "images"
    out_labels = OUT / split / "labels"
    out_images = OUT / split / "images"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    kept = skipped_unknown_only = missing_images = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = image_for(label_path, images_dir)
        if image_path is None:
            missing_images += 1
            continue

        known_lines: list[str] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            if cls in KNOWN_CLASSES:
                known_lines.append(raw)

        # Never turn an unknown-only damaged image into a negative example.
        if not known_lines:
            skipped_unknown_only += 1
            continue

        out_image = out_images / image_path.name
        out_label = out_labels / label_path.name
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(known_lines) + "\n", encoding="utf-8")
        kept += 1

    return kept, skipped_unknown_only, missing_images


def write_yaml() -> None:
    yaml = "\n".join(
        [
            "train: clean_7class/train/images",
            "val: clean_7class/valid/images",
            "test: clean_7class/test/images",
            "",
            f"nc: {len(CLASS_NAMES)}",
            f"names: {CLASS_NAMES!r}",
            "",
        ]
    )
    (ROOT / "data" / "clean_7class.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    total = [0, 0, 0]
    for split in ("train", "valid", "test"):
        stats = clean_split(split)
        total = [a + b for a, b in zip(total, stats)]
        print(
            f"{split}: kept={stats[0]}, unknown_only_skipped={stats[1]}, "
            f"missing_images={stats[2]}"
        )

    write_yaml()
    print(f"\nCreated: {OUT}")
    print(f"Created: {ROOT / 'data' / 'clean_7class.yaml'}")
    print(
        f"TOTAL: kept={total[0]}, unknown_only_skipped={total[1]}, "
        f"missing_images={total[2]}"
    )
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
