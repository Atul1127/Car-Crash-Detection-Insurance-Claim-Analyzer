"""Create a contact sheet of class-7 (legacy unknown) annotations.

Run from the repository root:
    python scripts/inspect_unknown.py

The output is useful for deciding whether class 7 is a legitimate class or
ambiguous/noisy annotation before changing the dataset.
"""
from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dataset_audit" / "unknown_samples.jpg"
CLASS_ID = 7
SPLITS = ("train", "valid", "test")
SAMPLES_PER_SPLIT = 8
TILE_W, TILE_H = 320, 240


def find_unknown(split: str) -> list[tuple[Path, list[tuple[float, float, float, float]]]]:
    labels_dir = ROOT / split / "labels"
    images_dir = ROOT / split / "images"
    rows = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        boxes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                vals = tuple(float(x) for x in parts[1:])
            except ValueError:
                continue
            if cls == CLASS_ID:
                boxes.append(vals)
        if not boxes:
            continue
        candidates = [images_dir / f"{label_path.stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp")]
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path:
            rows.append((image_path, boxes))
    return rows


def draw_sample(image_path: Path, boxes: list[tuple[float, float, float, float]], split: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((TILE_W, TILE_H))
    canvas = Image.new("RGB", (TILE_W, TILE_H + 28), "white")
    x = (TILE_W - image.width) // 2
    y = (TILE_H - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    sx, sy = image.width / Image.open(image_path).width, image.height / Image.open(image_path).height
    for xc, yc, w, h in boxes:
        x1 = x + (xc - w / 2) * image.width
        y1 = y + (yc - h / 2) * image.height
        x2 = x + (xc + w / 2) * image.width
        y2 = y + (yc + h / 2) * image.height
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    draw.text((8, TILE_H + 7), f"{split} | class 7 | {image_path.name[:34]}", fill="black")
    return canvas


def main() -> None:
    random.seed(42)
    selected: list[tuple[str, Path, list[tuple[float, float, float, float]]]] = []
    for split in SPLITS:
        rows = find_unknown(split)
        random.shuffle(rows)
        selected.extend((split, path, boxes) for path, boxes in rows[:SAMPLES_PER_SPLIT])

    if not selected:
        raise SystemExit("No class-7 annotations found.")

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE_W, rows * (TILE_H + 28)), "#dddddd")
    for i, (split, path, boxes) in enumerate(selected):
        tile = draw_sample(path, boxes, split)
        sheet.paste(tile, ((i % cols) * TILE_W, (i // cols) * (TILE_H + 28)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=90)
    print(f"Saved: {OUT}")
    print(f"Samples: {len(selected)}")


if __name__ == "__main__":
    main()
