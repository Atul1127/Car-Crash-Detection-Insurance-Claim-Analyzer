"""Evaluate the trained car-damage YOLO checkpoint on the test split.

Usage:
    python scripts/evaluate_yolo.py
    python scripts/evaluate_yolo.py --weights models/best.pt --device cpu

If ``--device`` is omitted, CPU is used when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO
from config import YOLO_MODEL_PATH

EXPECTED_CLASSES = [
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp", "unknown",
]


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.isdigit() and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(YOLO_MODEL_PATH))
    parser.add_argument("--data", default="data/damage_dataset.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    device = resolve_device(args.device)

    output_dir = PROJECT_ROOT / "reports" / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    names = {int(k): str(v) for k, v in model.names.items()}
    print("Model classes:")
    for idx, name in names.items():
        print(f"  {idx}: {name}")

    missing = set(EXPECTED_CLASSES) - set(names.values())
    if missing:
        raise SystemExit(f"Checkpoint is missing expected classes: {sorted(missing)}")

    print(f"Evaluation device: {device}")
    metrics = model.val(
        data=str(PROJECT_ROOT / args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        plots=True,
        project=str(output_dir),
        name=f"{Path(args.weights).stem}_{args.split}",
        exist_ok=True,
        verbose=True,
    )

    print("\n=== YOLO TEST SUMMARY ===")
    print(f"Precision:   {metrics.box.mp:.4f}")
    print(f"Recall:      {metrics.box.mr:.4f}")
    print(f"mAP@50:      {metrics.box.map50:.4f}")
    print(f"mAP@50-95:   {metrics.box.map:.4f}")

    print("\n=== PER-CLASS AP@50-95 ===")
    for class_id, ap in enumerate(metrics.box.maps):
        print(f"{names.get(class_id, str(class_id))}: {ap:.4f}")

    print(f"\nPlots and confusion matrix: {output_dir / (Path(args.weights).stem + '_' + args.split)}")


if __name__ == "__main__":
    main()
