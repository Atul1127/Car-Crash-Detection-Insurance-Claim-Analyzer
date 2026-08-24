"""Train the car-damage YOLO detector reproducibly.

Example:
    python scripts/train_yolo.py --epochs 100 --device 0

The dataset currently contains a legacy ``unknown`` class. Do not remove that
class from the YAML until the dataset has been audited; inference normalizes it
to ``unclassified_damage``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--data", default="data/damage_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="car_damage_v2")
    args = parser.parse_args()

    batch: int | str = int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=True,
        patience=20,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()
