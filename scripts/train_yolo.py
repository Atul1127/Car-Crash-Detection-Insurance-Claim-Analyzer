"""Train the car-damage YOLO detector reproducibly.

Examples:
    python scripts/train_yolo.py --data data/clean_7class.yaml --weights yolov8n.pt --epochs 100 --device auto --name car_damage_7class
    python scripts/train_yolo.py --data data/clean_7class.yaml --weights models/best.pt --epochs 100 --device cpu --name car_damage_7class

The clean 7-class experiment excludes the ambiguous legacy ``unknown`` class.
The original dataset remains untouched by the cleaning script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.isdigit() and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--data", default="data/damage_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="car_damage_v2")
    args = parser.parse_args()

    batch: int | str = int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch
    device = resolve_device(args.device)
    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
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
