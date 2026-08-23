"""YOLO damage detector adapter with portable device selection."""

from pathlib import Path

from ultralytics import YOLO

from claimvision.schemas import DamageAssessment, DamageDetection


class DamageDetector:
    """Wrap YOLO inference behind a stable application interface."""

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.iou = iou

    @staticmethod
    def resolve_device(device: str | int = "auto") -> str | int:
        if device != "auto":
            return device
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(
        self,
        image_path: str | Path,
        device: str | int = "auto",
    ) -> DamageAssessment:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            save=False,
            device=self.resolve_device(device),
            verbose=False,
        )

        detections: list[DamageDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    DamageDetection(
                        label=str(self.model.names[cls_id]),
                        confidence=float(box.conf[0]),
                        bbox=tuple(float(v) for v in box.xyxy[0].tolist()),
                    )
                )

        return DamageAssessment(detections=detections)
