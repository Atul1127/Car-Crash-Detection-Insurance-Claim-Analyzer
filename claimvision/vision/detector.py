"""YOLO damage detector adapter."""

from pathlib import Path

from ultralytics import YOLO

from claimvision.schemas import DamageAssessment, DamageDetection


class DamageDetector:
    """Wrap the existing YOLO model behind a stable application interface."""

    def __init__(self, weights_path: str | Path, confidence: float = 0.25):
        self.model = YOLO(str(weights_path))
        self.confidence = confidence

    def predict(self, image_path: str | Path, device: str | int = "cpu") -> DamageAssessment:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            save=False,
            device=device,
        )

        detections: list[DamageDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                coords = tuple(float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    DamageDetection(
                        label=str(self.model.names[cls_id]),
                        confidence=confidence,
                        bbox=coords,  # type: ignore[arg-type]
                    )
                )

        return DamageAssessment(detections=detections)
