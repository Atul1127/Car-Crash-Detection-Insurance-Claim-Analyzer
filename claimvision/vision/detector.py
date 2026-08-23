"""YOLO damage detector adapter."""

from pathlib import Path

from ultralytics import YOLO

from claimvision.schemas import DamageAssessment, DamageDetection
from config import YOLO_CONFIDENCE, YOLO_IOU, YOLO_DEVICE


class DamageDetector:
    """Wrap YOLO inference behind a stable application interface."""

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = YOLO_CONFIDENCE,
        iou: float = YOLO_IOU,
    ):
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.iou = iou
        self.last_result = None

    def predict(self, image_path: str | Path, device: str | int | None = None) -> DamageAssessment:
        inference_device = YOLO_DEVICE if device is None else device
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            save=False,
            device=inference_device,
        )
        if not results:
            self.last_result = None
            return DamageAssessment(detections=[])

        # Keep the result so the UI can render the same inference without
        # running YOLO a second time.
        self.last_result = results[0]

        detections: list[DamageDetection] = []
        result = results[0]
        if result.boxes is None:
            return DamageAssessment(detections=detections)

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

    def render_last_result(self):
        """Return the annotated numpy image from the most recent inference."""
        if self.last_result is None:
            raise RuntimeError("No YOLO inference result is available to render.")
        return self.last_result.plot()
