"""YOLO damage detector adapter."""

from pathlib import Path

from ultralytics import YOLO

from car_crash_claim_analyzer.schemas import DamageAssessment, DamageDetection
from config import YOLO_CONFIDENCE, YOLO_IOU, YOLO_DEVICE


KNOWN_DAMAGE_CLASSES = {
    "bumper_dent",
    "bumper_scratch",
    "door_dent",
    "door_scratch",
    "glass_shatter",
    "head_lamp",
    "tail_lamp",
}
UNCLASSIFIED_LABEL = "unclassified_damage"
LEGACY_UNKNOWN_LABELS = {"unknown", "unclassified", "other"}


class DamageDetector:
    """Wrap YOLO inference behind a stable application interface.

    The current trained checkpoint still contains a legacy ``unknown`` class.
    That class is not exposed as a business damage category. At inference time
    it is normalized to ``unclassified_damage`` so downstream claim reasoning
    cannot mistake it for a meaningful damage type or vehicle class.
    """

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
        self._validate_classes()

    def _validate_classes(self) -> None:
        names = {str(name).strip().lower() for name in self.model.names.values()}
        missing = KNOWN_DAMAGE_CLASSES - names
        if missing:
            raise ValueError(
                "YOLO model is missing expected damage classes: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _resolve_device(device: str | int | None) -> str | int:
        """Resolve auto to an actually available inference device."""
        if device is None or str(device).lower() == "auto":
            try:
                import torch
                return 0 if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = (label or "").strip().lower()
        if normalized in LEGACY_UNKNOWN_LABELS:
            return UNCLASSIFIED_LABEL
        return normalized

    def predict(self, image_path: str | Path, device: str | int | None = None) -> DamageAssessment:
        inference_device = self._resolve_device(YOLO_DEVICE if device is None else device)
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            iou=self.iou,
            save=False,
            device=inference_device,
            verbose=False,
        )
        if not results:
            self.last_result = None
            return DamageAssessment(detections=[])

        self.last_result = results[0]
        detections: list[DamageDetection] = []
        result = results[0]
        if result.boxes is None:
            return DamageAssessment(detections=detections)

        # Keep the annotated image semantically consistent with the normalized
        # application label instead of rendering the legacy ``unknown`` class.
        result.names = dict(result.names)

        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coords = tuple(float(v) for v in box.xyxy[0].tolist())
            raw_label = str(self.model.names[cls_id])
            label = self._normalize_label(raw_label)
            if raw_label.strip().lower() in LEGACY_UNKNOWN_LABELS:
                result.names[cls_id] = UNCLASSIFIED_LABEL

            detections.append(
                DamageDetection(
                    label=label,
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
