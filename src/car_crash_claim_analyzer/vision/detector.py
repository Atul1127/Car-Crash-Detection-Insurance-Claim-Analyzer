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

    The checkpoint still contains a legacy ``unknown`` class. It is never
    exposed as a business damage category. If the checkpoint returns only
    ``unknown`` detections, a single lower-threshold rescue pass is used to
    recover a supported damage class when the model has evidence for one.
    If no supported class reaches the rescue threshold, the result remains
    ``unclassified_damage`` and is sent to manual review.
    """

    RESCUE_CONFIDENCE = 0.10
    RESCUE_MIN_KNOWN_CONFIDENCE = 0.20
    MAX_RESCUE_DETECTIONS = 3

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

    def _collect_detections(self, result, only_known: bool = False) -> list[DamageDetection]:
        detections: list[DamageDetection] = []
        if result.boxes is None:
            return detections

        result.names = dict(result.names)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_label = str(self.model.names[cls_id])
            raw_normalized = raw_label.strip().lower()
            if only_known and raw_normalized not in KNOWN_DAMAGE_CLASSES:
                continue

            label = self._normalize_label(raw_label)
            if raw_normalized in LEGACY_UNKNOWN_LABELS:
                result.names[cls_id] = UNCLASSIFIED_LABEL

            coords = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(
                DamageDetection(
                    label=label,
                    confidence=confidence,
                    bbox=coords,  # type: ignore[arg-type]
                )
            )
        return detections

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

        result = results[0]
        detections = self._collect_detections(result)

        # The legacy unknown class can dominate even when the model has weak
        # evidence for a supported damage class. Only in that case, perform one
        # cheaper recovery pass instead of lowering the normal threshold for
        # every request.
        known_detections = [d for d in detections if d.label in KNOWN_DAMAGE_CLASSES]
        if not known_detections and detections:
            rescue_results = self.model.predict(
                source=str(image_path),
                conf=self.RESCUE_CONFIDENCE,
                iou=self.iou,
                save=False,
                device=inference_device,
                verbose=False,
            )
            if rescue_results:
                rescue_result = rescue_results[0]
                known_detections = [
                    d
                    for d in self._collect_detections(rescue_result, only_known=True)
                    if d.confidence >= self.RESCUE_MIN_KNOWN_CONFIDENCE
                ]
                known_detections.sort(key=lambda d: d.confidence, reverse=True)
                if known_detections:
                    self.last_result = rescue_result
                    return DamageAssessment(
                        detections=known_detections[: self.MAX_RESCUE_DETECTIONS]
                    )

        self.last_result = result
        return DamageAssessment(detections=detections)

    def render_last_result(self):
        """Return the annotated numpy image from the most recent inference."""
        if self.last_result is None:
            raise RuntimeError("No YOLO inference result is available to render.")
        return self.last_result.plot()
