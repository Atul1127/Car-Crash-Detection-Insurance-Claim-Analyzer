"""Interpretable baseline severity estimation from detection geometry."""

from car_crash_claim_analyzer.schemas import DamageAssessment


class SeverityEstimator:
    """Estimate coarse severity until a severity-labelled model is available."""

    PART_WEIGHTS = {
        "glass_shatter": 0.70,
        "head_lamp": 0.65,
        "tail_lamp": 0.65,
        "bumper_dent": 0.45,
        "bumper_scratch": 0.25,
        "door_dent": 0.45,
        "door_scratch": 0.25,
        "unknown": 0.35,
    }

    def estimate(self, assessment: DamageAssessment, image_width: int, image_height: int) -> DamageAssessment:
        if not assessment.detections or image_width <= 0 or image_height <= 0:
            assessment.severity = "none"
            assessment.severity_score = 0.0
            return assessment

        image_area = float(image_width * image_height)
        contributions = []
        for detection in assessment.detections:
            x1, y1, x2, y2 = detection.bbox
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1) / image_area)
            geometry_score = min(area_ratio / 0.20, 1.0)
            type_weight = self.PART_WEIGHTS.get(detection.label, 0.35)
            contributions.append(0.45 * detection.confidence + 0.35 * geometry_score + 0.20 * type_weight)

        score = min(1.0, sum(contributions) / len(contributions) + 0.08 * max(0, len(contributions) - 1))
        if score < 0.35:
            severity = "minor"
        elif score < 0.68:
            severity = "moderate"
        else:
            severity = "severe"

        assessment.severity = severity
        assessment.severity_score = round(score, 3)
        return assessment
