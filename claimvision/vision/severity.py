"""Initial severity estimation heuristic.

This is deliberately conservative: it provides a baseline until a labeled
severity dataset/model is introduced in a later phase.
"""

from claimvision.schemas import DamageAssessment


class SeverityEstimator:
    """Estimate coarse severity from detected damage classes/confidence."""

    HIGH_IMPACT = {"glass_shatter", "head_lamp", "tail_lamp"}
    MEDIUM_IMPACT = {"bumper_dent", "door_dent"}

    def estimate(self, assessment: DamageAssessment) -> DamageAssessment:
        if not assessment.detections:
            assessment.severity = "none"
            assessment.severity_score = 0.0
            return assessment

        weighted_scores = []
        for detection in assessment.detections:
            if detection.label in self.HIGH_IMPACT:
                weight = 0.9
            elif detection.label in self.MEDIUM_IMPACT:
                weight = 0.6
            else:
                weight = 0.35
            weighted_scores.append(weight * detection.confidence)

        score = min(1.0, max(weighted_scores))
        if score >= 0.75:
            severity = "high"
        elif score >= 0.45:
            severity = "medium"
        else:
            severity = "low"

        assessment.severity = severity
        assessment.severity_score = round(score, 3)
        return assessment
