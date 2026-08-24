from car_crash_claim_analyzer.schemas import DamageAssessment, DamageDetection
from car_crash_claim_analyzer.vision.severity import SeverityEstimator


def test_unclassified_damage_gets_stable_fallback_weight():
    assessment = DamageAssessment(
        detections=[
            DamageDetection(
                label="unclassified_damage",
                confidence=0.56,
                bbox=(0, 0, 300, 200),
            )
        ]
    )
    result = SeverityEstimator().estimate(assessment, 600, 400)
    assert result.severity in {"minor", "moderate", "severe"}
    assert 0.0 < result.severity_score <= 1.0


def test_no_detections_means_no_severity():
    result = SeverityEstimator().estimate(DamageAssessment(detections=[]), 600, 400)
    assert result.severity == "none"
    assert result.severity_score == 0.0
