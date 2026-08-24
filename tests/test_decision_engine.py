from car_crash_claim_analyzer.decision.engine import ClaimDecisionEngine
from car_crash_claim_analyzer.schemas import (
    ClaimInformation,
    DamageAssessment,
    DamageDetection,
    PolicyEvidence,
)


def _damage(label="head_lamp", severity="severe", score=0.8):
    return DamageAssessment(
        detections=[DamageDetection(label=label, confidence=0.8, bbox=(0, 0, 10, 10))],
        severity=severity,
        severity_score=score,
    )


def test_generic_coverage_word_does_not_establish_coverage():
    evidence = [PolicyEvidence(text="This policy contains conditions and coverage information.", page=2)]
    result = ClaimDecisionEngine().evaluate(_damage(), ClaimInformation(), evidence)
    assert result.coverage_status == "uncertain"
    assert result.decision == "manual_review"


def test_explicit_own_damage_provision_establishes_potential_coverage():
    evidence = [
        PolicyEvidence(
            text="Section I: The Company will indemnify the insured against loss of or damage to the insured vehicle.",
            page=2,
        )
    ]
    claim = ClaimInformation(policy_number="POL123", incident_date="2026-08-24")
    result = ClaimDecisionEngine().evaluate(_damage(), claim, evidence)
    assert result.coverage_status == "potentially_covered"
    assert result.decision == "preliminary_coverage_review"


def test_unclassified_damage_keeps_potential_coverage_but_forces_manual_review():
    evidence = [
        PolicyEvidence(
            text="The Company will indemnify the insured against loss of or damage to the insured vehicle.",
            page=2,
        )
    ]
    claim = ClaimInformation(policy_number="POL123", incident_date="2026-08-24")
    result = ClaimDecisionEngine().evaluate(_damage("unclassified_damage"), claim, evidence)
    assert result.coverage_status == "potentially_covered"
    assert result.decision == "manual_review"
    assert any("must not be interpreted as a vehicle class" in w for w in result.warnings)


def test_unrelated_exclusion_does_not_become_component_exclusion():
    evidence = [
        PolicyEvidence(
            text="Section I covers loss of or damage to the insured vehicle. Damage to tyres and tubes is excluded unless the vehicle is damaged at the same time.",
            page=6,
        )
    ]
    claim = ClaimInformation(policy_number="POL123", incident_date="2026-08-24")
    result = ClaimDecisionEngine().evaluate(_damage("head_lamp"), claim, evidence)
    assert result.coverage_status == "potentially_covered"
