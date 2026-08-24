"""Deterministic claim decision engine.

This layer does not invent policy coverage. It combines structured visual/claim
signals with explicitly applicable policy evidence and reports uncertainty.
"""

from car_crash_claim_analyzer.schemas import (
    ClaimDecision,
    ClaimInformation,
    DamageAssessment,
    PolicyEvidence,
)


_DAMAGE_TERMS = {
    "bumper_dent": ("bumper", "dent"),
    "bumper_scratch": ("bumper", "scratch"),
    "door_dent": ("door", "dent"),
    "door_scratch": ("door", "scratch"),
    "glass_shatter": ("glass", "shatter", "window", "windshield"),
    "head_lamp": ("headlamp", "head lamp", "headlight"),
    "tail_lamp": ("taillamp", "tail lamp", "tail light", "tail-light"),
}

_UNKNOWN_DAMAGE_LABELS = {"unknown", "unclassified", "unclassified_damage", "other"}


class ClaimDecisionEngine:
    """Convert evidence into a transparent preliminary claim assessment."""

    @staticmethod
    def _evidence_coverage_status(
        evidence: list[PolicyEvidence], damage: DamageAssessment
    ) -> tuple[str, int, int]:
        texts = [item.text.lower() for item in evidence]
        all_text = " ".join(texts)
        coverage_patterns = (
            "own damage",
            "loss of or damage to the vehicle",
            "loss of or damage to the insured vehicle",
            "accidental damage to the vehicle",
            "accidental damage to the insured vehicle",
            "damage to the insured vehicle",
            "loss or damage to the vehicle",
        )
        coverage_hits = sum(pattern in all_text for pattern in coverage_patterns)

        detected_labels = {(d.label or "").strip().lower() for d in damage.detections}
        detected_terms: set[str] = set()
        for label in detected_labels:
            detected_terms.update(_DAMAGE_TERMS.get(label, ()))

        exclusion_hits = 0
        if detected_terms:
            exclusion_markers = ("excluded", "exclusion", "not covered", "shall not cover")
            for text in texts:
                if any(marker in text for marker in exclusion_markers) and any(term in text for term in detected_terms):
                    exclusion_hits += 1

        if exclusion_hits > 0:
            return "potential_exclusion", coverage_hits, exclusion_hits
        if coverage_hits > 0:
            return "potentially_covered", coverage_hits, exclusion_hits
        return "uncertain", coverage_hits, exclusion_hits

    def evaluate(self, damage: DamageAssessment, claim: ClaimInformation, evidence: list[PolicyEvidence]) -> ClaimDecision:
        warnings: list[str] = []
        if not evidence:
            return ClaimDecision(
                decision="manual_review",
                coverage_status="unknown",
                risk_score=0.5,
                rationale="No policy evidence was retrieved; coverage cannot be established.",
                warnings=["Policy evidence is missing."],
            )

        if not claim.policy_number:
            warnings.append("Policy number is missing or could not be extracted.")
        if not claim.incident_date:
            warnings.append("Incident date is missing or could not be extracted.")
        if not damage.detections:
            warnings.append("No supported vehicle damage was detected.")

        unknown_damage = any((d.label or "").strip().lower() in _UNKNOWN_DAMAGE_LABELS for d in damage.detections)
        if unknown_damage:
            warnings.append(
                "The vision model detected damage but did not assign a supported damage category; "
                "the label must not be interpreted as a vehicle class or as a policy condition."
            )

        severity = damage.severity or "unknown"
        severity_score = damage.severity_score or 0.0
        coverage_status, coverage_hits, exclusion_hits = self._evidence_coverage_status(evidence, damage)

        # An unknown visual subtype does not erase explicit Section I own-damage
        # coverage. It requires manual review because the component/category is
        # unresolved, but the policy evidence can still establish potential coverage.
        # This prevents the detector's legacy catch-all class from incorrectly
        # turning clear own-damage policy language into "uncertain" coverage.

        risk_score = min(
            1.0,
            0.15
            + (0.25 if severity == "severe" else 0.12 if severity == "moderate" else 0.05)
            + (0.15 if not claim.policy_number else 0.0)
            + (0.10 if not claim.incident_date else 0.0)
            + (0.10 if exclusion_hits > 0 else 0.0)
            + (0.10 if unknown_damage else 0.0)
            + (0.05 if coverage_hits == 0 else 0.0),
        )

        # Unknown damage and missing claim metadata remain manual-review signals,
        # even when policy evidence establishes potential own-damage coverage.
        if warnings or coverage_status in {"uncertain", "potential_exclusion", "unknown"}:
            decision = "manual_review"
        else:
            decision = "preliminary_coverage_review"

        rationale = (
            f"Detected severity={severity} (score={severity_score:.2f}). "
            f"Retrieved policy evidence supports status={coverage_status}. "
            "Coverage status is based on explicit applicable policy language, not keyword presence. "
            "This is a preliminary evidence-based assessment, not an automatic approval or denial."
        )
        return ClaimDecision(
            decision=decision,
            coverage_status=coverage_status,
            risk_score=round(risk_score, 3),
            rationale=rationale,
            evidence=evidence,
            warnings=warnings,
        )
