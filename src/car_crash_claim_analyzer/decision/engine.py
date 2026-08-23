"""Deterministic claim decision engine.

This layer does not invent policy coverage. It combines structured visual/claim
signals with explicitly retrieved policy evidence and reports uncertainty.
"""

from car_crash_claim_analyzer.schemas import ClaimDecision, ClaimInformation, DamageAssessment, PolicyEvidence


class ClaimDecisionEngine:
    """Convert evidence into a transparent preliminary claim assessment."""

    def evaluate(
        self,
        damage: DamageAssessment,
        claim: ClaimInformation,
        evidence: list[PolicyEvidence],
    ) -> ClaimDecision:
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

        severity = damage.severity or "unknown"
        severity_score = damage.severity_score or 0.0
        evidence_text = " ".join(item.text.lower() for item in evidence)

        # Evidence presence is a retrieval signal, not proof of coverage.
        coverage_terms = sum(
            term in evidence_text
            for term in ("covered", "coverage", "own damage", "accidental damage")
        )
        exclusion_terms = sum(
            term in evidence_text
            for term in ("excluded", "exclusion", "not covered", "wear and tear")
        )

        if exclusion_terms > coverage_terms:
            coverage_status = "potential_exclusion"
        elif coverage_terms:
            coverage_status = "potentially_covered"
        else:
            coverage_status = "uncertain"

        risk_score = min(
            1.0,
            0.15
            + (0.25 if severity == "severe" else 0.12 if severity == "moderate" else 0.05)
            + (0.15 if not claim.policy_number else 0.0)
            + (0.10 if not claim.incident_date else 0.0)
            + (0.10 if exclusion_terms > coverage_terms else 0.0),
        )

        if warnings or coverage_status in {"uncertain", "potential_exclusion"}:
            decision = "manual_review"
        else:
            decision = "preliminary_coverage_review"

        rationale = (
            f"Detected severity={severity} (score={severity_score:.2f}). "
            f"Retrieved policy evidence suggests status={coverage_status}. "
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
