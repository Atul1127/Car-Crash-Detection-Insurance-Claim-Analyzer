"""Explainable claim-report rendering."""

from car_crash_claim_analyzer.schemas import ClaimDecision, ClaimInformation, DamageAssessment


class ClaimReportBuilder:
    """Render a human-readable preliminary claim assessment."""

    def build(
        self,
        damage: DamageAssessment,
        claim: ClaimInformation,
        decision: ClaimDecision,
    ) -> str:
        damage_lines = [
            f"- {item.label}: confidence={item.confidence:.2f}"
            for item in damage.detections
        ] or ["- No supported damage detected"]

        evidence_lines = []
        for item in decision.evidence:
            location = item.source or "policy"
            if item.page is not None:
                location += f", page {item.page}"
            evidence_lines.append(f"- {location}")

        warnings = "\n".join(f"- {item}" for item in decision.warnings) or "- None"

        return (
            "# Preliminary Claim Assessment\n\n"
            f"**Decision:** `{decision.decision}`\n\n"
            f"**Coverage status:** `{decision.coverage_status}`\n\n"
            f"**Risk score:** `{decision.risk_score:.2f}`\n\n"
            "## Vehicle Damage\n"
            + "\n".join(damage_lines)
            + "\n\n"
            "## Claim Information\n"
            f"- Claim ID: `{claim.claim_id or 'missing'}`\n"
            f"- Policy number: `{claim.policy_number or 'missing'}`\n"
            f"- Incident date: `{claim.incident_date or 'missing'}`\n\n"
            "## Rationale\n"
            f"{decision.rationale}\n\n"
            "## Policy Evidence\n"
            + ("\n".join(evidence_lines) or "- No evidence")
            + "\n\n"
            "## Warnings\n"
            + warnings
        )
