"""End-to-end decision orchestration."""

from claimvision.decision.engine import ClaimDecisionEngine
from claimvision.decision.report import ClaimReportBuilder
from claimvision.schemas import ClaimDecision, ClaimInformation, DamageAssessment, PolicyEvidence


class ClaimDecisionPipeline:
    """CV + claim metadata + policy evidence → decision + explanation."""

    def __init__(self):
        self.engine = ClaimDecisionEngine()
        self.report_builder = ClaimReportBuilder()

    def run(
        self,
        damage: DamageAssessment,
        claim: ClaimInformation,
        evidence: list[PolicyEvidence],
    ) -> tuple[ClaimDecision, str]:
        decision = self.engine.evaluate(damage, claim, evidence)
        report = self.report_builder.build(damage, claim, decision)
        return decision, report
