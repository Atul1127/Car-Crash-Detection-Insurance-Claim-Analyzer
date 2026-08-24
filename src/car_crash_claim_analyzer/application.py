import asyncio
import io
from pathlib import Path

from PIL import Image
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from ultralytics import YOLO

from car_crash_claim_analyzer.claim.pipeline import ClaimDocumentPipeline
from car_crash_claim_analyzer.decision.pipeline import ClaimDecisionPipeline
from car_crash_claim_analyzer.pipeline import CarCrashClaimAnalyzerPipeline
from car_crash_claim_analyzer.rag.context import format_context
from car_crash_claim_analyzer.rag.pipeline import PolicyRAGPipeline
from car_crash_claim_analyzer.vision.detector import DamageDetector
from car_crash_claim_analyzer.vision.severity import SeverityEstimator
from config import (
    FAISS_INDEX_PATH,
    LOCAL_EMBED_MODEL,
    MODEL_NAME,
    POLICY_DOCUMENT_PATH,
    RETRIEVAL_K,
    YOLO_MODEL_PATH,
)

EXPECTED_DAMAGE_CLASSES = {
    "bumper_dent", "bumper_scratch", "door_dent", "door_scratch",
    "glass_shatter", "head_lamp", "tail_lamp", "unknown",
}


class ClaimAnalysisApplication:
    """Application/service layer independent of the Chainlit UI."""

    def __init__(self) -> None:
        self.detector = DamageDetector(self._resolve_weights())
        self.severity_estimator = SeverityEstimator()
        self.vision_pipeline = CarCrashClaimAnalyzerPipeline()
        self.claim_pipeline = ClaimDocumentPipeline()
        self.decision_pipeline = ClaimDecisionPipeline()
        self.policy_rag = PolicyRAGPipeline(
            embedding_model=LOCAL_EMBED_MODEL,
            top_k=RETRIEVAL_K,
            index_path=FAISS_INDEX_PATH,
        )
        self.policy_rag.build(POLICY_DOCUMENT_PATH)
        self.llm = Ollama(model=MODEL_NAME)
        self.prompt = self._build_prompt()

    @staticmethod
    def _resolve_weights() -> Path:
        """Find the trained damage model and reject unrelated YOLO weights."""
        configured = Path(YOLO_MODEL_PATH)
        candidates = [
            configured,
            Path("runs/detect/train-3/weights") / configured.name,
            Path("runs/detect/train/weights") / configured.name,
        ]
        checked: list[str] = []
        for candidate in candidates:
            if not candidate.exists() or candidate in candidates[: candidates.index(candidate)]:
                continue
            try:
                model = YOLO(str(candidate))
                normalized = {str(value).strip().lower() for value in model.names.values()}
                checked.append(f"{candidate}: {sorted(normalized)}")
                if EXPECTED_DAMAGE_CLASSES.issubset(normalized):
                    return candidate
            except Exception as exc:
                checked.append(f"{candidate}: load failed ({exc})")
        raise FileNotFoundError(
            "No compatible trained car-damage YOLO weights were found. "
            "Expected the 8 damage classes from data/damage_dataset.yaml. "
            "Do not use a generic/object-detection best.pt.\n" + "\n".join(checked)
        )

    @staticmethod
    def _build_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """You are an expert insurance claim analysis assistant. Summarize and interpret ONLY the supplied evidence.

STRICT EVIDENCE RULES:
1. Never invent policy clauses, coverage, exclusions, limits, amounts, dates, or conditions.
2. A retrieved passage is evidence; determine applicability from its actual wording and section.
3. Never infer coverage merely because generic words such as 'coverage', 'covered', 'insured', or 'policy' appear.
4. If the evidence explicitly states that the insurer indemnifies the insured against loss of or damage to the insured vehicle, or explicitly covers accidental damage to the insured vehicle, that IS evidence of general own-damage coverage. Do not later claim that no coverage provision exists.
5. General own-damage coverage and claim-specific applicability are different questions. If general coverage is established but the vision model reports unclassified_damage, say that general coverage is evidenced but claim-specific applicability remains uncertain because the loss category is not reliably classified.
6. An exclusion requires an explicit applicable exclusion for the claim scenario. A generic exclusion, a third-party liability clause, towing clause, or owner-driver injury clause must NOT be presented as an exclusion for physical damage to the insured vehicle unless its wording actually applies to the current scenario.
7. A deductible/excess is a financial obligation. A notice/reporting requirement is a separate procedural condition. NEVER combine them into one clause.
8. Never describe a notice requirement as part of the deductible/excess. Never describe a deductible/excess as a notice requirement.
9. A deductible or excess is NOT an exclusion and must never be described with phrases such as 'the policy excludes the claim' unless the clause explicitly says so. Report it separately as a financial condition.
10. A repair estimate threshold is a repair-authorization condition, not automatically a coverage limit. A monetary amount is not a claim limit unless the supplied clause explicitly identifies it as one.
11. A towing provision is relevant only if the current claim involves towing or a towed vehicle/property. Do not list it merely because it appears in the retrieved context.
12. Do not use a condition from Section II/III/IV to negate Section I own-damage coverage unless the supplied wording explicitly connects them.
13. Distinguish: (a) coverage explicitly established, (b) exclusion explicitly established, (c) financial/repair condition, (d) procedural claim requirement, and (e) applicability unresolved. Never collapse these into one statement.
14. The deterministic decision engine is the source of truth for the structured decision. Do not contradict its manual-review/coverage result.

CLAUSE-BOUNDARY RULE:
- Treat each quoted policy clause/sentence as an independent evidence unit.
- Do not join a deductible sentence with an adjacent notice sentence.
- Do not attribute a page's general requirements to the deductible clause unless the actual deductible wording says so.
- If you cannot identify the exact wording supporting a statement, omit the statement rather than infer it.

CLAIM-SCOPE RULE:
- This assessment concerns accidental physical damage to the insured vehicle unless the supplied claim information explicitly describes another scenario.
- Do NOT discuss Total Loss (TL), Constructive Total Loss (CTL), IDV/market-value thresholds, vehicle age percentages, third-party liability, towing, owner-driver personal accident, or other sections merely because they appear in retrieved evidence.
- Mention TL/CTL or IDV only when the claim evidence explicitly indicates a total-loss scenario or the retrieved provision is directly necessary to resolve the current claim.
- A general policy definition is not itself a claim-specific finding.

VISION SEMANTICS:
- YOLO labels are damage categories only, never vehicle categories.
- 'unknown', 'unclassified_damage', and 'other' mean the detector did not map the damage to a supported category.
- Never interpret them as unknown vehicle, unknown vehicle class, uninsured vehicle, or a policy condition.
- Do not invent a component when the label is unclassified_damage.

EVIDENCE PRIORITY:
1. Explicit Section I / own-damage / loss-or-damage-to-the-insured-vehicle wording.
2. Explicit applicable exclusions or conditions for that same own-damage section.
3. Applicable deductible/excess and repair conditions, clearly labeled as financial or procedural terms rather than exclusions.
4. Other sections only when their wording explicitly applies to this claim.

OUTPUT RULES:
- Keep the assessment concise and suitable for an insurance reviewer.
- Cite source/page for every specific policy claim.
- Never say 'the policy does not provide coverage' when the supplied evidence explicitly establishes general own-damage coverage.
- Never call a deductible, excess, repair threshold, or unrelated towing clause an exclusion.
- Never claim that a deductible amount is unknown 'for unclassified damage'; a deductible is policy-level unless the evidence explicitly makes it damage-category-specific.
- If the deductible amount is not present in retrieved evidence, simply say the applicable amount was not retrieved; do not infer why.
- Never say 'the claim does not appear to fall under exclusions/conditions' unless the retrieved evidence and supplied claim facts are sufficient to establish that conclusion.
- If no applicable exclusion was identified, say exactly: 'No applicable exclusion was identified in the retrieved evidence; final applicability requires manual review because claim details and/or damage classification are incomplete.'
- Do not mention generic 'repair conditions' unless a specific repair condition and its source/page are actually retrieved and applicable.
- If general coverage is established but the claim cannot be classified, use: 'General own-damage coverage is evidenced in the retrieved policy, but claim-specific applicability cannot be determined from the current visual classification.'
- If no applicable coverage clause is retrieved, use: 'Coverage is not established from the retrieved evidence.'

Vehicle evidence:
{vehicle_evidence}

Claim information:
{claim_information}

Policy evidence:
{policy_context}

Question:
{question}

Return exactly these sections:
1. Findings
2. Coverage assessment
3. Conditions/exclusions actually supported by the evidence
4. Uncertainty / manual-review reasons
"""
        )

    async def analyze_image(self, path: str):
        report = await asyncio.to_thread(self.vision_pipeline.run, path, detector=self.detector)
        if not report.image_quality.valid:
            return report, None, None
        with Image.open(path) as source_image:
            width, height = source_image.size
        report.damage = self.severity_estimator.estimate(report.damage, image_width=width, image_height=height)
        image_element = None
        try:
            annotated = self.detector.render_last_result()
            output_image = Image.fromarray(annotated[..., ::-1])
            buffer = io.BytesIO()
            output_image.save(buffer, format="JPEG", quality=90)
            image_element = buffer.getvalue()
        except Exception:
            pass
        return report, image_element, [d.label for d in report.damage.detections]

    async def extract_claim(self, path: str):
        return await asyncio.to_thread(self.claim_pipeline.run, path)

    async def retrieve_policy(self, query: str):
        evidence = await asyncio.to_thread(self.policy_rag.retrieve, query)
        return evidence, format_context(evidence)

    async def reason(self, message_query: str, damage_report, claim_info, policy_context: str):
        if damage_report and damage_report.damage:
            vehicle_evidence = (
                f"Detected damage labels: {[d.label for d in damage_report.damage.detections]}; "
                f"severity: {damage_report.damage.severity}; severity_score: {damage_report.damage.severity_score}. "
                "The labels describe detected damage categories only, not vehicle classes."
            )
        else:
            vehicle_evidence = "No vehicle analysis available."
        claim_text = (
            f"policy_number={claim_info.policy_number}; claim_id={claim_info.claim_id}; "
            f"claimant={claim_info.claimant_name}; vehicle={claim_info.vehicle_registration}; "
            f"incident_date={claim_info.incident_date}"
        )
        return await self.llm.ainvoke(self.prompt.format(
            vehicle_evidence=vehicle_evidence,
            claim_information=claim_text,
            policy_context=policy_context,
            question=message_query,
        ))

    async def decide(self, damage_report, claim_info, evidence):
        return await asyncio.to_thread(self.decision_pipeline.run, damage_report.damage, claim_info, evidence)
