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
from car_crash_claim_analyzer.schemas import ClaimInformation
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
    "bumper_dent",
    "bumper_scratch",
    "door_dent",
    "door_scratch",
    "glass_shatter",
    "head_lamp",
    "tail_lamp",
    "unknown",
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
                names = model.names
                normalized = {str(value).strip().lower() for value in names.values()}
                checked.append(f"{candidate}: {sorted(normalized)}")
                if EXPECTED_DAMAGE_CLASSES.issubset(normalized):
                    return candidate
            except Exception as exc:
                checked.append(f"{candidate}: load failed ({exc})")

        details = "\n".join(checked)
        raise FileNotFoundError(
            "No compatible trained car-damage YOLO weights were found. "
            "Expected the 8 damage classes from data/damage_dataset.yaml. "
            "Do not use a generic/object-detection best.pt.\n"
            f"Checked:\n{details}"
        )

    @staticmethod
    def _build_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_template(
            """You are an expert insurance claim analysis assistant. Your job is to summarize and interpret ONLY the evidence supplied below.

STRICT EVIDENCE RULES:
1. Use ONLY the supplied policy evidence and structured claim/vehicle evidence.
2. Never invent policy clauses, coverage, exclusions, limits, amounts, dates, or conditions.
3. A retrieved policy passage is evidence, not proof that the passage applies to this claim. Check its wording and applicability before drawing a conclusion.
4. If the supplied evidence does not contain an explicit applicable coverage provision, say exactly: 'Coverage is not established from the retrieved evidence.' Recommend manual review.
5. Never infer coverage merely because words such as 'covered', 'coverage', 'insured', or 'policy' appear in a retrieved passage.
6. Never infer an exclusion merely because the word 'excluded' or 'exclusion' appears. The exclusion must apply to the detected loss/component and scenario.
7. Never treat a monetary amount in a retrieved passage as a claim limit unless the passage explicitly states that it applies to this claim and this type of loss.
8. Preserve the exact meaning and scope of policy conditions. If a condition applies to a different section, liability type, vehicle use, or scenario, do not apply it to own-damage coverage.
9. Distinguish between 'not mentioned in the retrieved evidence' and 'excluded by the policy'. The latter requires an explicit applicable exclusion.
10. The deterministic decision engine is the source of truth for the structured coverage status. Your explanation must not contradict it.

IMPORTANT VISION SEMANTICS:
- A YOLO damage label is a DAMAGE CATEGORY, not a vehicle category.
- 'unknown', 'unclassified_damage', and 'other' mean only that the detector did not map the detected damage to a supported damage category.
- NEVER interpret these labels as 'unknown vehicle', 'unknown vehicle class', 'uninsured vehicle', or any other vehicle/policy classification.
- Do not invent a specific damage type when the detector returns an unknown/unclassified label. Describe it only as detected vehicle damage with the supplied confidence and severity.

OUTPUT RULES:
- Separate facts directly supported by evidence from interpretation.
- Cite policy evidence by source/page when discussing a specific clause.
- For every important coverage/exclusion statement, explain which supplied evidence supports it.
- If evidence conflicts or is insufficient, state that clearly.
- Do not manufacture repair costs or financial values.
- Keep the assessment concise and suitable for an insurance claims reviewer.

Vehicle evidence:
{vehicle_evidence}

Claim information:
{claim_information}

Policy evidence:
{policy_context}

Question:
{question}

Return an evidence-grounded assessment with these sections:
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
        report.damage = self.severity_estimator.estimate(
            report.damage, image_width=width, image_height=height
        )

        image_element = None
        try:
            annotated = self.detector.render_last_result()
            output_image = Image.fromarray(annotated[..., ::-1])
            buffer = io.BytesIO()
            output_image.save(buffer, format="JPEG", quality=90)
            image_element = buffer.getvalue()
        except Exception:
            image_element = None
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
                f"severity: {damage_report.damage.severity}; "
                f"severity_score: {damage_report.damage.severity_score}. "
                "The labels describe detected damage categories only, not vehicle classes."
            )
        else:
            vehicle_evidence = "No vehicle analysis available."

        claim_text = (
            f"policy_number={claim_info.policy_number}; claim_id={claim_info.claim_id}; "
            f"claimant={claim_info.claimant_name}; vehicle={claim_info.vehicle_registration}; "
            f"incident_date={claim_info.incident_date}"
        )
        return await self.llm.ainvoke(
            self.prompt.format(
                vehicle_evidence=vehicle_evidence,
                claim_information=claim_text,
                policy_context=policy_context,
                question=message_query,
            )
        )

    async def decide(self, damage_report, claim_info, evidence):
        return await asyncio.to_thread(
            self.decision_pipeline.run,
            damage_report.damage,
            claim_info,
            evidence,
        )
