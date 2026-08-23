import io
from pathlib import Path

import chainlit as cl
from PIL import Image

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
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


def resolve_weights() -> Path:
    configured = Path(YOLO_MODEL_PATH)
    if configured.exists():
        return configured
    for candidate in (
        Path("runs/detect/train-3/weights") / configured.name,
        Path("runs/detect/train/weights") / configured.name,
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"YOLO weights not found: {YOLO_MODEL_PATH}. Configure YOLO_MODEL_PATH."
    )


# ── Application modules ───────────────────────────────────────────────────────
detector = DamageDetector(resolve_weights())
severity_estimator = SeverityEstimator()
vision_pipeline = CarCrashClaimAnalyzerPipeline()
claim_pipeline = ClaimDocumentPipeline()
decision_pipeline = ClaimDecisionPipeline()

policy_rag = PolicyRAGPipeline(
    embedding_model=LOCAL_EMBED_MODEL,
    top_k=RETRIEVAL_K,
    index_path=FAISS_INDEX_PATH,
)
policy_rag.build(POLICY_DOCUMENT_PATH)
llm = Ollama(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template(
    """You are an expert insurance claim analysis assistant.

Use ONLY the supplied policy evidence and structured claim/vehicle evidence.
If evidence is insufficient, explicitly say so.

Vehicle evidence:
{vehicle_evidence}

Claim information:
{claim_information}

Policy evidence:
{policy_context}

Question:
{question}

Return a concise, evidence-grounded assessment. Never invent policy clauses,
coverage, exclusions, or financial values."""
)


@cl.on_chat_start
async def start():
    cl.user_session.set("claim_info", ClaimInformation())
    cl.user_session.set("damage_report", None)
    await cl.Message(
        content=(
            "🚗 **Car Crash Detection & Insurance Claim Analyzer**\n\n"
            "Upload a vehicle image and optionally a claim document.\n\n"
            "**Pipeline:** Image Quality → YOLOv8 → Damage → Severity → OCR/Claim Info → Hybrid Policy RAG → Decision Engine"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    damage_report = cl.user_session.get("damage_report")
    claim_info = cl.user_session.get("claim_info") or ClaimInformation()
    detected_damages: list[str] = []
    message_query = message.content or ""

    # ── Vehicle image ─────────────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if "image" not in element.mime:
                continue

            await cl.Message(content="🔍 Running image analysis...").send()
            report = vision_pipeline.run(element.path, detector=detector)

            if not report.image_quality.valid:
                reasons = "\n".join(f"- {reason}" for reason in report.image_quality.reasons)
                await cl.Message(
                    content=(
                        "❌ **Image rejected by quality gate**\n\n"
                        f"{reasons}\n\nQuality score: **{report.image_quality.score:.2f}**"
                    )
                ).send()
                continue

            with Image.open(element.path) as source_image:
                image_width, image_height = source_image.size

            report.damage = severity_estimator.estimate(
                report.damage,
                image_width=image_width,
                image_height=image_height,
            )
            damage_report = report
            cl.user_session.set("damage_report", report)
            detected_damages = [d.label for d in report.damage.detections]

            try:
                annotated = detector.render_last_result()
                output_image = Image.fromarray(annotated[..., ::-1])
                buffer = io.BytesIO()
                output_image.save(buffer, format="JPEG", quality=90)
                buffer.seek(0)
                cl_image = cl.Image(
                    content=buffer.getvalue(),
                    name="damage_assessment.jpg",
                    display="inline",
                    size="large",
                )
                image_elements = [cl_image]
            except Exception as exc:
                image_elements = []
                await cl.Message(content=f"⚠️ Bounding-box preview unavailable: `{exc}`").send()

            damage_lines = [
                f"- **{d.label}** — confidence `{d.confidence:.2f}`"
                for d in report.damage.detections
            ] or ["- No trained damage class detected"]

            await cl.Message(
                content=(
                    "✅ **Image Analysis Complete**\n\n"
                    f"**Damage:**\n{'\n'.join(damage_lines)}\n\n"
                    f"**Severity:** `{report.damage.severity or 'unknown'}`\n"
                    f"**Severity score:** `{report.damage.severity_score or 0:.2f}`\n"
                    f"**Image quality:** `{report.image_quality.score:.2f}`"
                ),
                elements=image_elements,
            ).send()

            if detected_damages:
                message_query = (
                    f"Does the policy cover vehicle damage involving {', '.join(sorted(set(detected_damages)))}? "
                    f"The estimated severity is {report.damage.severity}. What conditions, exclusions, depreciation, and limitations apply?"
                )

    # ── Claim document ────────────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if "image" in element.mime or not getattr(element, "path", None):
                continue
            if not any(
                str(element.mime).lower().endswith(ext)
                for ext in ("pdf", "jpeg", "jpg", "png", "tiff", "webp")
            ):
                continue

            await cl.Message(content="📄 Extracting claim information with OCR...").send()
            try:
                claim_info, warnings = claim_pipeline.run(element.path)
                cl.user_session.set("claim_info", claim_info)
                fields = [
                    f"- Policy number: `{claim_info.policy_number or 'missing'}`",
                    f"- Claim ID: `{claim_info.claim_id or 'missing'}`",
                    f"- Claimant: `{claim_info.claimant_name or 'missing'}`",
                    f"- Vehicle registration: `{claim_info.vehicle_registration or 'missing'}`",
                    f"- Incident date: `{claim_info.incident_date or 'missing'}`",
                ]
                warning_text = "\n".join(f"- {w}" for w in warnings) or "- None"
                await cl.Message(
                    content=(
                        "📋 **Claim Information Extracted**\n\n"
                        + "\n".join(fields)
                        + f"\n\n**Validation warnings:**\n{warning_text}"
                    )
                ).send()
            except Exception as exc:
                await cl.Message(content=f"❌ OCR processing failed: `{exc}`").send()

    if not message_query.strip():
        return

    # ── Policy evidence ──────────────────────────────────────────────────────
    await cl.Message(content="📑 **Policy Retrieval:** retrieving and reranking evidence...").send()
    evidence = policy_rag.retrieve(message_query)
    policy_context = format_context(evidence)

    vehicle_evidence = "No vehicle analysis available."
    if damage_report and damage_report.damage:
        vehicle_evidence = (
            f"Damage classes: {[d.label for d in damage_report.damage.detections]}; "
            f"severity: {damage_report.damage.severity}; "
            f"severity_score: {damage_report.damage.severity_score}"
        )

    claim_text = (
        f"policy_number={claim_info.policy_number}; claim_id={claim_info.claim_id}; "
        f"claimant={claim_info.claimant_name}; vehicle={claim_info.vehicle_registration}; "
        f"incident_date={claim_info.incident_date}"
    )

    response = await llm.ainvoke(
        prompt.format(
            vehicle_evidence=vehicle_evidence,
            claim_information=claim_text,
            policy_context=policy_context,
            question=message_query,
        )
    )

    # ── Structured decision ───────────────────────────────────────────────────
    if damage_report and damage_report.damage:
        decision, _ = decision_pipeline.run(
            damage_report.damage,
            claim_info,
            evidence,
        )
        evidence_refs = "\n".join(
            f"- {item.source or 'policy'} — page {item.page if item.page is not None else 'N/A'}"
            for item in evidence
        )
        await cl.Message(
            content=(
                f"🤖 **Claim Assessment**\n\n{response}\n\n"
                f"## Structured Decision\n"
                f"- Decision: `{decision.decision}`\n"
                f"- Coverage status: `{decision.coverage_status}`\n"
                f"- Risk score: `{decision.risk_score:.2f}`\n\n"
                f"**Rationale:** {decision.rationale}\n\n"
                f"## Evidence\n{evidence_refs}\n\n"
                f"## Warnings\n{chr(10).join('- ' + w for w in decision.warnings) or '- None'}"
            )
        ).send()
    else:
        await cl.Message(content=f"🤖 **Policy Assessment**\n\n{response}").send()
