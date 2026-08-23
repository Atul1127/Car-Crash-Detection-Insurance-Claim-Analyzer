import io
from pathlib import Path

import chainlit as cl
from PIL import Image

from claimvision.pipeline import ClaimVisionPipeline
from claimvision.rag.context import format_context
from claimvision.rag.pipeline import PolicyRAGPipeline
from claimvision.vision.detector import DamageDetector
from claimvision.vision.severity import SeverityEstimator
from config import (
    LOCAL_EMBED_MODEL,
    MODEL_NAME,
    POLICY_DOCUMENT_PATH,
    RETRIEVAL_K,
    YOLO_DEVICE,
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


# ── ClaimVision computer vision ───────────────────────────────────────────────
detector = DamageDetector(resolve_weights())
severity_estimator = SeverityEstimator()
vision_pipeline = ClaimVisionPipeline()


# ── Advanced policy RAG ───────────────────────────────────────────────────────
policy_rag = PolicyRAGPipeline(
    embedding_model=LOCAL_EMBED_MODEL,
    top_k=RETRIEVAL_K,
)
policy_rag.build(POLICY_DOCUMENT_PATH)
llm = Ollama(model=MODEL_NAME)

prompt = ChatPromptTemplate.from_template(
    """You are an expert insurance claim analysis assistant.

Use ONLY the supplied policy evidence. If the evidence does not establish an
answer, explicitly say that the policy evidence is insufficient.

Policy evidence:
{context}

Claim question:
{question}

Return:
1. Coverage assessment
2. Relevant limitations/exclusions
3. Depreciation or conditions if present
4. Evidence references using the supplied source/page information

Do not invent policy clauses or financial values."""
)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "🚗 **Car Crash Detection & Insurance Claim Analyzer**\n\n"
            "Upload a vehicle image to run:\n"
            "**Image Quality → YOLOv8 → Damage → Severity → Hybrid Policy RAG**\n\n"
            "You can also ask policy questions directly."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    detected_damages: list[str] = []
    severity = None

    if message.elements:
        for element in message.elements:
            if "image" not in element.mime:
                continue

            await cl.Message(content="🔍 Running ClaimVision image analysis...").send()
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

            report.damage = severity_estimator.estimate(report.damage)
            detected_damages = [d.label for d in report.damage.detections]
            severity = report.damage.severity

            results = detector.model.predict(
                source=element.path,
                conf=detector.confidence,
                iou=detector.iou,
                save=False,
                device=YOLO_DEVICE,
            )
            result = results[0]
            output_image = Image.fromarray(result.plot()[..., ::-1])
            buffer = io.BytesIO()
            output_image.save(buffer, format="JPEG")
            buffer.seek(0)

            cl_image = cl.Image(
                content=buffer.read(),
                name="damage_assessment.jpg",
                display="inline",
                size="large",
            )

            if detected_damages:
                damage_lines = [
                    f"- **{d.label}** — confidence `{d.confidence:.2f}`"
                    for d in report.damage.detections
                ]
                summary = (
                    "✅ **Visual Analysis Complete**\n\n"
                    f"**Damage:**\n{'\n'.join(damage_lines)}\n\n"
                    f"**Severity:** `{severity}`\n"
                    f"**Severity score:** `{report.damage.severity_score:.2f}`\n"
                    f"**Image quality:** `{report.image_quality.score:.2f}`"
                )
            else:
                summary = "ℹ️ No trained damage class was detected."

            await cl.Message(content=summary, elements=[cl_image]).send()

    # ── Hybrid policy retrieval ───────────────────────────────────────────────
    if detected_damages:
        damage_list = ", ".join(sorted(set(detected_damages)))
        rag_query = (
            f"Does the policy cover vehicle damage involving {damage_list}? "
            f"The estimated damage severity is {severity}. "
            "What coverage conditions, exclusions, depreciation, and limitations apply?"
        )
    else:
        rag_query = message.content

    if not rag_query or not rag_query.strip():
        return

    await cl.Message(content="📑 **Hybrid Policy RAG:** retrieving and reranking evidence...").send()
    evidence = policy_rag.retrieve(rag_query)
    context = format_context(evidence)
    response = await llm.ainvoke(prompt.format(context=context, question=rag_query))

    evidence_refs = "\n".join(
        f"- {item.source or 'policy'} — page {item.page if item.page is not None else 'N/A'}"
        for item in evidence
    )
    await cl.Message(
        content=(
            f"🤖 **Policy Assessment**\n\n{response}\n\n"
            f"**Evidence used:**\n{evidence_refs}"
        )
    ).send()
