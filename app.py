import io
from pathlib import Path

import chainlit as cl
from PIL import Image

from claimvision.pipeline import ClaimVisionPipeline
from claimvision.vision.detector import DamageDetector
from claimvision.vision.severity import SeverityEstimator
from config import (
    FAISS_INDEX_PATH,
    LOCAL_EMBED_MODEL,
    MODEL_NAME,
    RETRIEVAL_K,
    YOLO_DEVICE,
    YOLO_MODEL_PATH,
)

# Existing policy RAG dependencies remain in the application layer for now.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ── ClaimVision CV pipeline ───────────────────────────────────────────────────

def resolve_weights() -> Path:
    configured = Path(YOLO_MODEL_PATH)
    if configured.exists():
        return configured

    candidates = [
        Path("runs/detect/train-3/weights") / configured.name,
        Path("runs/detect/train/weights") / configured.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"YOLO weights not found: {YOLO_MODEL_PATH}. "
        "Place the model in the project root or configure YOLO_MODEL_PATH."
    )


weights_path = resolve_weights()
detector = DamageDetector(weights_path)
severity_estimator = SeverityEstimator()
pipeline = ClaimVisionPipeline()


# ── Existing policy RAG ───────────────────────────────────────────────────────
embedding_model = HuggingFaceEmbeddings(
    model_name=LOCAL_EMBED_MODEL,
    model_kwargs={"device": "cpu"},
)
vector_db = FAISS.load_local(
    FAISS_INDEX_PATH,
    embedding_model,
    allow_dangerous_deserialization=True,
)
retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
llm = Ollama(model=MODEL_NAME)

prompt_template = """
You are an expert insurance claim adjuster AI. Analyze the vehicle damage against
only the policy clauses provided below.

Policy Context:
{context}

Question:
{question}

Return a professional assessment covering coverage, limitations, and relevant
policy conditions. Do not invent policy clauses.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "🚗 **Car Crash Detection & Insurance Claim Analyzer**\n\n"
            "Upload a vehicle image to run the ClaimVision computer-vision pipeline.\n\n"
            "**Pipeline:** Image Quality → YOLOv8 → Damage Classification → Severity → Policy RAG"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    detected_damages: list[str] = []

    if message.elements:
        for element in message.elements:
            if "image" not in element.mime:
                continue

            await cl.Message(content="🔍 Running image-quality and damage analysis...").send()

            report = pipeline.run(element.path, detector=detector)
            if not report.image_quality.valid:
                reasons = "\n".join(f"- {reason}" for reason in report.image_quality.reasons)
                await cl.Message(
                    content=(
                        "❌ **Image rejected by quality gate**\n\n"
                        f"{reasons}\n\n"
                        f"Quality score: **{report.image_quality.score:.2f}**"
                    )
                ).send()
                continue

            report.damage = severity_estimator.estimate(report.damage)
            detected_damages = [d.label for d in report.damage.detections]

            # Render YOLO boxes using the same model that generated the structured output.
            results = detector.model.predict(
                source=element.path,
                conf=detector.confidence,
                save=False,
                device=YOLO_DEVICE,
            )
            result = results[0]
            im_array = result.plot()
            output_image = Image.fromarray(im_array[..., ::-1])
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
                rows = []
                for detection in report.damage.detections:
                    rows.append(
                        f"- **{detection.label}** — confidence `{detection.confidence:.2f}`"
                    )
                damage_text = "\n".join(rows)
                summary = (
                    "✅ **Visual Analysis Complete**\n\n"
                    f"**Detected damage:**\n{damage_text}\n\n"
                    f"**Severity:** `{report.damage.severity}`\n"
                    f"**Severity score:** `{report.damage.severity_score:.2f}`\n"
                    f"**Image quality:** `{report.image_quality.score:.2f}`"
                )
            else:
                summary = (
                    "ℹ️ **No trained damage class detected.**\n\n"
                    f"Image quality score: `{report.image_quality.score:.2f}`"
                )

            await cl.Message(content=summary, elements=[cl_image]).send()

    # Preserve text-only policy questions and automatically query policy for CV hits.
    if detected_damages:
        damage_list = ", ".join(sorted(set(detected_damages)))
        rag_query = (
            f"Does the policy cover accidental vehicle damage involving {damage_list}? "
            "What coverage conditions, exclusions, depreciation, or limitations apply?"
        )
        await cl.Message(
            content=f"📑 **Policy RAG:** retrieving clauses for `{damage_list}`..."
        ).send()
    else:
        rag_query = message.content

    if rag_query and rag_query.strip():
        response = await rag_chain.ainvoke(rag_query)
        await cl.Message(
            content=f"🤖 **Policy Assessment**\n\n{response}"
        ).send()
