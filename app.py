import sys
from pathlib import Path

# Allow Chainlit to import the src-layout package when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import chainlit as cl
from PIL import Image

from car_crash_claim_analyzer.application import ClaimAnalysisApplication
from car_crash_claim_analyzer.schemas import ClaimInformation


application = ClaimAnalysisApplication()


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
    message_query = message.content or ""
    detected_damages: list[str] = []

    for element in message.elements or []:
        if "image" not in element.mime:
            continue

        await cl.Message(content="🔍 Running image analysis...").send()
        report, annotated_bytes, detected_damages = await application.analyze_image(element.path)

        if not report.image_quality.valid:
            reasons = "\n".join(f"- {reason}" for reason in report.image_quality.reasons)
            await cl.Message(
                content=(
                    "❌ **Image rejected by quality gate**\n\n"
                    f"{reasons}\n\nQuality score: **{report.image_quality.score:.2f}**"
                )
            ).send()
            continue

        damage_report = report
        cl.user_session.set("damage_report", report)

        image_elements = []
        if annotated_bytes:
            image_elements = [
                cl.Image(
                    content=annotated_bytes,
                    name="damage_assessment.jpg",
                    display="inline",
                    size="large",
                )
            ]

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
            damage_terms = ", ".join(sorted(set(detected_damages)))
            message_query = (
                "For a private motor insurance claim involving accidental damage to the insured vehicle "
                f"(detected damage category: {damage_terms}; severity: {report.damage.severity}), "
                "retrieve the policy provisions that establish the applicable vehicle/own-damage coverage, "
                "conditions, exclusions, depreciation, deductibles/excess, repair limitations, and relevant "
                "claim requirements. Do not require the policy to name the individual damaged component. "
                "Prioritize clauses that actually apply to accidental physical damage to the insured vehicle."
            )

    for element in message.elements or []:
        if "image" in element.mime or not getattr(element, "path", None):
            continue
        mime = str(element.mime).lower()
        if not any(mime.endswith(ext) for ext in ("pdf", "jpeg", "jpg", "png", "tiff", "webp")):
            continue

        await cl.Message(content="📄 Extracting claim information with OCR...").send()
        try:
            claim_info, warnings = await application.extract_claim(element.path)
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

    status = cl.Message(content="📑 **Policy Retrieval:** retrieving and reranking evidence...")
    await status.send()
    try:
        evidence, policy_context = await application.retrieve_policy(message_query)
        status.content = f"✅ **Policy Retrieval Complete** — {len(evidence)} evidence chunks found."
        await status.update()
    except Exception as exc:
        status.content = f"❌ **Policy Retrieval Failed:** `{type(exc).__name__}: {exc}`"
        await status.update()
        return

    llm_status = cl.Message(content="🤖 **Claim Reasoning:** generating policy-grounded assessment...")
    await llm_status.send()
    try:
        response = await application.reason(message_query, damage_report, claim_info, policy_context)
        llm_status.content = "✅ **Claim Reasoning Complete**"
        await llm_status.update()
    except Exception as exc:
        llm_status.content = f"❌ **LLM Reasoning Failed:** `{type(exc).__name__}: {exc}`"
        await llm_status.update()
        return

    if damage_report and damage_report.damage:
        try:
            decision, _ = await application.decide(damage_report, claim_info, evidence)
        except Exception as exc:
            await cl.Message(content=f"❌ Structured decision failed: `{type(exc).__name__}: {exc}`").send()
            return

        evidence_refs = "\n".join(
            f"- {item.source or 'policy'} — page {item.page if item.page is not None else 'N/A'}"
            for item in evidence
        )
        warnings = "\n".join(f"- {w}" for w in decision.warnings) or "- None"
        await cl.Message(
            content=(
                f"🤖 **Claim Assessment**\n\n{response}\n\n"
                f"## Structured Decision\n"
                f"- Decision: `{decision.decision}`\n"
                f"- Coverage status: `{decision.coverage_status}`\n"
                f"- Risk score: `{decision.risk_score:.2f}`\n\n"
                f"**Rationale:** {decision.rationale}\n\n"
                f"## Evidence\n{evidence_refs}\n\n"
                f"## Warnings\n{warnings}"
            )
        ).send()
