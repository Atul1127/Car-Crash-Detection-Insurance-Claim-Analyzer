"""Context assembly for downstream claim reasoning."""

from claimvision.schemas import PolicyEvidence
from langchain_core.documents import Document


def documents_to_evidence(documents: list[Document]) -> list[PolicyEvidence]:
    evidence: list[PolicyEvidence] = []
    for document in documents:
        evidence.append(
            PolicyEvidence(
                text=document.page_content,
                source=document.metadata.get("source"),
                page=document.metadata.get("page"),
            )
        )
    return evidence


def format_context(evidence: list[PolicyEvidence]) -> str:
    blocks = []
    for item in evidence:
        location = item.source or "policy"
        if item.page is not None:
            location += f", page {item.page}"
        blocks.append(f"[Source: {location}]\n{item.text}")
    return "\n\n".join(blocks)
