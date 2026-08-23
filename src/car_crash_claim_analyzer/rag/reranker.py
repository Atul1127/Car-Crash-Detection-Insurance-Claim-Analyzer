"""Policy-aware reranking and context compression."""

import re

from langchain_core.documents import Document


class EvidenceReranker:
    """Rerank evidence using lexical relevance plus policy-intent signals."""

    INTENT_GROUPS = {
        "own_damage": {
            "own", "damage", "loss", "vehicle", "insured", "indemnity", "repair", "accident"
        },
        "exclusion": {"exclusion", "excluded", "limitation", "not", "covered", "use", "condition"},
        "financial": {"depreciation", "deduction", "excess", "deductible", "estimate", "repair", "cost"},
    }

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []

        for document in documents:
            text = document.page_content.lower()
            terms = set(re.findall(r"[a-z0-9]+", text))
            overlap = len(query_terms.intersection(terms)) / max(len(query_terms), 1)

            intent_score = 0.0
            for group in self.INTENT_GROUPS.values():
                matched = len(group.intersection(terms))
                intent_score += min(matched / 4.0, 1.0)

            # Keep query relevance primary, but prevent a page containing one
            # repeated generic word such as "coverage" from outranking the
            # policy's actual vehicle-damage provisions.
            score = 0.60 * overlap + 0.40 * (intent_score / len(self.INTENT_GROUPS))
            scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class ContextCompressor:
    """Keep relevant sentences while preserving policy source metadata."""

    PRIORITY_TERMS = {
        "own", "damage", "loss", "vehicle", "insured", "indemnity", "accident",
        "exclusion", "excluded", "limitation", "condition", "depreciation",
        "excess", "deductible", "repair", "estimate", "claim", "section",
    }

    def compress(self, query: str, documents: list[Document], max_sentences: int = 7) -> list[Document]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        output: list[Document] = []

        for document in documents:
            sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
            ranked = sorted(
                sentences,
                key=lambda sentence: (
                    len(query_terms.intersection(re.findall(r"[a-z0-9]+", sentence.lower())))
                    + 0.5 * len(self.PRIORITY_TERMS.intersection(re.findall(r"[a-z0-9]+", sentence.lower())))
                ),
                reverse=True,
            )
            text = " ".join(ranked[:max_sentences]).strip()
            if text:
                output.append(Document(page_content=text, metadata=dict(document.metadata)))

        return output
