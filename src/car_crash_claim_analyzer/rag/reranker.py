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

            # Strongly prioritize the policy's governing own-damage provision.
            # This prevents repair limits or unrelated exclusions from replacing
            # the primary Section I coverage clause.
            own_damage_phrase = any(
                phrase in text
                for phrase in (
                    "section i",
                    "loss of or damage to the insured vehicle",
                    "loss or damage to the insured vehicle",
                    "loss of or damage to the vehicle",
                    "damage to the insured vehicle",
                    "own damage",
                )
            )
            section_i_bonus = 0.30 if own_damage_phrase else 0.0

            score = 0.50 * overlap + 0.30 * (intent_score / len(self.INTENT_GROUPS)) + section_i_bonus
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

    COVERAGE_PHRASES = (
        "loss of or damage to the insured vehicle",
        "loss or damage to the insured vehicle",
        "loss of or damage to the vehicle",
        "damage to the insured vehicle",
        "own damage",
    )

    def compress(self, query: str, documents: list[Document], max_sentences: int = 7) -> list[Document]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        output: list[Document] = []

        for document in documents:
            sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
            scored_sentences = []
            for sentence in sentences:
                lower = sentence.lower()
                score = (
                    len(query_terms.intersection(re.findall(r"[a-z0-9]+", lower)))
                    + 0.5 * len(self.PRIORITY_TERMS.intersection(re.findall(r"[a-z0-9]+", lower)))
                )
                if any(phrase in lower for phrase in self.COVERAGE_PHRASES):
                    score += 6.0
                if "section i" in lower:
                    score += 3.0
                scored_sentences.append((score, sentence))

            ranked = sorted(scored_sentences, key=lambda item: item[0], reverse=True)
            text = " ".join(sentence for _, sentence in ranked[:max_sentences]).strip()
            if text:
                output.append(Document(page_content=text, metadata=dict(document.metadata)))

        return output
