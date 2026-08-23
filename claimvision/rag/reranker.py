"""Lightweight lexical reranking and context compression."""

import re

from langchain_core.documents import Document


class EvidenceReranker:
    """Rerank retrieved policy chunks using query-term coverage."""

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for document in documents:
            terms = re.findall(r"[a-z0-9]+", document.page_content.lower())
            overlap = len(query_terms.intersection(terms)) / max(len(query_terms), 1)
            scored.append((overlap, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


class ContextCompressor:
    """Keep the most query-relevant sentences while preserving source metadata."""

    def compress(self, query: str, documents: list[Document], max_sentences: int = 5) -> list[Document]:
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        output: list[Document] = []
        for document in documents:
            sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(query_terms.intersection(re.findall(r"[a-z0-9]+", sentence.lower()))),
                reverse=True,
            )
            text = " ".join(ranked[:max_sentences]).strip()
            if text:
                output.append(Document(page_content=text, metadata=dict(document.metadata)))
        return output
