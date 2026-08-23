"""Build and query the advanced policy retrieval pipeline."""

from pathlib import Path

from claimvision.rag.chunking import PolicyChunker
from claimvision.rag.context import diversify_documents, documents_to_evidence
from claimvision.rag.document import PolicyDocumentLoader
from claimvision.rag.query import QueryExpander, QueryNormalizer
from claimvision.rag.reranker import ContextCompressor, EvidenceReranker
from claimvision.rag.retriever import HybridPolicyRetriever
from claimvision.schemas import PolicyEvidence


class PolicyRAGPipeline:
    """Policy PDF → chunking → hybrid retrieval → reranking → compression."""

    def __init__(self, embedding_model: str, chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 6):
        self.embedding_model = embedding_model
        self.chunker = PolicyChunker(chunk_size, chunk_overlap)
        self.top_k = top_k
        self.retriever: HybridPolicyRetriever | None = None
        self.normalizer = QueryNormalizer()
        self.expander = QueryExpander()
        self.reranker = EvidenceReranker()
        self.compressor = ContextCompressor()

    def build(self, policy_path: str | Path) -> None:
        pages = PolicyDocumentLoader().load(policy_path)
        chunks = self.chunker.split(pages)
        if not chunks:
            raise ValueError("No text could be extracted from the policy document.")
        self.retriever = HybridPolicyRetriever(chunks, self.embedding_model, self.top_k * 2)

    def retrieve(self, query: str) -> list[PolicyEvidence]:
        if self.retriever is None:
            raise RuntimeError("PolicyRAGPipeline has not been built yet.")

        normalized = self.normalizer.normalize(query)
        variants = self.expander.expand(normalized)

        candidates = []
        seen: set[tuple[str, object]] = set()
        for variant in variants:
            for document in self.retriever.retrieve(variant):
                key = (document.page_content, document.metadata.get("page"))
                if key not in seen:
                    seen.add(key)
                    candidates.append(document)

        reranked = self.reranker.rerank(normalized, candidates, top_k=self.top_k * 2)
        compressed = self.compressor.compress(normalized, reranked, max_sentences=5)
        diversified = diversify_documents(compressed, max_per_page=2)
        return documents_to_evidence(diversified[: self.top_k])
