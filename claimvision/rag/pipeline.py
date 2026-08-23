"""Build and query the advanced policy retrieval pipeline."""

from pathlib import Path

from claimvision.rag.chunking import PolicyChunker
from claimvision.rag.context import documents_to_evidence
from claimvision.rag.document import PolicyDocumentLoader
from claimvision.rag.retriever import HybridPolicyRetriever
from claimvision.schemas import PolicyEvidence


class PolicyRAGPipeline:
    """Policy PDF → page-aware chunks → hybrid retrieval → evidence."""

    def __init__(self, embedding_model: str, chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 6):
        self.embedding_model = embedding_model
        self.chunker = PolicyChunker(chunk_size, chunk_overlap)
        self.top_k = top_k
        self.retriever: HybridPolicyRetriever | None = None

    def build(self, policy_path: str | Path) -> None:
        pages = PolicyDocumentLoader().load(policy_path)
        chunks = self.chunker.split(pages)
        if not chunks:
            raise ValueError("No text could be extracted from the policy document.")
        self.retriever = HybridPolicyRetriever(chunks, self.embedding_model, self.top_k)

    def retrieve(self, query: str) -> list[PolicyEvidence]:
        if self.retriever is None:
            raise RuntimeError("PolicyRAGPipeline has not been built yet.")
        documents = self.retriever.retrieve(query)
        return documents_to_evidence(documents)
