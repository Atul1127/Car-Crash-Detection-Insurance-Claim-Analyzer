"""Build and query the advanced policy retrieval pipeline."""

import hashlib
import json
from pathlib import Path

from car_crash_claim_analyzer.rag.chunking import PolicyChunker
from car_crash_claim_analyzer.rag.context import diversify_documents, documents_to_evidence
from car_crash_claim_analyzer.rag.document import PolicyDocumentLoader
from car_crash_claim_analyzer.rag.query import QueryExpander, QueryNormalizer
from car_crash_claim_analyzer.rag.reranker import ContextCompressor, EvidenceReranker
from car_crash_claim_analyzer.rag.retriever import HybridPolicyRetriever
from car_crash_claim_analyzer.schemas import PolicyEvidence


class PolicyRAGPipeline:
    """Policy PDF → chunking → hybrid retrieval → reranking → compression."""

    def __init__(
        self,
        embedding_model: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 6,
        index_path: str | Path | None = None,
    ):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = PolicyChunker(chunk_size, chunk_overlap)
        self.top_k = top_k
        self.index_path = Path(index_path) if index_path else None
        self.retriever: HybridPolicyRetriever | None = None
        self.normalizer = QueryNormalizer()
        self.expander = QueryExpander()
        self.reranker = EvidenceReranker()
        self.compressor = ContextCompressor()

    def build(self, policy_path: str | Path) -> None:
        """Load a valid persisted index, otherwise build and persist one."""
        policy = Path(policy_path)
        if not policy.exists():
            raise FileNotFoundError(f"Policy document not found: {policy}")

        if self.index_path and self._cache_is_valid(policy):
            self.retriever = HybridPolicyRetriever.load(
                self.index_path,
                self.embedding_model,
                self.top_k * 2,
            )
            return

        pages = PolicyDocumentLoader().load(policy)
        chunks = self.chunker.split(pages)
        if not chunks:
            raise ValueError("No text could be extracted from the policy document.")

        self.retriever = HybridPolicyRetriever(chunks, self.embedding_model, self.top_k * 2)

        if self.index_path:
            self.retriever.save(self.index_path)
            self._write_manifest(policy)

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

    def _manifest_path(self) -> Path | None:
        if self.index_path is None:
            return None
        return self.index_path.parent / f"{self.index_path.name}.manifest.json"

    def _policy_fingerprint(self, policy: Path) -> str:
        digest = hashlib.sha256()
        digest.update(policy.read_bytes())
        return digest.hexdigest()

    def _manifest(self, policy: Path) -> dict:
        return {
            "policy_sha256": self._policy_fingerprint(policy),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
        }

    def _cache_is_valid(self, policy: Path) -> bool:
        if self.index_path is None:
            return False
        manifest_path = self._manifest_path()
        index_file = self.index_path / "index.faiss"
        store_file = self.index_path / "index.pkl"
        if not manifest_path or not manifest_path.exists() or not index_file.exists() or not store_file.exists():
            return False
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8")) == self._manifest(policy)
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _write_manifest(self, policy: Path) -> None:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(self._manifest(policy), indent=2),
            encoding="utf-8",
        )
