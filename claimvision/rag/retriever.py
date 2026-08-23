"""Hybrid policy retrieval: dense FAISS + sparse lexical matching."""

from collections import Counter
import math
import re

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class HybridPolicyRetriever:
    """Combine FAISS semantic retrieval with a lightweight BM25-style scorer."""

    def __init__(self, documents: list[Document], embedding_model: str, top_k: int = 6):
        self.documents = documents
        self.top_k = top_k
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        self.tokenized = [self._tokens(doc.page_content) for doc in documents]
        self.avgdl = sum(map(len, self.tokenized)) / max(len(self.tokenized), 1)

    def retrieve(self, query: str) -> list[Document]:
        dense = self.vector_db.similarity_search_with_score(query, k=min(self.top_k * 2, len(self.documents)))
        dense_scores = {id(doc): 1.0 / (1.0 + float(score)) for doc, score in dense}
        lexical_scores = self._bm25(query)

        ranked = []
        for index, document in enumerate(self.documents):
            score = 0.65 * dense_scores.get(id(document), 0.0) + 0.35 * lexical_scores[index]
            ranked.append((score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in ranked[: self.top_k]]

    def _bm25(self, query: str) -> list[float]:
        query_terms = self._tokens(query)
        df = Counter(term for terms in self.tokenized for term in set(terms))
        n = len(self.tokenized)
        scores: list[float] = []
        k1, b = 1.5, 0.75

        for terms in self.tokenized:
            counts = Counter(terms)
            score = 0.0
            dl = len(terms)
            for term in query_terms:
                if term not in counts:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                tf = counts[term]
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1))
                score += idf * (tf * (k1 + 1)) / denom
            scores.append(score)

        max_score = max(scores, default=1.0)
        return [score / max_score for score in scores]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
