"""Structure-aware policy chunking."""

import re

from langchain_core.documents import Document


class PolicyChunker:
    """Create overlapping chunks while retaining policy metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            text = document.page_content
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                content = text[start:end].strip()
                if content:
                    metadata = dict(document.metadata)
                    heading = self._heading(content)
                    if heading:
                        metadata["section"] = heading
                    chunks.append(Document(page_content=content, metadata=metadata))
                if end >= len(text):
                    break
                start = end - self.chunk_overlap
        return chunks

    @staticmethod
    def _heading(text: str) -> str | None:
        match = re.search(r"(?:^|\s)((?:SECTION|CHAPTER|CLAUSE)\s+[A-Z0-9. -]{2,80})", text, re.I)
        return match.group(1).strip() if match else None
