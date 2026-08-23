"""Policy-document loading, cleaning, and metadata preparation."""

from pathlib import Path
import re

from langchain_core.documents import Document
from pypdf import PdfReader


class PolicyDocumentLoader:
    """Load a PDF into page-aware LangChain documents."""

    def load(self, path: str | Path) -> list[Document]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Policy document not found: {pdf_path}")

        reader = PdfReader(str(pdf_path))
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self.clean(text)
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                        "document_type": "insurance_policy",
                    },
                )
            )
        return documents

    @staticmethod
    def clean(text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()
