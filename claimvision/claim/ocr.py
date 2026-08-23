"""OCR adapter with optional Tesseract backend.

OCR is intentionally isolated from extraction so the OCR engine can later be
replaced by PaddleOCR, EasyOCR, a document AI model, or a hosted OCR service.
"""

from pathlib import Path


class OCRExtractor:
    """Extract raw text from claim documents/images."""

    def __init__(self, language: str = "eng"):
        self.language = language

    def extract(self, file_path: str | Path) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OCR requires pytesseract and Pillow. Install the OCR dependencies "
                "and ensure the Tesseract executable is available."
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Claim document not found: {path}")

        with Image.open(path) as image:
            text = pytesseract.image_to_string(image, lang=self.language)

        return text.strip()
