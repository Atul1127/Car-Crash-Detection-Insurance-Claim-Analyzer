"""OCR adapter with preprocessing and optional PDF page support."""

from pathlib import Path


class OCRExtractor:
    """Extract readable text from claim documents/images using Tesseract."""

    def __init__(self, language: str = "eng"):
        self.language = language

    @staticmethod
    def _prepare_image(image):
        """Upscale and lightly normalize scans before OCR."""
        from PIL import ImageOps, ImageFilter

        image = ImageOps.exif_transpose(image).convert("L")
        scale = 2 if max(image.size) < 2200 else 1
        if scale > 1:
            image = image.resize((image.width * scale, image.height * scale))
        image = ImageOps.autocontrast(image)
        return image.filter(ImageFilter.SHARPEN)

    def _ocr_image(self, image, pytesseract) -> str:
        image = self._prepare_image(image)
        # PSM 6 works well for forms with multiple labeled fields.
        text = pytesseract.image_to_string(image, lang=self.language, config="--psm 6")
        if not text.strip():
            text = pytesseract.image_to_string(image, lang=self.language, config="--psm 3")
        return text.strip()

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

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pdf2image import convert_from_path
            except ImportError as exc:
                raise RuntimeError(
                    "PDF OCR requires pdf2image and a Poppler installation. "
                    "Install the document/OCR extras before uploading PDF claims."
                ) from exc
            pages = convert_from_path(path, dpi=200)
            texts = [self._ocr_image(page, pytesseract) for page in pages]
            return "\n\n".join(text for text in texts if text).strip()

        with Image.open(path) as image:
            return self._ocr_image(image, pytesseract)
