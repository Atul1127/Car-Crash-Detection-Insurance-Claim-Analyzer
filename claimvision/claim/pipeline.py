"""End-to-end claim-document processing pipeline."""

from pathlib import Path

from claimvision.claim.extractor import ClaimInformationExtractor
from claimvision.claim.normalizer import ClaimNormalizer
from claimvision.claim.ocr import OCRExtractor
from claimvision.claim.validator import ClaimValidator
from claimvision.schemas import ClaimInformation


class ClaimDocumentPipeline:
    """OCR → extraction → normalization → validation."""

    def __init__(self, ocr: OCRExtractor | None = None):
        self.ocr = ocr or OCRExtractor()
        self.extractor = ClaimInformationExtractor()
        self.normalizer = ClaimNormalizer()
        self.validator = ClaimValidator()

    def run(self, file_path: str | Path) -> tuple[ClaimInformation, list[str]]:
        raw_text = self.ocr.extract(file_path)
        claim = self.extractor.extract(raw_text)
        claim = self.normalizer.normalize(claim)
        warnings = self.validator.validate(claim)
        claim.metadata["validation_warnings"] = warnings
        return claim, warnings
