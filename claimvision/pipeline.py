"""Top-level ClaimVision orchestration layer."""

from pathlib import Path

from claimvision.schemas import ClaimReport
from claimvision.vision.quality import ImageQualityChecker


class ClaimVisionPipeline:
    """Coordinate quality checks and downstream modules without coupling them."""

    def __init__(self, quality_checker: ImageQualityChecker | None = None):
        self.quality_checker = quality_checker or ImageQualityChecker()

    def validate_image(self, image_path: str | Path) -> ClaimReport:
        quality = self.quality_checker.check(image_path)
        return ClaimReport(image_quality=quality, damage=None)  # type: ignore[arg-type]

    def run(self, image_path: str | Path, detector=None) -> ClaimReport:
        quality = self.quality_checker.check(image_path)
        if not quality.valid:
            return ClaimReport(image_quality=quality, damage=None)  # type: ignore[arg-type]

        if detector is None:
            raise ValueError("A damage detector is required after image validation.")

        damage = detector.predict(image_path)
        return ClaimReport(image_quality=quality, damage=damage)
