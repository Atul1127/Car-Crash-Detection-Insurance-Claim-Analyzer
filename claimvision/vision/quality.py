"""Lightweight image-quality gate used before expensive CV inference."""

from pathlib import Path

from PIL import Image, ImageStat

from claimvision.schemas import ImageQualityResult


class ImageQualityChecker:
    """Reject images that are clearly unsuitable for damage detection."""

    def __init__(self, min_width: int = 320, min_height: int = 240, min_brightness: float = 8.0):
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness

    def check(self, image_path: str | Path) -> ImageQualityResult:
        reasons: list[str] = []
        metrics: dict[str, float] = {}

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                metrics["width"] = float(width)
                metrics["height"] = float(height)
                brightness = sum(ImageStat.Stat(image).mean) / 3.0
                metrics["brightness"] = round(brightness, 2)

                if width < self.min_width or height < self.min_height:
                    reasons.append("Image resolution is too low for reliable analysis.")
                if brightness < self.min_brightness:
                    reasons.append("Image is extremely dark.")

        except Exception as exc:
            return ImageQualityResult(False, 0.0, [f"Unable to read image: {exc}"])

        valid = not reasons
        score = 1.0 if valid else max(0.0, 1.0 - 0.35 * len(reasons))
        return ImageQualityResult(valid, round(score, 3), reasons, metrics)
