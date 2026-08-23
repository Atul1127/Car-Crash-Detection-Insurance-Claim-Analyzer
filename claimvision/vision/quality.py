"""Image-quality gate used before expensive CV inference."""

from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

from claimvision.schemas import ImageQualityResult
from config import IMAGE_MIN_BRIGHTNESS, IMAGE_MIN_HEIGHT, IMAGE_MIN_WIDTH


class ImageQualityChecker:
    """Reject images that are clearly unsuitable for reliable damage detection."""

    def __init__(
        self,
        min_width: int = IMAGE_MIN_WIDTH,
        min_height: int = IMAGE_MIN_HEIGHT,
        min_brightness: float = IMAGE_MIN_BRIGHTNESS,
        min_sharpness: float = 2.0,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.min_brightness = min_brightness
        self.min_sharpness = min_sharpness

    def check(self, image_path: str | Path) -> ImageQualityResult:
        reasons: list[str] = []
        metrics: dict[str, float] = {}

        try:
            with Image.open(image_path) as source:
                image = source.convert("L")
                width, height = image.size
                metrics["width"] = float(width)
                metrics["height"] = float(height)

                brightness = float(ImageStat.Stat(image).mean[0])
                metrics["brightness"] = round(brightness, 2)

                # The previous implementation compared the mean brightness of
                # an image with a blurred copy. That is not a blur detector:
                # both means are normally almost identical. Use the mean
                # absolute high-frequency difference instead.
                blurred = image.filter(ImageFilter.GaussianBlur(radius=2))
                sharpness = ImageStat.Stat(
                    Image.fromarray(
                        __import__("PIL.ImageChops", fromlist=["ImageChops"])
                        .ImageChops.difference(image, blurred)
                    )
                ).mean[0]
                metrics["sharpness_proxy"] = round(float(sharpness), 3)

                if width < self.min_width or height < self.min_height:
                    reasons.append(
                        f"Image resolution is too low ({width}x{height}); "
                        f"minimum is {self.min_width}x{self.min_height}."
                    )
                if brightness < self.min_brightness:
                    reasons.append("Image is extremely dark.")
                if sharpness < self.min_sharpness:
                    reasons.append("Image may be too blurred for reliable damage detection.")

        except Exception as exc:
            return ImageQualityResult(False, 0.0, [f"Unable to read image: {exc}"])

        valid = not reasons
        score = max(0.0, 1.0 - 0.30 * len(reasons))
        return ImageQualityResult(valid, round(score, 3), reasons, metrics)
