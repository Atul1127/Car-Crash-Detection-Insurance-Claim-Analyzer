from pathlib import Path

from PIL import Image

from car_crash_claim_analyzer.vision.quality import ImageQualityChecker


def _make_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, (120, 120, 120)).save(path)


def test_412x237_image_passes_resolution_gate(tmp_path):
    path = tmp_path / "web_image.jpg"
    _make_image(path, (412, 237))
    result = ImageQualityChecker(min_sharpness=0).check(path)
    assert result.valid


def test_tiny_image_is_rejected(tmp_path):
    path = tmp_path / "tiny.jpg"
    _make_image(path, (200, 150))
    result = ImageQualityChecker(min_sharpness=0).check(path)
    assert not result.valid
    assert any("resolution is too low" in reason.lower() for reason in result.reasons)
