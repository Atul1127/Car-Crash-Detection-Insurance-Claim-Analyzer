"""Central configuration for the Car Crash Detection Insurance Claim Analyzer."""

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent

# Insurance policy documents. Override with POLICY_DOCUMENT_PATH when needed.
DEFAULT_POLICY_PATH = ROOT_DIR / "data" / "policies" / "private_car_policy.pdf"
POLICY_DOCUMENT_PATH = Path(os.getenv("POLICY_DOCUMENT_PATH", str(DEFAULT_POLICY_PATH)))
SOURCES: list[str] = [str(POLICY_DOCUMENT_PATH)]
SOURCE_LABEL = "insurance-policy"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Models
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2:3b")
LOCAL_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(ROOT_DIR / "models" / "best.pt"))

# Computer vision inference
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto")

# Image quality gate
IMAGE_MIN_WIDTH = int(os.getenv("IMAGE_MIN_WIDTH", "320"))
IMAGE_MIN_HEIGHT = int(os.getenv("IMAGE_MIN_HEIGHT", "224"))
IMAGE_MIN_BRIGHTNESS = float(os.getenv("IMAGE_MIN_BRIGHTNESS", "8.0"))
# The blur metric is intentionally permissive for compressed/web-uploaded images.
IMAGE_MIN_SHARPNESS = float(os.getenv("IMAGE_MIN_SHARPNESS", "1.0"))

# Retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))

# FAISS persistence
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(ROOT_DIR / ".cache" / "faiss_index"))
FAISS_INDEX_URL_MAP_PATH = os.getenv(
    "FAISS_INDEX_URL_MAP_PATH", str(ROOT_DIR / ".cache" / "faiss_url_map.json")
)
